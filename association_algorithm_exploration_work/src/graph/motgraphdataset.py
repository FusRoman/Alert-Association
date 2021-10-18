# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:13:47 2021

@author: Roman
"""


from spektral.transforms import AdjToSpTensor
from spektral.data import Dataset
from spektral.transforms.normalize_one import NormalizeOne
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from astropy.coordinates import SkyCoord
import astropy.units as u
from src.graph.motgraph import MOTGraph
import time as t


def sliding_window(base_value, window_size = 4, overlap = 2, copy = False):
    """
    build an array containing multiple view of base_value in order to create a sliding window with overlap

    Parameters
    ----------
    base_value : numpy array
        values used to make the window.
    window_size : int, optional
        size of the window. The default is 4.
    overlap : int, optional
        number of value in the overlaping gap. The default is 2.
    copy : bool, optional
        return a copy of the base_value or not. The default is False.

    Returns
    -------
    numpy array
        multiple view that compose the window.

    """
    sh = (base_value.size - window_size + 1, window_size)
    st = base_value.strides * 2
    view = np.lib.stride_tricks.as_strided(base_value, strides = st, shape = sh)[0::overlap]
    if copy:
        return view.copy()
    else:
        return view
    
    
def loadSSOdata(path, _class, point_limit):
    """
    load solar system alerts from local file

    Parameters
    ----------
    month : string
        a string number used to specify which file will be loaded.
    _class : string
        specify which class object wil be loaded, values can only be 'Solar System MPC' or 'Solar System Candidate'.
    point_limit : int
        a value to limit the number of alerts loaded by taken only the object seen more than point_limit times, work only when _class is 'Solar System MPC'.

    Returns
    -------
    dataframe
        all alerts seen in the month, belonging to _class and seen more than point_limit times 

    """
    
    df_sso = pd.read_pickle(path)
    df_class = df_sso[df_sso['fink_class'] == _class]
    if _class == 'Solar System MPC':
        mpc_trajectory = df_class.groupby(['ssnamenr']).count()
        mpc_index = mpc_trajectory[mpc_trajectory['ra'] >= point_limit].index
        feature = ['ra', 'dec', 'jd', 'nid', 'dcmag', 'fid', 'ssnamenr', 'candid']
        return df_class[df_class['ssnamenr'].isin(mpc_index)][feature]
    else:
        feature = ['ra', 'dec', 'jd', 'nid', 'dcmag', 'fid', 'candid']
        return df_class[feature]

class EdgeNormalizeOne:
    r"""
    Normalizes the edge attributes by dividing each row by its sum, so that it
    sums to 1:
    $$
        \X_i \leftarrow \frac{\X_i}{\sum_{j=1}^{N} \X_{ij}}
    $$
    """

    def __call__(self, graph):
        e_sum = np.sum(graph.e, -1)
        e_sum[e_sum == 0] = 1
        graph.e = graph.e / e_sum[..., None]

        return graph

class MOTGraphDataset(Dataset):
    def __init__(self, date, load_candidates, lightcurves_point_limit, window_params = None, **kwargs):
        """
        Build graph dataset from local solar system alert dataset

        Parameters
        ----------
        date : string
            a string number used to specify which file will be loaded.
        load_candidates : string
            specify which class object wil be loaded, values can only be 'Solar System MPC' or 'Solar System Candidate'.
        lightcurves_point_limit : int
            a value to limit the number of alerts loaded by taken only the object seen more than point_limit times, work only when _class is 'Solar System MPC'.
        window_params : int tuple, optional
            parameter of the window, first is size, second is overlap. The default is None.

        Returns
        -------
        None.

        """
        self.date = date
        self.lcpl = lightcurves_point_limit
        self.load_candidates = load_candidates
        self.window_params = window_params
        
        super().__init__(**kwargs)
    
    def read(self):
        """
        method call by the class internally, perform file reading and graph building in order to create graph dataset

        Returns
        -------
        output : graph list
            all the graph build from the overlaping window.

        """
        print("reading data...")
        output = []
        
        df_sso = loadSSOdata(self.date, self.load_candidates, self.lcpl)
        
        print("number of sso_alert remaining after limitation by number of point in lightcurves: {}"\
             .format(len(df_sso)))
    
        nid = np.unique(df_sso['nid'])
        window = 10
        overlap = 5
        if self.window_params is not None:
            window, overlap = self.window_params
        frames_window = sliding_window(nid, window, overlap)
        
        print("construct graph by overlapping window on night id")
        print("number of graph: {}".format(len(frames_window)))
        nb_graph = 1
        
        for frames in frames_window:
            df_frames = df_sso[df_sso['nid'].isin(frames)]
            
            df_frames = df_frames.assign(candid_idx=pd.Series(np.arange(len(df_frames))).values)
            df_frames = df_frames.assign(label=pd.Series(np.zeros(len(df_frames))).values)

            tmp_df = pd.merge(df_frames, df_frames, on='label')
            graph_prune = tmp_df[(tmp_df['candid_x'] != tmp_df['candid_y'])\
                               & (tmp_df['nid_x'] != tmp_df['nid_y'])\
                                 
                               & (((tmp_df['dcmag_x'] - tmp_df['dcmag_y']) / (tmp_df['jd_x'] - tmp_df['jd_y'])) <= 1.0)
                                ]
            
            del tmp_df
            
            ra_x, dec_x = np.array(graph_prune['ra_x']), np.array(graph_prune['dec_x'])
            ra_y, dec_y = np.array(graph_prune['ra_y']), np.array(graph_prune['dec_y']) 
            
            c1 = SkyCoord(ra_x, dec_x, unit = u.degree)
            c2 = SkyCoord(ra_y, dec_y, unit = u.degree)

            alerts_sep = c1.separation(c2).degree

            graph_prune['alert_sep'] = alerts_sep

            graph_prune = graph_prune[graph_prune['alert_sep'] <= 0.8]        
            
            print("constructing graph nb {} with {} nodes and {} edges"\
                  .format(nb_graph, len(df_frames), len(graph_prune)))
            
            # take edges where extremity nodes are the same mpc object
            same_mpc = graph_prune[graph_prune['ssnamenr_x'] == graph_prune['ssnamenr_y']]
            # take edges where the left node have been created before the right node
            forward_same_mpc = same_mpc[same_mpc['nid_x'] < same_mpc['nid_y']]
            # take only one edge if multiple exists
            idx_label = forward_same_mpc.groupby(['ssnamenr_x', 'nid_x'])['nid_y'].idxmin()
            # create the training label
            graph_prune.loc[same_mpc.loc[idx_label].index, 'label'] = 1
                        
            edge_label = graph_prune['label'].to_numpy().astype(np.int32)
            
            row = list(graph_prune['candid_idx_x'])
            col = list(graph_prune['candid_idx_y'])
            data = np.ones(len(col))
            sparse_adj_mat = coo_matrix((data, (row, col)), shape=(len(df_frames), len(df_frames))).tocsr()
            node_feature = df_frames[['ra', 'dec', 'jd', 'dcmag', 'nid', 'fid']].to_numpy()
            
            edge_feature = np.c_[np.array(np.abs(graph_prune['dcmag_x'] - graph_prune['dcmag_y'])),
            np.array(graph_prune['jd_x'] - graph_prune['jd_y']),
            np.array(graph_prune['alert_sep']),
            np.array(graph_prune['nid_x'] - graph_prune['nid_y'])]

            past_index = np.where(edge_feature[:, -1] > 0)[0]
            past_index = past_index.reshape((len(past_index), 1))
            futur_index = np.where(edge_feature[:, -1] < 0)[0]
            futur_index = futur_index.reshape((len(futur_index), 1))
            
            if self.load_candidates == 'Solar System MPC':
                graph_prune = graph_prune[['candid_x', 'nid_x', 'ssnamenr_x',
                                           'candid_y', 'nid_y', 'ssnamenr_y', 'label']]
            else:
                graph_prune = graph_prune[['candid_x', 'nid_x', 'candid_y', 'nid_y']]

            
            g = MOTGraph(node_feature, sparse_adj_mat, edge_feature, edge_label.reshape((len(edge_label), 1)),
                        graph_prune, past_index, futur_index)
            
            output.append(g)
            nb_graph += 1
        
        print()
        print("end reading")
        return output
    
    
    
if __name__ == "__main__":
    print("test the motgraphdataset building...\n can take some time.")


    t_before = t.time()
    tr_dataset = MOTGraphDataset("../../../data/month=03", 'Solar System MPC', 18, window_params=(5, 2),
                                transforms=[EdgeNormalizeOne(), NormalizeOne(), AdjToSpTensor()])
    print("tr_dataset construct time: ", t.time() - t_before)
    
    print("number of graph in the dataset :", len(tr_dataset))

    print("label sum (number of true label in each graph of the dataset):")
    for g in tr_dataset:
        print(g.y.sum())
    