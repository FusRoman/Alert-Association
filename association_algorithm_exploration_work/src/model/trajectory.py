# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:51:27 2021

@author: Roman
"""

import pandas as pd
import numpy as np
import time as t
import src.graph.motgraphdataset as motdataset
from spektral.transforms import AdjToSpTensor
from spektral.transforms.normalize_one import NormalizeOne

def mean_edge_pred_window(all_graph):
    """
    Compute the prediction mean of duplicates edges. A sliding windows with overlappping is applied during the graph building process which produce duplicates edges

    Parameters
    ----------
    all_graph : dataframe
        contains all edges features plus the prediction column return from MOTModel
    
    Return
    ------
    mean_merge : dataframe
        a dataframe where each edges have his mean prediction over all his duplicates edges
    """

    frames = [g.edge_info for g in all_graph]
    df_concat = pd.concat(frames).reset_index(drop=True)
    df_gb_test = df_concat.groupby(['candid_x', 'candid_y']).agg({
        'pred': 'mean'
    })
    df_gb_test = df_gb_test['pred']
    
    mean_merge = pd.merge(df_concat, df_gb_test, left_on=['candid_x', 'candid_y'], 
                          right_index=True, how='left')
    
    mean_merge['pred_x'] = mean_merge['pred_x'].mask(
        np.logical_not(pd.isna(mean_merge['pred_y'])), mean_merge['pred_y'])
    
    return mean_merge.drop(columns=['pred_y']).rename(columns={'pred_x': 'pred'}).drop_duplicates()


def greedy_constraints(edge_mean_pred):
    """
    Keep only the max egde prediction for the past edge and futur edge for all nodes one by one

    Parameters
    ----------
    edge_mean_pred : dataframe
        results from mean_edge_pred_window function

    Return
    ------
    backward_trajectories : dataframe
        dataframe where each node have only one edge to the past and one edge to the futur and each edge is the max among all the others.
    """
    active_edge = edge_mean_pred[edge_mean_pred['pred'] >= 0.5]
    forward = active_edge[active_edge['nid_x'] < active_edge['nid_y']]
    backward = active_edge[active_edge['nid_x'] > active_edge['nid_y']]
    swap_backward = backward.reindex(columns=["candid_y", "nid_y", "ssnamenr_y",
                             "candid_x", "nid_x", "ssnamenr_x",
                             "label", "pred"])
    swap_backward = swap_backward.rename(columns={
        "candid_x": "candid_y",
        "nid_x": "nid_y",
        "ssnamenr_x": "ssnamenr_y",
        "candid_y": "candid_x",
        "nid_y": "nid_x",
        "ssnamenr_y": "ssnamenr_x"
    })
    forward = pd.concat([forward, swap_backward])
    forward[forward['ssnamenr_x'] == forward['ssnamenr_y']]
    gb_max = forward.groupby(['candid_x'])['pred'].idxmax()
    forward_trajectories = forward.loc[gb_max]
    break_constraint = forward_trajectories.groupby(['candid_y'])['pred'].idxmax()
    backward_trajectories = forward_trajectories.loc[break_constraint]
    return backward_trajectories

def set_track_id(hypotheses_edge, row, track_id, track_id_list):
    """
    Assign an id to all track return by greedy_constraints function
    Perform recursion over the predict association in the dataframe.
    Parameters
    ----------
    hypotheses_edge : dataframe
        results from greedy_constraints function
    row : A Series column
        assigns a trajectory id to this row if they already don't have it.
    track_id : integer
        the current trajectory id assign to this trajectory.
    track_id_list : integer list
        all trajectories id already given to previous trajectories

    Return
    ------
    track_id_list : integer list
        list of all trajectories id already given
    """
    index = row[0]
    if track_id_list[index] == -1:
        track_id_list[index] = track_id
        next_hop = row[4]
        next_row = list(hypotheses_edge[hypotheses_edge['candid_x'] == next_hop].itertuples())
        if len(next_row) > 0:
            return set_track_id(hypotheses_edge, next_row[0], track_id, track_id_list)
        else:
            return track_id_list
    else:
        return track_id_list
    
    
def build_trajectories(active_edge):
    """
    Build trajectories from all active edge return by MOTModel. An edge is set to active if his prediction is greater than 0.5 .
    Trajecories id assignation is perform inplace in the active edge dataframe

    Parameters
    ----------
    active_edge : dataframe
        all edges that have their predictions above 0.5
    
    Return
    ------
    track_id_list : integer list
        list of all trajectories id already given
    """
    track_id_list = np.zeros(len(active_edge)) + (-1)
    
    track_id = 0
    for row in active_edge.itertuples():
        track_id_list = set_track_id(active_edge, row, track_id, track_id_list)
        track_id += 1
    return track_id_list


def trajectory_metrics(trajectories):
    """
    Compute metrics to measure the trajectories prediction performance

    Parameters
    ----------
    trajectories : dataframe
        dataframe which contains all features edges and trajectories id computed by build_trajectories.
    
    Return
    ------
    metrics : dict
        a dictionnary which contains two trajectories metrics

            accuracy measure the mean of the number of positive label in trajectories
            consistency measure the mean of the number of mpc name in trajectories

            note : consistency seems more interesting than accuracy
    """
    all_track_id = np.unique(trajectories['track_id'])
    
    traj_sum = 0
    change_mpc = 0
    for track_id in all_track_id:
        test_traj = trajectories[trajectories["track_id"] == track_id]
        change_mpc += (1 / len(np.union1d(test_traj['ssnamenr_x'], test_traj['ssnamenr_y']))) * 100
        traj_sum += (test_traj['label'].sum() / len(test_traj) * 100)
        
    return {"accuracy" : traj_sum / len(all_track_id),
            "consistency" : change_mpc / len(all_track_id)}


if __name__ == "__main__":
    print("test")
    
    t_before = t.time()
    # AdjToSpTensor()
    tr_dataset = motdataset.MOTGraphDataset("../../../data/month=03", 'Solar System MPC', 15, window_params=(5, 2),
                                transforms=[motdataset.EdgeNormalizeOne(), NormalizeOne(), AdjToSpTensor()])
    print("dataset construct time: ", t.time() - t_before)
    
    
    for g in tr_dataset:
        edge_info = g.edge_info
        edge_info['pred'] = np.random.random_sample((len(g.y),))
        
    t_before = t.time()
    edge_mean_pred = mean_edge_pred_window(tr_dataset)
    print(t.time() - t_before)
    
    t_before = t.time()
    df_edge = greedy_constraints(edge_mean_pred)
    df_edge = df_edge.reset_index(drop=True)
    print(t.time() - t_before)
    
    trajectories = build_trajectories(df_edge)
    
    df_edge['track_id'] = trajectories
    
    metrics_res = trajectory_metrics(df_edge)
    
    print(metrics_res)