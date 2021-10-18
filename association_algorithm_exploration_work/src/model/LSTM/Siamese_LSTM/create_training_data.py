#goal of this lstm
#return a score between two tracklets, close observation during the same exposition.
#highest is the score, better is the association
#use a siamese LSTM network to achieved goal

import numpy as np
from numpy.lib.type_check import real
from src.graph.motgraphdataset import loadSSOdata
import pandas as pd
from src.graph.motgraphdataset import sliding_window
import time as t




if __name__ == "__main__":

    df_sso = loadSSOdata("..\..\..\..\data\month=03", "Solar System MPC", 0)
    df_sso2 = loadSSOdata("..\..\..\..\data\month=04", "Solar System MPC", 0)
    df_sso3 = loadSSOdata("..\..\..\..\data\month=05", "Solar System MPC", 0)
    df_sso4 = loadSSOdata("..\..\..\..\data\month=06", "Solar System MPC", 0)
    #df_sso5 = loadSSOdata("..\..\..\..\data\month=07", "Solar System MPC", 0)

    asteroid_alert = pd.concat([df_sso, df_sso2, df_sso3, df_sso4])

    real_trajectory = asteroid_alert\
        .groupby(['ssnamenr'])\
            .agg(
                    {
                        'ra':list, 
                        'dec':list, 
                        'dcmag':list, 
                        'fid':list,
                        'nid':list,
                        'jd':list,
                        'ssnamenr':list, 
                        'candid':lambda x: len(list(x))
                    }
                )

    print(np.unique(real_trajectory['candid']))

    for traj_length in range(3, 20):
        print("traj_length: {}".format(traj_length))
        t_before = t.time()
        trajectory = []
        observation = []

        current_trajectory = real_trajectory[real_trajectory['candid'] >= traj_length]
        print("number of trajectory: {}".format(len(current_trajectory)))

        for _, rows in current_trajectory.iterrows():
            windows = sliding_window(np.arange(rows['candid']), traj_length, 2)
            ra = rows['ra']
            dec = rows['dec']
            dcmag = rows['dcmag']
            fid = rows['fid']
            nid = rows['nid']
            jd = rows['jd']
            ssnamenr = rows['ssnamenr']
            for windows_idx in windows:
                trajectory_sequence = []
                for i in windows_idx[:-1]:
                    trajectory_sequence.append([ra[i], dec[i], dcmag[i], fid[i], nid[i], jd[i], ssnamenr[i]])
                last_value = windows_idx[-1]
                observation.append([
                    ra[last_value], 
                    dec[last_value], 
                    dcmag[last_value], 
                    fid[last_value], 
                    nid[last_value], 
                    jd[last_value], 
                    ssnamenr[last_value]
                    ])
                trajectory.append(trajectory_sequence)

        traj = np.array(trajectory)

        true_obs = np.array(observation)

        #random sample alerts from observation datatset to construct false association
        false_obs = np.array(asteroid_alert.sample(n=np.shape(traj)[0])[['ra', 'dec', 'dcmag', 'fid', 'nid', 'jd', 'ssnamenr']])

        #remove true association from false association created by random sampling of trajectory dataframe
        true_in_false = traj[:, :, -2][:, 0] == false_obs[:, -2]
        keep_false = np.where(true_in_false == False)[0]
        false_obs = false_obs[keep_false]
        false_traj = traj[keep_false]
        
        traj_dataset = np.concatenate((traj, false_traj))
        obs_dataset = np.concatenate((true_obs, false_obs))
        
        # 0 : trajectory and observation is associated
        # 1 : trajectory and observation should not be associated
        true_label = np.zeros(np.shape(true_obs)[0])
        false_label = np.ones(np.shape(false_obs)[0])

        label_dataset = np.concatenate((true_label, false_label))

        print(np.shape(traj_dataset))
        print(np.shape(obs_dataset))
        print(np.shape(label_dataset))
        print("elapsed_time: {}".format(t.time() - t_before))

        np.save("data/trajectory_dataset_length={}".format(traj_length), traj_dataset[:, :, :-1])
        np.save("data/observation_dataset_length={}".format(traj_length), obs_dataset[:, :-1])
        np.save("data/label_dataset_length={}".format(traj_length), label_dataset)

        print()

    


