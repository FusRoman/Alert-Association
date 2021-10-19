import numpy as np
import pandas as pd
from src.graph.motgraphdataset import loadSSOdata
from astropy.coordinates import SkyCoord
import astropy.units as u
import time as t
import astropy.time as astro_time


def alert_association(df1, df2, search_radius):
    c1 = SkyCoord(df1['ra'], df1['dec'], unit=u.degree)
    c2 = SkyCoord(df2['ra'], df2['dec'], unit=u.degree)

    c2_idx, c1_idx, sep2d, _ = c1.search_around_sky(c2, search_radius)

    left_assoc = df1.iloc[c1_idx]
    right_assoc = df2.iloc[c2_idx]
    return left_assoc, right_assoc, sep2d


def tracklets_associations(first_night, second_night, max_trajectory_id):

    #detect closer alert in the same night
    trackA, trackB, sep1 = alert_association(first_night, first_night, separation_in_the_same_night)
    trackA_next, trackB_next, sep2 = alert_association(second_night, second_night, separation_in_the_same_night)

    #remove identical alert from the same night association
    remove_identical_alert_idx1 = np.where(sep1.arcsecond > 0)[0]
    remove_identical_alert_idx2 = np.where(sep2.arcsecond > 0)[0]

    trackA = trackA.iloc[remove_identical_alert_idx1].reset_index(drop=True)
    trackB = trackB.iloc[remove_identical_alert_idx1].reset_index(drop=True)

    #give a trajectory id to the tracklets
    trajectory_id = np.arange(max_trajectory_id, len(trackA) + max_trajectory_id)
    max_trajectory_id += len(trackA)
    trackA['trajectory_id'] = trajectory_id
    trackB['trajectory_id'] = trajectory_id


    trackA_next = trackA_next.iloc[remove_identical_alert_idx2]
    trackB_next = trackB_next.iloc[remove_identical_alert_idx2]

    res_track_assoc = []
    for track1 in [trackA, trackB]:
        for track2 in [trackA_next, trackB_next]:
            track_left, track_right, _ = alert_association(track1, track2, max_speed_sso * u.degree)
            track_right['trajectory_id'] = track_left['trajectory_id'].values
            res_track_assoc.append(track_right)


    trajectory_df = pd.concat(res_track_assoc).drop_duplicates(['candid'])

    remain_trackA = trackA_next[~trackA_next['candid'].isin(trajectory_df['candid'])]
    remain_trackB = trackB_next[~trackB_next['candid'].isin(trajectory_df['candid'])]
    trajectory_id = np.arange(max_trajectory_id, len(remain_trackA) + max_trajectory_id)

    remain_trackA['trajectory_id'] = trajectory_id
    remain_trackB['trajectory_id'] = trajectory_id
    max_trajectory_id += len(remain_trackA)
    
    all_trajectory = pd.concat([trackA, trackB, remain_trackA, remain_trackB, trajectory_df])
    return all_trajectory, max_trajectory_id



if __name__ == "__main__":
    df_sso = loadSSOdata("../../data/month=03", "Solar System MPC", 0)
    all_night_id = np.unique(df_sso['nid'])

    first_night = df_sso[df_sso['nid'] == all_night_id[0]]
    second_night = df_sso[df_sso['nid'] == all_night_id[1]]
    third_night = df_sso[df_sso['nid'] == all_night_id[5]]

    max_trajectory_id = 0
    max_speed_sso = 0.6 #0.6 deg/day in the sky (in average)
    separation_in_the_same_night = 16*u.arcsecond #in arcsecond, separation between close observation in the same night 
                                      #that can correspond to the same object

    night_sep = all_night_id[5] - all_night_id[0] #night difference between the observations
    search_radius = max_speed_sso * night_sep #search radius if it has more than one night between the observations


    t_before = t.time()
    all_trajectory, new_max_trajectory_id = tracklets_associations(first_night, second_night, max_trajectory_id)
    print(t.time() - t_before)

    print(all_trajectory)

    