# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:13:47 2021

@author: Roman
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from numpy.lib.type_check import real
import pandas as pd
import time as t
import matplotlib.pyplot as plt
import os

def sep_df(x):

    ra, dec, jd = x[1], x[2], x[3]

    c1 = SkyCoord(ra, dec, unit = u.degree)

    sep = c1[0:-1].separation(c1[1:]).radian * (180/np.pi) / np.diff(jd)

    return sep

def mag_df(x):

    mag, jd = x[0], x[1]

    delta_mag = np.abs(np.divide(np.diff(mag), np.diff(jd)))

    return delta_mag

def diff_mag_df(x):
    mag_x, mag_y = x[0], x[2]

    return np.subtract(np.mean(mag_x), np.mean(mag_y))


if __name__=="__main__":

    #result = MPC.query_objects('asteroid')
    #pprint(result)

    all_df = []
    for i in range(3, 7):
        path = '../../data/month=0{}'.format(i)
        df_sso = pd.read_pickle(path)
        all_df.append(df_sso)
    

    df_sso = pd.concat(all_df)

    real_trajectory = df_sso\
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

    path = "sso_time_series_analyse/"
    i = 1
    for idx, rows in real_trajectory.iterrows():
        print(len(real_trajectory), i)
        i += 1
        if rows['candid'] >= 60:
            path_traj = path + "traj_length={}".format(rows['candid'])
            path_coord = path_traj + "/coord"
            path_mag = path_traj + "/magnitude"
            try:
                os.makedirs(path_coord)
                os.makedirs(path_mag)
            except:
                None
                #print("directory already exist")

            plt.scatter(rows['ra'], rows['dec'])
            plt.xlabel('ra')
            plt.ylabel('dec')
            plt.title('trajectory')
            plt.savefig(path_coord + "/{}".format(rows['ssnamenr'][0]))
            plt.close()

            dcmag_fid_1 = np.array(rows['dcmag'])[np.where(np.array(rows['fid']) == 1)[0]]
            dcmag_fid_2 = np.array(rows['dcmag'])[np.where(np.array(rows['fid']) == 2)[0]]
            
            plt.scatter(np.arange(len(dcmag_fid_1)), dcmag_fid_1, label='filtre 1')
            plt.scatter(np.arange(len(dcmag_fid_2)), dcmag_fid_2, label='filtre 2')
            plt.legend()
            plt.title('lightcurve')
            plt.savefig(path_mag + "/{}".format(rows['ssnamenr'][0]))
            plt.close()


    exit()
    df_sso['ssnamenr'] = df_sso['ssnamenr'].astype("string")
    print(df_sso['ssnamenr'])

    mpc_database = pd.read_json('../../data/mpc_database/mpcorb_extended.json')
    mpc_database['Number'] = mpc_database['Number'].astype("string").str[1:-1]

    print(mpc_database['Number'])
    print()
    orbit_param = df_sso.merge(mpc_database, how='inner', left_on='ssnamenr', right_on='Number')
    orbit_sep = orbit_param.explode(['sep'])
    print(orbit_param)

    #print(np.unique(orbit_sep['a']))
    #print(np.unique(orbit_sep['sep']))

    low_speed_object = orbit_sep[orbit_sep['sep'] <= 0.16]
    medium_speed_object = orbit_sep[orbit_sep['sep'] > 0.16]

    plt.hist(low_speed_object['Orbit_type'])
    plt.xticks(rotation=90)
    plt.show()

    plt.hist(medium_speed_object['Orbit_type'])
    plt.xticks(rotation=90)
    plt.show()


    """
    Ce plot permet d'observer un premier pique avant les 2 UA, il s'agit du groupe d'asteroide des Hungaria
    Le gros blobs du milieu avec trois gros piques sont les asteroides de la main belt.Notamment, le dernier pique 
    du gros blob est le groupe d'asteroide des cybèle
    Le dernier groupe proche de la main belt avant les 4 UA est le groupe des Hilda.
    Afin le dernier groupe vers les 5 UA sont les Jupiter Trojan.
    """
    plt.scatter(orbit_sep['a'], orbit_sep['sep'])
    plt.xlabel('a (UA)')
    plt.xlim(0, 6)
    plt.ylabel('velocity (deg/day)')
    plt.show()

    plt.scatter(orbit_sep['e'], orbit_sep['sep'])
    plt.xlabel('e')
    plt.ylabel('velocity (deg/day)')
    plt.show()

    plt.scatter(orbit_sep['a'], orbit_sep['e'])
    plt.xlim(0, 6)
    plt.xlabel('a (UA)')
    plt.ylabel('e')
    plt.show()

    plt.scatter(orbit_sep['i'], orbit_sep['sep'])
    plt.xlabel('i (deg)')
    plt.ylabel('velocity (deg/day)')
    plt.show()

    plt.scatter(orbit_sep['a'], orbit_sep['i'])
    plt.xlabel('a (UA)')
    plt.ylabel('i (deg)')
    plt.show()

    """
    #save separation data
    all_month = ["03", "04", "05", "06"]

    for month in all_month:

        print(month)
        path = 'solar_system_object_analyse/month=' + month


        df_sso = loadSSOdata(month, "Solar System MPC", 0).sort_values(['jd'])

        df_mpc = df_sso.groupby(['ssnamenr'])

        df_sep = df_mpc.agg({'ra':list, 'dec':list, 'jd':list}).reset_index()

        print("separation calculation running")
        df_sep['sep'] = df_sep.apply(sep_df, axis=1)

        print("save data")
        df_sep.to_pickle(path + "/mpc_separation_data")
    """


    """
    compute analysis of solar system object and save the plots
    all_month = ["03", "04", "05", "06", "07"]

    for month in all_month:
        print("current month: " + month)

        t_before = t.time()

        path = 'solar_system_object_analyse/month=' + month

        if not os.path.isdir(path):
            os.mkdir(path)

        df_sso = loadSSOdata(month, "Solar System MPC", 0).sort_values(['jd'])

        df_mpc = df_sso.groupby(['ssnamenr'])
        
        df_sep = df_mpc.agg({'ra':list, 'dec':list, 'jd':list})

        df_sep['sep'] = df_sep.apply(sep_df, axis=1)

        df_fid_1 = df_sso[df_sso['fid'] == 1]
        df_fid_2 = df_sso[df_sso['fid'] == 2]


        df_mpc_fid_1 = df_fid_1.groupby(['ssnamenr']).agg({'dcmag':list, 'jd':list})
        df_mpc_fid_2 = df_fid_2.groupby(['ssnamenr']).agg({'dcmag':list, 'jd':list})

        separate_fid = pd.merge(df_mpc_fid_1, df_mpc_fid_2, how='inner', on='ssnamenr')

        separate_fid['diff_mag_fid'] = separate_fid.apply(diff_mag_df, axis=1)

        df_mpc_fid_1['delta_mag'] = df_mpc_fid_1.apply(mag_df, axis=1)
        df_mpc_fid_2['delta_mag'] = df_mpc_fid_2.apply(mag_df, axis=1)

        plt.title('Distribution de la séparation engulaire entre deux observation consécutives des objets MPC')
        plt.ylabel('séparation angulaire normalisé en jour (jd)')
        plt.hist(df_sep.explode(['sep'])['sep'].values, bins=200, range=(0, 1))

        #plt.show()
        plt.savefig(path + '/sep')
        plt.tight_layout()
        plt.close()

        plt.title('Distribution de l\'écart de magnitude entre deux observations consécutives des objets MPC pour le filtre 1')
        plt.ylabel('magnitude normalisé en jour (jd)')
        plt.hist(df_mpc_fid_1.explode(['delta_mag'])['delta_mag'].values, bins=200, range=(0, 1))

        #plt.show()
        plt.savefig(path + '/fid1')
        plt.tight_layout()
        plt.close()

        plt.title('Distribution de l\'écart de magnitude entre deux observations consécutives des objets MPC pour le filtre 2')
        plt.ylabel('magnitude normalisé en jour (jd)')
        plt.hist(df_mpc_fid_2.explode(['delta_mag'])['delta_mag'].values, bins=200, range=(0, 1))

        #plt.show()
        plt.savefig(path + '/fid2')
        plt.tight_layout()
        plt.close()

        plt.title('Distribution de l\'écart de la magnitude moyenne entre les filtres 1 et 2')
        plt.ylabel('magnitude moyenne')
        plt.hist(separate_fid['diff_mag_fid'].values, bins=200, range=(-2, 2))

        #plt.show()
        plt.savefig(path + '/diffmag')
        plt.tight_layout()
        plt.close()

        print(t.time() - t_before)

    """
