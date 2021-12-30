
# coding: utf-8

# In[78]:


import numpy as np
import numpy.linalg as LA
import pandas as pd
import copy
import os
import time


# In[124]:


def hr_file_to_df_hop_wiht_band_index( hr_file, n_skip ):
    col_name_list = ["a1","a2","a3","n_band1","n_band2","hop_real","hop_imag"]
    df_hop = pd.read_table(hr_file, skiprows=[ i for i in range(0,n_skip)], header=None,
                       sep="\s+", names= col_name_list)
    df_hop["hop"] = df_hop["hop_real"] + 1j*df_hop["hop_imag"]
    return df_hop.drop( ["hop_real","hop_imag"], axis=1 )

def band_index_to_sublatt_orbital_spin_indexes( df_hop, orbital_list ):
    df_hop["n_band1"] -= 1
    df_hop["n_band2"] -= 1
    df_hop["sub1"] = df_hop["n_band1"]//( 2*len(orbital_list) )
    df_hop["sub2"] = df_hop["n_band2"]//( 2*len(orbital_list) )
    df_hop["orb_index1"] = df_hop["n_band1"]%( 2*len(orbital_list) )//2
    df_hop["orb_index2"] = df_hop["n_band2"]%( 2*len(orbital_list) )//2
    df_hop["spin_index1"] = df_hop["n_band1"]%2
    df_hop["spin_index2"] = df_hop["n_band2"]%2
    f_orb = lambda index: orbital_list[index]
    df_hop["orb1"] = df_hop["orb_index1"].map(f_orb)
    df_hop["orb2"] = df_hop["orb_index2"].map(f_orb)
    f_spin = lambda index: "up" if index==0 else "down"
    df_hop["spin1"] = df_hop["spin_index1"].map(f_spin)
    df_hop["spin2"] = df_hop["spin_index2"].map(f_spin)
    return df_hop.drop( ["orb_index1","orb_index2","spin_index1","spin_index2"], axis=1 )

def add_distance_col( df_hop, bravais, unitcell ):
    f_r_unitcell = lambda index: unitcell[index]
    f_r_bravais = lambda index, n_bra: index*bravais[n_bra]
    f_distance = lambda vec: LA.norm(vec,2)
    df_hop["displace"] = df_hop["sub2"].map(f_r_unitcell) - df_hop["sub1"].map(f_r_unitcell)
    df_hop["displace"] += df_hop["a1"].apply( f_r_bravais, args=(0,) ) + df_hop["a2"].apply( f_r_bravais, args=(1,) ) + df_hop["a3"].apply( f_r_bravais, args=(2,) )
    df_hop["distance"] = df_hop["displace"].map(f_distance)
    return df_hop.drop(["displace"],axis=1)

def hr_file_to_df_hop( hr_file, n_skip, bravais, unitcell, orbital_list ):
    df_hop = hr_file_to_df_hop_wiht_band_index( hr_file, n_skip )
    df_hop = band_index_to_sublatt_orbital_spin_indexes( df_hop, orbital_list )
    return add_distance_col( df_hop, bravais, unitcell )

def df_hop_to_hr_file( filename, df_hop ):
    df_out = df_hop[ ["a1","a2","a3","n_band1","n_band2","hop"] ]
    df_out["hop_real"] = np.real( df_out["hop"] )
    df_out["hop_imag"] = np.imag( df_out["hop"] )
    df_out["n_band1"] += 1
    df_out["n_band2"] += 1
    df_out = df_out.drop(["hop"],axis=1)
    df_out.to_csv(filename,index=False,sep=" ")
    return
