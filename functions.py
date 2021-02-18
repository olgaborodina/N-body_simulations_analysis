import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.pyplot import cm

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import integrate
from csaps import csaps

from pathlib import Path

def get_filename(snapshot):
    snapstring = str(int(snapshot))
    while len(snapstring) < 6:
        snapstring = '0' + snapstring

    return snapstring + '.dat'

def king_profile(r, r_tidal, r_center, k):
    profile = k * (1 / (np.sqrt(1 + (r / r_center) ** 2)) -
                     1 / (np.sqrt(1 + (r_tidal / r_center) ** 2))) ** 2
    profile[r > r_tidal] = 0
    
    return profile

def dehnen_profile_2D(r, ro0, a, gamma):
    zeta = np.geomspace(1e-10, 30000, 10001)
    eta = r / a
    ETA, ZETA = np.meshgrid(eta, zeta)
    S = integrate.simps(np.power(ZETA ** 2 + ETA ** 2, - gamma/2) * np.power(1 + np.sqrt(ZETA ** 2 + ETA ** 2), gamma - 4), x=zeta, axis=0)
    return 2 * ro0 * a * S

def dehnen_profile_3D(r, ro0, a, gamma):
    return ro0 * (r / a) ** (- gamma) * (1 + r / a) ** (gamma - 4)

def dmdr_profile_2D(r, ro0, a, gamma):
    return dehnen_profile_2D(r, ro0, a, gamma) * 2 * np.pi * r

def dmdr_profile_3D(r, ro0, a, gamma):
    return dehnen_profile_3D(r, ro0, a, gamma) * 4 * np.pi * r ** 2

def get_dmdr(folder, i, r_e):
    
    names = np.array(['M', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'M_ini', 'Z', 'Nan', 'Event',
                  'M_', 'Nan3', 'Nan4', 'Nan5', 'Nan6', 'Nan7', 'Nan8'])
    
    cluster = pd.read_csv(folder / get_filename(i), 
                          index_col=0, delimiter='\s+', header=3, names=names)
    cluster = normalize_cluster(cluster)

    density_cs = pd.read_csv(folder / 'def-dc.dat', delimiter='\s+', index_col=0, header=None)
    xc, yc, zc = density_cs.iloc[i, 1:4]
    
    r_sort = np.sort(np.sqrt((cluster['x'] - xc) ** 2 + 
                             (cluster['y'] - yc) ** 2 + 
                             (cluster['z'] - zc) ** 2))
    
    n = np.arange(len(r_sort)) + 1
    n_smoothed = csaps(r_sort, n, smooth=1 - 1e-4)
    dmdr = n_smoothed(r_e, 1)
    return dmdr


def normalize_cluster(cluster):
    
    R_norm = 1.2262963200 #pc
    V_norm = 4.587330615 #km/s
    M_norm = 6.0e3 #Msun
    
    cluster['x'] *= R_norm
    cluster['y'] *= R_norm
    cluster['z'] *= R_norm

    cluster['vx'] *= V_norm
    cluster['vy'] *= V_norm
    cluster['vz'] *= V_norm

    cluster['M'] *= M_norm
    cluster['M_ini'] *= M_norm
    
    return cluster


def spherical_coords(cluster, density_cs, snapshot):
    xc, yc, zc = density_cs.iloc[snapshot, 1:4]
    vxc, vyc, vzc = density_cs.iloc[snapshot, 4:7]
    
    r = np.sqrt((cluster['x'] - xc) ** 2 + (cluster['y'] - yc) ** 2 + (cluster['z'] - zc) ** 2)
    phi = np.arctan2((cluster['y'] - yc), (cluster['x'] - xc))
    theta = np.arccos((cluster['z'] - zc) / r)
    
    vr = (cluster['vx'] - vxc) * np.cos(phi) * np.sin(theta) + \
         (cluster['vy'] - vyc) * np.sin(phi) * np.sin(theta) + \
         (cluster['vz'] - vzc) * np.cos(theta)
    return r, vr


def cluster_center_coords(cluster, density_cs, snapshot):
    xc, yc, zc = density_cs.iloc[snapshot, 1:4]
    vxc, vyc, vzc = density_cs.iloc[snapshot, 4:7]
    
    r = np.sqrt((cluster['x'] - xc) ** 2 + (cluster['y'] - yc) ** 2 + (cluster['z'] - zc) ** 2)
    
    alpha = np.arctan2(yc, xc)
    
    x_new =   np.cos(alpha) * (cluster['x'] - xc) + np.sin(alpha) * (cluster['y'] - yc)
    
    vx_new =   np.cos(alpha) * (cluster['vx'] - vxc) + np.sin(alpha) * (cluster['vy'] - vyc)
    vy_new = - np.sin(alpha) * (cluster['vx'] - vxc) + np.cos(alpha) * (cluster['vy'] - vyc)
    vz_new =   cluster['vz'] - vzc
    
    plt.hist(vx_new * abs(x_new) / x_new, bins=250, label=snapshot)
    plt.xlim(-30,30)
    plt.legend()
    plt.show()

    return r, vx_new