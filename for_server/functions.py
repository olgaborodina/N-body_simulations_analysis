import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def dehnen_profile(r, ro0, a, gamma, beta, alpha, dim):
    if dim == 2:
        zeta = np.geomspace(1e-10, 30000, 10001)
        eta = r / a
        ETA, ZETA = np.meshgrid(eta, zeta)
        S = integrate.simps(np.power(ZETA ** 2 + ETA ** 2, - gamma / 2) * 
                            np.power(1 + (ZETA ** 2 + ETA ** 2) ** (alpha / 2), (gamma - beta) / alpha),
                            x=zeta, axis=0)
        return 2 * ro0 * a * S
    elif dim == 3:
        return ro0 * (r / a) ** (- gamma) * (1 + (r / a) ** alpha) ** ((gamma - beta) / alpha)
    else: raise ValueError ('Wrong dimension')
        

def dmdr_profile(r, ro0, a, gamma, beta, alpha, dim):
    if dim == 2:
        return dehnen_profile(r, ro0, a, gamma, beta, alpha, dim) * 2 * np.pi * r
    elif dim == 3:
        return dehnen_profile(r, ro0, a, gamma, beta, alpha, dim) * 4 * np.pi * r ** 2
    else: raise ValueError ('Wrong dimension')

def dn_profile(r, ro0, a, gamma, beta, alpha, dim):
    r_mean, dr = np.diff(r)/2 + r[:-1], np.diff(r)
    rho = dehnen_profile(r_mean, ro0, a, gamma, beta, alpha, dim) 
    if dim == 2:
        return rho * 2 * np.pi * r_mean * dr
    elif dim == 3:
        return rho * 4 * np.pi * r_mean ** 2 * dr
    else: raise ValueError ('Wrong dimension')

def log_dn_profile(r, ro0, a, gamma, beta, alpha, dim):
    return np.log10(dn_profile(r, ro0, a, gamma, beta, alpha, dim))
        
def log_dmdr_profile(r, ro0, a, gamma, beta, alpha, dim):
    return np.log10(dmdr_profile(r, ro0, a, gamma, beta, alpha, dim))

def get_r_list(cluster, xc, yc, zc, dim):
    if dim == 2:
        return list(np.sqrt((cluster['x'] - xc) ** 2 + 
                            (cluster['y'] - yc) ** 2))
    elif dim == 3:
        return list(np.sqrt((cluster['x'] - xc) ** 2 + 
                            (cluster['y'] - yc) ** 2 + 
                            (cluster['z'] - zc) ** 2))
    else: raise ValueError ('Wrong dimension')

def get_dmdr(INPUT_DIR, snap, r_e, gamma_ini, sfe, dim, n_time):

    n_randoms = 9 * (2 * n_time + 1)
    r =[]

    for random in [11, 12, 13, 21, 22, 23, 31, 32, 33]:

        if gamma_ini == '':
            folder = Path(f'{INPUT_DIR}/run-{sfe}-{random}')
        else:
            folder = Path(f'{INPUT_DIR}/run-{gamma_ini}-{sfe}-{random}')

        density_cs = pd.read_csv(folder / 'def-dc.dat', delimiter='\s+', index_col=0, header=None)
        for i in range(-n_time, n_time + 1):
            try:
                xc, yc, zc = density_cs.loc[snap + i, 2:4]
                cluster = limit_by_status(folder, snap + i)
                r += get_r_list(cluster, xc, yc, zc, dim)
            except: 
                n_randoms -= 1
    r_sort = np.sort(np.array(r))
    n = (np.arange(len(r_sort)) + 1) / n_randoms
    n_smoothed = csaps(r_sort, n, smooth=1 - 1e-4)
    dmdr= n_smoothed(r_e, 1)

    mask_negative = (dmdr <= 0)
    dmdr[mask_negative] = 1e-6
    
    return dmdr, r_mean

def get_dn(INPUT_DIR, snap, random, r_e, gamma_ini, sfe, dim, n_time):

    n_randoms = 2 * n_time + 1
    r =[]

    if gamma_ini == '':
        folder = Path(f'{INPUT_DIR}/run-{sfe}-{random}')
    else:
        folder = Path(f'{INPUT_DIR}/run-{gamma_ini}-{sfe}-{random}')

    for i in range(-n_time, n_time + 1):
        try:
            density_cs = pd.read_csv(folder / 'def-dc.dat', delimiter='\s+', index_col=0, header=None)
            xc, yc, zc = density_cs.loc[snap + i, 2:4]
            cluster = limit_by_status(folder, snap + i)
            r += get_r_list(cluster, xc, yc, zc, dim)
        except: 
            n_randoms -= 1

    dn, _ = np.histogram(r, r_e)
    r_mean = np.diff(r_e) / 2 + r_e[:-1]
    dn = dn / n_randoms
    
    mask_negative = (dn == 0)
    dn[mask_negative] = 1e-6
    
    return dn, r

def limit_by_status(folder, i):
    names = np.array(['M', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'M_ini', 'Z', 'Nan', 'Event',
                  'M_', 'Nan3', 'Nan4', 'Nan5', 'Nan6', 'Nan7', 'Nan8'])
    names_vir = np.array(['index', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'd', 'M', 'n_cum', 'Status', 
                      'U', 'K', 'vir', 'count', 'pot', 'ax', 'ay', 'az', 'potc', 'pot_ext','ax_ext', 'ay_ext',
                      'az_ext', 'potc_extc'])
    cluster = pd.read_csv(folder / get_filename(i), 
                          index_col=0, delimiter='\s+', header=2, names=names)
    cluster = normalize_cluster(cluster)
    
    cluster_vir = pd.read_csv(folder / Path(get_filename(i).split('.')[0] + '.vir'), 
                      index_col=0, delimiter='\s+', names=names_vir)
    
    assert len(cluster_vir) == len(cluster), '.dat and .vir files have different size'

    mask_status = (cluster_vir.sort_values('index')['Status'] > 1)
    
    return cluster[mask_status]
    

def normalize_cluster(cluster):
    
    R_norm =  1.2262963200 # 0.93985917 1.2262963200  #pc
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
