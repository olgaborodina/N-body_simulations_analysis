### usage:  python r_interval.py 3 3


import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import functions as f
from csaps import csaps
from scipy import interpolate
from scipy import signal
from tqdm import trange

parser = argparse.ArgumentParser(description="App for r_min and r_max calculation!")
parser.add_argument("dim", help="The fitting dimension")
parser.add_argument("n_time", help="The number of +- averaged snapshots")

args = parser.parse_args()

dim, n_time = int(args.dim), int(args.n_time)


def r_min_calc(INPUT_DIR, gamma_ini, sfe, snap, dim, n_time):
    n_randoms = (2 * n_time + 1)
    r = []

    for random in [11]:

        if gamma_ini == '':
            folder = Path(f'{INPUT_DIR}/run-{sfe}-{random}')
        else:
            folder = Path(f'{INPUT_DIR}/run-{gamma_ini}-{sfe}-{random}')

        density_cs = pd.read_csv(folder / 'def-dc.dat', delimiter='\s+', index_col=0, header=None)
        for i in range(-n_time, n_time + 1):
            try: 
                xc, yc, zc = density_cs.loc[snap + i, 2:4]

                cluster = f.limit_by_status(folder, snap + i)
                r += f.get_r_list(cluster, xc, yc, zc, dim)

            except: 
                n_randoms -= 1
                print('uups trouble')
    if len(r) > 0:
        r_sort = np.sort(np.array(r))
        n = (np.arange(len(r_sort)) + 1) / n_randoms
        if len(r_sort[n>10]) > 0:
            r_min = r_sort[n>10][0]
        else: 
    	    print('the maximum n is', n.max())
    	    r_min = 0.5
    else: r_min = 0.5
    return r_min
    
def r_max_calc(INPUT_DIR, gamma_ini, sfe, snap, dim, n_time):
    omega    = 231.38/8173 # km/s / pc
    kappa    = 1.37 * omega
    gamma_sq = 4 * omega**2 - kappa**2
    G      = 4.3009125e-3 # pc * (km/s)^2 / Msun

    r_jacobi = []
    for random in [11]:

        if gamma_ini == '':
            folder = Path(f'{INPUT_DIR}/run-{sfe}-{random}')
        else:
            folder = Path(f'{INPUT_DIR}/run-{gamma_ini}-{sfe}-{random}')

        density_cs = pd.read_csv(folder / 'def-dc.dat', delimiter='\s+', index_col=0, header=None)

        for i in range(-n_time, n_time + 1):
            try:
                xc, yc, zc = density_cs.loc[snap + i, 2:4]
                cluster = f.limit_by_status(folder, snap + i)
                r = f.get_r_list(cluster, xc, yc, zc, dim)

                mass = cluster['M'].to_numpy()
            except: 
             	print('No such file with gamma, sfe, snapshot', gamma_ini, sfe, snap + i)
             	continue#print('Exception in r_max calculation')

            try:
                cum_mass = np.cumsum(mass[np.argsort(r)])
                rjs = (G * cum_mass / gamma_sq) ** (1/3)
                r_sorted = np.sort(r)
                diff = interpolate.interp1d(rjs - r_sorted, r_sorted)
                r_jacobi.append(float(diff(0)))

            except:
               print('Cant find Rj for gamma, sfe, snapshot', gamma_ini, sfe, snap + i)
               r_jacobi.append(rjs[-1])

    if len(r_jacobi) > 0:
        r_max = np.median(r_jacobi)
    else:
        r_max = 25
    return r_max

for gamma_ini in ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5']:
    INPUT_DIR = Path(f'/golowood-homes/Dehnen-mkhalo/gamma={gamma_ini}/')
    OUTPUT_DIR = Path(f'./bounds/gamma={gamma_ini}/')
    try:
        os.stat(OUTPUT_DIR)
    except:
        os.mkdir(OUTPUT_DIR)

    for sfe in ['0.05', '0.10', '0.20']:
        result = pd.DataFrame(columns=['snap', 'r_min','r_max'])
        for snap in trange(0, 800):
            r_min = r_min_calc(INPUT_DIR, gamma_ini, sfe, snap, dim, n_time)
            r_max = r_max_calc(INPUT_DIR, gamma_ini, sfe, snap, dim, n_time)
            result_i = pd.DataFrame(data={'snap':[snap], 'r_min': [r_min],'r_max': [r_max]})
            result = result.append(result_i)
            #print('snapshot', snap, 'is ready')
        b, a = signal.butter(8, 0.125)
        r_min_smooth = signal.filtfilt(b, a, result['r_min'])
        result['r_min_smooth'] = r_min_smooth
        b, a = signal.butter(6, 0.125)
        r_max_smooth = signal.filtfilt(b, a, result['r_max'])
        result['r_max_smooth'] = r_max_smooth
        print(f'{sfe} is ready')
        result.to_csv(OUTPUT_DIR / f'r_bounds_gamma={gamma_ini}_{sfe}_{dim}_{n_time}.csv', index=False, sep=' ')


INPUT_DIR = Path('/golowood-homes/Dehnen-mkhalo/Plummer/')

#for gamma_ini in ['']:
#    OUTPUT_DIR = Path(f'./bounds/gamma={gamma_ini}/')
#    try:
#        os.stat(OUTPUT_DIR)
#    except:
#        os.mkdir(OUTPUT_DIR)
#      
#    for sfe in ['0.15', '0.20', '0.25']:
#        result = pd.DataFrame(columns=['snap', 'r_min','r_max'])
#        for snap in trange(0, 1000):
#            r_min = r_min_calc(INPUT_DIR, gamma_ini, sfe, snap, dim, n_time)
#            r_max = r_max_calc(INPUT_DIR, gamma_ini, sfe, snap, dim, n_time)
#            result_i = pd.DataFrame(data={'snap':[snap], 'r_min': [r_min],'r_max': [r_max]})
#            result = result.append(result_i)
#           # print('snapshot', snap, 'is ready')
#
#        b, a = signal.butter(8, 0.125)
#        r_min_smooth = signal.filtfilt(b, a, result['r_min'])
#        result['r_min_smooth'] = r_min_smooth
#        b, a = signal.butter(6, 0.125)
#        r_max_smooth = signal.filtfilt(b, a, result['r_max'])
#        result['r_max_smooth'] = r_max_smooth
#        print(f'{sfe} is ready')
#        result.to_csv(OUTPUT_DIR / f'r_bounds_gamma={gamma_ini}_{sfe}_{dim}_{n_time}.csv', index=False, sep=' ')




