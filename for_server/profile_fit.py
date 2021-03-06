import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import functions as f
from lmfit import Model

parser = argparse.ArgumentParser(description="App for dmdr calculation!")
parser.add_argument("gamma_ini", help="The model's initial gamma")
parser.add_argument("sfe", help="The model's star formation efficiency")
parser.add_argument("snap", help="The model's snapshot")
parser.add_argument("--dim", help="The fitting dimension, default 3")
parser.add_argument("--dir", help="input files' directory. Default is /golowood-homes/Dehnen-mkhalo/gamma={gamma_ini}/")

args = parser.parse_args()

gamma_ini, sfe, snap = args.gamma_ini, args.sfe, int(args.snap)

if args.dim:
    dim = int(args.dim)
else:
    dim = 3

if args.dir:
    INPUT_DIR = Path(f'{args.dir}')
else:
    INPUT_DIR = Path(f'/golowood-homes/Dehnen-mkhalo/gamma={gamma_ini}/')

OUTPUT_DIR = Path(f'./gamma={gamma_ini}/')
try:
    os.stat(OUTPUT_DIR)
except:
    os.mkdir(OUTPUT_DIR)

r_min_d = 0.05 # r min for dmdr calculation
r_max_d = 25   # r max for dmdr calculation

BOUNDS_DIR = Path(f'./bounds/gamma={gamma_ini}')
bounds = pd.read_csv(BOUNDS_DIR / f'r_bounds_gamma={gamma_ini}_{sfe}_{dim}_0.csv', delimiter=' ')
r_min_b = bounds.loc[snap, 'r_min'] # r min from boundary calculation
r_max_b = bounds.loc[snap, 'r_max'] # r max from boundary calculation


# r min and max for fitting
if np.isnan(r_min_b):
    r_min_f = r_min_d
else:
    r_min_f = np.max(r_min_d, r_min_b)

if np.isnan(r_max_b):
    r_max_f = r_max_d
else:
    r_max_f = np.min(r_max_d, r_max_b)


r_e = np.geomspace(r_min_d, r_max_d, 101)
mask_fit = ((r_e >= r_min_f) & (r_e <= r_max_f))

output = pd.DataFrame(data={'r':r_e})
popts = pd.DataFrame()
n_time = 3

output['dmdr'] = f.get_dmdr(INPUT_DIR, snap, r_e, gamma_ini, sfe, dim, n_time)

fmodel = Model(f.dmdr_profile)
fmodel.set_param_hint('ro0',   value=500,  min=1,     max=40000.0)
fmodel.set_param_hint('a',     value=1.0,  min=0.1,   max=10.0)
fmodel.set_param_hint('gamma', value=0.1,  min=1e-10, max=3.5)
fmodel.set_param_hint('beta',  value=4.0,  min=2.0,   max=7.5)
fmodel.set_param_hint('alpha', value=2.0,  min=0.5,   max=6.0)

params = fmodel.make_params()

params['dim'].vary = False
params['dim'].value = dim

# params['alpha'].vary = False
# params['alpha'].value = 2

# params['beta'].vary = False
# params['beta'].value = 4

best_fit = fmodel.fit(data=output[mask_fit]['dmdr'], params=params, r=r_e[mask_fit])

ro0, a, gamma, beta, alpha, dim = list(best_fit.params.valuesdict().values())

try:
    ro0_err   = float(best_fit.fit_report().split('\n')[12].split('+/-')[1].split('(')[0])
    a_err     = float(best_fit.fit_report().split('\n')[13].split('+/-')[1].split('(')[0])
    gamma_err = float(best_fit.fit_report().split('\n')[14].split('+/-')[1].split('(')[0])
    beta_err  = float(best_fit.fit_report().split('\n')[15].split('+/-')[1].split('(')[0])
    alpha_err = float(best_fit.fit_report().split('\n')[16].split('+/-')[1].split('(')[0])
except: 
    report = best_fit.fit_report()
    print(f"problems with split! fit_report: \n {report}")
    ro0_err = a_err = gamma_err = 'nan'

output['fit'] = f.dmdr_profile(output['r'], ro0, a, gamma, beta, alpha, dim)
output.to_csv(OUTPUT_DIR / f'fit_dmdr_gamma={gamma_ini}_{sfe}_{snap}_{dim}.csv', 
              index=False, sep=' ')

popts_columns = ['snap', 'ro0', 'a', 'gamma', 'beta', 'alpha', 
                 'ro0_err', 'a_err', 'gamma_err', 'beta_err', 'alpha_err']
popts = pd.DataFrame([[snap, ro0, a, gamma, beta, alpha, 
                       ro0_err, a_err, gamma_err, beta_err, alpha_err]], columns=popts_columns)
try:
    popts_output = pd.read_csv(OUTPUT_DIR / f'fit_popts_gamma={gamma_ini}_{sfe}_{dim}.csv', delimiter=' ', header=0)
    popts_output = popts_output.append(popts)
except:
    popts_output = popts.copy()

popts_output.to_csv(OUTPUT_DIR / f'fit_popts_gamma={gamma_ini}_{sfe}_{dim}.csv', index=False, sep=' ')
