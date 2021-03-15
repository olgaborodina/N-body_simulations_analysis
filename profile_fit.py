import argparse
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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

args = parser.parse_args()

gamma_ini, sfe, snap = args.gamma_ini, args.sfe, int(args.snap)

if args.dim:
    dim = int(args.dim)
else:
    dim = 3

INPUT_DIR = Path(f'./../gamma={gamma_ini}/')
OUTPUT_DIR = Path(f'./gamma={gamma_ini}/')
try:
    os.stat(OUTPUT_DIR)
except:
    os.mkdir(OUTPUT_DIR)

r_e = np.geomspace(15e-2, 25, 101)
output = pd.DataFrame(data={'r':r_e})
popts = pd.DataFrame()
dmdr_ = 0

for random in [11, 12, 13, 21, 22, 23, 31, 32, 33]:
    folder = Path(f'{INPUT_DIR}/run-{gamma_ini}-{sfe}-{random}')
    dmdr_ += f.get_dmdr(folder, snap, r_e, dim)

output['dmdr'] = dmdr_ / 9

fmodel = Model(f.dmdr_profile)
fmodel.set_param_hint('ro0',   value=500,  min=10,   max=30000.0)
fmodel.set_param_hint('a',     value=1,    min=0.2,  max=2.0)
fmodel.set_param_hint('gamma', value=0.3,  min=1e-9, max=2.5)
fmodel.set_param_hint('beta',  value=3.85, min=3,    max=7.1)
fmodel.set_param_hint('alpha', value=1.0,  min=0.1,  max=4)

params = fmodel.make_params()

params['dim'].vary = False
params['dim'].value = dim

best_fit = fmodel.fit(data=output['dmdr'], params=params, r=r_e)

ro0, a, gamma, beta, alpha, dim = list(best_fit.params.valuesdict().values())

ro0_err   = float(best_fit.fit_report().split('\n')[12].split('-')[1].split('(')[0])
a_err     = float(best_fit.fit_report().split('\n')[13].split('-')[1].split('(')[0])
gamma_err = float(best_fit.fit_report().split('\n')[14].split('-')[1].split('(')[0])
beta_err  = float(best_fit.fit_report().split('\n')[15].split('-')[1].split('(')[0])
alpha_err = float(best_fit.fit_report().split('\n')[16].split('-')[1].split('(')[0])

output['fit'] = f.dmdr_profile(output['r'], ro0, a, gamma, beta, alpha, dim)
output.to_csv(OUTPUT_DIR / f'fit_dmdr_gamma={gamma_ini}_{snap}_{dim}.csv', 
              index=False, sep=' ')

popts_columns = ['snap', 'ro0', 'a', 'gamma', 'beta', 'alpha', 
                 'ro0_err', 'a_err', 'gamma_err', 'beta_err', 'alpha_err']
popts = pd.DataFrame([[snap, ro0, a, gamma, beta, alpha, 
                       ro0_err, a_err, gamma_err, beta_err, alpha_err]], columns=popts_columns)
try:
    popts_output = pd.read_csv(OUTPUT_DIR / f'fit_popts_gamma={gamma_ini}_{dim}.csv', delimiter=' ', header=0)
    popts_output = popts_output.append(popts)
except:
    popts_output = popts.copy()

popts_output.to_csv(OUTPUT_DIR / f'fit_popts_gamma={gamma_ini}_{dim}.csv', index=False, sep=' ')