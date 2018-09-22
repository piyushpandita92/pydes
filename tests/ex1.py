"""
Test the multi-objective optimization algorithm.

"""

import matplotlib
matplotlib.use('PS')
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pydes
import numpy as np
import design
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
from example_objective_functions import ObjFunc1
import shutil

if __name__ == '__main__':
    assert len(sys.argv)==3
    sigma = sys.argv[1]
    n = int(sys.argv[2])
    out_dir = 'ex1_results_n={0:d}_sigma={1:s}'.format(n,sys.argv[1])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    dim = 2
    obj_funcs = ObjFunc1(sigma=sigma, n_samp=1)
    obj_funcs_true = ObjFunc1(sigma=sigma, n_samp=100)
    X_init = design.latin_center(n, dim, seed=123455)
    Y_init = np.array([obj_funcs(x) for x in X_init])
    X_d_true = design.latin_center(10000, dim, seed=12345)
    Y_true = np.array([obj_funcs_true(x) for x in X_d_true]) 
    ehvi_opt_bounds = ((0, 1), ) * dim
    trans_function = lambda y:y
    p = pydes.ParetoFront(X_init, Y_init, obj_funcs, obj_funcs_true, 
        Y_true=Y_true,
        ehvi_opt_bounds=ehvi_opt_bounds,
        X_design=1000,
        max_it=100,
        do_posterior_samples=True,
        gp_fixed_noise=None,
        kernel_type=GPy.kern.Matern32,
        verbosity=1,
        how='max',
        lim=None,
        pareto_how='max',
        trans_function=trans_function,
        figname=os.path.join(out_dir,'ex1'))
    p.optimize(plot=True)