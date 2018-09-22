# pydes
Multi-objective Stochastic Bayesian Optimization with Quantified Uncertainties on the Pareto Front

This module is called pydes. 
The simple examples ex1.py, ex2.py provide a self explanatory overview of using pydes.
This code works for maximizing two functions (so the user would have to include that in their function object).\\

The user mainly needs to specify the objective function ```obj_func``` as an object, number of iterations (samples to be collected depending on the budget) ```max_it```, number of designs of the discretized input space (for calculating the value of the EEIHV criterion) ```X_design```. The methodology should be used with the inputs transformed to [0, 1]^{d} cube and outputs roughly normalized to a standard normal.\\

After each iteration a plot depicting the state of the Pareto Frontier is generated, this can be controlled by a make_plots flag  

More documentation to follow:

To install the package do the following:
pip install git+git://github.com/PredictiveScienceLab/GPy.git@pymc
