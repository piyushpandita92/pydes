# pydes
This module is called pydes.

Multi-objective Stochastic Bayesian Optimization with Quantified Uncertainties on the Pareto Frontier.

This package was used in following paper, demonstrating the details of the methodology:


Pandita, Piyush, Ilias Bilionis, Jitesh Panchal, B. P. Gautham, Amol Joshi, and Pramod Zagade. "STOCHASTIC MULTIOBJECTIVE OPTIMIZATION ON A BUDGET: APPLICATION TO MULTIPASS WIRE DRAWING WITH QUANTIFIED UNCERTAINTIES." International Journal for Uncertainty Quantification 8, no. 3 (2018).

 
It needs the support of the following PYTHON packages.
1. pyDOE 
2. GPy
3. seaborn
4. tqdm

To install the package do the following:
pip install git+git://github.com/piyushpandita92/pydes.git  

or clone the repository and run python setup.py install.

Import the package like as follows:
 ```import pydes```

The simple examples ex1.py, ex2.py provide a self explanatory overview of using pydes.
This code works for maximizing two functions (so the user would have to include that in their function object).

The user mainly needs to specify the objective function ```obj_func``` as an object, number of iterations (samples to be collected depending on the budget) ```max_it```, number of designs of the discretized input space (for calculating the value of the EEIHV criterion) ```X_design```. 

Note: The methodology should be used with the inputs transformed to [0, 1]^{d} cube and outputs roughly normalized to a standard normal.

For sequential design  (one suggested design/experiment at a time):
Running the code: the examples in the ```tests``` directory can be called from the command line with a set of arguments as follows: python tests/ex1.py ```noise_inputs``` ```num_init_data```.

The examples have been setup with two example functions that have input variable stochasticity (the methodology is not suited just for this case of noise/uncertainty, this is one such case). The ```noise_inputs``` is basically the noise variance of an assumed Gaushesian noise in the inputs. For an unknown function the noise can come from any source and the methodology should be able to treat unequivocally.

After each iteration a plot depicting the state of the Pareto Frontier is generated, this can be controlled by a make_plots flag  

For multiple sequential designs  (multiple suggested design/experiment in a batch):

Replace the p.optimize in examples ex1.py and ex2.py with p.suggest(num_add) this returns two objects. The first one with the array of inputs of size (num_add, dimensionality) and the second with the value of the EEIHV at each of the ```num_add``` suggested inputs. ex3.py demonstrates how the methodology suggests multiple experiments. In case of a user supplied unknown function the noise argument to be given to the driver code can be a placeholder, for example:
python tests/ex1.py 0.0 ```num_init_data```.

More documentation to follow.


