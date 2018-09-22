"""
Implementation of the algorithm that gradually disovers the Pareto front.

"""

import numpy as np
import GPy
from scipy.optimize import minimize
import design
import tqdm
from . import get_idx_of_observed_pareto_front
from . import ehvi_2d_func
from . import DistributedObject
from . import ParallelizedGPRegression
from . import distributed_xrange
from . import reduce_max
from . import parallel_eval
from . import get_yref
from . import get_empirical_attainment_function
from . import get_Vorobev_expectation
from . import get_symmetric_deviation_function
#from mpi4py import MPI as mpi
import copy

__all__ = ['ParetoFront']


class ParetoFront(DistributedObject):

    """
    Initialize the class.

    :param X:                   Input points - num_points x num_dim
    :param Y:                   Objectives - num_points x num_obj
    :param obj_funcs:           The objective functions to evaluate.
    :param obj_funcs_true:      The true objective functions for the noisy
                                problem.
    :param Y_true:              Finitely large number of observations for
                                an approximate pareto front.
    :param X_design:            A set of design points over which we look for
                                the maximum EHVI. Alternatively, we use them as
                                starting points for L-BFGS. If X_design is
                                an integer, then we draw such points randomly
                                from a hypercube.
    :param y_ref:               The reference point in the objective space used
                                for computing the hypervolume.
    :param how:                 Do you want to maximize or minimize the
                                objectives?
    :param ehvi_opt_method:     The method used for the optimization of the
                                EHVI. Any of the methods available to
                                `scipy.optimize.minimize_`. If you provide
                                any bounds or constraints you must make sure
    :param ehvi_opt_bounds:     The bounds of the EHVI optimization. Just like
                                the bounds in `scipy.optimize.minimize`_.
    :param ehvi_opt_constraints:The constraints of the EHVI optimization.
    :param ehvi_opt_options:    A dictionary of solver options, just like in
                                `scipy.optimize.minimize`_.
    :param max_it:              Maximum number of iterations of the algorithm.
    :param rtol:                Relative tolerance for terminating the
                                algorithm.
    :param add_at_least:        The algorithm will not terminate until at least
                                this many simulations have been performed.
    :param add_in_parallel:     How many points to add in parallel.
    :param verbosity:             The greater this integer is, the more
                                information we print for the progress of the
                                algorithm.
    :param kernel_type:         The kernel used in GP regression.
    :param gp_regression_type:  The GP regression model.
    :param gp_opt_num_restarts: Number of restarts for the GP optimization.
    :param gp_opt_verbosity >= 1: Should we print information about the GP
                                optimization.
    :param gp_opt_verbosity:      If the user wants to print the value of the log
                                likelihood of the GP during each stage of the
                                maximization of the likelihood.
    :param gp_fixed_noise:      What level of noise should we assume for the GP?
                                Select ``None`` if you want the GP regression
                                to estimate the noise level. Select, a small
                                value if your objectives are deterministic.
    :param do_posterior_samples:Do posterior samples or not.
    :param figname:             Name of the figure, may also include full name
                                of the
                                path to the directory where the figures have to be
                                saved.
    :param get_fig:             A figure object that can be used to plot the
                                status of the optimization at each stage.
    :param lim:                 Limits specified as tuples for the two
                                axis of the 'plot status' figures.
    :param make_plot_status:    Do you want the plot_status figures to be made.
    :param comm:                The MPI communicator.
    :param trans_function:      A function to transform the scaled measurements for
                                obtaining the desired plots.
    :param pareto_how:          The mode ```min``` or ```max``` to plot the pareto frontier.

    .. _scipy.optimize.minimize: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    """

    # Observed objectives
    Y = None

    # Input points corresponding to the observed objectives
    X = None

    # The indexes of the points that correspond to the Pareto front
    idx = None

    # The reference point used for the computation of the expected improvement
    y_ref = None

    @property
    def Y_pareto(self):
        """
        :getter: The objectives on the Pareto front.
        """
        return self.Y_projected[self.idx, :]

    @property
    def X_pareto(self):
        """
        :getter: The design points on the Pareto front.
        """
        return self.X[self.idx, :]

    @property
    def num_obj(self):
        """
        :getter: The number of objectives.
        """
        return self.Y.shape[1]

    @property
    def num_dim(self):
        """
        :getter: The number of input dimensions.
        """
        return self.X.shape[1]

    @property
    def num_pareto(self):
        """
        :getter: The number of points on the Pareto front.
        """
        return self.idx.shape[0]

    def _update_yref(self, value=None):
        """
        Update the value of the reference point.
        """
        if value is None:
            mx = np.max(self.Y_projected, axis=0)
            mn = np.min(self.Y_projected, axis=0)
            if self.how == 'min':
                value = mx + 0.1 * (mx - mn)
            else:
                value = mn - 0.1 * (mx - mn)
        if self.how == 'min':
            t = max
        else:
            t = min
        if self.y_ref is None:
            self.y_ref = value
        else:
            for k in xrange(self.num_obj):
                self.y_ref[k] = t(self.y_ref[k], value[k])

    def __init__(self, X, Y, obj_funcs,
                 obj_funcs_true=None,
                 Y_true=None,
                 X_design=None,
                 y_ref=None,
                 how='max',
                 ehvi_opt_method='TNC',
                 ehvi_opt_bounds=None,
                 ehvi_opt_constraints=None,
                 ehvi_opt_options=None,
                 max_it=50,
                 rtol=1e-3,
                 add_at_least=1,
                 add_in_parallel=1,
                 verbosity=0,
                 kernel_type=GPy.kern.Matern32,
                 gp_regression_type=ParallelizedGPRegression,
                 gp_opt_num_restarts=10,
                 gp_opt_verbosity=False,
                 gp_fixed_noise=None,
                 do_posterior_samples=False,
                 figname='ex1/moo',
                 get_fig=None,
                 lim=None,
                 make_plot_status=True,
                 comm=None,
                 trans_function=None,
                 label=('Objective 1', 'Objective 2'),
                 pareto_how='max'):
        super(ParetoFront, self).__init__(comm=comm, verbosity=verbosity)
        assert X.ndim == 2
        self.X = X
        assert Y.ndim == 2
        assert Y.shape[0] == X.shape[0]
        self.Y = Y
        assert self.num_obj == 2, 'Currently working only for two objectives'
        self.X_design = X_design
        self.obj_funcs = obj_funcs
        assert how == 'max' or how == 'min'
        self.how = how
        self.ehvi_opt_method = ehvi_opt_method
        self.ehvi_opt_bounds = ehvi_opt_bounds
        self.ehvi_opt_constraints = ehvi_opt_constraints
        self.ehvi_opt_options = ehvi_opt_options
        assert isinstance(max_it, int) and max_it >= 1
        assert isinstance(add_at_least, int) and add_at_least >= 1 \
               and add_at_least <= max_it
        assert isinstance(rtol, float) and rtol >= 0.
        self.max_it = max_it
        self.rtol = rtol
        self.add_at_least = add_at_least
        self.add_in_parallel = add_in_parallel
        assert isinstance(verbosity, int) and verbosity >= 0
        self.kernel_type = kernel_type
        self.gp_regression_type = gp_regression_type
        self.gp_opt_num_restarts = gp_opt_num_restarts
        self.gp_opt_verbosity = gp_opt_verbosity
        assert gp_fixed_noise is None or \
              isinstance(gp_fixed_noise, float) and gp_fixed_noise >= 0.
        self.gp_fixed_noise = gp_fixed_noise
        self.do_posterior_samples = do_posterior_samples
        self._surrogates = None
        self.train_surrogates()
        self._update_yref(y_ref)
        self.figname = figname
        self.get_fig = get_fig
        self.Y_true = Y_true
        self.lim = lim
        self.obj_funcs_true = obj_funcs_true
        self.make_plot_status = make_plot_status
        self.trans_function = trans_function
        self.label = label
        assert pareto_how == 'max' or pareto_how == 'min'
        self.pareto_how = pareto_how

    @property
    def surrogates(self):
        """
        Get the surrogates. Train them if this hasn't happened yet.
        """
        if self._surrogates is None:
            self.train_surrogates()
        return self._surrogates

    def train_surrogates(self):
        """
        Train the surrogates.
        """
        self._surrogates = []
        self.Y_projected = self.Y.copy()
        for i in xrange(self.num_obj):
            k = self.kernel_type(self.num_dim, ARD=True)
            gp = self.gp_regression_type(self.X, self.Y[:, i][:, None], k,
                                         comm=self.comm,
                                         verbosity=self.gp_opt_verbosity)
            if self.gp_fixed_noise is not None:
                fixed_noise = self.gp_fixed_noise * np.std(self.Y[:, i])
                gp.Gaussian_noise.variance.unconstrain()
                gp.Gaussian_noise.variance.fix(fixed_noise ** 2)
            # The following can be parallelized
            gp.optimize_restarts(self.gp_opt_num_restarts)
            self.Y_projected[:, i] = gp.predict(self.X)[0][:, 0]
            self._surrogates.append(gp)
        self.idx = get_idx_of_observed_pareto_front(self.Y_projected, how=self.how)

    def _optimize_ehvi(self, x0):
        """
        Optimize the EHVI starting at ``x0``.
        """
        args = (self.surrogates, self.Y_pareto, self.y_ref, self.how)
        def func(x, *args):
            r, dr = ehvi_2d_func(x, *args)
            return -r, -dr
        res = minimize(func, x0,
                       jac=True,
                       args=args,
                       method=self.ehvi_opt_method,
                       bounds=self.ehvi_opt_bounds,
                       constraints=self.ehvi_opt_constraints,
                       options=self.ehvi_opt_options)
        return res.x, -res.fun

    def optimize_ehvi(self, X_design):
        """
        Optimize the expected hypervolume improvement starting at all the
        points in X_design.
        """
        if isinstance(X_design, int):
            num_design = X_design
            if self.rank == 0:
                X_design = design.latin_center(num_design, self.num_dim)
            else:
                X_design = None
            if self.use_mpi:
                X_design = self.comm.bcast(X_design)
            if self.ehvi_opt_bounds is not None:
                b = np.array(self.ehvi_opt_bounds)
                X_design = b[:, 0] + (b[:, 1] - b[:, 0]) * X_design
        x_best = None
        ei_max = 0.
        if self.verbosity >= 1:
            pbar = tqdm.tqdm(total=X_design.shape[0])
        for i in distributed_xrange(X_design.shape[0], comm=self.comm):
            x0 = X_design[i, :]
            if self.verbosity >= 1:
                pbar.update(self.size)
            if self.verbosity >= 2:
                print '\t\t> computing EHVI at design point:\n', x0
            x, ei = self._optimize_ehvi(x0)
            if self.verbosity >= 2:
                print '\t\t> final design point:\n', x
                print '\t\t> found: ', ei
            if ei > ei_max:
                ei_max = ei
                x_best = x
        if self.verbosity >= 1:
            pbar.close()
        return reduce_max(x_best, ei_max, comm=self.comm)

    def suggest(self, k, X_design=None):
        """
        Suggest k points to be simulated.
        """
        if X_design is None:
            X_design = self.X_design
        X0 = self.X.copy()
        Y0 = self.Y.copy()
        x_best = []
        eis = []
        for j in xrange(k):
            for i in xrange(self.num_obj):
                gp = self.surrogates[i]
                gp.set_XY(self.X, self.Y[:, i][:, None])
                self.Y_projected = self.Y.copy()
                self.Y_projected[:, i] = gp.predict(self.X)[0][:, 0]
            self.idx = get_idx_of_observed_pareto_front(self.Y_projected,
                                                        how=self.how)
            self._update_yref()
            x, ei = self.optimize_ehvi(X_design)
            eis.append(ei)
            y = [gp.posterior_samples(x[None, :], 1)[0, 0]
                 for gp in self.surrogates]
            x_best.append(x)
            if self.verbosity >=1:
                print '\t\t> add (EI=%f):' % ei
                print x
            self.X = np.vstack([self.X, x])
            self.Y = np.vstack([self.Y,[y]])
        x_best = np.array(x_best)
        self.X = X0
        self.Y = Y0
        self.Y_projected = self.Y.copy()
        for i in xrange(self.num_obj):
            self.surrogates[i].set_XY(X0, Y0[:, i][:, None])
            self.Y_projected[:, i] = self.surrogates[i].predict(self.X)[0][:, 0]
        self.idx = get_idx_of_observed_pareto_front(self.Y_projected,
                                                    how=self.how)
        return x_best, eis[0]

    def plot_status(self, it):
        """
        Plot the status of the algorithm.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        from _plot_pareto import plot_pareto
        sns.set_style("white")
        if self.get_fig is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = self.get_fig()
        if self.trans_function is None:
            self.trans_function = lambda y:y
        if self.do_posterior_samples:
            Y_p = self.sample_pareto_fronts()
            #y_ref = get_yref(np.vstack(Y_p), how=self.how)
            y_ref = self.y_ref
            Y1, Y2, A, beta, Q_beta = get_Vorobev_expectation(Y_p, y_ref, n=64,
                                                              how=self.how)
            D = get_symmetric_deviation_function(Y1, Y2, Q_beta, Y_p,
                                                 how=self.how)
            c = ax.contour(self.trans_function(Y1), self.trans_function(Y2), self.trans_function(Q_beta), levels=[0.99999],
                           colors=(sns.color_palette()[2],
                                   'green', 'blue', (1, 1, 0),
                                   '#afeeee', '0.5'),
                            linestyle='--')
            c = ax.contourf(self.trans_function(Y1), self.trans_function(Y2), D, interpolation='nearest')
            plt.colorbar(c)
        else:
            y_ref = self.y_ref
        if self.Y_true is not None:
            plot_pareto(self.trans_function(self.Y_true),self.trans_function(y_ref),
                        ax=ax, style='-',
                        color=sns.color_palette()[0],
                        how=self.pareto_how)
        if it==self.max_it-1:
            #plot_pareto(self.trans_function(self.Y_projected[:, :]), self.trans_function(y_ref),
            #            ax=ax, style='--', color=sns.color_palette()[1],how=self.pareto_how)
            ax.plot(self.trans_function(self.Y_projected[:, 0]), self.trans_function(self.Y_projected[:, 1]), 'd', color=sns.color_palette()[1], markersize=10)
            if self.obj_funcs_true is not None:
                Y_true_noiseless = np.array([self.obj_funcs_true(x) for x in self.X_pareto])
                plot_pareto(self.trans_function(Y_true_noiseless[:, :]), self.trans_function(y_ref), ax=ax, style='-', color=sns.color_palette()[4],how=self.pareto_how)
                ax.plot(self.trans_function(Y_true_noiseless[:, 0]), self.trans_function(Y_true_noiseless[:, 1]), 'o', color=sns.color_palette()[4], markersize=8)
        else:
            plot_pareto(self.trans_function(self.Y_projected[:-1, :]), self.trans_function(y_ref),
                        ax=ax, style='-', color=sns.color_palette()[1],
                        how=self.pareto_how)
            ax.plot(self.trans_function(self.Y_projected[:-1, 0]), self.trans_function(self.Y_projected[:-1, 1]), 'd',
                    color=sns.color_palette()[1], markersize=10)
            ax.plot(self.trans_function(self.Y_projected[-self.add_in_parallel:, 0]),
                    self.trans_function(self.Y_projected[-self.add_in_parallel:, 1]),
                    'o', markersize=10,
                    color=sns.color_palette()[2])
        if self.lim is not None:
            ax.set_xlim(self.lim[0][0], self.lim[1][0])
            ax.set_ylim(self.lim[0][1], self.lim[1][1])
        ax.set_xlabel(str(self.label[0]),fontsize=15)
        ax.set_ylabel(str(self.label[1]),fontsize=15)
        figname = self.figname + '_' + str(it).zfill(len(str(self.max_it))) \
                  + '.pdf'
        if self.verbosity>=1:
            print '\t> writing:', figname
        fig.savefig(figname)
        plt.close(fig)

    def optimize(self, plot=False):
        """
        Optimize the objectives, i.e., discover the Pareto front.
        """
        if plot:
            import matplotlib
            import matplotlib.pyplot as plt
            if self.rank == 0:
                plt.ion()
                fig, ax = plt.subplots()
                ax.plot(self.Y_projected[:, 0], self.Y_projected[:, 1], 'bo')
            plt.pause(0.05)
        self.ei_values = []
        for it in xrange(self.max_it):
            if self.verbosity >= 1:
                print 'step {0:s}'.format(str(it).zfill(len(str(self.max_it))))
                print '\t> training surrogates'
            if self.verbosity >= 1:
                print '\t> done'
                print '\t> optimizing EHVI'
            x_best, ei_max = self.suggest(self.add_in_parallel)
            if self.verbosity >= 1:
                print '\t> done'
            self.ei_values.append(ei_max)
            rel_ei_max = ei_max / self.ei_values[0]
            if self.verbosity >= 1:
                print '\t> rel_ei_max = {0:1.3f}'.format(rel_ei_max)
            if it >= self.add_at_least and rel_ei_max < self.rtol:
                if self.verbosity >= 1:
                    print '*** Converged (rel_ei_max = {0:1.7f} < rtol = {1:1.2e})'.format(rel_ei_max, self.rtol)
                    print '\t> writing final status'
                break
            if self.verbosity >= 1:
                print '\t> adding best design point'
                print '\t> x_best', x_best
                print '\t> starting simulation'
            y = np.array([self.obj_funcs(x) for x in x_best])
            #y = parallel_eval(self.obj_funcs, x_best)
            self.add_new_observations(x_best, y)
            if self.verbosity>1:
                if self.rank == 0:
                    ax.plot(y[:, 0], y[:, 1], 'rx', markersize=5,
                            markeredgewidth=2)
                plt.pause(0.05)
            if self.make_plot_status and self.rank == 0:
                self.plot_status(it)
            if self.verbosity >= 1:
                print '\t> done'

    def add_new_observations(self, x, y):
        """
        Add new observations and make sure all the quantities defined in
        __init__ are re-initialized.
        """
        self.X = np.vstack([self.X, x])
        self.Y = np.vstack([self.Y,y])
        self.train_surrogates()
        self._update_yref()

    def sample_pareto_fronts(self, num_of_design_samples=10,
                             num_of_gp=10,
                             num_of_design_points=1000, verbosity=0):
        """
        Samples a plaussible pareto front.
        NOTE: Only works if design is the unit hyper-cube.
        """
        Y_p = []
        if self.verbosity >= 1:
            print '\t> sampling Pareto'
            pbar = tqdm.tqdm(total=num_of_design_samples)
        old_variances = []
        for _ in xrange(num_of_design_samples):
            X_design = design.latin_center(num_of_design_points,
                                           self.X.shape[1])
            X_design = np.vstack((X_design,self.X)) #Adding the observed designs to the set of sampled designs.
            b = np.array(self.ehvi_opt_bounds)
            X_design = b[:, 0] + (b[:, 1] - b[:, 0]) * X_design
            Y = []
            for m in self._surrogates:
                #y = m.posterior_samples(X_design, num_of_gp, full_cov=True)
                _m, C = m.predict(X_design, full_cov=True)
                # Subtract the variance
                C -= m.likelihood.variance * np.eye(C.shape[0])
                val, vec = np.linalg.eigh(C)
                val[val <= 0] = 0.0
                _y = np.dot(vec * np.sqrt(val),
                            np.random.randn(vec.shape[0], num_of_gp))
                y = _m + _y
                Y.append(y)
            Y = np.array(Y)
            for i in xrange(Y.shape[2]):
                idx = get_idx_of_observed_pareto_front(Y[:, :, i].T,
                                                       how=self.how)
                y_p = Y[:, idx, i].T
                Y_p.append(y_p)
            if self.verbosity >= 1:
                pbar.update(1)
        if self.verbosity >= 1:
            pbar.close()
        return Y_p