"""
Parallelizes the multi-start optimization procedure of GPy models.

Author:
    Ilias Bilionis

Date:
    4/15/2015

"""


import numpy as np
import scipy
import scipy.linalg
from GPy.models import GPRegression
from . import DistributedObject
from . import reduce_max


__all__ = ['Parallelizer', 'ParallelizedGPRegression']


class Parallelizer(DistributedObject):
    """
    Parallelize the ``optimize_restarts()`` function.
    """

    def __init__(self, **kwargs):
        if not kwargs.has_key('comm'):
            comm = None
        else:
            comm = kwargs['comm']
        if not kwargs.has_key('verbosity'):
            verbosity = 0
        else:
            verbosity = kwargs['verbosity']
        super(Parallelizer, self).__init__(comm=comm, verbosity=verbosity)


    def optimize_restarts(self, num_restarts=10, **kwargs):
        """
        Optimize restarts using MPI.

        :param comm:    The MPI communicator

        When we return, we guarantee that every core has the right model.
        """
        size = self.size
        rank = self.rank
        comm = self.comm
        my_num_restarts = num_restarts / size
        if my_num_restarts == 0:
            my_num_restarts = 1
        num_restarts = my_num_restarts * size
        if self.verbosity >= 2:
            print '> optimizing hyper-parameters using multi-start'
            print '> num available cores:', size
            print '> num restarts:', num_restarts
            print '> num restarts per core:', my_num_restarts
        # Let everybody work with its own data
        self.randomize()
        super(Parallelizer, self).optimize_restarts(num_restarts=my_num_restarts,
                                                    verbose=self.verbosity>=2,
                                                    **kwargs)
        if self.use_mpi:
            best_x_opt, log_like = reduce_max(self.optimizer_array.copy(),
                                              self.log_likelihood(),
                                              comm=comm)
            if self.verbosity >= 2:
                print '> best hyperparameters:', best_x_opt
            self.optimizer_array = best_x_opt


class ParallelizedGPRegression(Parallelizer, GPRegression):
    """
    A parallelized version of GPRegression.
    """

    @property
    def W(self):
        return np.eye(self.input_dim)

    @property
    def theta(self):
        theta = np.hstack([self.kern.variance,
                           self.kern.lengthscale,
                           self.likelihood.variance])
        return theta

    def __init__(self, X, Y, k, Y_mean=0., Y_std=1., comm=None, verbosity=0, **kwargs):
        Parallelizer.__init__(self, comm=comm, verbosity=verbosity)
        GPRegression.__init__(self, X, Y, k, **kwargs)
        self.Y_mean = Y_mean
        self.Y_std = Y_std

    def to_array(self):
        """
        Turn the self to an array.
        """
        W = self.W
        theta = self.theta
        X = self.X
        Z = np.dot(X, W)
        Y = np.array(self.Y)
        num_samples = X.shape[0]
        K = self.kern.K(X) + self.likelihood.variance * np.eye(num_samples)
        L = scipy.linalg.cho_factor(K, lower=True)
        b = scipy.linalg.cho_solve(L, Y).flatten()
        Ki = scipy.linalg.cho_solve(L, np.eye(num_samples))
        tmp = []
        tmp.append(W.flatten())
        tmp.append(Z.flatten())
        tmp.append(b.flatten())
        tmp.append(Ki.flatten())
        tmp.append(theta.flatten())
        num_input = self.input_dim
        num_samples = self.num_data
        num_active = W.shape[1]
        return np.hstack([[num_input, num_samples, num_active],
                          np.hstack(tmp), [self.Y_mean, self.Y_std]])
