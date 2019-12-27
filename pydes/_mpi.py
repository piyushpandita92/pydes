"""
Some useful parallelization methods/classes.
"""


__all__ = ['DistributedObject', 'distributed_xrange', 'reduce_max',
           'parallel_eval']


import numpy as np


class DistributedObject(object):

    """
    An object that is aware of the parallelization environment.
    :param comm:        An MPI communicator or ``None``.
    :param verbosity:     The verbosity level desired. It will be automatically
                        be zero if the rank of the task storing the objecti
                        is not zero.
    """

    @property
    def use_mpi(self):
        """
        Are we using MPI or not?
        """
        return self.comm is not None

    @property
    def verbosity(self):
        if self.rank == 0:
            return self._verbosity
        else:
            return 0

    def __init__(self, comm=None, verbosity=0):
        self.comm = comm
        self._verbosity = verbosity
        if self.use_mpi:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.rank = 0
            self.size = 1


def distributed_xrange(n, comm=None):
    """
    If ``comm`` is a MPI communicator, then we distributed the range to
    the processors as evenly as possible.
    """
    if comm is None:
        return xrange(n)
    rank = comm.Get_rank()
    size = comm.Get_size()
    r = n % size
    my_n = n / size
    if rank < r:
        my_n += 1
    my_start = rank * my_n
    if rank >= r:
        my_start += r
    my_end = my_start + my_n
    return xrange(my_start, my_end)


def reduce_max(obj, val, comm=None):
    """
    Return the obj that corresponds to the maximum val among a group of processors.
    """
    if comm is None:
        return obj, val
    rank = comm.Get_rank()
    m = np.hstack(comm.allgather(np.array([val])))
    max_rank = np.argmax(m)
    if rank == max_rank:
        best_obj = obj
    else:
        best_obj = None
    best_obj = comm.bcast(best_obj, root=max_rank)
    return best_obj, m[max_rank]


def parallel_eval(fun, X, comm=None):
    """
    Evaluate the function in parallel at each one of the rows of X.
    """
    if comm is None:
        return np.array([fun(x) for x in X])
    n = X.shape[0]
    y = np.array([fun(X[i, :])
                  for i in distributed_xrange(n, comm=comm)])
    return np.vstack(comm.allgather(y))