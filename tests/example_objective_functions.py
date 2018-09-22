"""
Some example objective functions.

"""

import numpy as np
import copy
import math
__all__ = ['ObjFunc1', 'ObjFunc2']


class ObjFunc1(object):

    """
    A stochastic objective function with controllable stochaticity.

    The objective function can work with an arbitrary number of design
    dimensions. This objective function is a slightly modified form of the function given on
    page 88 of the PhD. thesis
    Parr (cited in the draft of the MOO paper).
    
    This is example 1 of the paper. I have coded the negative of the above mentioned function.

    :param sigma:   The standard deviation of the noise.
    :param n_samp:  The number of samples used in computing the average
                    objective value.
    """
    def f1(self,x):
        y = 0
        for _ in xrange(self.n_samp):
	    xi = copy.copy(x)
            xi = xi + float(self.sigma) * np.random.randn((np.size(x)),)
	    b1 = 15. * xi[0] - 5.
            b2 = 15. * xi[1]
            k = (b2 - 5.1 / 4. / math.pi ** 2 * b1 ** 2 + 5. / math.pi * b1 - 6.) ** 2. \
            + 10. * ((1. - 1. / 8. / math.pi) * math.cos(b1) + 1.)
            y = y + k
        return -((float(y)/self.n_samp)-150)/75.

    def f2(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            xi = copy.copy(x)
            xi = xi + float(self.sigma) * np.random.randn(np.size(x),)
            b1 = 15. * xi[0] - 5.
            b2 = 15. * xi[1]
            k = - np.sqrt(np.abs((10.5 - b1) * ((b1 + 5.5)) * (b2 + 0.5))) \
            - 1. / 30. * (b2 - 5.1 / 4. / math.pi ** 2 * b1 ** 2 - 6.) ** 2 \
            - 1. / 3. * ((1. - 1. / 8. / math.pi) * math.cos(b1) + 1.)
            y = y + k
        return -((float(y)/self.n_samp)-30.)/10. 

    def __init__(self,sigma=0.,n_samp=1.):
        self.n_samp = n_samp
        self.sigma = sigma

    def __call__(self,x):
        return self.f1(x), self.f2(x)

class ObjFunc2(object):

    """
    A stochastic objective function with controllable stochaticity.

    The objective function can work with an arbitrary number of design
    dimensions.

    This is example 2 of the paper. I have coded the negative of the above mentioned function.

    :param sigma:   The standard deviation of the noise.
    :param n_samp:  The number of samples used in computing the average
                    objective value.
    """

    def f1(self, x):
        """
        Where did you find this function?
        from Knowles et al. ref 16 of the paper.
        """
        y = 0.
        for _ in xrange(self.n_samp):
            xi = x + float(self.sigma) * np.random.randn(np.size(x),)
            g = 100. * (((xi[1:6] - 0.5) ** 2 - np.cos(2. * np.pi * (xi[1:6]-0.5))).sum() + 5.)
            k = 0.5 * (xi[0]) * (g + 1.)
            y += k
        return -((y / self.n_samp)-150.)/80.

    def f2(self, x):
        """
        This seemed to be exactly the same as f1. Why re-implement it?
        """
        y = 0.
        for _ in xrange(self.n_samp):
            xi = x + float(self.sigma) * np.random.randn(np.size(x),)
            g = 100. * (((xi[1:6] - 0.5) ** 2 - np.cos(2. * np.pi * (xi[1:6] - 0.5))).sum() + 5.)
            k = 0.5 * (1. - xi[0]) * (g + 1.)
            y += k
        return -((y / self.n_samp)-150.)/80.

    def __init__(self, sigma=0., n_samp=1):
        self.n_samp = n_samp
        self.sigma = sigma

    def __call__(self,x):
        return self.f1(x), self.f2(x)