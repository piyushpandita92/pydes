"""
This has the plot_pareto method for tplotting the
figures at each stage of the optimization process.
"""


import numpy as np
from . import get_idx_of_observed_pareto_front
from numba import jit


__all__ = ['get_yref', 'plot_pareto']



def get_yref(Y, how='max'):
    """
    Update the value of the reference point.
    """
    mx = np.max(Y, axis=0)
    mn = np.min(Y, axis=0)
    if how == 'min':
        value = mx + 0.1 * (mx - mn)
    else:
        value = mn - 0.1 * (mx - mn)
    if how == 'min':
        t = max
    else:
        t = min
    return value


def plot_pareto(Y, y_ref,
                ax=None, style='-',
                color='r', linewidth=2,
                how='max'):
    """
    Plot the pareto front.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    if how == 'min':
        m = np.max(Y, axis=0)
        max_obj = m + .1 * m
        minmax_obj = max_obj
    elif how == 'max':
        m = np.min(Y, axis=0)
        min_obj = m - .1 * m
        minmax_obj = min_obj
    m = Y.shape[1]
    assert m == 2, 'Only works with 2 objectives.'
    idx = get_idx_of_observed_pareto_front(Y, how=how)
    Y = Y[idx, :]
    n = Y.shape[0]
    #ax.plot([minmax_obj[0], Y[0, 0]],
    #        [Y[0, 1], Y[0, 1]], style, color=color, linewidth=linewidth)
    ax.plot([y_ref[0], Y[0, 0]],
            [Y[0, 1], Y[0, 1]], style, color=color, linewidth=linewidth)
    for i in range(n-1):
        ax.plot([Y[i, 0], Y[i, 0], Y[i + 1, 0]],
                [Y[i, 1], Y[i + 1, 1], Y[i + 1, 1]], style,
                color=color,
                linewidth=linewidth)
    ax.plot([Y[-1, 0], Y[-1, 0]],
            [Y[-1, 1], y_ref[1]], style, color=color, linewidth=linewidth)
    return ax.get_figure(), ax
