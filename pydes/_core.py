"""
Some core functions.

"""


from numba import jit
import numpy as np
import math


__all__ = ['gausspdf', 'dgausspdf',
            'get_idx_of_observed_pareto_front_2d_min',
            'get_idx_of_observed_pareto_front_2d_max',
            'get_idx_of_observed_pareto_front_min',
            'get_idx_of_observed_pareto_front_max',
            'get_idx_of_observed_pareto_front',
            'compute_ehvi_2d_max',
            'ehvi_2d_func',
            'calculate_S',
            'monte_carlo_ehvi_2d',
            'plot_pareto',
            'get_empirical_attainment_function',
            'get_Vorobev_expectation',
            'get_symmetric_deviation_function'
            ]


SQRT_TWO = 1.41421356237 # sqrt(2)
SQRT_TWOPI_NEG = 0.3989422804 # 1/sqrt(2*PI)
HALFPI = 1.57079632679; # pi/2


@jit(cache=True)
def _remove_dominated_2d(idx, Y):
    idxp = [idx[0]]
    yp = Y[idx[0], 0]
    for i in idx[1:]:
        if Y[i, 0] >= yp:
            continue
        yp = Y[i, 0]
        idxp.append(i)
    return idxp


def get_idx_of_observed_pareto_front_2d_min(Y):
    """
    Fast algorithm for the pareto front for the case of two objectives.

    :param Y: The set of observed objectives. 2D numpy array. Rows are the
              number of observations and columns the numbers of objectives.
    :returns: The indices of the points that belong to the Pareto front.
    """
    assert Y.shape[1] == 2
    idx = np.lexsort(Y.T)
    idxp = _remove_dominated_2d(idx, Y)
    return idxp


def get_idx_of_observed_pareto_front_2d_max(Y):
    """
    Fast algorithm for the pareto front for the case of two objectives.

    :param Y: The set of observed objectives. 2D numpy array. Rows are the
              number of observations and columns the numbers of objectives.
    :returns: The indices of the points that belong to the Pareto front.
    """
    return get_idx_of_observed_pareto_front_2d_min(-Y)


@jit(cache=True)
def _all_greater_than(x1, x2):
    """
    :param x1:   A vector.
    :param x2:   A value.
    :returns:    ``True`` if the all elements of ``x1`` are greater than all
                 elements of ``x2``.
    """
    for k in xrange(x1.shape[0]):
        if x1[k] <= x2[k]:
            return False
    return True


@jit(cache=True)
def _get_indices_to_be_eliminated(Y):
    idx_to_be_eliminated = [0] # This is just a trick to help numba figure
                               # out the type of the list
    idx_to_be_eliminated.pop()
    num_obj = Y.shape[1]
    num_obs = Y.shape[0]
    for i in xrange(num_obs):
        for j in xrange(i + 1, num_obs):
            if _all_greater_than(Y[i, :], Y[j, :]):
                idx_to_be_eliminated.append(i)
            elif _all_greater_than(Y[j, :], Y[i, :]):
                idx_to_be_eliminated.append(j)
    return idx_to_be_eliminated


def get_idx_of_observed_pareto_front_min(Y):
    """
    Slow algorithm for the Pareto front that works for an arbitrary number of
    objectives.

    :param Y: The set of observed objectives. 2D numpy array. Rows are the
              number of observations and columns the numbers of objectives.
    :returns: The indices of the points that belong to the Pareto front.
    """
    num_obj = Y.shape[1]
    if num_obj == 2:
        return get_idx_of_observed_pareto_front_2d_min(Y)
    num_obs = Y.shape[0]
    idx_to_be_eliminated = set(_get_indices_to_be_eliminated(Y))
    all_idx = set(np.arange(num_obs))
    idx_to_be_kept = all_idx.difference(idx_to_be_eliminated)
    return np.array([i for i in idx_to_be_kept])

def get_idx_of_observed_pareto_front_max(Y):
    """
    Slow algorithm for the Pareto front that works for an arbitrary number of
    objectives.

    :param Y: The set of observed objectives. 2D numpy array. Rows are the
              number of observations and columns the numbers of objectives.
    :returns: The indices of the points that belong to the Pareto front.
    """
    return get_idx_of_observed_pareto_front_min(-Y)

def get_idx_of_observed_pareto_front(Y, how='max'):
    """
    Slow algorithm for the Pareto front that works for an arbitrary number of
    objectives.

    :param Y:   The set of observed objectives. 2D numpy array. Rows are the
                number of observations and columns the numbers of objectives.
    :param how: Do you want to minimize or maximize?
    :returns:   The indices of the points that belong to the Pareto front.
    """
    assert how == 'max' or how == 'min'
    if how == 'max':
        return get_idx_of_observed_pareto_front_max(Y)
    else:
        return get_idx_of_observed_pareto_front_min(Y)

def compute_sorted_list_of_pareto_points(Y, y_ref):
    """
    Compute and return the sorted list of all the i-th coordinates of a
    set of Pareto points.

    This is the ``b`` of Emerich (2008). See page 5.
    """
    m = Y.shape[1]
    return np.concatenate([[[-np.inf for _ in xrange(m)]],
                           np.sort(Y, axis=0),
                           y_ref[None, :],
                           [[np.inf for _ in xrange(m)]]], axis=0)


def plot_pareto(Y, ax=None, style='-',
                color='r', linewidth=2,
                max_obj=None):
    """
    Plot the pareto front.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    if max_obj is None:
        m = np.max(Y, axis=0)
        max_obj = m + .5 * m
    m = Y.shape[1]
    assert m == 2, 'Only works with 2 objectives.'
    idx = get_idx_of_observed_pareto_front_min(Y)
    Y = Y[idx, :]
    n = Y.shape[0]
    ax.plot([max_obj[0], Y[0, 0]],
            [Y[0, 1], Y[0, 1]], style, color=color, linewidth=linewidth)
    for i in xrange(n-1):
        ax.plot([Y[i, 0], Y[i, 0], Y[i + 1, 0]],
                [Y[i, 1], Y[i + 1, 1], Y[i + 1, 1]], style,
                color=color,
                linewidth=linewidth)
    ax.plot([Y[-1, 0], Y[-1, 0]],
            [Y[-1, 1], max_obj[1]], style, color=color, linewidth=linewidth)
    return ax.get_figure(), ax

@jit(cache=True)
def gausspdf(x):
    """
    Probability density function for the normal distribution.
    """
    return SQRT_TWOPI_NEG * math.exp(-.5 * x ** 2)


@jit(cache=True)
def dgausspdf(x):
    """
    Derivative of the probability density function for the normal distribution.
    """
    r = SQRT_TWOPI_NEG * math.exp(-.5 * x ** 2)
    dr = -x * r
    return r, dr


@jit(cache=True)
def gausscdf(x):
    """
    Cumulative distribution function for the normal distribution.
    """
    return 0.5 * (1 + math.erf(x / SQRT_TWO))


@jit(cache=True)
def dgausscdf(x):
    """
    Derivative of the cumulative distribution function for the normal distribution.
    """
    return gausspdf(x)


@jit(cache=True)
def exipsi(fmax, cellcorner, mu, s):
    """
    Partial expected improvement function 'psi'.
    """
    return (s * gausspdf((cellcorner-mu)/s)) + \
            ((fmax-mu) * gausscdf((cellcorner-mu)/s))


@jit(cache=True)
def dexipsi(fmax, cellcorner, mu, s):
    """
    Deerivative of the partial expected improvement function 'psi'
    with respect to mu and s.

    :returns:  The result of the function, derivative with respect to mu,
               and derivative with respect to s.
    """
    tmp1 = (cellcorner - mu) / s
    tmp2 = fmax - mu
    p1, p2 = dgausspdf(tmp1)
    p3 = gausscdf(tmp1)
    r = s * p1 + tmp2 * p3
    dr_dmu = - tmp2 / s * p1 - p2 - p3
    dr_ds = p1 - tmp1 * p2 - tmp1 * tmp2 * p1 / s
    return r, dr_dmu, dr_ds


@jit(cache=True)
def compute_ehvi_2d_max(Yp, r, mu, s):
    """
    Compute the EHVI for a 2D objective that are maximized.

    :param Yp:  The Pareto front sorted along the first objective.
    :param r:   The reference point.
    :param mu:  The predictive mean.
    :param s:   The predictive standard deviation.
    :returns:   The EHVI and its gradients with respect to mu and s.
    """
    # The final answer
    answer = 0.
    # Stores the gradient of the expected improvement with respect to mu
    da_dmu = np.zeros((2, ))
    # Stores the gradient of the expected improvement with respect to s
    da_ds = np.zeros((2, ))
    # Number of points
    k = Yp.shape[0]
    Sstart = k - 1
    Shorizontal = 0
    fmax = np.zeros((2,))
    for i in xrange(k+1):
        Sminus = 0.
        Shorizontal = Sstart
        for j in xrange(k-i, k+1):
            if j == k:
                fmax[1] = r[1]
                cu1 = 1e99
            else:
                fmax[1] = Yp[j, 1]
                cu1 = Yp[j, 0]
            if i == k:
                fmax[0] = r[0]
                cu2 = 1e99
            else:
                fmax[0] = Yp[k-i-1, 0]
                cu2 = Yp[k-i-1, 1]
            if j == 0:
                cl1 = r[0]
            else:
                cl1 = Yp[j-1, 0]
            if i == 0:
                cl2 = r[1]
            else:
                cl2 = Yp[k-i, 1]
            if Shorizontal > Sstart:
                Sminus += (Yp[Shorizontal, 0] - fmax[0]) * \
                          (Yp[Shorizontal, 1] - fmax[1])
            Shorizontal += 1
            # And then we integrate.
            rl1, drl1_dmu, drl1_ds = dexipsi(fmax[0],cl1,mu[0],s[0])
            ru1, dru1_dmu, dru1_ds = dexipsi(fmax[0],cu1,mu[0],s[0])
            rl2, drl2_dmu, drl2_ds = dexipsi(fmax[1],cl2,mu[1],s[1])
            ru2, dru2_dmu, dru2_ds = dexipsi(fmax[1],cu2,mu[1],s[1])

            psi1 = rl1 - ru1
            dpsi1_dmu = drl1_dmu - dru1_dmu
            dpsi1_ds = drl1_ds - dru1_ds
            psi2 = rl2 - ru2
            dpsi2_dmu = drl2_dmu - dru2_dmu
            dpsi2_ds = drl2_ds - dru2_ds

            tmpu1 = (cu1 - mu[0]) / s[0]
            dtmpu1_dmu = -1. / s[0]
            dtmpu1_ds = -tmpu1 / s[0]
            gausscdfu1 = gausscdf(tmpu1)
            gausspdfu1 = gausspdf(tmpu1)
            dgausscdfu1_dmu = gausspdfu1 * dtmpu1_dmu
            dgausscdfu1_ds = gausspdfu1 * dtmpu1_ds

            tmpl1 = (cl1 - mu[0]) / s[0]
            dtmpl1_dmu = -1. / s[0]
            dtmpl1_ds = -tmpl1 / s[0]
            gausscdfl1 = gausscdf(tmpl1)
            gausspdfl1 = gausspdf(tmpl1)
            dgausscdfl1_dmu = gausspdfl1 * dtmpl1_dmu
            dgausscdfl1_ds = gausspdfl1 * dtmpl1_ds

            gausscdf1 = gausscdfu1 - gausscdfl1
            dgausscdf1_dmu = dgausscdfu1_dmu - dgausscdfl1_dmu
            dgausscdf1_ds = dgausscdfu1_ds - dgausscdfl1_ds

            tmpu2 = (cu2 - mu[1]) / s[1]
            dtmpu2_dmu = -1. / s[1]
            dtmpu2_ds = -tmpu2 / s[1]
            gausscdfu2 = gausscdf(tmpu2)
            gausspdfu2 = gausspdf(tmpu2)
            dgausscdfu2_dmu = gausspdfu2 * dtmpu2_dmu
            dgausscdfu2_ds = gausspdfu2 * dtmpu2_ds

            tmpl2 = (cl2 - mu[1]) / s[1]
            dtmpl2_dmu = -1. / s[1]
            dtmpl2_ds = -tmpl2 / s[1]
            gausscdfl2 = gausscdf(tmpl2)
            gausspdfl2 = gausspdf(tmpl2)
            dgausscdfl2_dmu = gausspdfl2 * dtmpl2_dmu
            dgausscdfl2_ds = gausspdfl2 * dtmpl2_ds

            gausscdf2 = gausscdfu2 - gausscdfl2
            dgausscdf2_dmu = dgausscdfu2_dmu - dgausscdfl2_dmu
            dgausscdf2_ds = dgausscdfu2_ds - dgausscdfl2_ds

            ss = (psi1*psi2) - (Sminus*gausscdf1*gausscdf2)
            dss_dmu1 = dpsi1_dmu * psi2 - Sminus * dgausscdf1_dmu * gausscdf2
            dss_ds1 = dpsi1_ds * psi2 - Sminus * dgausscdf1_ds * gausscdf2
            dss_dmu2 = psi1 * dpsi2_dmu - Sminus * gausscdf1 * dgausscdf2_dmu
            dss_ds2 = psi1 * dpsi2_ds - Sminus * gausscdf1 * dgausscdf2_ds
            if ss > 0.:
                answer += ss
                da_dmu[0] += dss_dmu1
                da_dmu[1] += dss_dmu2
                da_ds[0] += dss_ds1
                da_ds[1] += dss_ds2
        Sstart -= 1
    return answer, da_dmu, da_ds


def compute_ehvi_2d_min(Yp, r, mu, s):
    """
    Compute the EHVI for a 2D objective that are maximized.

    :param Yp:  The Pareto front sorted along the first objective.
    :param r:   The reference point.
    :param mu:  The predictive mean.
    :param s:   The predictive standard deviation.
    :returns:   The EHVI and its gradients with respect to mu and s.
    """
    return compute_ehvi_2d_max(-Yp, -r, -mu, s)


def compute_ehvi_2d(Yp, r, mu, s, how='max'):
    """
    Compute the EHVI for a 2D objective.

    :param Yp:  The Pareto front sorted along the first objective.
    :param r:   The reference point.
    :param mu:  The predictive mean.
    :param s:   The predictive standard deviation.
    :param how: Do you want to minimize or maximize?
    :returns:   The EHVI and its gradients with respect to mu and s.
    """
    num_obj = Yp.shape[1]
    assert num_obj == 2
    assert how == 'max' or how == 'min'
    if how == 'max':
        return compute_ehvi_2d_max(Yp, r, mu, s)
    else:
        return compute_ehvi_2d_min(Yp, r, mu, s)


def ehvi_2d_func(x, objs, Yp, r, how='max'):
    """
    The EHVI as a function of x.

    :param objs:    List of GPy.GPRegression objects that contain the fitted
                    surrogates.
    :param Yp:  The Pareto front sorted along the first objective.
    :param r:   The reference point.
    :param how: Do you want to minimize or maximize?
    :returns:   The EHVI.
    """
    num_dim = x.shape[0]
    mu = np.ndarray((2,))
    s = np.ndarray((2,))
    dmu_dx = np.ndarray((2, num_dim))
    ds_dx = np.ndarray((2, num_dim))
    for k in xrange(len(objs)):
        f = objs[k]
        m, v = f.predict(x[None, :])
        mu[k] = m[0, 0]
	s[k] = math.sqrt(v[0, 0] - f.likelihood.variance)
        dm_dx, dv_dx = f.predictive_gradients(x[None, :])
        dmu_dx[k, :] = dm_dx[0, :, 0]
        ds_dx[k, :] = .5 * dv_dx[0, :] / s[k]
    ei, dei_dmu, dei_ds = compute_ehvi_2d(Yp, r, mu, s, how=how)
    dei_dx = np.einsum('i,ij->j', dei_dmu, dmu_dx) + \
             np.einsum('i,ij->j', dei_ds, ds_dx)
    return ei, dei_dx


@jit(cache=True)
def calculate_S(Yp, r):
    """
    Returns the 2d hypervolume for the population P with reference
    point r.
    """
    answer = 0;
    n = Yp.shape[0]
    if n >= 1:
        answer += (Yp[n-1, 0] - r[0]) * (Yp[n-1, 1] - r[1])
        for i in xrange(n-2, -1, -1):
            answer += (Yp[i, 0] - r[0]) * (Yp[i, 1] - Yp[i+1, 1])
    return answer


@jit(cache=True)
def _sample_new_S_2d(Yp_and_work, r, mu, s, idx):
    """
    Compute the EHVI using MC.
    """
    n = Yp_and_work.shape[0] - 1
    num_obj = Yp_and_work.shape[1]
    res = np.zeros((num_obj, ))
    for k in xrange(num_obj):
        res[k] = mu[k] + s[k] * np.random.randn()
        if res[k] <= r[k]:
            return 0.
    Yp_and_work[-1, :] = res
    c = 0
    for i in xrange(n):
        if res[0] > Yp_and_work[i, 0]:
            if res[1] > Yp_and_work[i, 1]:
                idx[c] = -1
                c += 1
                for j in xrange(i+1, n):
                    if res[0] < Yp_and_work[j, 0]:
                        for k in xrange(j, n):
                            idx[c] = k
                            c += 1
                        break
                break
        else:
            for k in xrange(i, n):
                idx[c] = k
                c += 1
            break
        idx[c] = i
        c += 1
    return calculate_S(Yp_and_work[idx[:c], :], r)


@jit(cache=True)
def _monte_carlo_new_S_2d(num_samples, Yp_and_work, r, mu, s, idx):
    ss = 0.
    for i in xrange(num_samples):
        ss += _sample_new_S_2d(Yp_and_work, r, mu, s, idx)
    return ss / num_samples


def monte_carlo_ehvi_2d(num_samples, Yp, r, mu, s):
    """
    Monte Carlo computation of the EHVI for maximizing the objectives.

    :param objs:    List of GPy.GPRegression objects that contain the fitted
                    surrogates.
    :param Yp:  The Pareto front sorted along the first objective.
    :param r:   The reference point.
    :param mu:  The predictive mean.
    :param s:   The predictive standard deviation.
    :returns:   The EHVI.
    """
    Yp_and_work = np.vstack([Yp, np.zeros((1, Yp.shape[1]))])
    idx = np.ndarray((Yp.shape[0] + 1, ), dtype='int64')
    ss = _monte_carlo_new_S_2d(num_samples, Yp_and_work, r, mu, s, idx)
    return ss - calculate_S(Yp, r)


@jit(cache=True)
def get_empirical_attainment_function(Y, y_ref, n=10, how='max'):
    """
    Get the empirical attainment function.
    """
    assert how == 'max'
    sY = np.vstack(Y)
    mx = np.max(sY, axis=0)
    mn = np.min(sY, axis=0)
    if how == 'max':
        y1 = np.linspace(y_ref[0], mx[0], n)
        y2 = np.linspace(y_ref[1], mx[1], n)
    else:
        y1 = np.linspace(mn[0], y_ref[0], n)
        y2 = np.linspace(mn[1], y_ref[1], n)
    Y1, Y2 = np.meshgrid(y1, y2)
    A = np.zeros(Y1.shape)
    for i in range(n):
        for j in range(n):
            # Loop over sampled Pareto fronts
            for k in range(len(Y)):
                Y_p = Y[k]
                # Check if Y1[i, j], Y2[i, j] is dominated by Y_p
                for s in range(Y_p.shape[0]):
                    if Y_p[s, 0] >= Y1[i, j] and Y_p[s, 1] >= Y2[i, j]:
                        A[i, j] += 1.0
                        break
    A /= len(Y)
    return Y1, Y2, A


@jit(cache=True)
def get_symmetric_deviation_function(Y1, Y2, Q, Y, how='max'):
    """
    Get the symmetric deviation function.
    """
    assert how == 'max'
    D = np.zeros(Y1.shape)
    n = Y1.shape[0]
    for i in range(n):
        for j in range(n):
            # Loop over sampled Pareto fronts
            for k in range(len(Y)):
                Y_p = Y[k]
                # Check if Y1[i, j], Y2[i, j] is in the symmetric difference
                z_in_Y = False
                for s in range(Y_p.shape[0]):
                    if Y_p[s, 0] >= Y1[i, j] and Y_p[s, 1] >= Y2[i, j]:
                        z_in_Y = True
                        break
                z_in_Q = Q[i, j] > 0.0
                if (z_in_Q or z_in_Y) and not (z_in_Q and z_in_Y):
                    D[i, j] += 1.0
    D /= len(Y)
    return D


@jit(cache=True)
def get_Vorobev_expectation(Y, y_ref, n=100, how='max'):
    """
    Get the Vorob'ev expecation of the random set Pareto fronts in Y.
    """
    Y1, Y2, A = get_empirical_attainment_function(Y, y_ref, how=how, n=n)
    exp_mu_Y = 0.
    N = len(Y)
    for i in range(N):
        exp_mu_Y += calculate_S(Y[i], y_ref)
    exp_mu_Y /= N
    a = 0.0
    b = 1.0
    sv = (Y1[0, 1] - Y1[0, 0]) * (Y2[1, 0] - Y2[0, 0])
    while b - a > 1e-3:
        x = .5 * (a + b)
        idx = A >= x
        mu_Q = np.sum(idx) * sv
        if mu_Q < exp_mu_Y:
            b = .5 * (a + b)
        else:
            a = .5 * (a + b)
    beta = .5 * (a + b)
    Q_beta = np.zeros(A.shape)
    Q_beta[A >= beta] = 1.
    return Y1, Y2, A, beta, Q_beta