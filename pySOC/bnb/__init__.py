import copy
import logging
import timeit

import numpy as np
import scipy as sp
from scipy.linalg import sqrtm

from ..utils.matrixdivide import mldivide, mrdivide

logging.basicConfig(level=logging.CRITICAL,
                    filename='python.log', filemode='w')
np.set_printoptions(formatter={'int': '{:d}'.format})

__all__ = ['pb3wc']

def pb3wc(gy: np.ndarray, gyd: np.ndarray, wd: np.ndarray, wn: np.ndarray,
          juu: np.ndarray, jud: np.ndarray, n: int, tlimit: int = np.Inf,
          nc: int = 1):
    """Partial Bidirectional Branch and Bound algorithm for Worst-Case
    Scenario.

    Implementation of PB3 algorithm to select measurements, whose combinations
    can be used as controlled variables. The measurements are selected to
    provide minimum worst-case loss in terms of Self-Optimization Control based
    on the following static problem:

    .. math::
        \\newline

        \\text{minimize:}

        J = f(u,d,y)

        \\newline

        \\text{subject to:}

        y = G^{y} \\times u + G^{y}_{d} \\times W_{d} \\times d + W_{n} \\times e

        \\newline

    Where :math:`J` is the objective function, :math:`y` is measurement vector
    of size ny, :math:`u` is the input vector of size ny, :math:`d` the
    disturbances, :math:`G^{y}` is the process model (gain matrix of y with
    respect to u), :math:`G^{y}_{d}` is the disturbance model (gain matrix of y
    with respect to d), :math:`W_{d}` is the disturbances magnitude,
    :math:`W_{n}` is the implementatione errors magnitude and :math:`e` is the
    control error.

    Parameters
    ----------
    gy : np.ndarray
        Process model (ny-by-nu Gain matrix of :math:`y` with respect to
        :math:`u`).

    gyd : np.ndarray
        Disturbance model (ny-by-nd Gain matrix of :math:`y` with respect to
        :math:`d`)

    wd : np.ndarray
        1D array containing nd magnitudes of disturbances.

    wn : np.ndarray
        1D array containing n magnitudes of implementation errors.

    juu : np.ndarray
        Hessian matrix of :math:`J` relative :math:`u`, evaluated at optimum
        (nu-by-nu).

    jud : np.ndarray
        Hessian matrix of :math:`J` relative to :math:`u` and :math:`d`,
        evaluated at optimum (nu-by-nd).

    n : int
        Number of measurements to be selected. Must follow nu <= n <= ny.

    tlimit : int, optional
        Time limit to run the code, by default np.Inf.

    nc : int, optional
        Number of best subsets to be selected, by default 1.

    Returns
    -------
    B : ndarray
        1D array of size nc with the worst-case loss of selected subsets.

    sset : ndarray
        nc-by-n indices of selected subsets.

    ops : ndarray
        1D array with 4 elements containing the number of nodes evaluated.
        counters: 0) terminal; 1) nodes; 2) sub-nodes; 3) calls.

    ctime : float
        Computation time used.

    flag : bool
        Status flag of success.
        0 if successful, 1 otherwise (`tlimit` < np.inf)

    Raises
    ------
    ValueError
        When `n` < nu. Number of measurements must be greater than the number 
        of degrees of freedom.

    Notes
    -----
    Formal definition of `Juu`:

    .. math:: J_{uu} = \\frac{\\partial^2 J}{\\partial u^2}

    Formal definition of `Jud`:

    .. math:: J_{ud} = \\frac{\\partial^2 J}{\\partial u\\partial d}

    References
    ----------
    .. [1] I. J. Halvorsen, S. Skogestad, J. C. Morud, and V. Alstad. Optimal 
        selection of controlled variables. Ind. Eng. Chem. Res.,
        42(14):3273-3284, 2003.

    .. [2] V. Kariwala, Y. Cao, and S. Janardhanan. Local self-optimizing 
        control with average loss minimization. Ind. Eng. Chem. Res., 
        47(4):1150-1158, 2008.

    .. [3] V. Kariwala and Y. Cao, Bidirectional Branch and Bound for 
        Controlled Variable Selection: Part II. Exact Local Method for 
        Self-optimizing Control, Computers and Chemical Engineering, 
        33(8):1402:1412, 2009.
    """
    # --------------------------- input sanitation ----------------------------
    # array dimensions
    # Gy, Gyd must be 2D
    if gy.ndim != 2 or gyd.ndim != 2:
        raise ValueError("Gy and Gyd must be 2D arrays!")
    else:
        # they are 2D, ensure they aren't scalars
        if gy.size == 1 or gyd.size == 1:
            raise ValueError(("This implementation does not allow scalar gain "
                              "matrices Gy or Gyd."))

    ny, nu = gy.shape
    _, nd = gyd.shape

    # Juu and Jud must be scalar or 2D
    if juu.ndim != 2 or jud.ndim != 2:
        raise ValueError("Juu and Jud must be 2D arrays!")
    else:
        # check their dimensions
        if juu.shape[0] != juu.shape[1]:
            raise ValueError("Juu must be a square matrix.")

        if jud.shape[0] != nu or jud.shape[1] != nd:
            raise ValueError(("Jud is not a nu-by-nd matrix."
                              "\nnu={0:d}\nnd={1:d}".format(nu, nd)))

    # Wd and Wn must be 1D
    if wd.ndim != 1 or wn.ndim != 1:
        raise ValueError("Wd and Wn must be 1D arrays.")
    else:
        # they are 1D, check their dimensions
        if wd.size != nd:
            raise ValueError("Wd must have nd={0:d} elements.".format(nd))

        if wn.size != ny:
            raise ValueError("Wn must have ny={0:d} elements.".format(ny))

    # transform in diagonal matrix
    wd = np.diag(wd)
    wn = np.diag(wn)

    # check if nu <= n < ny
    if not nu <= n <= ny:
        raise ValueError(("n must follow nu <= n <= ny "
                          "({0:d} <= n <= {1:d}).".format(nu, ny)))

    # ------------------------------ calculations -----------------------------
    flag = False
    ctime0 = timeit.default_timer()

    r, m = gy.shape

    if n < m:
        raise ValueError("n must be larger than number of inputs (nu).")

    # prepare matrices
    Y = np.hstack(((mrdivide(gy, juu) @ jud - gyd) @ wd, wn))
    G = mrdivide(gy, sqrtm(juu))
    Y2 = Y @ Y.T
    Gd = copy.deepcopy(G)
    Xd = copy.deepcopy(Y2)
    Xu = copy.deepcopy(Xd)
    h2 = np.diag(Y2)
    q2 = np.diag(G @ G.T) / h2
    p2 = copy.deepcopy(q2)

    # counters: 1) terminal; 2) nodes; 3) sub-nodes; 4) calls
    ops = np.zeros((4,), dtype=int)

    B = np.zeros((nc,))
    sset = np.zeros((nc, n))
    ib = 0
    bound = 0
    fx = np.full((r,), False)
    rem = np.full((r,), True)
    down_v = copy.deepcopy(fx)
    down_r = copy.deepcopy(fx)
    nf = 0
    n2 = n
    m2 = r

    # initalize
    f = False
    bf = False
    downf = False
    downff = False
    upf = False
    upff = True

    params = {'flag': flag, 'ctime0': ctime0, 'tlimit': tlimit, 'n': n, 'm': m,
              'G': G, 'Y2': Y2, 'Gd': Gd, 'Xd': Xd, 'Xu': Xu, 'h2': h2,
              'q2': q2, 'p2': p2, 'ops': ops, 'B': B, 'sset': sset, 'ib': ib,
              'bound': bound, 'fx': fx, 'rem': rem, 'down_v': down_v,
              'down_r': down_r, 'nf': nf, 'n2': n2, 'm2': m2, 'f': f, 'bf': bf,
              'downf': downf, 'downff': downff, 'upf': upf, 'upff': upff}

    params = BnBParams(params)
    _bbL3sub(fx, rem, params)
    idx = np.argsort(0.5 / B, axis=None)
    B = np.sort(0.5 / B, axis=None)

    # the +1 to change from 0 to 1 index
    sset = np.sort(sset[idx, :], axis=1) + 1
    ctime = timeit.default_timer() - ctime0

    # TODO: write test routine to compare results with matlab

    return B, sset, ops, ctime, flag


class BnBParams(object):
    """Parameters storage object. Converts dictionary keys to attributes.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing the Branch and Bound parameters.
    """

    def __init__(self, param_dict: dict):
        self.__dict__.update(param_dict)


def _bbL3sub(fx0: np.ndarray, rem0: np.ndarray, params: BnBParams):
    # unpacking parameters values
    p = params

    logging.debug('BBL3SUB\t\t- %s | %s | %s', p.ops, p.fx.sum(), p.rem.sum())

    # recursive solver
    bn = 0
    if timeit.default_timer() - p.ctime0 > p.tlimit:
        params['flag'] = True
        return bn

    p.ops[3] += 1
    p.fx = copy.deepcopy(fx0)
    p.rem = copy.deepcopy(rem0)
    p.nf = np.sum(p.fx)
    p.m2 = np.sum(p.rem)
    p.n2 = p.n - p.nf

    while not p.f and 0 <= p.n2 < p.m2:  # loop for second branches
        while not p.f and 0 < p.n2 < p.m2 and \
                (not p.downf or not p.upf or not p.bf):
            # loop for bidirectional pruning
            if (not p.upf or not p.bf) and p.n2 <= p.m and p.bound:
                # upwards pruning
                _upprune(p)
            else:
                p.upf = True

            if not p.f and p.m2 > p.n2 and p.m2 > 0 and \
                    (not p.downf or not p.bf) and p.bound:
                # downwards pruning
                _downprune(p)
            else:
                p.downf = True

            p.bf = True
            # end pruning loop

        if p.f or p.m2 < p.n2 or p.n2 < 0:
            # pruned cases
            return bn
        elif p.m2 == p.n2 or not p.n2:
            # terminal cases
            break

        if p.n2 == 1:  # one or more element to be fixed
            if p.upff:
                p.q2[np.logical_not(p.rem)] = 0
                idk = np.argmax(p.q2)
            else:
                p.p2[np.logical_not(p.rem)] = np.Inf
                idk = np.argmin(p.p2)

            s = copy.deepcopy(p.fx)
            s[idk] = True
            bn = _update(s, p) - 1

            if bn > 0:
                bn += np.sum(fx0) - p.nf
                return bn

            p.rem[idk] = False
            p.m2 -= 1
            p.downf = False
            p.upf = True
            continue  # line 178

        if p.m2 - p.n2 == 1:  # one or more element to be removed
            if p.downff:
                p.p2[np.logical_not(p.rem)] = 0
                idk = np.argmax(p.p2)
            else:
                p.q2[np.logical_not(p.rem)] = np.Inf
                idk = np.argmin(p.q2)

            p.rem[idk] = False
            s = np.logical_or(p.fx, p.rem)
            _update(s, p)
            p.fx[idk] = True
            p.nf += 1
            p.n2 -= 1
            p.m2 -= 1
            p.upf = False
            p.downf = True
            continue  # line 200

        # save data for bidirectional branching
        fx1 = copy.deepcopy(p.fx)
        rem1 = copy.deepcopy(p.rem)
        p0 = copy.deepcopy(p.p2)
        q0 = copy.deepcopy(p.q2)
        D0 = copy.deepcopy(p.Xd)
        U0 = copy.deepcopy(p.Xu)
        dV0 = copy.deepcopy(p.down_v)
        dR0 = copy.deepcopy(p.down_r)

        if p.n2 - p.m < 0.75 * p.m2:  # upward branching
            if p.bound:
                p.p2[np.logical_not(p.rem)] = np.Inf
                idd = np.argmin(p.p2)
            else:
                p.q2[np.logical_not(p.rem)] = 0
                idd = np.argmax(p.q2)

            p.fx[idd] = True
            rem1[idd] = False
            p.downf = True
            p.upf = False
            bn = _bbL3sub(p.fx, rem1, p) - 1
            p.downf = False
            p.upf = True
        else:  # downward branching
            if p.bound:
                p.p2[np.logical_not(p.rem)] = 0
                idd = np.argmax(p.p2)
            else:
                p.q2[np.logical_not(p.rem)] = np.Inf
                idd = np.argmin(p.q2)

            fx1[idd] = True
            rem1[idd] = False
            p.downf = False
            p.upf = True
            bn = _bbL3sub(p.fx, rem1, p) - 1
            p.downf = True
            p.upf = False

            if q0[idd] <= p.bound and p.n == p.m:
                return bn  # line 244

        # check pruning conditions
        if bn > 0:
            bn += - np.sum(rem0) + p.m2
            return bn

        # recover data saved before first branch
        p.fx = copy.deepcopy(fx1)
        p.rem = copy.deepcopy(rem1)
        p.f = False
        p.p2 = copy.deepcopy(p0)
        p.q2 = copy.deepcopy(q0)
        p.nf = np.sum(p.fx)
        p.n2 = p.n - p.nf
        p.m2 = np.sum(p.rem)
        p.Xd = copy.deepcopy(D0)
        p.Xu = copy.deepcopy(U0)
        p.down_v = copy.deepcopy(dV0)
        p.down_r = copy.deepcopy(dR0)
        # end loop for second branches

    if not p.f:  # terminal cases
        if p.m2 == p.n2:
            bn = _update(np.logical_or(p.fx, p.rem), p) - 1
        elif not p.n2:
            bn = _update(p.fx, p) - 1

    bn += np.sum(fx0) - p.nf
    return bn


# ------------------------------ local functions ------------------------------
def _upprune(params: BnBParams) -> None:
    # parameters unpacking
    p = params

    logging.debug('UPPRUNE\t\t- %s | %s | %s', p.ops, p.fx.sum(), p.rem.sum())

    # partially upwards pruning
    p.upf = True
    try:
        R1 = sp.linalg.cholesky(p.Y2[np.ix_(p.fx, p.fx)])
    except sp.linalg.LinAlgError:
        p.f = True
        return
    else:
        p.f = False

    X1 = mldivide(R1.T, p.G[p.fx, :])
    D = X1.T @ X1
    tD = np.trace(D)

    if tD < p.bound and p.n2 < p.m:
        p.ops[1] += 1
        p.f = True

        return

    if tD / p.m > p.bound and p.n2 == p.m:
        return

    if p.m > 2:
        # general cases
        bf0 = np.sum(np.linalg.eig(D)[0] < p.bound)
        p.ops[0] += 1

        if bf0 > p.n2:
            p.f = True

            return

        if bf0 != p.n2:
            return

        D = np.eye(p.m) * p.bound - D

    else:
        # special cases without using eig
        D = np.eye(p.m) * p.bound - D
        try:
            R = sp.linalg.cholesky(D)
        except sp.linalg.LinAlgError:
            p.f = True
        else:
            p.f = False

        p.ops[1] += 1

        if (p.f and p.n2 > 1) or not p.f and p.n2 < p.m:
            p.f = not p.f

            return

        if p.n2 == 1:
            try:
                R = sp.linalg.cholesky(-D)
            except sp.linalg.LinAlgError:
                p.f = True
            else:
                p.f = False

            if not p.f:
                # m eigen values > bound, no pruning
                return

            # otherwise only one eigen value < bound
            p.f = not p.f

        # end if m > 2

    R2 = mldivide(R1.T, p.Y2[np.ix_(p.fx, p.rem)])
    R3 = p.h2[p.rem] - np.sum(R2 * R2, axis=0).T
    X2 = p.G[p.rem, :] - R2.T @ X1
    p.ops[2] += p.m2
    p.q2[:] = np.Inf

    if p.n2 == 1 or p.m > 2:
        p.q2[p.rem] = np.sum(X2.T * mldivide(D, X2.T), axis=0).T / R3 - 1
    else:
        X = mldivide(R.T, X2.T)
        p.q2[p.rem] = np.sum(X * X, axis=0).T / R3 - 1

    p.upff = True
    L = p.q2 <= 0
    if np.any(L):
        # upwards pruning
        p.downf = False
        p.downff = False
        p.rem[L] = False
        p.m2 = np.sum(p.rem)
        p.q2[L] = np.Inf


def _downprune(params: BnBParams) -> None:
    # parameters unpacking
    p = params

    logging.debug('DOWNPRUNE\t- %s | %s | %s', p.ops, p.fx.sum(), p.rem.sum())

    # downwards pruning
    p.downf = True
    s0 = np.logical_or(p.fx, p.rem)
    t = np.logical_xor(p.down_v, s0)

    if p.bf and np.sum(t) == 1 and p.down_r[t]:
        # single update
        D = p.Xd[np.ix_(p.rem, p.rem)]
        x = p.Xd[np.ix_(p.rem, t)]
        D = D - x @ mrdivide(x.T, p.Xd[np.ix_(t, t)])
        U = p.Xu[np.ix_(p.rem, p.rem)]
        x = p.Xu[np.ix_(p.rem, t)]
        U = U - x @ mrdivide(x.T, p.Xu[np.ix_(t, t)])
        p.down_v = copy.deepcopy(s0)

    elif p.bf and np.array_equal(p.down_v, s0) and \
            np.sum(np.logical_and(p.down_r, p.rem)) == p.m2:
        # no pruning
        return

    else:
        # normal cases
        try:
            R1 = sp.linalg.cholesky(p.Y2[np.ix_(s0, s0)])
        except sp.linalg.LinAlgError:
            p.f = True
            return
        else:
            p.f = False

        Q = mldivide(R1, mldivide(R1.T, np.eye(p.m2 + p.nf)))
        p.down_v = copy.deepcopy(s0)
        Yinv = np.zeros((s0.size, s0.size))
        Yinv[np.ix_(s0, s0)] = Q
        V = p.G[s0, :]
        U = Q @ V
        p.Gd[s0, :] = U

        try:
            R = sp.linalg.cholesky(V.T @ U - np.eye(p.m) * p.bound)
        except sp.linalg.LinAlgError:
            p.f = True
            p.ops[1] += 1
            return
        else:
            p.f = False

        U = mldivide(R.T, p.Gd[p.rem, :].T)
        D = Yinv[np.ix_(p.rem, p.rem)]
        U = D - U.T @ U

    p.ops[2] += p.m2
    p.down_r = copy.deepcopy(p.rem)
    p.p2[p.rem] = np.diag(U) / np.diag(D)
    p.Xd[np.ix_(p.rem, p.rem)] = D
    p.Xu[np.ix_(p.rem, p.rem)] = U
    p.p2[np.logical_not(p.rem)] = np.Inf
    p.downff = True
    L = p.p2 <= 0
    if np.any(L):
        # downwards pruning
        p.upff = False
        p.upf = False
        p.fx[L] = True
        p.rem[L] = False
        p.nf = np.sum(p.fx)
        p.m2 = np.sum(p.rem)
        p.n2 = p.n - p.nf


def _update(s: np.ndarray, params: BnBParams):
    # parameters unpacking
    p = params

    logging.debug('UPDATE - %s | %s | %s', p.ops, p.fx.sum(), p.rem.sum())

    # terminal cases to update the bound
    X = mldivide(sp.linalg.cholesky(p.Y2[np.ix_(s, s)]).T, p.G[s, :])
    lmbda = np.linalg.eig(X.T @ X)[0]
    p.ops[0] += 1
    bf0 = np.sum(lmbda < p.bound)
    if not bf0:
        p.B[p.ib] = np.min(lmbda)
        p.sset[p.ib, :] = np.nonzero(s)[0]
        bound0 = p.bound
        p.ib = np.argmin(p.B)
        p.bound = np.min(p.B)
        p.bf = bound0 == p.bound

    return bf0
