import numpy as np
import scipy as sp
from scipy.linalg import sqrtm
import timeit
from pySOC.utils.matrixdivide import mldivide, mrdivide


def pb3wc(Gy, Gyd, Wd, Wn, Juu, Jud, n, tlimit=np.inf, nc=1):
    """ Partial Bidirectional Branch and Bound algorithm for Worst-Case Scenario.

    Implementation of PB3 algorithm to select measurement, whose combinations can be used as controlled variables.
    The measurements are select do provide minimum worst-case loss in terms of self-optimization control based on the
    following static optimization problem:

    .. math::
        \\newline

        \\text{minimize:}

        J = f(u,d,y)

        \\newline

        \\text{subject to:}

        y = G_{y} \\times u + G^{y}_{d} \\times W_{d} \\times d + W_{n} \\times e

        \\newline

    Where u are the degrees of freedom, d the disturbances, y the measurements and e is the control error.

    Parameters
    ----------
    Gy : ndarray
        Process model (ny-by-nu).
    Gyd : ndarray
        Disturbance model (ny-by-nd).
    Wd : ndarray
        Diagonal matrix containing magnitudes of disturbances (nd-by-nd).
    Wn : ndarray
        Diagonal matrix containing magnitudes of implementation errors (n-by-n).
    Juu : ndarray
        Hessian matrix of J relative u, evaluated at optimum (nu-by-nu).
    Jud : ndarray
        Hessian matrix of J relative to u and d, evaluated at optimum (nu-by-nd).
    n : int
        Number of measurements to be selected. Must be nu <= n <= ny.
    tlimit : {int, None}, optional
        Time limit to run the code (default is np.inf).
    nc : int, optional
        Number of best subsets to be selected (default is 1).

    Returns
    -------
    B : ndarray
        nc-by-1 vector of the worst-case loss of selected subsets.
    sset : ndarray
        nc-by-nu indices of selected subsets.
    ops : ndarray
        1-by-4 array containing the number of nodes evaluated.
        counters: 0) terminal; 1) nodes; 2) sub-nodes; 3) calls.
    ctime : float
        Computation time used.
    flag : bool
        Status flag of success.
        0 if successful, 1 otherwise (`tlimit` < np.inf)

    Raises
    ------
    ValueError
        When `n` < nu. Number of measurements must be greater than the number of degrees of freedom.

    Notes
    -----
    Formal definition of `Juu`:

    .. math:: J_{uu} = \\frac{\partial^2 J}{\partial u^2}

    Formal definition of `Jud`:

    .. math:: J_{ud} = \\frac{\partial^2 J}{\partial u\partial d}

    References
    ----------
    .. [1] I. J. Halvorsen, S. Skogestad, J. C. Morud, and V. Alstad. Optimal selection of controlled variables.
        Ind. Eng. Chem. Res., 42(14):3273-3284, 2003.
    .. [2] V. Kariwala, Y. Cao, and S. Janardhanan. Local self-optimizing control with average loss minimization.
        Ind. Eng. Chem. Res., 47(4):1150-1158, 2008.
    .. [3] V. Kariwala and Y. Cao, Bidirectional Branch and Bound for Controlled Variable Selection: Part II.
        Exact Local Method for Self-optimizing Control, Computers and Chemical Engineering, 33(8):1402:1412, 2009.


    """
    # function definitions
    def bbL3sub(fx0, rem0):
        nonlocal ctime0, tlimit, flag  # to access upper scope
        # recursive solver
        bn = 0
        if timeit.default_timer() - ctime0 > tlimit:
            flag = True
            return bn

        nonlocal ops, fx, rem, nf, m2, n2, n, nf
        ops[0, 3] += 1
        fx = fx0.copy()
        rem = rem0.copy()
        nf = np.sum(fx)
        m2 = np.sum(rem)
        n2 = n - nf  # line 134

        nonlocal f, downf, upf, bf, bound
        while not f and m2 > n2 and n2 >= 0:  # loop for second branches
            while not f and m2 > n2 and n2 > 0 and (not downf or not upf or not bf):
                # loop for bidirection pruning
                if (not upf or not bf) and n2 <= m and bound:
                    # if np.array_equal(np.array([[21, 9, 63, 9]]), ops):
                        #a = 1
                    upprune()
                else:
                    upf = True

                if not f and m2 > n2 and m2 > 0 and (not downf or not bf) and bound:
                    # downwards pruning
                    downprune()
                else:
                    downf = True

                bf = True
            # pruning loop end

            if f or m2 < n2 or n2 < 0:
                # pruned cases
                return bn
            elif m2 == n2 or not n2:
                # terminal cases
                break

            if n2 == 1:  # one or more elements to be fixed
                nonlocal upff, q2, p2
                if upff:
                    q2[np.logical_not(rem).reshape(-1), 0] = 0
                    idk = np.argmax(q2)
                else:
                    p2[np.logical_not(rem).reshape(-1), 0] = np.inf
                    idk = np.argmin(p2)

                s = fx.copy()
                s[0, idk] = True
                bn = update(s) - 1
                if bn > 0:
                    bn += np.sum(fx0) - nf
                    return bn

                rem[0, idk] = False
                m2 -= 1  # line 175
                downf = False
                upf = True
                continue

            if m2 - n2 == 1:  # one more element to be removed
                if downff:  # line 181
                    p2[np.logical_not(rem).reshape(-1), 0] = 0
                    idk = np.argmax(p2)
                else:
                    q2[np.logical_not(rem).reshape(-1), 0] = np.inf
                    idk = np.argmin(q2)

                rem[0, idk] = False
                s = np.logical_or(fx, rem)  # line 189
                update(s)
                fx[0, idk] = True
                nf += 1
                n2 -= 1
                m2 -= 1
                upf = False
                downf = True
                continue

            # save data from bidirectional branching
            nonlocal Xd, Xu, downV, downR
            fx1 = fx.copy()
            rem1 = rem.copy()
            p0 = p2.copy()
            q0 = q2.copy()
            D0 = Xd.copy()
            U0 = Xu.copy()
            dV0 = downV.copy()
            dR0 = downR.copy()

            if n2 - m < 0.75 * m2:  # upwards branching
                if bound:
                    p2[np.logical_not(rem).reshape(-1), 0] = np.inf
                    idd = np.argmin(p2)
                else:
                    q2[np.logical_not(rem).reshape(-1), 0] = 0
                    idd = np.argmax(q2)

                fx[0, idd] = True  # line 220
                rem1[0, idd] = False
                downf = True
                upf = False
                bn = bbL3sub(fx, rem1) - 1
                downf = False
                upf = True
            else:  # downward branching
                if bound:  # line 228
                    p2[np.logical_not(rem), 0] = 0
                    idd = np.argmax(p2)
                else:
                    q2[np.logical_not(rem), 0] = np.inf
                    idd = np.argmin(q2)

                fx1[0, idd] = True  # line 235
                downf = False
                rem1[0, idd] = False
                downf = False
                upf = True
                bn = bbL3sub(fx, rem1) - 1
                downf = True
                upf = False
                if q0[idd, 0] <= bound and n == m:
                    return bn

            # check pruning conditions
            if bn > 0:
                bn += -np.sum(rem0) + m2
                return bn

            # recover data saved before the first branch
            fx = fx1.copy()  # line 253
            rem = rem1.copy()
            f = False
            p2 = p0.copy()
            q2 = q0.copy()
            nf = np.sum(fx)
            n2 = n - nf
            m2 = np.sum(rem)
            Xd = D0.copy()
            Xu = U0.copy()
            downV = dV0.copy()
            downR = dR0.copy()

        if not f:  # terminal cases
            if m2 == n2:
                bn = update(np.logical_or(fx, rem)) - 1
            elif not n2:
                bn = update(fx) - 1

        bn += np.sum(fx0) - nf
        return bn

    def upprune():
        # partially upwards branching
        nonlocal upf, Y2, fx, f
        upf = True
        try:
            R1 = sp.linalg.cholesky(Y2[np.ix_(fx.reshape(-1), fx.reshape(-1))])
            f = False
        except sp.linalg.LinAlgError:
            f = True
            return

        nonlocal G
        X1 = mldivide(R1.conj().T,  G[fx.reshape(-1), :])
        D = X1.conj().T @ X1
        tD = np.trace(D)  # line 285
        nonlocal bound, n2, m, ops
        if tD < bound and n2 < m:  # m eigen values < bound
            ops[0, 1] += 1
            f = True
            return

        if tD / m > bound and n2 == m:  # at least one eigen value > bound
            return

        if m > 2:  # general cases
            bf0 = np.sum(np.linalg.eig(D)[0] < bound)
            ops[0, 0] += 1
            if bf0 > n2:  # not feasible
                f = True
                return

            if bf0 != n2:  # no pruning
                return

            D = np.eye(m) * bound - D

        else:  # special cases without using eig
            D = np.eye(m) * bound - D
            try:
                R = sp.linalg.cholesky(D)
                f = False
            except sp.linalg.LinAlgError:
                f = True

            ops[0, 1] += 1
            # ~f: m eigen values < bound
            # f: at least 1 eigen value > bound
            if (f and n2 > 1) or not f and n2 < m:
                f = not f
                return

            if n2 == 1:  # for m = 2, nf = 1
                try:
                    R = sp.linalg.cholesky(-D)
                    f = False
                except sp.linalg.LinAlgError:
                    f = True

                if not f:  # m eigen values > bound, no pruning
                    return

                # otherwise only one eigen value < bound
                f = not f

        nonlocal rem, h2, m2, q2
        R2 = mldivide(R1.conj().T, Y2[np.ix_(fx.reshape(-1), rem.reshape(-1))])
        R3 = h2[rem.reshape(-1)] - np.sum(R2 * R2, axis=0,
                                          keepdims=True).conj().T
        X2 = G[rem.reshape(-1), :] - R2.conj().T @ X1
        ops[0, 2] += m2
        q2[:] = np.inf
        if n2 == 1 or m > 2:
            q2[rem.reshape(-1)] = np.sum(X2.conj().T * mldivide(D, X2.conj().T), axis=0, keepdims=True).conj().T / R3 \
                - 1
        else:
            X = mldivide(R.conj().T, X2.conj().T)  # line 333
            q2[rem.reshape(-1)] = np.sum(X * X, axis=0,
                                         keepdims=True).conj().T / R3 - 1

        nonlocal upff, downf, downff
        upff = True  # line 336
        L = q2 <= 0
        if L.any():
            # upwards pruning
            downf = False
            downff = False
            rem[0, L.reshape(-1)] = False
            m2 = np.sum(rem)
            q2[L.reshape(-1), 0] = np.inf

    def downprune():
        # downwards pruning
        nonlocal downf, fx, rem, downV, bf, downR, Xd, Xu, Y2, f, m2, nf, G, Gd, m, bound, ops, p2
        downf = True
        s0 = np.logical_or(fx, rem)
        t = np.logical_xor(downV, s0)
        if bf and np.sum(t) == 1 and downR[t]:
            # single update
            D = Xd[np.ix_(rem.reshape(-1), rem.reshape(-1))]
            x = Xd[np.ix_(rem.reshape(-1), t.reshape(-1))]
            D = D - x @ (mrdivide(x.conj().T,
                                  Xd[np.ix_(t.reshape(-1), t.reshape(-1))]))
            U = Xu[np.ix_(rem.reshape(-1), rem.reshape(-1))]
            x = Xu[np.ix_(rem.reshape(-1), t.reshape(-1))]
            U = U - x @ (mrdivide(x.conj().T,
                                  Xu[np.ix_(t.reshape(-1), t.reshape(-1))]))
            downV = s0.copy()
        elif bf and np.array_equal(downV, s0) and np.sum(np.logical_and(downR, rem)) == m2:
            # no pruning
            return
        else:
            # normal cases
            try:
                R1 = sp.linalg.cholesky(
                    Y2[np.ix_(s0.reshape(-1), s0.reshape(-1))])
                f = False
            except sp.linalg.LinAlgError:
                f = True
                return

            Q = mldivide(R1, mldivide(
                R1.conj().T, np.eye(m2 + nf)))  # line 371
            downV = s0.copy()
            Yinv = np.zeros((s0.size, s0.size))
            Yinv[np.ix_(s0.reshape(-1), s0.reshape(-1))] = Q
            V = G[s0.reshape(-1), :]
            U = Q @ V
            Gd[s0.reshape(-1), :] = U
            try:
                R = sp.linalg.cholesky(V.conj().T @ U - np.eye(m) * bound)
                f = False
            except sp.linalg.LinAlgError:
                f = True
                ops[0, 1] += 1
                return

            U = mldivide(R.conj().T, Gd[rem.reshape(-1), :].conj().T)
            D = Yinv[np.ix_(rem.reshape(-1), rem.reshape(-1))]
            U = D - U.conj().T @ U

        ops[0, 2] += m2
        downR = rem.copy()
        p2[rem.reshape(-1), :] = np.expand_dims(np.diag(U),
                                                axis=1) / np.expand_dims(np.diag(D), axis=1)
        Xd[np.ix_(rem.reshape(-1), rem.reshape(-1))] = D
        Xu[np.ix_(rem.reshape(-1), rem.reshape(-1))] = U
        p2[np.logical_not(rem).reshape(-1), :] = np.inf

        nonlocal downff, upff, upf, n2, n
        downff = True
        L = p2 <= 0
        if L.any():
            # downwards pruning
            upff = False
            upf = False
            fx[0, L.reshape(-1)] = True
            rem[0, L.reshape(-1)] = False
            nf = np.sum(fx)
            m2 = np.sum(rem)
            n2 = n - nf

    def update(s):
        # terminal cases to update the bound
        nonlocal Y2, G, ops, bound, B, ib, sset, bf
        X = mldivide(sp.linalg.cholesky(
            Y2[np.ix_(s.reshape(-1), s.reshape(-1))]).conj().T, G[s.reshape(-1), :])
        lambda_v = np.linalg.eig(X.conj().T @ X)[0]
        ops[0, 0] += 1
        bf0 = np.sum(lambda_v < bound)
        if not bf0:
            B[ib - 1, 0] = np.min(lambda_v)  # avoid sorting
            sset[ib - 1, :] = np.nonzero(s)[1]
            bound0 = bound
            bound = np.min(B)
            ib = np.argmin(B) + 1
            bf = bound0 == bound

        return bf0

    # default inputs and outputs
    flag = False
    ctime0 = timeit.default_timer()
    r, m = Gy.shape

    if n < m:
        raise ValueError("n must be larger than the number of inputs.")

    # prepare matrices
    Y = np.hstack(((mrdivide(Gy, Juu) @ Jud - Gyd) @ Wd, Wn))
    G = mrdivide(Gy, sqrtm(Juu))
    Y2 = Y @ Y.conj().T
    Gd = G.copy()
    Xd = Y2.copy()
    Xu = Xd.copy()
    # expand_dims to make column vector
    h2 = np.expand_dims(np.diag(Y2), axis=1)
    q2 = np.expand_dims(np.diag(G @ G.conj().T), axis=1) / h2
    p2 = q2.copy()
    # counters: 1) terminal; 2) nodes; 3) sub-nodes; 4) calls
    ops = np.zeros((1, 4))
    B = np.zeros((nc, 1))
    sset = np.zeros((nc, n))
    ib = 1
    bound = 0
    fx = np.zeros((1, r), dtype=bool)
    rem = np.ones((1, r), dtype=bool)
    downV = fx.copy()
    downR = fx.copy()
    nf = 0
    n2 = n
    m2 = r
    # Initialize flags
    f = False
    bf = False
    downf = False
    downff = False
    upf = False
    upff = True
    # the recursive solver
    bbL3sub(fx, rem)  # line 116
    idx = np.argsort(0.5 / B, axis=None)
    B = np.sort(0.5 / B, axis=0)
    sset = np.sort(sset[idx, :], axis=1)
    ctime = timeit.default_timer() - ctime0

    return B, sset, ops, ctime, flag
