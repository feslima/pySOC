import warnings
from itertools import combinations

import numpy as np
from numpy.linalg import cond, norm, svd, LinAlgError
from scipy.linalg import sqrtm, pinv

from pysoc.bnb import pb3wc
from pysoc.utils.matrixdivide import mldivide, mrdivide


def helm(Gy: np.ndarray, Gyd: np.ndarray, Juu: np.ndarray, Jud: np.ndarray,
         md: np.ndarray, me: np.ndarray, ss_size: int, nc_user: int = 1):
    """H evaluation for Exact Local method (HELM).

    Parameters
    ----------
    Gy : np.ndarray
        Process model (ny-by-nu Gain matrix).

    Gyd : np.ndarray
        Disturbance model (ny-by-nd Gain matrix).

    Juu : np.ndarray
        Hessian with respect to unconstrained DOFs (nu-by-nu).

    Jud : np.ndarray
        Hessian with respect to unconstrained DOFs and disturbances (nu-by-nd).

    md : np.ndarray
        Disturbances magnitude (1D array with nd elements).

    me : np.ndarray
        Measurement errors (1D array with ny elements).

    ss_size : int
        Subset size.

    nc_user : int
        Number of best SOC estructures do be returned.

    Returns
    -------
    worst_loss_list: np.ndarray
        Worst case losses for each possible control structure, ascending order.

    average_loss_list: np.ndarray
        Average loss for each possible control structure. Following
        `worst_loss_list` order.

    sset_bnb: np.ndarray
        Subsets CV indexes for each possible control structure,
        following `worst_loss_list` order.

    cond_list: np.ndarray
        Conditional number for each possible control structure in `Gy_list`,
        following `worst_loss_list` order.

    H_list: list
        List of H matrices for each possible control structure, following
        `worst_loss_list` order.

    Gy_list: list
        List containg Gy for each possible control structure, following
        `worst_loss_list` order.

    Gyd_list: list
        List containg Gyd for each possible control structure, following
        `worst_loss_list` order.

    F_list : list
        List of locally optimal sensitivity matrices, following
        `worst_loss_list` order. The elements of F represent the optimal
        change in y (measurements) due to changes in d (disturbances).

    References
    ----------
    .. [1] V. Alstad, S. Skogestad, E. S. Hori. Optimal measurement
        combinations as controlled variables. Journal of Process Control,
        19(1):138-148, 2009.

    .. [2] G. P. Rangaiah, V. Kariwala. Plantwide Control: Recent Developments
        and Applications. 2012.

    .. [3] V. Kariwala and Y. Cao, Bidirectional Branch and Bound for
        Controlled Variable Selection: Part II. Exact Local Method for
        Self-optimizing Control, Computers and Chemical Engineering,
        33(8):1402:1412, 2009.
    """
    # input sanitation
    if Gy.ndim != 2 or Gyd.ndim != 2:
        raise ValueError("Both Gy and Gyd must be 2D arrays.")

    nyt, nu = Gy.shape
    _, nd = Gyd.shape

    if (nu, nu) != Juu.shape:
        raise ValueError("Juu has to be square, and its number of "
                         "rows/columns has to be the same number of "
                         "columns of Gy.")

    if (nu, nd) != Jud.shape:
        raise ValueError("Jud must have nu rows by nd columns.")
    # matrices dimension checking

    if Juu.ndim != 2 or Jud.ndim != 2:
        raise ValueError("Both Juu and Jud must be 2D arrays.")

    if me.ndim != 1 or md.ndim != 1:
        raise ValueError("Both me and Wd must be 1D arrays.")

    # check for zero elements, replace by eps
    if np.any(Gy == 0) or np.any(Gyd == 0) or np.any(Juu == 0) \
            or np.any(Jud == 0):
        Gy[Gy == 0] = np.spacing(1)
        Gyd[Gyd == 0] = np.spacing(1)
        Juu[Juu == 0] = np.spacing(1)
        Jud[Jud == 0] = np.spacing(1)

    if np.any(md == 0) or np.any(me == 0):
        raise ValueError("Neither the disturbance magnitudes (Wd) or "
                         "measurement errors (me) can be exactly 0.")
    # Generating diagonal matrix for disturbances magnitudes
    Wd = np.diag(md)

    # check if nu <=  ss_size < ny
    if not nu <= ss_size <= nyt:
        raise ValueError("n must follow nu <= n <= ny "
                         "({0:d} <= ss_size <= {1:d}).".format(nu, nyt))

    index_CVs_ss = np.asarray(list(combinations(np.arange(nyt), ss_size)))
    c_size, _ = np.shape(index_CVs_ss)
    _, sset_bnb, _, _, flag_bnb = pb3wc(Gy, Gyd, md, me, Juu, Jud, ss_size,
                                        tlimit=np.inf, nc=nc_user)
    sset_bnb_size, _ = np.shape(sset_bnb)

    if flag_bnb:
        raise ValueError('Branch and Bound search could not converge.')

    if sset_bnb_size > c_size:
        raise ValueError("You tried to analyse more subsets than the possible "
                         "combinations (number of sets you tried = {0:d}. "
                         "Number of all possible combinations = {1:d}."
                         .format(sset_bnb_size, c_size))

    index_Wny_ss = me[sset_bnb[:, 0] - 1]

    for c in np.arange(1, ss_size):

        index_Wny_ss = np.c_[index_Wny_ss, me[sset_bnb[:, c] - 1]]
    H_list = []
    Gy_list = []
    Gyd_list = []
    F_list = []
    worst_loss_list = np.full((sset_bnb.shape[0], ), np.NaN)
    average_loss_list = np.full((sset_bnb.shape[0], ), np.NaN)
    cond_list = np.full((sset_bnb.shape[0], ), np.NaN)

    for u in np.arange(sset_bnb.shape[0]):
        Gy_ss = Gy[sset_bnb[u, :]-1, :]
        Gyd_ss = Gyd[sset_bnb[u, :]-1, :]
        Wny_ss = np.diag(index_Wny_ss[u, :])

        try:
            F = -(mrdivide(Gy_ss, Juu) @ Jud - Gyd_ss)
            Ft = np.c_[F @ Wd, Wny_ss]

            # Simplified H evaluation, based on Yelchuru and Skogestad (2012)
            H = mrdivide(Gy_ss.T, (Ft @ Ft.T))

            H = H / norm(H, ord='fro')

            # Loss calculation
            G = H @ Gy_ss
            Md = -mrdivide(sqrtm(Juu), G) @ H @ F @ Wd
            Mny = -mrdivide(sqrtm(Juu), G) @ H @ Wny_ss
            M = np.c_[Md, Mny]
            worst_loss = 0.5 * (np.max(svd(M)[1], axis=0)) ** 2
            avg_loss = (1 / (6 * (ss_size + nd))) * (norm(M, ord="fro") ** 2)
        except LinAlgError:
            H = np.full((nu, nyt), np.NaN)
            F = np.full(Jud.shape, np.NaN)
            worst_loss = np.Inf
            avg_loss = np.Inf

        # append matrices
        H_list.append(H)
        Gy_list.append(Gy_ss)
        Gyd_list.append(Gyd_ss)
        F_list.append(F)
        worst_loss_list[u] = worst_loss
        average_loss_list[u] = avg_loss
        cond_list[u] = cond(Gy_ss)

    return worst_loss_list, average_loss_list, sset_bnb, cond_list,
    H_list, Gy_list, Gyd_list, F_list


def hen(Gy: np.ndarray, Gyd: np.ndarray, Juu: np.ndarray, Jud: np.ndarray,
        md: np.ndarray, me: np.ndarray, ss_size: int):
    """H evaluation for Extendend Nullspace method (HEN).

    The null space method ignores the implementation error. The loss depends
    entirely on the setpoint error. The central idea of the null space method
    is that the loss due to setpoint error can be reduced to zero if the
    optimal value of the CVs does not change with disturbances.

    Therefore, H is selected such that:

    .. math:: H \\times F = 0

    Parameters
    ----------
    Gy : np.ndarray
        Process model (ny-by-nu Gain matrix).

    Gyd : np.ndarray
        Disturbance model (ny-by-nd Gain matrix).

    Juu : np.ndarray
        Hessian with respect to unconstrained DOFs (nu-by-nu).

    Jud : np.ndarray
        Hessian with respect to unconstrained DOFs and disturbances (nu-by-nd).

    md : np.ndarray
        Disturbances magnitude (1D array with nd elements).

    me : np.ndarray
        Measurement errors (1D array with ny elements).

    ss_size : int
        Subset size.

    nc_user : int
        Number of best SOC estructures do be returned.

    Returns
    -------
    worst_loss_list: np.ndarray
        Worst case losses for each possible control structure, ascending order.

    average_loss_list: np.ndarray
        Average loss for each possible control structure. Following
        `worst_loss_list` order.

    sset_bnb: np.ndarray
        Subsets CV indexes for each possible control structure,
        following `worst_loss_list` order.

    cond_list: np.ndarray
        Conditional number for each possible control structure in `Gy_ss_list`,
        following `worst_loss_list` order.

    H_list: list
        List of H matrices for each possible control structure, following
        `worst_loss_list` order.

    Gy_list: list
        List containg Gy for each possible control structure, following
        `worst_loss_list` order.

    Gyd_list: list
        List containg Gyd for each possible control structure, following
        `worst_loss_list` order.

    F_list : list
        List of locally optimal sensitivity matrices, following
        `worst_loss_list` order. The elements of F represent the optimal
        change in y (measurements) due to changes in d (disturbances).

    References
    ----------
    .. [1] V. Alstad, S. Skogestad. Null Space Method for Selecting Optimal
        Measurement Combinations as Controlled Variables. Ind. Eng. Chem. Res.,
        46(3):846-853, 2007.
    .. [2] G. P. Rangaiah, V. Kariwala. Plantwide Control: Recent Developments
        and Applications. 2012.
    """
    # input sanitation
    if Gy.ndim != 2 or Gyd.ndim != 2:
        raise ValueError("Both Gy and Gyd must be 2D Arrays.")

    nyt, nu = Gy.shape
    _, nd = Gyd.shape

    if (nu, nu) != Juu.shape:
        raise ValueError("Juu has to be square, and its number of "
                         "rows/columns has to be the same number of "
                         "columns of Gy.")

    if (nu, nd) != Jud.shape:
        raise ValueError("Jud must have nu rows by nd columns.")
    # matrices dimension checking

    if Juu.ndim != 2 or Jud.ndim != 2:
        raise ValueError("Both Juu and Jud must be 2D Arrays.")

    if me.ndim != 1 or md.ndim != 1:
        raise ValueError("Both me and Wd must be 1D Arrays.")

    # check for zero elements, replace by eps
    if np.any(Gy == 0) or np.any(Gyd == 0) or np.any(Juu == 0) \
            or np.any(Jud == 0):
        Gy[Gy == 0] = np.spacing(1)
        Gyd[Gyd == 0] = np.spacing(1)
        Juu[Juu == 0] = np.spacing(1)
        Jud[Jud == 0] = np.spacing(1)

    if np.any(md == 0) or np.any(me == 0):
        raise ValueError("Neither the disturbance magnitudes (Wd) or "
                         "measurement errors (me) can be exactly 0.")
    # Generating diagonal matrix for disturbances magnitudes
    Wd = np.diag(md)

    index_CVs_null = np.asarray(list(combinations(np.arange(nyt), ss_size)))
    index_Wny_null = me[index_CVs_null[:, 0]]

    for c in np.arange(1, ss_size):
        index_Wny_null = np.c_[index_Wny_null, me[index_CVs_null[:, c]]]

    H_list = []
    Gy_list = []
    Gyd_list = []
    F_list = []
    worst_loss_list = np.full((index_CVs_null.shape[0], ), np.NaN)
    average_loss_list = np.full((index_CVs_null.shape[0], ), np.NaN)
    cond_list = np.full((index_CVs_null.shape[0],), np.NaN)

    if nyt < nu + nd:
        warnings.warn("The number of measurements is less than the sum of the "
                      "number of disturbances and degrees of freedom of the "
                      "unconstrained problem(ny < nu+nd). Thus, there are too "
                      "few measurements to have Md = 0. Instead, the "
                      "expression for H is minimizing ||E||_f but not both "
                      "||E||_f and Md = 0 (consideration of the nullspace "
                      "method).\nBe careful in this special case, with the "
                      "order of the CVs obtained. Consider if possible, "
                      "choosing ny >= nu + nd")

    for u in np.arange(index_CVs_null.shape[0]):
        Gy_null = Gy[index_CVs_null[u, :], :]
        Gyd_null = Gyd[index_CVs_null[u, :], :]
        Wny = np.diag(index_Wny_null[u, :])
        Gt = np.c_[Gy_null, Gyd_null]
        Mn = np.eye(Gy_null.shape[1])
        Jt = np.c_[sqrtm(Juu), mrdivide(sqrtm(Juu), Juu) @ Jud]
        F = -(mrdivide(Gy_null, Juu) @ Jud - Gyd_null)

        if nyt > nu + nd:
            H = mldivide(Mn, Jt) @ mrdivide(pinv(mldivide(Wny, Gt)), Wny)
            Mny = mrdivide(sqrtm(Juu), (H @ Gy_null)) @ H @ Wny

        elif nyt == (nu + nd):
            H = mrdivide(mldivide(Mn, Jt), Gt)
            Mny = mrdivide(-Jt, Gt) @ Wny

        else:
            H = mrdivide(mldivide(Mn, Jt), Gy_null)
            Mny = mrdivide(sqrtm(Juu), (H @ Gy_null)) @ H @ Wny

        H = H/norm(H, ord='fro')
        Md = -mrdivide(sqrtm(Juu), H @ Gy_null) @ H @ F @ Wd
        M = np.c_[Md, Mny]

        worst_loss = 0.5 * (np.max(svd(M)[1], axis=0)) ** 2
        avg_loss = (1 / (6 * (ss_size + nd))) * (norm(M, ord="fro") ** 2)

        H_list.append(H)
        Gy_list.append(Gy_null)
        Gyd_list.append(Gyd_null)
        F_list.append(F)

        worst_loss_list[u] = worst_loss
        average_loss_list[u] = avg_loss
        cond_list[u] = cond(Gy_null)

    index_sorted = np.argsort(worst_loss_list)

    worst_loss_list = worst_loss_list[index_sorted]
    average_loss_list = average_loss_list[index_sorted]
    cond_list = cond_list[index_sorted]

    # generate sorted lists to return
    zip_pairs = zip(index_sorted.tolist(), H_list, Gy_list, Gyd_list, F_list)
    H_list, Gy_list, Gyd_list, F_list = zip(*[(h, gy, gyd, f)
                                              for idx, h, gy, gyd, f in
                                              sorted(zip_pairs,
                                                     key=lambda pair: pair[0])]
                                            )

    return worst_loss_list, average_loss_list, cond_list, \
        H_list, Gy_list, Gyd_list, F_list
