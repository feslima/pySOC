import warnings
from itertools import combinations

import numpy as np
import scipy as sp
from numpy.linalg import cond, norm, svd
from scipy.linalg import sqrtm

from pySOC.bnb import pb3wc
from pySOC.utils.matrixdivide import mldivide, mrdivide


def h_exact_local_method(Gy, Gyd, Juu, Jud, d, Wd, Wny):
    """ Determination of H matrix for c=Hy, explicit extended null space method
    exact local method based on Alstad et al., for all measurements available (2009) """
    nyt, nu = Gy.shape  # Total Number of measurements and Remaining Degrees of Freedom
    nd = d.shape[1]  # N° of Disturbances
    # Eq (15) Alstad et al. (2009): Optimal Sensitivity Matrix
    F = -((mrdivide(Gy, Juu)) @ Jud - Gyd)
    Gt = np.c_[Gy, Gyd]  # G tilde , Augmented Plant G.
    Ft = np.c_[F @ Wd, Wny]  # Eq (27) Alstad et al. Definition (F~)

    # Simplified H evaluation, based on Yelchuru and Skogestad (2012)
    H_exact = mrdivide(Gy.conj().T, (Ft @ Ft.conj().T))

    H_exact = (H_exact / np.linalg.norm(H_exact, ord='fro'))
    G_exact = H_exact @ Gy
    Mn_exact = mrdivide(sqrtm(Juu), G_exact)
    Md_exact = -mrdivide(sqrtm(Juu), G_exact) @ H_exact @ F @ Wd
    Mny_exact = -mrdivide(sqrtm(Juu), G_exact) @ H_exact @ Wny
    M_exact = np.c_[Md_exact, Mny_exact]
    worst_loss_exact = 0.5 * (np.max(np.linalg.svd(M_exact)[1], axis=0)) ** 2
    avg_loss_exact = (1 / (6 * (nyt + nd))) * \
        ((np.linalg.norm(M_exact, ord="fro")) ** 2)
    results_exact = {"H_exact": H_exact,
                     "G_exact": G_exact,
                     "M_exact": M_exact,
                     "worst_loss_exact": worst_loss_exact,
                     "avg_loss_exact": avg_loss_exact}
    return results_exact


def helm(Gy: np.ndarray, Gyd: np.ndarray, Juu: np.ndarray, Jud: np.ndarray,
         md: np.ndarray, me: np.ndarray, ss_size: int, nc_user: int = 1):
    """H evaluation for Exact local method (HELM) (possibly evaluating 
    subsets).

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
        Disturbances magnitude (1D Array with nd elements).

    me : np.ndarray
        Measurement errors (1D Array with ny elements).

    ss_size : int
        Subset size.

    nc_user : int
        Number of best SOC estructures do be returned.

    Returns
    -------
    Loss_list: np.ndarray
        Losses for each possible control structure, asceding order.

    sset_bnb: np.ndarray
        Subsets CV indexes for each possible control structure, 
        following `Loss_list` order.

    cond_list: np.ndarray
        Conditional number for each possible control structure in `Gy_ss_list`,
        following `Loss_list` order.

    H_matrix_list: list
        H matrix for each possible control structure, following `Loss_list`
        order.

    Gy_ss_list: list
        List containg Gy for each possible control structure, following
        `Loss_list` order.

    Gyd_ss_list: list
        List containg Gyd for each possible control structure, following
        `Loss_list` order.

    """
    # input sanitation
    nyt, nu = Gy.shape
    _, nd = Gyd.shape

    if (nu, nu) != Juu.shape:
        raise ValueError(("Juu has to be square, and its number of "
                          "rows/columns has to be the same number of "
                          "columns of Gy."))

    if (nu, nd) != Jud.shape:
        raise ValueError(("Jud must have nu rows by nd columns."))
    # matrices dimension checking

    if Juu.ndim != 2 or Jud.ndim != 2:
        raise ValueError(("Both Juu and Jud must be 2D Arrays."))

    if me.ndim != 1 or md.ndim != 1:
        raise ValueError(("Both me and Wd must be 1D Arrays."))

    # check for zero elements, replace by eps
    # if np.any(Gy == 0) or np.any(Gyd == 0) or np.any(Juu == 0) \
    #         or np.any(Jud == 0):
    #     Gy[Gy == 0] = np.spacing(1)
    #     Gyd[Gyd == 0] = np.spacing(1)
    #     Juu[Juu == 0] = np.spacing(1)
    #     Jud[Jud == 0] = np.spacing(1)

    if np.any(md == 0) or np.any(me == 0):
        raise ValueError(("Neither the disturbance magnitudes (Wd) or "
                          "measurement erros (me) can be exactly 0."))
    # Generating diagonal matrix for disturbances magnitudes
    Wd = np.diag(md)

    # check if nu <=  ss_size < ny
    if not nu <= ss_size <= nyt:
        raise ValueError(("n must follow nu <= n <= ny "
                          "({0:d} <= ss_size <= {1:d}).".format(nu, nyt)))

    index_CVs_ss = np.asarray(list(combinations(np.arange(nyt), ss_size)))
    c_size, _ = np.shape(index_CVs_ss)
    result_bnb = pb3wc(Gy, Gyd, md, me, Juu, Jud, ss_size, tlimit=np.inf,
                       nc=nc_user)
    _, sset_bnb, _, _, flag_bnb = result_bnb
    sset_bnb_size, _ = np.shape(sset_bnb)
    if flag_bnb:
        raise ValueError('Branch and Bound search could not converge...')
    if sset_bnb_size > c_size:
        raise ValueError(f'You tried to analyse more subsets than the possible combinations (number of sets you tried = {sset_bnb_size}'
                         f', Number of all possible combinations = {c_size}')

    index_Wny_ss = me[sset_bnb[:, 0]-1]

    for c in np.arange(1, ss_size):

        index_Wny_ss = np.c_[index_Wny_ss, me[sset_bnb[:, c]-1]]
    H_matrix_list = []
    Gy_ss_list = []
    Gyd_ss_list = []
    Loss_list = np.empty((sset_bnb.shape[0], 2))
    cond_list = np.empty((sset_bnb.shape[0], 1))

    for u in np.arange(sset_bnb.shape[0]):
        Gy_ss = Gy[sset_bnb[u, :]-1, :]
        Gyd_ss = Gyd[sset_bnb[u, :]-1, :]
        Wny_ss = np.diag(index_Wny_ss[u, :])
        F = -(mrdivide(Gy_ss, Juu) @ Jud - Gyd_ss)
        Ft = np.c_[F @ Wd, Wny_ss]
        # Simplified H evaluation, based on Yelchuru and Skogestad (2012)
        H = mrdivide(Gy_ss.T, (Ft @ Ft.T))

        H = (H / np.linalg.norm(H, ord='fro'))

        # Loss calculation
        G = H @ Gy_ss
        Md = - mrdivide(sqrtm(Juu), G) @ H @ F @ Wd
        Mny = -mrdivide(sqrtm(Juu), G) @ H @ Wny_ss
        M = np.c_[Md, Mny]
        worst_loss = 0.5 * (np.max(svd(M)[1], axis=0)) ** 2
        avg_loss = (1 / (6 * (ss_size + nd))) * ((norm(M, ord="fro")) ** 2)
        loss = np.c_[worst_loss, avg_loss]

        # append matrices
        H_matrix_list.append(H)
        Gy_ss_list.append(Gy_ss)
        Gyd_ss_list.append(Gyd_ss)
        Loss_list[[u], :] = loss
        cond_list[[u], :] = cond(Gy_ss)

    index_sorted = np.argsort(Loss_list[:, 0])
    cond_list = cond_list[index_sorted, :]

    H_matrix_list = [H_matrix_list[i] for i in index_sorted]
    Gy_ss_list = [Gy_ss_list[i] for i in index_sorted]
    Gyd_ss_list = [Gyd_ss_list[i] for i in index_sorted]
    Loss_list = Loss_list[index_sorted, :]
    sset_bnb = sset_bnb[index_sorted, :]

    return Loss_list, sset_bnb, cond_list, H_matrix_list, Gy_ss_list, \
        Gyd_ss_list


def h_extended_nullspace(Gy, Gyd, Juu, Jud, d, Wd, me, ss_size):
    nyt, nu = Gy.shape
    nd = d.shape[1]
    index_CVs_null = np.asarray(list(combinations(np.arange(nyt), ss_size)))
    index_Wny_null = me[0, index_CVs_null[:, 0]]

    for c in np.arange(1, ss_size):
        index_Wny_null = np.c_[index_Wny_null, me[0, index_CVs_null[:, c]]]

    H_null_list = []
    Gy_null_list = []
    Gyd_null_list = []
    Loss_null_list = np.empty([index_CVs_null.shape[0], 2])
    cond_null_list = np.empty([index_CVs_null.shape[0], 1])

    if nyt < nu + nd:
        warnings.warn('The number of measurements is less than the sum of the number of disturbances'
                      'and degrees of Freedom of the unconstrained problem (ny<nu+nd). Thus,'
                      'there are too few measurements to have Md = 0, be careful. '
                      'Instead, the expression for H is minimizing'
                      '||E||_f but not both ||E||_f and Md = 0 (consideration of the nullspace method).'
                      'Be careful in this special case, with the order of the CVs obtained.'
                      'Consider if possible, choosing ny >= nu + nd')

    for u in np.arange(index_CVs_null.shape[0]):
        Gy_null = Gy[index_CVs_null[u, :], :]
        Gyd_null = Gyd[index_CVs_null[u, :], :]
        Wny_null = np.diag(index_Wny_null[u, :])
        Gt_null = np.c_[Gy_null, Gyd_null]
        Mn_null = np.eye(np.shape(Gy_null)[1])
        Jt_null = np.c_[sp.linalg.sqrtm(Juu), mrdivide(
            sp.linalg.sqrtm(Juu), Juu)@Jud]
        F_null = -(mrdivide(Gy_null, Juu) @ Jud - Gyd_null)

        if nyt > nu+nd:
            H_null = mldivide(Mn_null, Jt_null) @ mrdivide(
                sp.linalg.pinv(mldivide(Wny_null, Gt_null)), Wny_null)
            Mny_null = mrdivide(sp.linalg.sqrtm(
                Juu), (H_null @ Gy_null)) @ H_null @ Wny_null

        if nyt == (nu + nd):
            H_null = mrdivide(mldivide(Mn_null, Jt_null), Gt_null)
            Mny_null = mrdivide(-Jt_null, Gt_null) @ Wny_null
        if nyt < (nu+nd):
            H_null = mrdivide(mldivide(Mn_null, Jt_null), Gy_null)
            Mny_null = mrdivide(sp.linalg.sqrtm(
                Juu), (H_null @ Gy_null)) @ H_null @ Wny_null

        aux_H = H_null
        H_null = H_null/sp.linalg.norm(H_null, ord='fro')
        Erf_md = aux_H @ Gt_null - Jt_null
        Md_null = - \
            sp.linalg.sqrtm(Juu)@ sp.linalg.inv(
                H_null@Gy_null) @ H_null @ F_null @ Wd
        M_null = np.c_[Md_null, Mny_null]

        worst_loss_null = 0.5 * (np.max(np.linalg.svd(M_null)[1], axis=0)) ** 2
        avg_loss_null = (1 / (6 * (ss_size + nd))) * \
            ((np.linalg.norm(M_null, ord="fro")) ** 2)
        Loss_null = np.c_[worst_loss_null, avg_loss_null]

        H_null_list.append(H_null)
        Gy_null_list.append(Gy_null)
        Gyd_null_list.append(Gyd_null)
        Loss_null_list[[u], :] = Loss_null
        cond_null_list[[u], :] = np.linalg.cond(Gy_null)

    index_sorted_null = np.argsort(Loss_null_list[:, 0])
    cond_null_list = cond_null_list[index_sorted_null, :]
    H_null_list = [H_null_list[i] for i in index_sorted_null]
    Gy_null_list = [Gy_null_list[i] for i in index_sorted_null]
    Gyd_null_list = [Gyd_null_list[i] for i in index_sorted_null]
    return np.c_[Loss_null_list[index_sorted_null, :], index_CVs_null[index_sorted_null, :], cond_null_list], \
        H_null_list, Gy_null_list, Gyd_null_list, index_sorted_null
