import numpy as np
from aux_files.matrixdivide import mldivide, mrdivide
from scipy.linalg import sqrtm
from itertools import combinations
from bnb.bnb import pb3wc
import scipy as sp
import warnings

def h_exact_local_method(Gy, Gyd, Juu, Jud, d, Wd, Wny):
    """ Determination of H matrix for c=Hy, explicit extended null space method
    exact local method based on Alstad et al., for all measurements available (2009) """
    nyt, nu = Gy.shape  # Total Number of measurements and Remaining Degrees of Freedom
    nd = d.shape[1]  # N° of Disturbances
    F = -((mrdivide(Gy, Juu)) @ Jud - Gyd)  # Eq (15) Alstad et al. (2009): Optimal Sensitivity Matrix
    Gt = np.c_[Gy, Gyd]  # G tilde , Augmented Plant G.
    Ft = np.c_[F @ Wd, Wny]  # Eq (27) Alstad et al. Definition (F~)
    H_exact = mrdivide((mldivide(Ft @ Ft.conj().T, Gy)), mrdivide(Gy.conj().T, (Ft @ Ft.conj().T)) @ Gy) @ sqrtm(
        Juu)  # Eq (31) Alstad et al. Exact local method;
    H_exact = (H_exact / np.linalg.norm(H_exact, ord=2)).conj().T
    G_exact = H_exact @ Gy
    Mn_exact = mrdivide(sqrtm(Juu), G_exact)
    Md_exact = -mrdivide(sqrtm(Juu), G_exact) @ H_exact @ F @ Wd
    Mny_exact = -mrdivide(sqrtm(Juu), G_exact) @ H_exact @ Wny
    M_exact = np.c_[Md_exact, Mny_exact]
    worst_loss_exact = 0.5 * (np.max(np.linalg.svd(M_exact)[1], axis=0)) ** 2
    avg_loss_exact = (1 / (6 * (nyt + nd))) * ((np.linalg.norm(M_exact, ord="fro")) ** 2)
    results_exact = {"H_exact": H_exact,
                     "G_exact": G_exact,
                     "M_exact": M_exact,
                     "worst_loss_exact": worst_loss_exact,
                     "avg_loss_exact": avg_loss_exact}
    return results_exact


def h_exact_local_method_ss(Gy, Gyd, Juu, Jud, d, Wd, me, ss_size, nc_user):
    nyt, nu = Gy.shape
    nd = d.shape[1]
    index_CVs_ss = np.asarray(list(combinations(np.arange(nyt), ss_size)))
    c_size = np.shape(index_CVs_ss)[0]
    result_bnb = pb3wc(Gy, Gyd, Wd, np.diag(me.reshape(-1)), Juu, Jud, ss_size, tlimit=np.inf, nc=nc_user)
    sset_bnb = result_bnb[1].astype(np.int64)
    sset_bnb_size = np.shape(sset_bnb)[0]
    flag_bnb = result_bnb[-1]
    if flag_bnb:
        raise ValueError('Branch and Bound search could not converge...')
    if sset_bnb_size > c_size:
        raise ValueError(f'You tried to analyse more subsets than the possible combinations (number of sets you tried = {sset_bnb_size}'
            f', Number of all possible combinations = {c_size}')

    index_Wny_ss = me[0, sset_bnb[:, 0]]

    for c in np.arange(1, ss_size):
        # index_Wny_ss = np.c_[index_Wny_ss, me[0, index_CVs_ss[:, c]]]
        index_Wny_ss = np.c_[index_Wny_ss, me[0, sset_bnb[:, c]]]
    H_matrix_list = []
    Gy_ss_list = []
    Gyd_ss_list = []
    Loss_list = np.empty([sset_bnb.shape[0], 2])
    cond_list = np.empty([sset_bnb.shape[0], 1])
    res_eq = np.empty([sset_bnb.shape[0], 1])

    for u in np.arange(sset_bnb.shape[0]):
        Gy_ss = Gy[sset_bnb[u, :], :]
        Gyd_ss = Gyd[sset_bnb[u, :], :]
        Wny_ss = np.diag(index_Wny_ss[u, :])
        F_ss = -(mrdivide(Gy_ss, Juu) @ Jud - Gyd_ss)
        Ft_ss = np.c_[F_ss @ Wd, Wny_ss]
        H_exact_ss = mrdivide((mldivide(Ft_ss @ Ft_ss.conj().T, Gy_ss)),
                              mrdivide(Gy_ss.conj().T, (Ft_ss @ Ft_ss.conj().T)) @ Gy_ss) @ sqrtm(Juu)
        H_exact_ss = (H_exact_ss / np.linalg.norm(H_exact_ss, ord='fro')).conj().T


        # Loss calculation
        G_exact_ss = H_exact_ss @ Gy_ss
        Md_exact_ss = -mrdivide(sqrtm(Juu), G_exact_ss) @ H_exact_ss @ F_ss @ Wd
        Mny_exact_ss = -mrdivide(sqrtm(Juu), G_exact_ss) @ H_exact_ss @ Wny_ss
        M_exact_ss = np.c_[Md_exact_ss, Mny_exact_ss]
        worst_loss_exact_ss = 0.5 * (np.max(np.linalg.svd(M_exact_ss)[1], axis=0)) ** 2
        avg_loss_exact_ss = (1 / (6 * (ss_size + nd))) * ((np.linalg.norm(M_exact_ss, ord="fro")) ** 2)
        Loss_ss = np.c_[worst_loss_exact_ss, avg_loss_exact_ss]

        # append matrices
        H_matrix_list.append(H_exact_ss)
        Gy_ss_list.append(Gy_ss)
        Gyd_ss_list.append(Gyd_ss)
        Loss_list[[u], :] = Loss_ss
        cond_list[[u], :] = np.linalg.cond(Gy_ss)

    index_sorted = np.argsort(Loss_list[:, 0])
    cond_list = cond_list[index_sorted, :]
    # res_eq = res_eq[index_sorted, :]
    H_matrix_list = [H_matrix_list[i] for i in index_sorted]
    Gy_ss_list = [Gy_ss_list[i] for i in index_sorted]
    Gyd_ss_list = [Gyd_ss_list[i] for i in index_sorted]
    return np.c_[Loss_list[index_sorted, :], sset_bnb[index_sorted, :], cond_list], \
           H_matrix_list, Gy_ss_list, Gyd_ss_list, index_sorted

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
        Gy_null  = Gy[index_CVs_null[u, :], :]
        Gyd_null = Gyd[index_CVs_null[u, :], :]
        Wny_null = np.diag(index_Wny_null[u, :])
        Gt_null  = np.c_[Gy_null, Gyd_null]
        Mn_null  = np.eye(np.shape(Gy_null)[1])
        Jt_null  = np.c_[sp.linalg.sqrtm(Juu), mrdivide(sp.linalg.sqrtm(Juu),Juu)@Jud]
        F_null = -(mrdivide(Gy_null, Juu) @ Jud - Gyd_null)

        if nyt > nu+nd:
            H_null = mldivide(Mn_null, Jt_null) @ mrdivide(sp.linalg.pinv(mldivide(Wny_null, Gt_null)), Wny_null)
            Mny_null = mrdivide(sp.linalg.sqrtm(Juu), (H_null @ Gy_null)) @ H_null @ Wny_null

        if nyt == (nu + nd):
            H_null   = mrdivide(mldivide(Mn_null, Jt_null), Gt_null)
            Mny_null = mrdivide(-Jt_null, Gt_null) @ Wny_null
        if nyt < (nu+nd):
            H_null  = mrdivide(mldivide(Mn_null, Jt_null), Gy_null)
            Mny_null = mrdivide(sp.linalg.sqrtm(Juu), (H_null @ Gy_null)) @ H_null @ Wny_null

        aux_H   = H_null
        H_null  = H_null/sp.linalg.norm(H_null, ord='fro')
        Erf_md  = aux_H @ Gt_null - Jt_null
        Md_null = - sp.linalg.sqrtm(Juu)@ sp.linalg.inv(H_null@Gy_null) @ H_null @ F_null @ Wd
        M_null  = np.c_[Md_null,Mny_null]


        worst_loss_null = 0.5 * (np.max(np.linalg.svd(M_null)[1], axis=0)) ** 2
        avg_loss_null = (1 / (6 * (ss_size + nd))) * ((np.linalg.norm(M_null, ord="fro")) ** 2)
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

