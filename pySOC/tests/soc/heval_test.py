import scipy.io as spio
import numpy as np
from soc.h_eval import h_exact_local_method, h_exact_local_method_ss, h_extended_nullspace


matrices_cont = spio.loadmat("matrices_self_opt_100pts.mat")
Gy = matrices_cont["Gy"]
Gyd = matrices_cont["Gyd"]
Juu = matrices_cont["Juu"]
Jud = matrices_cont["Jud"]
d = matrices_cont["d"]
Wd = matrices_cont["Wd"]
Wny = matrices_cont["Wny"]

me = np.array([[1.285, 1, 1, 0.027, 0.189, 1, 0.494, 0.163, 4.355, 0.189]])
ss_size = 10  # This will be an user input (From the GUI): How many CVs as linear combinations?
nc_user = 1  # Number of best sets (from lower to higher order) to be considered among the all possible combinations
Result_exact_full = h_exact_local_method(Gy, Gyd, Juu, Jud, d, Wd, Wny)
Res_exact_ss = h_exact_local_method_ss(Gy, Gyd, Juu, Jud, d, Wd, me, ss_size, nc_user)
Res_null = h_extended_nullspace(Gy, Gyd, Juu, Jud, d, Wd, me, ss_size)