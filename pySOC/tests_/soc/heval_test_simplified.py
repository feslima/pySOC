import scipy.io as spio
import numpy as np
from pySOC.soc import helm

import pathlib

mat_filepath = pathlib.Path(__file__).parent
matrices_cont = spio.loadmat(mat_filepath / "matrices_self_opt_100pts.mat")
Gy = matrices_cont["Gy"]
Gyd = matrices_cont["Gyd"]
Juu = matrices_cont["Juu"]
Jud = matrices_cont["Jud"]
#d = matrices_cont["d"]
Wd = np.diag(matrices_cont["Wd"])
#Wny = matrices_cont["Wny"]

me = np.array([1.285, 1, 1, 0.027, 0.189, 1, 0.494, 0.163, 4.355, 0.189])
# This will be an user input (From the GUI): How many CVs as linear combinations?
ss_size = 2
# Number of best sets (from lower to higher order) to be considered among the all possible combinations
nc_user = 10
result = helm(Gy, Gyd, Juu, Jud, Wd, me, ss_size, nc_user=nc_user)
result2 = helm(Gy, Gyd, Juu, Jud, Wd, me, ss_size=10)
a = 1
