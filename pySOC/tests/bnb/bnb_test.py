import numpy as np
import scipy as sp
from bnb.bnb import pb3wc
import scipy.io as spio

matrices_cont = spio.loadmat("python_bnb.mat")
Gy = matrices_cont["Gy"]
Gyd = matrices_cont["Gyd"]
Juu = matrices_cont["Juu"]
Jud = matrices_cont["Jud"]
Wd = matrices_cont["Wd"]
Wny = matrices_cont["Wny"]


var_test = pb3wc(Gy,Gyd,Wd,Wny,Juu,Jud,2,np.inf,45)
a = 1