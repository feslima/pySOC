import pathlib
import re

import numpy as np
import pytest
from scipy.io import loadmat

from pySOC.bnb import pb3wc
from pySOC.bnb.bnb import pb3wc as pb2
from pySOC.tests_ import BNB_FOLDERPATH

BNB_MAT_FILES_FOLDERPATH = pathlib.Path(__file__).parent.parent / 'mat_files'
PB3WC_FOLDERPATH = BNB_MAT_FILES_FOLDERPATH / 'pb3wc'
PB3WC_INPUT_FOLDERPATH = PB3WC_FOLDERPATH / 'input'
PB3WC_RESULTS_FOLDERPATH = PB3WC_FOLDERPATH / 'results'
CAO_INPUT_FILENAME = 'ny40_nu15_nd5_input.mat'
CAO_RESULT_FILENAME = 'ny40_nu15_nd5_n15_nc5.mat'

_matfiles_list = [mat.name for mat in PB3WC_RESULTS_FOLDERPATH.glob('*.mat')]
_param_tuple_list = [(CAO_INPUT_FILENAME, res_file)
                     for res_file in _matfiles_list]
_param_tuple_list = [(CAO_INPUT_FILENAME, CAO_RESULT_FILENAME)]


@pytest.mark.parametrize("input_filename,result_filename", _param_tuple_list)
def test_same_result(input_filename, result_filename):
    """Test routine that verifies that results from python routine are in 
    conformity with matlab results.
    """
    input_contents = loadmat(PB3WC_INPUT_FOLDERPATH / input_filename)
    results_contents = loadmat(PB3WC_RESULTS_FOLDERPATH / result_filename)

    # input variables
    Gy = input_contents["G"]
    Gyd = input_contents["Gd"]
    Juu = input_contents["Juu"]
    Jud = input_contents["Jud"]
    Wd = input_contents["Wd"]
    Wd = np.diag(Wd)
    Wny = input_contents["Wn"]
    Wny = np.diag(Wny)

    # matlab results
    res_to_load = ['B', 'sset', 'ops', 'flag']
    B_mat, sset_mat, ops_mat, flag_mat = tuple(
        results_contents[var] for var in res_to_load)

    # generate python results
    n = int(re.search(r'_n(\d+)_', result_filename).group(1))
    nc = int(re.search(r'_nc(\d+)', result_filename).group(1))
    B_py, sset_py, ops_py, _, flag_py = pb3wc(Gy, Gyd, Wd, Wny, Juu, Jud, n,
                                              nc=nc)
    B_py2, sset_py2, ops_py2, _, flag_py2 = pb2(Gy, Gyd, np.diag(Wd), np.diag(Wny), Juu, Jud, n,
                                                nc=nc)

    assert np.allclose(B_py, B_mat.flatten()), "B failed"
    assert np.allclose(sset_py, sset_mat), "sset failed"
    assert np.allclose(ops_py, ops_mat.flatten()), "ops failed"
    assert flag_py == flag_mat, 'flag failed'


if __name__ == "__main__":
    test_same_result(CAO_INPUT_FILENAME, CAO_RESULT_FILENAME)
