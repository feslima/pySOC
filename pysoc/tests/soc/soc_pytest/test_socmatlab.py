import pathlib
import numpy as np
import pytest
from scipy.io import loadmat
from scipy.special import binom
import re

from pysoc.soc import helm

MAT_FILES_FOLDERPATH = pathlib.Path(__file__).parent.parent / 'mat_files'
HELM_INPUT_FOLDERPATH = MAT_FILES_FOLDERPATH / 'input'
HELM_OUTPUT_FOLDERPATH = MAT_FILES_FOLDERPATH / 'output' / 'helm'
HELM_INPUT_FILENAME = 'helm_hen_input.mat'

_matfiles_list = [mat.name for mat in HELM_OUTPUT_FOLDERPATH.glob('*.mat')]
_param_tuple_list = [(HELM_INPUT_FILENAME, res_file)
                     for res_file in _matfiles_list]


@pytest.mark.parametrize("input_filename,output_filename", _param_tuple_list)
def test_helm_result(input_filename, output_filename):
    input_contents = loadmat(HELM_INPUT_FOLDERPATH / input_filename)
    results_contents = loadmat(HELM_OUTPUT_FOLDERPATH / output_filename)

    # input variables
    Gy = input_contents["Gy"]
    Gyd = input_contents["Gyd"]
    Juu = input_contents["Juu"]
    Jud = input_contents["Jud"]
    md = input_contents["md"].flatten()
    me = input_contents["me"].flatten()

    # matlab results
    res_to_load = ['Avg_Loss', 'Loss_Exact', 'index_CV']
    avg_loss_mat, worst_loss_mat, index_CV_mat = tuple(
        results_contents[var] for var in res_to_load)

    # generate python results
    ss_size = int(re.search(r'ss(\d+)', output_filename).group(1))
    nc_user = int(binom(10, ss_size))
    worst_loss_py, avg_loss_py, index_CV_py, *_ = helm(Gy, Gyd, Juu, Jud,
                                                       md, me, ss_size=ss_size,
                                                       nc_user=nc_user)

    assert np.allclose(index_CV_py, index_CV_mat), "index cv fail"
    assert np.allclose(worst_loss_py, worst_loss_mat.flatten()), "wc loss fail"
    assert np.allclose(avg_loss_py, avg_loss_mat.flatten()), "avg loss fail"
