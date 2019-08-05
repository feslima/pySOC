import numpy as np


def assert_equal(A: np.ndarray, B: np.ndarray):
    """Asserts that the arrays A and B are the same. Verifies and throws
    exceptions if the following are not the same:

    * Dimensions;

    * Shape;

    * Values
    """

    if A.ndim != B.ndim:
        raise ValueError("A has different dimension of B.")

    if A.shape != B.shape:
        raise ValueError("A has different shape of B.")

    if not np.allclose(A, B):
        raise ValueError("A is not equal to B.")
