import pytest
import numpy as np
import numpy.testing as nptest
from modules import FEMBeam

r1 = np.array([5, 6, 8, 3, 2.5, 0])
r2 = np.array([5, 8, 9, 9, 5, 0])
coordinate = np.array([0, 3, 5, 6, 7.5, 9])
ext_stress = np.array([0, 0, 0, 0, 0, 0])
fi = 0

def test_gen_elements():
    R1 = np.array([5.5, 7, 5.5, 2.75, 1.25])
    R2 = np.array([6.5, 8.5, 9, 7, 2.5])
    l = np.array([3, 2, 1, 1.5, 1.5])

    elements = FEMBeam(coordinate, r1, r2, 0)

    nptest.assert_allclose(elements.R1, R1, rtol=1e-5)
    nptest.assert_allclose(elements.R2, R2, rtol=1e-5)
    nptest.assert_allclose(elements.l, l, rtol=1e-5)