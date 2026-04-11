"""Tests for scalar multiplication operators on SparseHist."""

import hist
import numpy as np
import scipy.sparse

from wums.sparse_hist import SparseHist


def _make_sh():
    ax0 = hist.axis.Regular(3, 0, 3, underflow=False, overflow=False, name="x")
    ax1 = hist.axis.Regular(2, 0, 2, underflow=False, overflow=False, name="y")
    vals = np.array([2.0, 4.0, -1.5])
    coo = scipy.sparse.coo_array(
        (vals, (np.array([0, 0, 0]), np.array([0, 3, 5]))), shape=(1, 6)
    )
    return SparseHist(coo, [ax0, ax1]), vals.copy()


def test_mul_returns_new_sparsehist():
    sh, ref = _make_sh()
    sh2 = sh * 2.5
    assert isinstance(sh2, SparseHist)
    assert np.allclose(sh2._values, ref * 2.5)
    assert np.array_equal(sh2._flat_indices, sh._flat_indices)
    assert np.allclose(sh._values, ref), "mul must not modify the original"


def test_rmul():
    sh, ref = _make_sh()
    sh3 = 3 * sh
    assert isinstance(sh3, SparseHist)
    assert np.allclose(sh3._values, ref * 3)
    assert np.allclose(sh._values, ref), "rmul must not modify the original"


def test_imul_in_place():
    sh, ref = _make_sh()
    buf = sh._values
    sh *= 2.0
    assert np.allclose(sh._values, ref * 2)
    assert sh._values is buf, "imul must modify the underlying values buffer"


def test_mul_numpy_scalar():
    sh, ref = _make_sh()
    sh4 = sh * np.float64(1.5)
    assert np.allclose(sh4._values, ref * 1.5)


if __name__ == "__main__":
    test_mul_returns_new_sparsehist()
    test_rmul()
    test_imul_in_place()
    test_mul_numpy_scalar()
    print("All SparseHist scalar-mul tests passed")
