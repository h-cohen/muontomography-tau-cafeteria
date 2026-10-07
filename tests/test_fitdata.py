import numpy as np
import pytest

from cafetomo.fitdata import FitData, RowIndex


def _rows(sx, sy, pos) -> RowIndex:
    sx = np.asarray(sx, dtype=float)
    return RowIndex(
        position_ids=("pos0", "pos1"),
        pos_of_row=np.asarray(pos, dtype=np.int64),
        sx=sx,
        sy=np.asarray(sy, dtype=float),
        sky_flat=np.arange(sx.size, dtype=np.int64),
    )


def test_mask_for_selects_one_positions_rows():
    rows = _rows([0, 0.1, 0.2, 0.3], [0, 0, 0, 0], [0, 0, 1, 1])
    assert rows.n_rows == 4
    assert rows.mask_for("pos0").sum() == 2
    assert rows.mask_for("pos1").tolist() == [False, False, True, True]


def test_restricted_zeroes_weights_and_keeps_the_row_layout():
    rows = _rows([0, 0.1, 0.2], [0, 0, 0], [0, 0, 0])
    data = FitData(lam=np.array([1.0, 2.0, 3.0]), w=np.ones(3), rows=rows)
    r = data.restricted(np.array([True, False, True]))
    assert r.rows is data.rows
    np.testing.assert_allclose(r.lam, data.lam)
    np.testing.assert_allclose(r.w, [1.0, 0.0, 1.0])


def test_t_reach_is_the_largest_tangent():
    rows = _rows([0.1, -0.8, 0.3], [0.2, 0.0, 0.5], [0, 0, 1])
    assert rows.t_reach() == pytest.approx(0.8)


def test_directions_are_unit_vectors_along_sx_sy_1():
    rows = _rows([0.0, 0.6], [0.0, 0.0], [0, 0])
    d = rows.directions()
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    np.testing.assert_allclose(d[0], [0, 0, 1])
    assert d[1, 0] / d[1, 2] == pytest.approx(0.6)


def test_row_index_key_changes_with_the_rows():
    a = _rows([0.0, 0.1], [0, 0], [0, 0])
    b = _rows([0.0, 0.2], [0, 0], [0, 0])
    assert a.key() == _rows([0.0, 0.1], [0, 0], [0, 0]).key()
    assert a.key() != b.key()
