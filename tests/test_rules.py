import numpy as np

from dronehacks import rules


def test_max_points_is_60():
    assert rules.MAX_RUN_POINTS == 60


def test_reference_scorer():
    full = rules.run_points([True] * 5, [False] * 5, True, True, t_finish=20, tref_finish=20)
    assert full == 60
    assert rules.run_points([True] * 5, [False] * 5, True, True, t_finish=20, tref_finish=20, returned=True) == 75
    # checkpoint 2 reached over a wall earns 15%
    over = rules.run_points([True, True, False, False, False], [False, True, False, False, False], False, False,
                            t_last=10, tref_last=10)
    assert np.isclose(over, 4 + 7 * 0.15 + 7)
    assert rules.run_points([False] * 5, [False] * 5, False, False, t_last=1, tref_last=1) == 0


def test_landmarks_land_in_their_sectors():
    assert rules.sector_index(rules.LAUNCH_MARK[None])[0] == 0          # launch is in the first corridor
    assert rules.sector_index(rules.PAD_CENTER[None])[0] == 4           # pad is in the landing area


def test_gate_normals_are_unit_and_perpendicular():
    for g in rules.GATES:
        assert np.isclose(np.linalg.norm(g['n']), 1.0)
        assert abs(np.dot(g['n'], g['t'])) < 1e-9
