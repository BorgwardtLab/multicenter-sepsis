"""Test the physionet 2019 score."""
import numpy as np
from src.evaluation.physionet2019_score import physionet2019_utility


def test_perfect_physionet_score():
    label = [([0] * 20) + [1]]
    pred = [([0] * 9) + ([1] * 12)]
    assert physionet2019_utility(label, pred) == 1.0


def test_perfect_physionet_score_with_gaps():
    # Shows that the score does not care about NaN prediction outside the 12
    # hour window
    label = [([0] * 5) + ([np.NaN] * 4) + ([0] * 11) + [1]]
    pred = [([0] * 5) + ([np.NaN] * 4) + ([1] * 12)]
    assert physionet2019_utility(label, pred) == 1.0


def test_perfect_physionet_score_with_control():
    # Checks if a control without any predictions gets a zero score.
    # We do need a positive example otherwise the score will return 0/0 = NaN.
    label = [([0] * 20) + [1], [0, 0, np.NaN, 0, 0]]
    pred = [([0] * 9) + ([1] * 12), [0, 0, np.NaN, 0, 0]]
    assert physionet2019_utility(label, pred) == 1.0


def test_noninteract_physionet_score():
    label = [([0] * 20) + [1]]
    pred = [[0] * 21]
    assert physionet2019_utility(label, pred) == 0.0
