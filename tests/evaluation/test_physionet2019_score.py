"""Test the physionet 2019 score."""

from src.evaluation.physionet2019_score import physionet2019_utility


def test_perfect_physionet_score():
    label = [([0] * 20) + [1]]
    pred = [([0] * 9) + ([1] * 12)]
    assert physionet2019_utility(label, pred) == 1.0


def test_noninteract_physionet_score():
    label = [([0] * 20) + [1]]
    pred = [[0] * 21]
    assert physionet2019_utility(label, pred) == 0.0
