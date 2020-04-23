"""Evaluation scores and utilities."""
from .physionet2019_score import physionet2019_utility
from .sklearn_utils import (
    OnlineScoreWrapper, StratifiedPatientKFold, shift_onset_label)
from sklearn.metrics import make_scorer


def get_physionet2019_scorer(shift):
    """Get physionet2019 scorer which corrects for shift.

    Args:
        shift: Number of hours to shift the label into the future (in order to
               compensate for label propagation).

    Returns:
        scorer

    """
    return make_scorer(OnlineScoreWrapper(physionet2019_utility, shift))

__all__ = [
    'physionet2019_utility', 'get_physionet2019_scorer',
    'OnlineScoreWrapper', 'StratifiedPatientKFold', 'shift_onset_label']
