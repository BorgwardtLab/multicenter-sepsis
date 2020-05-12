"""Test the online score wrapper for scoring scikit-learn models."""
import pytest

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer

from src.evaluation.sklearn_utils import (
    OnlineScoreWrapper,
    StratifiedPatientKFold,
    NotOnsetLabelError,
    NaNInEvalError,
    NotAlignedError
)
import tests.evaluation.mock as mock


def test_stratified_patient_kfold():
    cv = StratifiedPatientKFold(n_splits=2)
    for train_indices, test_indices in cv.split(mock.MOCK_X, mock.MOCK_Y):
        train_data = mock.MOCK_X.iloc[train_indices]
        train_patients = train_data.index.get_level_values('id').unique()
        test_data = mock.MOCK_X.iloc[test_indices]
        test_patients = test_data.index.get_level_values('id').unique()
        train_patients = set(train_patients)
        test_patients = set(test_patients)
        assert len(train_patients.intersection(test_patients)) == 0
        assert len(train_patients.union(test_patients)) == (
            len(train_patients) + len(test_patients))


def test_online_score_wrapper():
    per_patient_label = [
        np.array([0, 0, 1, 1, 1, 1]),
        np.array([0, 1, 1, 1, 1])
    ]
    labels = pd.DataFrame(
        np.concatenate(per_patient_label, axis=0),
        index=pd.MultiIndex.from_arrays(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
            ],
            names=('id', 'time')
        )
    )

    def score(y_true, y_pred):
        all_same = True
        for i, lab in enumerate(y_true):
            all_same &= bool(np.all(per_patient_label[i] == lab))
        return all_same

    wrapped_score = OnlineScoreWrapper(score)
    assert wrapped_score(labels, labels) is True


def test_with_gridsearch_cv():
    wrapped_scorer = OnlineScoreWrapper(mock.ALL_EQUAL_SCORE)

    clf = LogisticRegression()
    grid = {
        'C': [0.1]
    }
    gridsearch = GridSearchCV(
        clf,
        grid,
        cv=StratifiedPatientKFold(n_splits=2),
        scoring=make_scorer(wrapped_scorer)
    )
    gridsearch.fit(mock.MOCK_X, mock.MOCK_Y)


def test_nan_check():
    mock_with_nan = mock.MOCK_Y.copy()
    index = mock_with_nan.index[0]
    mock_with_nan[index] = np.NaN
    wrapped_scorer = OnlineScoreWrapper(mock.ALL_EQUAL_SCORE)
    with pytest.raises(NaNInEvalError):
        wrapped_scorer(mock_with_nan, mock.MOCK_Y)

    with pytest.raises(NaNInEvalError):
        wrapped_scorer(mock.MOCK_Y, mock_with_nan)


def test_aligned_check():
    unaligned_index = mock.MOCK_Y.index.values.copy()
    unaligned_index[0] = (unaligned_index[0][0], 5)
    unaligned_index = pd.MultiIndex.from_tuples(
        unaligned_index, names=['id', 'time'])
    mock_unaligned = pd.Series(
        mock.MOCK_Y.values.copy(), index=unaligned_index)
    wrapped_scorer = OnlineScoreWrapper(mock.ALL_EQUAL_SCORE)
    with pytest.raises(NotAlignedError):
        wrapped_scorer(mock_unaligned, mock.MOCK_Y)
    with pytest.raises(NotAlignedError):
        wrapped_scorer(mock.MOCK_Y, mock_unaligned)


def test_onset_label_check():
    with pytest.raises(NotOnsetLabelError):
        wrapped_scorer = OnlineScoreWrapper(
            mock.ALL_EQUAL_SCORE, shift_onset_label=2)
        wrapped_scorer(mock.MOCK_Y_ONSET_ERROR, mock.MOCK_Y_ONSET_ERROR)


def test_onset_label_shift():
    mock_index = pd.MultiIndex.from_arrays(
        [[0, 0, 0, 0], [0, 1, 2, 3]],
        names=('id', 'time')
    )
    mock_y = pd.DataFrame([0, 0, 0, 1], index=mock_index)
    mock_pred = pd.DataFrame([0, 0, 1, 1], index=mock_index)
    wrapped_scorer = OnlineScoreWrapper(
        mock.ALL_EQUAL_SCORE, shift_onset_label=-1)
    score = wrapped_scorer(mock_y, mock_pred)
    assert score is True
