"""Test the online score wrapper for scoring scikit-learn models."""
import pytest

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer

from src.evaluation.sklearn_utils import (
    OnlineScoreWrapper, StratifiedPatientKFold, NotConsecutiveError)

MOCK_LENGTHS = [5, 3, 10, 12]
MOCK_INDEX = pd.MultiIndex.from_arrays(
    [
        np.concatenate(
            [np.repeat(np.random.randint(0, 100000), l) for l in MOCK_LENGTHS],
            axis=0),
        np.concatenate([np.arange(l) for l in MOCK_LENGTHS], axis=0)
    ],
    names=('id', 'time')
)
MOCK_INDEX_WRONG = pd.MultiIndex.from_arrays(
    [
        np.concatenate(
            [np.repeat(np.random.randint(0, 100000), l) for l in MOCK_LENGTHS],
            axis=0),
        np.concatenate(
            [np.random.randint(10, size=l) for l in MOCK_LENGTHS], axis=0)
    ],
    names=('id', 'time')
)
MOCK_X = pd.DataFrame(
    np.random.random(size=(sum(MOCK_LENGTHS), 10)),
    index=MOCK_INDEX
)
MOCK_Y = pd.Series(
    np.random.randint(0, 1+1, size=sum(MOCK_LENGTHS)),
    index=MOCK_INDEX
)
MOCK_Y_WRONG = pd.Series(
    np.random.randint(0, 1+1, size=sum(MOCK_LENGTHS)),
    index=MOCK_INDEX_WRONG
)


def MOCK_SCORE(y_label, y_pred):
    return np.mean(
        [np.all(label, pred) for label, pred in zip(y_label, y_pred)])


def test_with_gridsearch_cv():
    wrapped_scorer = OnlineScoreWrapper(MOCK_SCORE)

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
    gridsearch.fit(MOCK_X, MOCK_Y)


def test_consecutive_check():
    with pytest.raises(NotConsecutiveError):
        wrapped_scorer = OnlineScoreWrapper(MOCK_SCORE)
        wrapped_scorer(MOCK_Y_WRONG, MOCK_Y_WRONG)




