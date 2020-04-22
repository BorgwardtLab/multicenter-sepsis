"""Mock data and functions."""
import pandas as pd
import numpy as np

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

# BUILD mock onset label
labels = []
for l in MOCK_LENGTHS:
    onset = np.random.randint(0, l+1)
    labels.append(
        np.concatenate((np.zeros(onset), np.ones(l - onset)), axis=0))
MOCK_Y = pd.Series(
    np.concatenate(labels, axis=0),
    index=MOCK_INDEX
)
MOCK_Y_CONSECUTIVE_ERROR = pd.Series(
    np.random.randint(0, 1+1, size=sum(MOCK_LENGTHS)),
    index=MOCK_INDEX_WRONG
)
MOCK_Y_ONSET_ERROR = pd.Series(
    np.random.randint(0, 1+1, size=sum(MOCK_LENGTHS)),
    index=MOCK_INDEX
)


def ALL_EQUAL_SCORE(y_label, y_pred):
    return bool(
        np.all([np.all(label == pred) for label, pred in zip(y_label, y_pred)])
    )

