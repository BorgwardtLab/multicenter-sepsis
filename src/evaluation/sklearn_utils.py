"""Module containing wrappers for scikit-learn online predictors."""
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold


class NotConsecutiveError(Exception):
    """Error for non-consecutive instances."""

    def __init__(self, instance_id):
        self.instance_id = instance_id

    def __str__(self):
        return "Instance {} contains non-consecutive elements.".format(
            self.instance_id)


class NotOnsetLabelError(Exception):
    """Error for labels which cannot be considered a onset."""

    def __init__(self, instance_id):
        self.instance_id = instance_id

    def __str__(self):
        return "Label of instance {} cannot be considered a onset.".format(
            self.instance_id)


def ensure_consecutive(patient_id, instance):
    """Ensure that we have observations every hour."""
    times = instance.index
    time_differences = np.unique(np.diff(times))
    if len(time_differences) != 1 or time_differences[0] != 1:
        raise NotConsecutiveError(patient_id)


def shift_onset_label(patient_id, label, shift):
    """Shift the label onset."""
    onset = np.argmax(label)
    # Check if label is a onset
    if not np.all(label.iloc[onset:]):
        raise NotOnsetLabelError(patient_id)
    new_onset = onset + shift
    new_onset = min(max(0, new_onset), len(label))
    new_label = np.zeros(len(label), dtype=label.dtype)
    new_label[new_onset:] = 1
    return pd.Series(new_label, index=label.index)


class OnlineScoreWrapper:
    """Wraps an online prediction scores to process pandas DataFrames."""

    def __init__(self, score_func, shift_onset_label=0):
        self.score_func = score_func
        self.shift_onset_label = shift_onset_label

    @property
    def __name__(self):
        return self.score_func.__name__

    def __call__(self, y_true, y_pred):
        """Call score_func on dataframe input."""
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            # Drop additional dimension in case of one dimensional label
            y_pred = pd.Series(y_pred.iloc[:, 0], index=y_true.index)
        elif len(y_pred.shape) == 2:
            y_pred = pd.DataFrame(y_pred, index=y_true.index)
        elif len(y_pred.shape) == 1:
            y_pred = pd.Series(y_pred, index=y_true.index)
        else:
            raise ValueError(
                'Unexpected shape of predictions: {}'.format(y_pred.shape))

        if len(y_true.shape) == 2 and y_true.shape[1] == 1:
            # Drop additional dimension in case of one dimensional label
            y_true = pd.Series(y_true.iloc[:, 0], index=y_true.index)
        elif len(y_true.shape) == 2:
            pass
        elif len(y_true.shape) == 1:
            pass
        else:
            raise ValueError(
                'Unexpected shape of labels: {}'.format(y_pred.shape))

        patients = y_true.index.get_level_values('id').unique()
        per_patient_preds = []
        per_patient_y = []
        for patient in patients:
            patient_y = y_true.loc[patient]
            patient_preds = y_pred.loc[patient]
            ensure_consecutive(patient, patient_preds)
            if self.shift_onset_label != 0:
                patient_y = shift_onset_label(
                    patient, patient_y, self.shift_onset_label)
            per_patient_y.append(patient_y.values)
            per_patient_preds.append(patient_preds.values)

        return self.score_func(per_patient_y, per_patient_preds)


class StratifiedPatientKFold(StratifiedKFold):
    """Stratified KFold on patient level."""

    def split(self, X, y, groups=None):
        """Split X and y into stratified folds based on patient ids."""
        patient_ids = X.index.get_level_values('id')
        unique_patient_ids = patient_ids.unique()
        patient_labels = [np.any(y.loc[pid]) for pid in unique_patient_ids]
        for train, test in super().split(unique_patient_ids, patient_labels):
            train_patients = unique_patient_ids[train]
            test_patients = unique_patient_ids[test]
            yield (
                np.where(patient_ids.isin(train_patients))[0],
                np.where(patient_ids.isin(test_patients))[0]
            )
