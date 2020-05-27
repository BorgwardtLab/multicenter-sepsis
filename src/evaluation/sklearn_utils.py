"""Module containing wrappers for scikit-learn online predictors."""
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.model_selection import StratifiedKFold


def nanany(array: np.ndarray):
    """Any operation which ignores NaNs.

    Numpy by default interprets NaN as True, thus returning True for any on
    arrays containing NaNs.

    """
    array = np.asarray(array)
    return np.any(array[~np.isnan(array)])


class NotAlignedError(Exception):
    """Error for non-aligned predictions and labels."""

    def __init__(self, instance_id):
        self.instance_id = instance_id

    def __str__(self):
        return "Predictions and labels of instance {} are not aligned.".format(
            self.instance_id)


class NotOnsetLabelError(Exception):
    """Error for labels which cannot be considered a onset."""

    def __init__(self, instance_id):
        self.instance_id = instance_id

    def __str__(self):
        return "Label of instance {} cannot be considered a onset.".format(
            self.instance_id)


class NaNInEvalError(Exception):
    """Error for NaNs in either labels of model predictions."""

    def __init__(self, instance_id, where):
        self.instance_id = instance_id
        self.where = where

    def __str__(self):
        return "Found NaNs in {} of instance {}.".format(
            self.where,
            self.instance_id
        )


def ensure_aligned(patient_id, labels, predictions):
    """Ensure labels and predictions are aligned."""
    if not np.all(labels.index == predictions.index):
        raise NotAlignedError(patient_id)


def ensure_not_NaN(patient, labels, predictions):
    """Ensure we do not have any predictions or labels with NaN."""
    if np.any(np.isnan(labels)):
        raise NaNInEvalError(patient, 'labels')
    if np.any(np.isnan(predictions)):
        raise NaNInEvalError(patient, 'predictions')


def make_consecutive(labels, predictions):
    """Make predictions and labels with gaps consecutive.

    This makes the predictions and labels consecutive by filling missing values
    with NaNs.

    """
    min_time, max_time = predictions.index.min(), predictions.index.max()
    consecutive_times = np.arange(min_time, max_time+1)
    predictions = predictions.reindex(
        consecutive_times, method=None, fill_value=np.NaN)
    labels = labels.reindex(
        consecutive_times, method=None, fill_value=np.NaN)
    return labels, predictions


def shift_onset_label(patient_id, label, shift):
    """Shift the label onset.

    Args:
        patient_id: The patient_id for more informative errors
        labels: pd.Series of the labels
        shift: The number of hours to shift the label. Positive values
               correspond to shifting into the future.

    Returns:
        pd.Series with labels shifted.

    """
    # We need to exclude NaNs as they are considered positive. This would lead
    # to treating controls as cases.
    is_case = nanany(label)
    if is_case:
        onset = np.nanargmax(label.values)
        # Check if label is a onset
        if not np.all(label.iloc[onset:]):
            raise NotOnsetLabelError(patient_id)

        new_onset = onset + shift
        new_onset = min(max(0, new_onset), len(label))
        old_onset_segment = label.values[new_onset:]
        new_onset_segment = np.ones(len(label) - new_onset)
        # NaNs should stay NaNs
        new_onset_segment[np.isnan(old_onset_segment)] = np.NaN
        new_label = np.concatenate(
            [label.values[:new_onset], new_onset_segment], axis=0)
        return pd.Series(new_label, index=label.index)
    else:
        return label


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
        prediction_index = \
            y_pred.index if hasattr(y_pred, 'index') else y_true.index
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            # Drop additional dimension in case of one dimensional label
            y_pred = pd.Series(y_pred.iloc[:, 0], index=prediction_index)
        elif len(y_pred.shape) == 2:
            y_pred = pd.DataFrame(y_pred, index=prediction_index)
        elif len(y_pred.shape) == 1:
            y_pred = pd.Series(y_pred, index=prediction_index)
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
            ensure_not_NaN(patient, patient_y, patient_preds)
            ensure_aligned(patient, patient_y, patient_preds)
            patient_y, patient_preds = make_consecutive(
                patient_y, patient_preds)
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
        #determine if data is a dask dataframe:
        if type(X) == dd.DataFrame:
            #using dask:
            patient_ids = X.index.compute().get_level_values('id')
            unique_patient_ids = patient_ids.unique()
            y = y.compute()
            patient_labels = [np.any(y.loc[pid]) for pid in unique_patient_ids]
            from IPython import embed; embed()
        else: 
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
