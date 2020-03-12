"""Dataset processing functionality."""
import abc
import glob
import math
import os
import pickle

import numpy as np
import pandas as pd


class Dataset(abc.ABC):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.

    This is basically a copy of the pytorch dataset class and was included to
    reduce dependencies of this implementation on pytorch.
    """

    @abc.abstractmethod
    def __getitem__(self, index):
        """Get a single data instance."""

    @abc.abstractmethod
    def __len__(self):
        """Get number of data instances."""


class Physionet2019Dataset(Dataset):
    """Physionet 2019 Dataset for Sepsis early detection in the ICU."""

    STATIC_COLUMNS = [
        'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    TIME_COLUMN = 'ICULOS'
    TS_COLUMNS = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
        'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
        'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
        'WBC', 'Fibrinogen', 'Platelets'
    ]
    LABEL_COLUMN = 'SepsisLabel'

    def __init__(self, root_dir='datasets/physionet2019/data/extracted',
                 split_file='datasets/physionet2019/data/split_info.pkl',
                 split='train', split_repetition=0, transform=None):
        """Physionet 2019 Dataset.

        Args:
            root_dir: Path to patient files as provided by physionet
            transform: Tranformation that should be applied to each instance
        """
        self.root_dir = root_dir

        split_repetition_name = f'split_{split_repetition}'

        with open(split_file, 'rb') as f:
            d = pickle.load(f)
        patients = d[split_repetition_name][split]

        self.files = [
            # Patient ids are int but files contain leading zeros
            os.path.join(root_dir, f'p{patient_id:06d}.psv')
            for patient_id in patients
        ]
        self.transform = transform

    def __len__(self):
        """Get number of instances in dataset."""
        return len(self.files)

    @classmethod
    def _split_instance_data_into_dict(cls, instance_data):
        """Convert the pandas dataframe into a dict.

        The resulting dict has the keys: ['statics', 'times', 'ts', 'labels'].
        """
        static_variables = instance_data[cls.STATIC_COLUMNS].values
        times = instance_data[cls.TIME_COLUMN].values
        ts_data = instance_data[cls.TS_COLUMNS].values
        labels = instance_data[cls.LABEL_COLUMN].values

        return {
            # Statics are repeated, only take first entry
            'statics': static_variables[0],
            'times': times,
            'ts': ts_data,
            'labels': labels
        }

    def __getitem__(self, idx):
        """Get instance from dataset."""
        filename = self.files[idx]
        instance_data = pd.read_csv(filename, sep='|')
        instance_dict = self._split_instance_data_into_dict(instance_data)

        if self.transform:
            instance_dict = self.transform(instance_dict)

        return instance_dict


class MIMIC3Dataset(Physionet2019Dataset):
    STATIC_COLUMNS = []
    TIME_COLUMN = 'charttime'
    TS_COLUMNS = [
        'O2Sat', 'FiO2', 'Temp', 'SBP', 'DBP', 'MAP',
        'Resp', 'HR', 'Glucose', 'Alkalinephos', 'AST', 'HCO3',
        'Bilirubin_total', 'Chloride', 'Creatinine', 'Potassium', 'BUN', 'Hct',
        'Hgb', 'Platelets', 'WBC', 'Lactate', 'PTT', 'Calcium', 'Magnesium',
        'Phosphate', 'BaseExcess', 'PaCO2', 'pH', 'Bilirubin_direct',
        'Fibrinogen'
    ]
    def __init__(self, root_dir='datasets/mimic3/data/extracted',
                 split_file='datasets/mimic3/data/split_info.pkl',
                 split='train', split_repetition=0, transform=None):
        super().__init__(
            root_dir=root_dir, split_file=split_file, split=split,
            split_repetition=split_repetition, transform=transform
        )


# pylint: disable=R0903
class PositionalEncoding():
    """Apply positional encoding to instances."""

    def __init__(self, min_timescale, max_timescale, n_channels,
                 positions_key='times'):
        """PositionalEncoding.

        Args:
            min_timescale: minimal scale of values
            max_timescale: maximal scale of values
            n_channels: number of channels to use to encode position
        """
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.n_channels = n_channels
        self.positions_key = positions_key

        self._num_timescales = self.n_channels // 2
        self._inv_timescales = self._compute_inv_timescales()

    def _compute_inv_timescales(self):
        log_timescale_increment = (
            math.log(float(self.max_timescale) / float(self.min_timescale))
            / (float(self._num_timescales) - 1)
        )
        inv_timescales = (
            self.min_timescale
            * np.exp(
                np.arange(self._num_timescales)
                * -log_timescale_increment
            )
        )
        return inv_timescales

    def __call__(self, instance):
        """Apply positional encoding to instances."""
        instance = instance.copy()  # We only want a shallow copy
        positions = instance[self.positions_key]
        scaled_time = (
            positions[:, np.newaxis] *
            self._inv_timescales[np.newaxis, :]
        )
        signal = np.concatenate(
            (np.sin(scaled_time), np.cos(scaled_time)),
            axis=1
        )
        positional_encoding = np.reshape(signal, (-1, self.n_channels))
        instance[self.positions_key] = positional_encoding
        return instance


def to_observation_tuples(instance_dict):
    """Convert time series to tuple representation.

    Basically replace all NaNs in the ts field with zeros, add a measurement
    indicator vector and combine both with the time field.
    """
    instance_dict = instance_dict.copy()  # We only want a shallow copy
    time = instance_dict['times']
    if len(time.shape) != 2:
        time = time[:, np.newaxis]

    ts_data = instance_dict['ts']
    valid_measurements = np.isfinite(ts_data)
    ts_data = np.nan_to_num(ts_data)  # Replace NaNs with zero

    # Combine into a vector
    combined = np.concatenate((time, ts_data, valid_measurements), axis=-1)
    # Replace time series data with new vectors
    instance_dict['ts'] = combined
    return instance_dict


# pylint: disable=R0903
class ComposeTransformations():
    """Chain multiple transformations together."""

    def __init__(self, transformations):
        """ComposeTransformations.

        Args:
            transformations: List of transformations
        """
        self.transformations = transformations

    def __call__(self, instance):
        """Apply transformations to instance."""
        out = instance
        for transform in self.transformations:
            out = transform(out)
        return out
