"""Dataset processing functionality."""
import abc
import os
import pickle

from sklearn.model_selection import train_test_split

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

    STATIC_COLUMNS = ['age', 'sex']
    TIME_COLUMN = 'stay_time'
    TS_COLUMNS = [
        'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
        'etco2', 'be', 'bicar', 'fio2', 'ph', 'pco2', 'ast', 
        'bun', 'alp', 'ca', 'cl', 'crea', 'bili_dir', 'glu', 
        'lact', 'mg', 'phos', 'k', 'bili', 'tri', 'hct', 'hgb', 
        'ptt', 'wbc', 'fgn', 'plt', 'alb', 'alt', 'basos', 
        'bnd', 'cai', 'ck', 'ckmb', 'crp', 'eos', 'esr', 'hbco',
        'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'methb', 'na', 
        'neut', 'po2', 'pt', 'rbc', 'rdw', 'tco2', 'tnt', 'sirs', 
        'news', 'mews'
    ]
    LABEL_COLUMN = 'sep3'

    def __init__(self, root_dir='datasets/physionet2019/data/extracted',
                 split_file='datasets/physionet2019/data/split_info.pkl',
                 split='train', split_repetition=0, as_dict=True,
                 transform=None, custom_path=None):
        """Physionet 2019 Dataset.

        Args:
            root_dir: Path to extracted patient files as provided by physionet
            split_file: Path to split file
            transform: Tranformation that should be applied to each instance
            custom_path: when working from a different repo, custom path can be used to replace the root of the other paths
        """
        if custom_path is not None:
           root_dir = os.path.join(custom_path, os.path.split(root_dir)[1])
           split_file = os.path.join(custom_path, os.path.split(split_file)[1]) 
        self.root_dir = root_dir
        self.as_dict = as_dict

        split_repetition_name = f'split_{split_repetition}'

        with open(split_file, 'rb') as f:
            d = pickle.load(f)
        self.patients = d[split_repetition_name][split]

        self.files = [
            # Patient ids are int but files contain leading zeros
            os.path.join(root_dir, f'p{patient_id}.psv') #{patient_id:06d}
            for patient_id in self.patients
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
        if self.as_dict:
            instance_data = self._split_instance_data_into_dict(instance_data)
        else: #feeding the ids to sklearn pipeline
            instance_data = self.patients[idx], instance_data

        if self.transform:
            instance_data = self.transform(instance_data)

        return instance_data


class DemoDataset(Physionet2019Dataset):
    """
    Demo dataset (based on subset of MIMIC) for quick testing of pipeline steps.
    """

    def __init__(self, root_dir='datasets/demo/data/extracted',
                 split_file='datasets/demo/data/split_info.pkl', split='train',
                 split_repetition=0, as_dict=True, transform=None,
                 custom_path=None):
        super().__init__(
            root_dir=root_dir, split_file=split_file, split=split,
            split_repetition=split_repetition, as_dict=as_dict, transform=transform,
            custom_path=custom_path
        )


class MIMIC3Dataset(Physionet2019Dataset):
    def __init__(self, root_dir='datasets/mimic3/data/extracted',
                 split_file='datasets/mimic3/data/split_info.pkl',
                 split='train', split_repetition=0, as_dict=True,
                 transform=None, custom_path=None):
        super().__init__(
            root_dir=root_dir, split_file=split_file, split=split,
            split_repetition=split_repetition, as_dict=as_dict, transform=transform,
            custom_path=custom_path
        )


class HiridDataset(Physionet2019Dataset):
    def __init__(self, root_dir='datasets/hirid/data/extracted',
                 split_file='datasets/hirid/data/split_info.pkl',
                 split='train', split_repetition=0, as_dict=True,
                 transform=None, custom_path=None):
        super().__init__(
            root_dir=root_dir, split_file=split_file, split=split,
            split_repetition=split_repetition, as_dict=as_dict, transform=transform,
            custom_path=custom_path
        )


class EICUDataset(Physionet2019Dataset):
    def __init__(self, root_dir='datasets/eicu/data/extracted',
                 split_file='datasets/eicu/data/split_info.pkl', split='train',
                 split_repetition=0, as_dict=True, transform=None,
                 custom_path=None):
        super().__init__(
            root_dir=root_dir, split_file=split_file, split=split,
            split_repetition=split_repetition, as_dict=as_dict, transform=transform,
            custom_path=custom_path
        )


class PreprocessedDataset(Dataset):
    LABEL_COLUMN = 'SepsisLabel'
    TIME_COLUMN = 'time'

    def __init__(self, prefix, split='train', drop_pre_icu=True, transform=None):
        self.file_path = '{}_{}.pkl'.format(prefix, split)

        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)

        self.patients = list(data.index.unique())
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def get_stratified_split(self, random_state=None):
        per_instance_labels = [
            np.any(self.data.loc[[patient_id], self.LABEL_COLUMN])
            for patient_id in self.patients
        ]
        train_indices, test_indices = train_test_split(
            range(len(per_instance_labels)),
            train_size=0.8,
            stratify=per_instance_labels,
            random_state=random_state
        )
        return train_indices, test_indices

    def _get_instance(self, idx):
        patient_id = self.patients[idx]
        patient_data = self.data.loc[[patient_id]]
        time = patient_data[self.TIME_COLUMN].values
        labels = patient_data[self.LABEL_COLUMN].values
        ts_data = patient_data.drop(
            columns=[self.LABEL_COLUMN]).values
        return {
            'times': time,
            'ts': ts_data,
            'labels': labels.astype(float)
        }

    def __getitem__(self, idx):
        instance = self._get_instance(idx)
        if self.transform:
            instance = self.transform(instance)
        return instance


class PreprocessedDemoDataset(PreprocessedDataset):
    def __init__(self,
                 prefix='datasets/demo/data/sklearn/processed/X_filtered',
                 **kwargs):
        super().__init__(prefix=prefix, **kwargs)


class PreprocessedPhysionet2019Dataset(PreprocessedDataset):
    def __init__(self,
                 prefix='datasets/physionet2019/data/sklearn/processed/X_filtered',
                 **kwargs):
        super().__init__(prefix=prefix, **kwargs)


class PreprocessedMIMIC3Dataset(PreprocessedDataset):
    def __init__(self,
                 prefix='datasets/mimic3/data/sklearn/processed/X_filtered',
                 **kwargs):
        super().__init__(prefix=prefix, **kwargs)


class PreprocessedHiridDataset(PreprocessedDataset):
    def __init__(self,
                 prefix='datasets/hirid/data/sklearn/processed/X_filtered',
                 **kwargs):
        super().__init__(prefix=prefix, **kwargs)


class PreprocessedEICUDataset(PreprocessedDataset):
    def __init__(self,
                 prefix='datasets/eicu/data/sklearn/processed/X_filtered',
                 **kwargs):
        super().__init__(prefix=prefix, **kwargs)
