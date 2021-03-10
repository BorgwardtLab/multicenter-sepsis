import pytest
from src.torch.datasets import (
    AUMC, EICU, Hirid, MIMIC, MIMICDemo, Physionet2019)


@pytest.mark.parametrize(
    'dataset_cls', [MIMICDemo, Physionet2019, MIMIC, Hirid, EICU, AUMC])
@pytest.mark.parametrize('split', ['train', 'test'])  # train and val use same code path
@pytest.mark.parametrize('feature_set', ['small', 'large'])
@pytest.mark.parametrize('only_physionet_features', [False])
@pytest.mark.parametrize('fold', [0])
def test_read_first_instance(dataset_cls, split, feature_set, only_physionet_features, fold):
    data = dataset_cls(split, feature_set, only_physionet_features, fold)
    data[0]
