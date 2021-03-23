import random
import timeit
import pytest
from src.torch.datasets import (
    AUMC, EICU, Hirid, MIMIC, MIMICDemo, Physionet2019)


@pytest.mark.parametrize(
    'dataset_cls', [MIMICDemo, Physionet2019, MIMIC, Hirid, EICU, AUMC])
@pytest.mark.parametrize('split', ['train', 'test'])  # train and val use same code path
@pytest.mark.parametrize('feature_set', ['small', 'large'])
@pytest.mark.parametrize('only_physionet_features', [True, False])
@pytest.mark.parametrize('fold', [0])
def test_read_first_instance(dataset_cls, split, feature_set, only_physionet_features, fold):
    data = dataset_cls(split, feature_set, only_physionet_features, fold)
    data[0]


@pytest.mark.parametrize(
    'dataset_cls', [MIMICDemo, Physionet2019, MIMIC, Hirid, EICU, AUMC])
@pytest.mark.parametrize('split', ['train', 'test'])  # train and val use same code path
@pytest.mark.parametrize('feature_set', ['small', 'large'])
@pytest.mark.parametrize('only_physionet_features', [True, False])
@pytest.mark.parametrize('fold', [0])
def test_random_access_speed(dataset_cls, split, feature_set, only_physionet_features, fold):
    data = dataset_cls(split, feature_set, only_physionet_features, fold)
    max_index = len(data) - 1
    rand = random.Random()
    time_per_instance = timeit.timeit(
        'data[rand.randint(0, max_index)]',
        globals={'data': data, 'max_index': max_index, 'rand': rand},
        number=1000
    ) / 1000
    assert time_per_instance < 0.1
