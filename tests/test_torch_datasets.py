import pytest
from src.torch.datasets import Physionet2019, MIMICDemo, Hirid, EICU, AUMC


@pytest.mark.parametrize(
    'dataset_cls', [Physionet2019, MIMICDemo, Hirid, EICU, AUMC])
@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('feature_set', ['small', 'large'])
@pytest.mark.parametrize('only_physionet_features', [True, False])
@pytest.mark.parametrize('fold', [0, 1, 2, 3, 4])
def test_read_first_instance(dataset_cls, split, feature_set, only_physionet_features, fold):
    data = dataset_cls(split, feature_set, only_physionet_features, fold)
    data[0]
