from typing import Optional, Dict, Callable, Union
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.core.saving import (
    load_hparams_from_tags_csv, load_hparams_from_yaml, update_hparams)


class FixedLightningModule(LightningModule):
    """Subclass of LightningModule implementing diverse fixes."""
    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: str,
            *args,
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            hparams_file: Optional[str] = None,
            tags_csv: Optional[str] = None,  # backward compatible, todo: remove in v0.9.0
            hparam_overrides: Optional[Dict] = None,
            **kwargs
    ):
        """
        Fixes https://github.com/PyTorchLightning/pytorch-lightning/issues/1886
        """
        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # add the hparams from csv file to checkpoint
        if tags_csv is not None:
            hparams_file = tags_csv
            rank_zero_warn('`tags_csv` argument is deprecated in v0.7.6. Will be removed v0.9.0', DeprecationWarning)

        if hparams_file is not None:
            extension = hparams_file.split('.')[-1]
            if extension.lower() in ('csv'):
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ('yml', 'yaml'):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError('.csv, .yml or .yaml is required for `hparams_file`')

            hparams['on_gpu'] = False

            # overwrite hparams by the given file
            checkpoint['hparams'] = hparams

        # override the hparam keys that were passed in
        if hparam_overrides is not None:
            update_hparams(checkpoint['hparams'], hparam_overrides)

        model = cls._load_model_state(checkpoint, *args, **kwargs)
        return model

