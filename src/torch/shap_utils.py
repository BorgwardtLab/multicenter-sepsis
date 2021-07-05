"""Utility functions for interacting with Shapley values."""

import json
import pickle
import os
import tempfile
import wandb

import numpy as np
import pandas as pd

from src.torch.eval_model import compute_md5hash
from src.torch.eval_model import device

from src.torch.torch_utils import ComposeTransformations

import src.torch.models

wandb_api = wandb.Api()


def feature_to_name(feature):
    """Map feature abbreviation to 'nicer' name."""
    abbreviation_to_name = {
        'age': 'patient age',
        'alb': 'albumin',
        'alp': 'alkaline phosphatase',
        'alt': 'alanine aminotransferase',
        'ast': 'aspartate aminotransferase',
        'basos': 'basophils',
        'be': 'base excess',
        'bicar': 'bicarbonate',
        'bili': 'total bilirubin',
        'bili_dir': 'bilirubin direct',
        'bnd': 'band form neutrophils',
        'bun': 'blood urea nitrogen',
        'ca': 'calcium',
        'cai': 'calcium ionized',
        'ck': 'creatine kinase',
        'ckmb': 'creatine kinase MB',
        'cl': 'chloride',
        'crea': 'creatinine',
        'crp': 'C-reactive protein',
        'dbp': 'diastolic blood pressure',
        'eos': 'eosinophils',
        'esr': 'erythrocyte sedimentation rate',
        'etco2': 'endtidal CO2',
        'fgn': 'fibrinogen',
        'fio2': 'fraction of inspired oxygen',
        'glu': 'glucose',
        'hbco': 'carboxyhemoglobin',
        'hct': 'hematocrit',
        'height': 'patient height',
        'hgb': 'hemoglobin',
        'hr': 'heart rate',
        'inr_pt': 'prothrombin time/international normalized ratio',
        'k': 'potassium',
        'lact': 'lactate',
        'lymph': 'lymphocytes',
        'map': 'mean arterial pressure',
        'mch': 'mean cell hemoglobin',
        'mchc': 'mean corpuscular hemoglobin concentration',
        'mcv': 'mean corpuscular volume',
        'methb': 'methemoglobin',
        'mg': 'magnesium',
        'na': 'sodium',
        'neut': 'neutrophils',
        'o2sat': 'oxygen saturation',
        'pco2': 'CO2 partial pressure',
        'ph': 'pH of blood',
        'phos': 'phosphate',
        'plt': 'platelet count',
        'po2': 'O2 partial pressure',
        'po2/fio2': 'PaO2/FiO2',
        'pt': 'prothrombine time',
        'ptt': 'partial thromboplastin time',
        'rbc': 'red blood cell count',
        'rdw': 'erythrocyte distribution width',
        'resp': 'respiratory rate',
        'sbp': 'systolic blood pressure',
        'sex': 'patient sex',
        'tco2': 'totcal CO2',
        'temp': 'temperature',
        'tnt': 'troponin t',
        'tri': 'troponin I',
        'urine': 'urine output',
        'wbc': 'white blood cell count',
        'weight': 'patient weight',
    }

    tokens = feature.split('_', maxsplit=1)
    base = tokens[0]
    category = tokens[-1]

    # Ensure that we are not splitting something that we should not be
    # splitting in the first place.
    if category not in ['count', 'derived', 'indicator', 'raw']:
        base = feature
        category = ''
    else:
        category = f'({category})'

    name = abbreviation_to_name.get(base, base)
    name = name[0].upper() + name[1:]

    # Local adjustments that I do not want to perform manually all the
    # time.
    if name == 'SOFAdeterioration':
        name = 'SOFA deterioration'

    return name + ' ' + category


def get_run_id(filename):
    """Extract run ID from filename."""
    run_id = os.path.basename(filename)
    run_id = os.path.splitext(run_id)[0]
    run_id = run_id[:8]
    run_id = f'sepsis/mc-sepsis/runs/{run_id}'
    return run_id


def extract_model_information(run_path, tmp):
    """Get model information from wandb run."""
    run = wandb_api.run(run_path)
    run_info = run.config
    checkpoint_path = None
    for f in run.files():
        if f.name.endswith('.ckpt'):
            file_desc = f.download(tmp)
            checkpoint_path = file_desc.name
            file_desc.close()
    if checkpoint_path is None:
        raise RuntimeError(
            f'Run "{run_path}" does not have a stored checkpoint file.')

    model_checksum = compute_md5hash(checkpoint_path)
    dataset_kwargs = {}
    for key in run_info.keys():
        if 'dataset_kwargs' in key:
            new_key = key.split('/')[-1]
            dataset_kwargs[new_key] = run_info[key]
    return run, {
        "model": run_info['model'],
        "run_id": run_path,
        "model_path": checkpoint_path,
        "model_checksum": model_checksum,
        "model_params": run_info,
        "dataset_train": run_info['dataset'],
        "task": run_info['task'],
        "label_propagation": run_info['label_propagation'],
        "rep": run_info['rep'],
        "dataset_kwargs": dataset_kwargs
    }


def get_model_and_dataset(run_id, return_out=False):
    """Get model and dataset from finished wandb run."""
    with tempfile.TemporaryDirectory() as tmp:
        # Download checkpoint to temporary directory
        run, out = extract_model_information(run_id, tmp)

        model_cls = getattr(src.torch.models, out['model'])
        model = model_cls.load_from_checkpoint(
            out['model_path'],
            dataset=out['dataset_train']
        )
    model.to(device)
    dataset = model.dataset_cls(
        split='test',
        transform=ComposeTransformations(model.transforms),
        **out['dataset_kwargs']
    )

    if return_out:
        return model, dataset, out
    else:
        return model, dataset


def pool(lengths, shapley_values, feature_values, hours_before=None):
    """Pool Shapley values and feature values.

    This function performs the pooling step required for Shapley values
    and feature values. Each time step of a time series will be treated
    as its own sample instance.

    Parameters
    ----------
    lengths:
        Vector of lengths.

    shapley_values:
        Shapley values.

    feature_values:
        Feature values.

    hours_before : int or `None`, optional
        If not `None`, only uses (at most) the last `hours_before`
        observations when reporting the Shapley values.

    Returns
    -------
    Tuple of pooled Shapely values and feature values.
    """
    # This is the straightforward way of doing it. Please don't judge,
    # or, if you do, don't judge too much :)
    shapley_values_pooled = []
    feature_values_pooled = []

    for index, (s_values, f_values) in enumerate(
        zip(shapley_values, feature_values)
    ):
        length = lengths[index]

        # Check whether we can remove everything but the last hours
        # before the maximum prediction score.
        if hours_before is not None:
            if length >= hours_before:
                s_values = s_values[length - hours_before:length, :]
                f_values = f_values[length - hours_before:length, :]

        # Just take *everything*.
        else:
            s_values = s_values[:length, :]
            f_values = f_values[:length, :]

        # Need to store mask for subsequent calculations. This is
        # required because we must only select features for which
        # the Shapley value is defined!
        mask = ~np.isnan(s_values).all(axis=1)

        s_values = s_values[mask, :]
        f_values = f_values[mask, :]

        shapley_values_pooled.append(s_values)
        feature_values_pooled.append(f_values)

    shapley_values_pooled = np.vstack(shapley_values_pooled)
    feature_values_pooled = np.vstack(feature_values_pooled)

    return shapley_values_pooled, feature_values_pooled


def get_aggregation_function(name):
    """Return aggregation function."""
    if name == 'absmax':
        def absmax(x):
            return max(x.min(), x.max(), key=abs)
        return absmax
    elif name == 'max':
        return np.max
    elif name == 'min':
        return np.min
    elif name == 'mean':
        return np.mean
    elif name == 'median':
        return np.median


def get_pooled_shapley_values(
    filename,
    ignore_indicators_and_counts=False,
    hours_before=None,
    return_normalised_features=True,
):
    """Process file and return pooled Shapley values.

    Parameters
    ----------
    filename : str
        Input filename.

    ignore_indicators_and_counts : bool, optional
        If set, ignores indicator and count features, thus reducing the
        number of features considered.

    hours_before : int or `None`, optional
        If not `None`, only uses (at most) the last `hours_before`
        observations when reporting the Shapley values.

    return_normalised_features : bool, optional
        If set, returns normalised features, corresponding to the
        values the model saw. If set to `False`, will calculate
        the original (i.e. measured) values.

    Returns
    -------
    Tuple of pooled Shapley values, features, feature names, and data
    set name.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    _, _, out = get_model_and_dataset(
        get_run_id(filename), return_out=True
    )

    dataset_name = out['model_params']['dataset']
    model_name = out['model']

    assert model_name == 'AttentionModel', RuntimeError(
        'Shapley analysis is currently only supported for the '
        'AttentionModel class.'
    )

    feature_names = data['feature_names']
    lengths = data['lengths'].numpy()

    # TODO: we might want to rename this now since it ignores
    # effectively everything *but* the raw features.
    if ignore_indicators_and_counts:
        keep_features = [
            True if not col.endswith('indicator') and
            not col.endswith('count') and
            not col.endswith('derived')
            else False
            for col in feature_names
        ]

        important_indices = np.where(keep_features)[0]
        selected_features = np.array(feature_names)[important_indices]
    else:
        important_indices = np.arange(0, len(feature_names))
        selected_features = feature_names

    # FIXME: ignore positional encoding
    #
    # This operation makes the remainder of the code incompatible with
    # other models. We are stuck with the `AttentionModel` for the time
    # being.
    important_indices += 10

    # Tensor has shape `n_samples, max_length, n_features`, where
    # `max_length` denotes the maximum length of a time series in
    # the batch.
    shap_values = data['shap_values']
    shap_values = shap_values[:, :, important_indices]
    features = data['input'].numpy()
    features = features[:, :, important_indices]

    if not return_normalised_features:
        name = dataset_name.lower()
        rep = out['rep']
        filename = f'normalizer_{name}_rep_{rep}.json'

        with open(f'./config/normalizer/{filename}') as f:
            normaliser = json.load(f)
            means = normaliser['means']
            sdevs = normaliser['stds']

            means = np.asarray(
                [means.get(name, 0.0) for name in selected_features]
            ).astype(float)
            sdevs = np.asarray(
                [sdevs.get(name, 1.0) for name in selected_features]
            ).astype(float)

            means[np.isnan(means)] = 0.0
            means[np.isnan(sdevs)] = 1.0

            features = features * sdevs
            features = features + means

    shap_values_pooled, features_pooled = pool(
        lengths,
        shap_values,
        features,
        hours_before,
    )

    return (shap_values_pooled,
            features_pooled,
            selected_features,
            dataset_name)


def collapse_and_aggregate(
    shap_values,
    feature_names,
    aggregation_function
):
    """Collapse Shapley values over variables instead of features.

    Parameters
    ----------
    shap_values:
        Shapley values.

    feature_names:
        Feature names (needs to be compatible with the last dimension of
        the Shapley values).

    aggregation_function : str
        The name of an aggregation function.

    Returns
    -------
    Tuple of data frame (containing collapsed and aggregated values),
    the Shapley values, and the corresponding feature names.
    """
    df = pd.DataFrame(shap_values, columns=feature_names)

    def feature_to_var(column):
        """Rename feature name to variable name."""
        column = column.replace('_count', '')
        column = column.replace('_raw', '')
        column = column.replace('_indicator', '')
        return column

    aggregation_fn = get_aggregation_function(aggregation_function)

    print(
        f'Collapsing features to variables using '
        f'{aggregation_function}...'
    )

    df = df.rename(feature_to_var, axis=1)
    df = df.groupby(level=0, axis=1).apply(
        lambda x: x.apply(aggregation_fn, axis=1)
    )

    shap_values = df.to_numpy()
    feature_names = df.columns

    return df, shap_values, feature_names
