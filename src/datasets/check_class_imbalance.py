"""Compute the class imbalance of a dataset."""
import argparse
import numpy as np
from torch.utils.data import Subset

from tqdm import tqdm

from src.torch.torch_utils import ComposeTransformations, LabelPropagation
import src.datasets as datasets


def keep_only_labels(instance):
    return instance['labels']


def main(dataset, label_propagation, seed):
    dataset_cls = getattr(datasets, dataset)
    transform = ComposeTransformations(
        [LabelPropagation(-label_propagation), keep_only_labels])

    d = dataset_cls(split='train', transform=transform)
    print('Positive weight (1-p)/p:', d.class_imbalance_factor)
    train_indices, online_val_indices = d.get_stratified_split(seed)
    train_dataset = Subset(d, train_indices)
    online_val_dataset = Subset(d, online_val_indices)
    val_dataset = dataset_cls(split='validation', transform=transform)

    data = {
        'train': train_dataset,
        'online_val': online_val_dataset,
        'validation': val_dataset
    }

    for name, d in data.items():
        all_labels = []
        all_lengths = []
        n_pos = []
        for label in tqdm(d, total=len(d)):
            all_labels.append(label)
            all_lengths.append(len(label))
            n_pos.append(np.sum(label))

        all_lengths = np.array(all_lengths)
        combined_labels = np.concatenate(all_labels, axis=0)

        class_imbalance = np.sum(combined_labels) / len(combined_labels)
        length_mean = np.mean(all_lengths)
        length_std = np.std(all_lengths)
        length_median = np.median(all_lengths)

        class_imbalances = \
            np.array([np.sum(l) for l in all_labels]) / all_lengths
        mean_imbalance = np.mean(class_imbalances)
        std_imbalance = np.std(class_imbalances)

        cases = np.array([np.any(l) for l in all_labels])
        length_cases_mean = np.mean(all_lengths[cases])
        length_cases_std = np.std(all_lengths[cases])
        length_controls_mean = np.mean(all_lengths[~cases])
        length_controls_std = np.std(all_lengths[~cases])

        print('Split: {}'.format(name))
        print('    Instances: {}'.format(len(all_lengths)))
        print('    Total time points: {}'.format(len(combined_labels)))
        print('    Prevalence: {:.3f}'.format(np.sum(cases) / len(cases)))
        print('    Prevalance (TP): {:.3f}'.format(class_imbalance))
        print('    Ideal positive weight: {:.3f}'.format((1 - class_imbalance) / class_imbalance))
        print('    Length: {:.1f} +/- {:.1f}'.format(length_mean, length_std))
        print('    Median length: {:.1f}'.format(length_median))
        print('    Length (cases): {:.1f} +/- {:.1f}'.format(
            length_cases_mean, length_cases_std))
        print('    Length (controls): {:.1f} +/- {:.1f}'.format(
            length_controls_mean, length_controls_std))
        print('    Class imbalance (per instance): {:.3f} +/- {:.3f}'.format(
            mean_imbalance, std_imbalance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=datasets.__all__)
    parser.add_argument('--label-propagation', type=int, default=-6)
    parser.add_argument('--seed', default=87346583)
    args = parser.parse_args()

    main(args.dataset, args.label_propagation, args.seed)
