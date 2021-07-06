"""Extract model information from run."""

import argparse

from src.torch.shap_utils import get_pooled_shapley_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        nargs='+',
        type=str,
        help='Input file(s)`. Must contain Shapley values.',
    )

    args = parser.parse_args()

    for filename in args.FILE:
        print(filename)

        get_pooled_shapley_values(filename)
