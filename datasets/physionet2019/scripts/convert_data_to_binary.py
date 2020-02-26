"""Convert data from physionet format to python binary."""
"""Author: Max Horn """

import argparse
import os
from glob import glob

import pandas as pd
from tqdm import tqdm



def read_patient(filename):
    """Read patient file and add patient id to dataframe."""
    patient_data = pd.read_csv(filename, sep='|')
    # Extract name and remove leading p
    patientstr = os.path.splitext(os.path.basename(filename))[0]
    patientid = int(patientstr[1:])
    patient_data['patientid'] = patientid
    return patient_data


def main():
    """Execute main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', nargs='+', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    aggregated_data = []
    for folder in args.folders:
        print(f'Processing folder {folder}')
        iterator = glob(os.path.join(folder, '*.psv'))
        iterator_with_progress = tqdm(iterator)
        for patient_datafile in iterator_with_progress:
            iterator_with_progress.set_description(patient_datafile)
            aggregated_data.append(read_patient(patient_datafile))

    aggregated_data = pd.concat(aggregated_data)
    aggregated_data.to_pickle(args.output)


if __name__ == '__main__':
    main()
