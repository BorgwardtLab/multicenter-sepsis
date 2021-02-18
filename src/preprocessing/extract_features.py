"""Feature extraction pipeline."""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import dask.dataframe
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from src.variables.mapping import VariableMapping
from src.sklearn.data.transformers import LookbackFeatures, MeasurementCounter

import ipdb
import warnings
# warnings.filterwarnings("error")

VM_CONFIG_PATH = \
    str(Path(__file__).parent.parent.parent.joinpath('config/variables.json'))

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


class DerivedFeatures(TransformerMixin, BaseEstimator):
    """
    This class is based on J. Morill's code base:
    https://github.com/jambo6/physionet_sepsis_challenge_2019/blob/master/src/data/transformers.py

    Adds any derived features thought to be useful
        - Shock Index: hr/sbp
        - Bun/crea ratio: Bun/crea
        - Hepatic SOFA: Bilirubin SOFA score

    # Can add renal and neruologic sofa
    """

    def __init__(self, vm=None, suffix='locf'):
        """
        Args:
            - vm: variable mapping object
            - suffix: which variables to use [raw or locf]
        """
        vm.suffix = suffix
        self.vm = vm

    def fit(self, df, y=None):
        return self

    def sirs_criteria(self, df):
        # Create a dataframe that stores true false for each category
        # df_sirs = pd.DataFrame(index=df.index, columns=[
        #                        'temp', 'hr', 'rr.paco2', 'wbc_'])

        # determine variables:
        vm = self.vm
        temp = vm('temp')
        hr = vm('hr')
        resp = vm('resp')
        pco2 = vm('pco2')
        wbc = vm('wbc')

        # Calculate score:
        temp_data = ((df[temp] > 38) | (df[temp] < 36))
        temp_data.name = 'temp'
        hr_data = df[hr] > 90
        hr_data.name = 'hr'
        paco2_data = ((df[resp] > 20) | (df[pco2] < 32))
        paco2_data.name = 'rr.paco2'
        wbc_data = ((df[wbc] < 4) | (df[wbc] > 12))
        wbc_data.name = 'wbc_'

        df_sirs = dask.dataframe.multi.concat(
            [temp_data, hr_data, paco2_data, wbc_data], axis=1)

        # Sum each row, if >= 2 then mar as SIRS
        sirs = (df_sirs.sum(axis=1) >= 2) * 1

        # Leave the binary and the path sirs
        sirs_df = dask.dataframe.multi.concat(
            [sirs, df_sirs.sum(axis=1)], axis=1)
        sirs_df.columns = ['SIRS', 'SIRS_path']

        return sirs_df

    def mews_score(self, df):
        # mews = np.zeros(shape=df.shape[0])
        # determine variables:
        vm = self.vm
        sbp = vm('sbp')
        hr = vm('hr')
        resp = vm('resp')
        temp = vm('temp')

        # sbp
        sbp = df[sbp]
        # mews[sbp <= 70] += 3
        # mews[(70 < sbp) & (sbp <= 80)] += 2
        # mews[(80 < sbp) & (sbp <= 100)] += 1
        # mews[sbp >= 200] += 2
        mews = (
            (sbp <= 70) * 3
            + ((70 < sbp) & (sbp <= 80)) * 2
            + ((80 < sbp) & (sbp <= 100)) * 1
            + (sbp >= 200) * 2
        )

        # hr
        hr = df[hr]
        # mews[hr < 40] += 2
        # mews[(40 < hr) & (hr <= 50)] += 1
        # mews[(100 < hr) & (hr <= 110)] += 1
        # mews[(110 < hr) & (hr < 130)] += 2
        # mews[hr >= 130] += 3
        mews += (
            ((40 < hr) & (hr <= 50)) * 1
            + ((100 < hr) & (hr <= 110)) * 1
            + ((110 < hr) & (hr < 130)) * 2
            + (hr >= 130) * 3
        )

        # resp
        resp = df[resp]
        # mews[resp < 9] += 2
        # mews[(15 < resp) & (resp <= 20)] += 1
        # mews[(20 < resp) & (resp < 30)] += 2
        # mews[resp >= 30] += 3
        mews += (
            (resp < 9) * 2
            + ((15 < resp) & (resp <= 20)) * 1
            + ((20 < resp) & (resp < 30)) * 2
            + (resp >= 30) * 3
        )

        # temp
        temp = df[temp]
        # mews[temp < 35] += 2
        # mews[(temp >= 35) & (temp < 38.5)] += 0
        # mews[temp >= 38.5] += 2
        mews += (
            (temp < 35) * 2
            + (temp >= 38.5) * 2
        )
        return mews

    def qSOFA(self, df):
        vm = self.vm
        resp = vm('resp')
        sbp = vm('sbp')

        # qsofa = np.zeros(shape=df.shape[0])
        # qsofa[df[resp].values >= 22] += 1
        # qsofa[df[sbp].values <= 100] += 1
        return (df[resp] >= 22) * 1 + (df[sbp] <= 100) * 1

    def SOFA(self, df):
        vm = self.vm
        plt = vm('plt')
        bili = vm('bili')
        map = vm('map')
        crea = vm('crea')

        # sofa = np.zeros(shape=df.shape[0])

        # Coagulation
        platelets = df[plt]
        # sofa[platelets >= 150] += 0
        # sofa[(100 <= platelets) & (platelets < 150)] += 1
        # sofa[(50 <= platelets) & (platelets < 100)] += 2
        # sofa[(20 <= platelets) & (platelets < 50)] += 3
        # sofa[platelets < 20] += 4
        sofa = (
            ((100 <= platelets) & (platelets < 150)) * 1
            + ((50 <= platelets) & (platelets < 100)) * 2
            + ((20 <= platelets) & (platelets < 50)) * 3
            + (platelets < 20) * 4
        )

        # Liver
        bilirubin = df[bili]
        # sofa[bilirubin < 1.2] += 0
        # sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
        # sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
        # sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
        # sofa[bilirubin > 11.9] += 4

        sofa += (
            ((1.2 <= bilirubin) & (bilirubin <= 1.9)) * 1
            + ((1.9 < bilirubin) & (bilirubin <= 5.9)) * 2
            + ((5.9 < bilirubin) & (bilirubin <= 11.9)) * 3
            + (bilirubin > 11.9) * 4
        )

        # Cardiovascular
        map = df[map]
        # sofa[map >= 70] += 0
        # sofa[map < 70] += 1
        sofa += (map < 70) * 1

        # crea
        creatinine = df[crea]
        # sofa[creatinine < 1.2] += 0
        # sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
        # sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
        # sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
        # sofa[creatinine > 4.9] += 4
        sofa += (
            ((1.2 <= creatinine) & (creatinine <= 1.9)) * 1
            + ((1.9 < creatinine) & (creatinine <= 3.4)) * 2
            + ((3.4 < creatinine) & (creatinine <= 4.9)) * 3
            + (creatinine > 4.9) * 4
        )

        return sofa

    def SOFA_deterioration(self, s):
        def check_24hr_deterioration(s):
            """ Check the max deterioration over the last 24 hours, if >= 2 then mark as a 1"""
            prev_23_hrs = pd.concat([s.shift(i)
                                     for i in range(1, 24)], axis=1).values
            tfr_hr_min = np.nanmin(prev_23_hrs, axis=1)
            return pd.Series(index=s.index, data=(s.values - tfr_hr_min))
        sofa_det = s.groupby(
            self.vm('id'),
            sort=False).apply(check_24hr_deterioration, meta=('SOFA_derived', 'f8'))
        # Remove negative sofa values
        sofa_det = (sofa_det < 0) * 0 + (sofa_det >= 0) * sofa_det
        return sofa_det

    def septic_shock(self, df):
        vm = self.vm
        map = vm('map')
        lact = vm('lact')

        # shock = np.zeros(shape=df.shape[0])
        # shock[df[map].values < 65] += 1
        # shock[df[lact].values > 2] += 1
        return (df[map] < 65) * 1 + (df[lact] > 2) * 1

    def transform(self, df):
        vm = self.vm
        hr = vm('hr')
        sbp = vm('sbp')
        bun = vm('bun')
        crea = vm('crea')
        po2 = vm('po2')
        fio2 = vm('fio2')
        plt = vm('plt')
        map = vm('map')
        bili = vm('bili')

        # Ratios:
        df['ShockIndex_derived'] = df[hr] / df[sbp]
        df['bun/cr_derived'] = df[bun] / df[crea]
        df['po2/fio2_dervied'] = df[po2] / \
            df[fio2]  # shouldnt it be PaO2/Fi ratio?

        # SOFA
        df['SOFA_derived'] = self.SOFA(df[[plt, map, crea, bili]])
        df['SOFA_deterioration_derived'] = self.SOFA_deterioration(
            df['SOFA_derived'])
        df['qSOFA_derived'] = self.qSOFA(df)
        df['SepticShock_derived'] = self.septic_shock(df)

        # Other scores
        sirs_df = self.sirs_criteria(df)
        df['MEWS_derived'] = self.mews_score(df)
        df['SIRS_derived'] = sirs_df['SIRS']
        return df


def convert_bool_to_float(ddf):
    bool_cols = [col for col in ddf.columns if ddf[col].dtype == bool]
    ddf[bool_cols] = ddf[bool_cols].astype(float)
    return ddf


def sort_time(df):
    res = df.sort_values(VM_DEFAULT('time'), axis=0)
    return res


def check_time_sorted(df):
    return (df[VM_DEFAULT('time')].diff() <= 0).sum() == 0


def main(input_filename, split_filename, output_filename):
    raw_data = dask.dataframe.read_parquet(
        input_filename,
        columns=VM_DEFAULT.core_set,
        engine='pyarrow-dataset'
    )
    # Sort fist according to time, then according to ID. This should lead to
    # completely sorted partitions.
    raw_data = raw_data \
        .set_index(VM_DEFAULT('id'), sorted=False, nparitions='auto')
    ipdb.set_trace()
    raw_data = raw_data \
        .groupby(VM_DEFAULT('id'), group_keys=False) \
        .apply(sort_time, meta=raw_data) \
        .persist()
    ipdb.set_trace()
    # Just to be sure, check that time is sorted
    sorted_check = raw_data \
        .groupby(VM_DEFAULT('id'), dropna=False) \
        .apply(check_time_sorted, meta=bool) \
        .compute()
    assert sorted_check.all()
    raw_data = convert_bool_to_float(raw_data).persist()
    data_pipeline = Pipeline([
        ('derived_features', DerivedFeatures(VM_DEFAULT, suffix='locf')),
        ('lookback_features', LookbackFeatures(
            vm=VM_DEFAULT, suffices=['_raw', '_derived'])),
        ('measurement_counts', MeasurementCounter(vm=VM_DEFAULT, suffix='_raw'))
    ])
    raw_data = data_pipeline.fit_transform(raw_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_data',
        type=str,
        help='Path to parquet file or folder with parquet files containing the '
        'raw data.'
    )
    parser.add_argument(
        '--split-file',
        type=str,
        # required=True,
        help='Json file containing split information. Needed to ensure '
        'normalization is only computed using the dev split.'
    )
    parser.add_argument(
        '--output',
        type=str,
        # required=True,
        help='Output file path to write parquet file with features.'
    )
    args = parser.parse_args()
    assert Path(args.input_data).exists()
    # assert os.path.exists(args.split_file)
    # assert os.path.exists(os.path.dirname(args.output))
    main(args.input_data, args.split_file, args.output)
