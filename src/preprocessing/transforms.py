from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import dask
import dask.dataframe as dd
import numpy as np

from kymatio.numpy import Scattering1D


def strip_cols(columns, suffices):
    """
    Utility function to sequentially strip suffices from columns.
    Observed that removing _string directly leads to 
    unexpected behaviour, therefore removing first string, then _.
    Here, we assume that the suffices are provied as: '_<string>'
    """
    # removing '_' from suffix and adding it as a separate suffix to remove
    suffices = [x.lstrip('_') for x in suffices]
    suffices.append('_')
    for suffix in suffices:
        columns = columns.str.rstrip(suffix)
    return columns


class DaskPersist(TransformerMixin, BaseEstimator):
    """Transform to persist a dask dataframe, i.e. to hold it in memory."""

    def fit(self, X):
        return self

    def transform(self, X: dd.DataFrame):
        return X.persist()


class DaskIDTransformer(TransformerMixin, BaseEstimator):
    """
    Dask-based Parallelized Base class when performing transformations over ids. The child class requires to have a transform_id method.
    """

    def __init__(self, vm):
        self.vm = vm

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self

    def transform(self, dask_df):
        """ Parallelized transform
        """
        result = dask_df.groupby(self.vm('id'),
                                 group_keys=False).apply(self.transform_id)
        return result


class MeasurementCounterandIndicators(BaseEstimator, TransformerMixin):
    """ Adds a count of the number of measurements up to the given timepoint. """

    def __init__(self, suffix='_raw'):
        # we only count raw time series measurements
        self.col_suffix = suffix  # apply look-back stats to time series columns

    def fit(self, df, labels=None):
        return self

    def transform(self, df: dd.DataFrame):
        # Make a counts frame
        drop_cols = [x for x in df.columns if self.col_suffix not in x]

        indicators = (
            ~(df.drop(drop_cols, axis=1).isna())).astype(int)
        counts = indicators \
            .groupby(indicators.index.name, sort=False, group_keys=False) \
            .cumsum()

        indicators = indicators.rename(columns=lambda col: col+'_indicator')
        counts = counts.rename(columns=lambda col: col+'_count')

        return dd.multi.concat([df, indicators, counts], axis=1)


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

        df_sirs = dd.multi.concat(
            [temp_data, hr_data, paco2_data, wbc_data], axis=1)

        # Sum each row, if >= 2 then mar as SIRS
        sirs = (df_sirs.sum(axis=1) >= 2) * 1.

        # Leave the binary and the path sirs
        sirs_df = dd.multi.concat(
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
            (sbp <= 70) * 3.
            + ((70 < sbp) & (sbp <= 80)) * 2.
            + ((80 < sbp) & (sbp <= 100)) * 1.
            + (sbp >= 200) * 2.
        )

        # hr
        hr = df[hr]
        # mews[hr < 40] += 2
        # mews[(40 < hr) & (hr <= 50)] += 1
        # mews[(100 < hr) & (hr <= 110)] += 1
        # mews[(110 < hr) & (hr < 130)] += 2
        # mews[hr >= 130] += 3
        mews += (
            ((40 < hr) & (hr <= 50)) * 1.
            + ((100 < hr) & (hr <= 110)) * 1.
            + ((110 < hr) & (hr < 130)) * 2.
            + (hr >= 130) * 3.
        )

        # resp
        resp = df[resp]
        # mews[resp < 9] += 2
        # mews[(15 < resp) & (resp <= 20)] += 1
        # mews[(20 < resp) & (resp < 30)] += 2
        # mews[resp >= 30] += 3
        mews += (
            (resp < 9) * 2.
            + ((15 < resp) & (resp <= 20)) * 1.
            + ((20 < resp) & (resp < 30)) * 2.
            + (resp >= 30) * 3.
        )

        # temp
        temp = df[temp]
        # mews[temp < 35] += 2
        # mews[(temp >= 35) & (temp < 38.5)] += 0
        # mews[temp >= 38.5] += 2
        mews += (
            (temp < 35) * 2.
            + (temp >= 38.5) * 2.
        )
        return mews

    def qSOFA(self, df):
        vm = self.vm
        resp = vm('resp')
        sbp = vm('sbp')

        # qsofa = np.zeros(shape=df.shape[0])
        # qsofa[df[resp].values >= 22] += 1
        # qsofa[df[sbp].values <= 100] += 1
        return (df[resp] >= 22) * 1. + (df[sbp] <= 100) * 1.

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
            ((100 <= platelets) & (platelets < 150)) * 1.
            + ((50 <= platelets) & (platelets < 100)) * 2.
            + ((20 <= platelets) & (platelets < 50)) * 3.
            + (platelets < 20) * 4.
        )

        # Liver
        bilirubin = df[bili]
        # sofa[bilirubin < 1.2] += 0
        # sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
        # sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
        # sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
        # sofa[bilirubin > 11.9] += 4

        sofa += (
            ((1.2 <= bilirubin) & (bilirubin <= 1.9)) * 1.
            + ((1.9 < bilirubin) & (bilirubin <= 5.9)) * 2.
            + ((5.9 < bilirubin) & (bilirubin <= 11.9)) * 3.
            + (bilirubin > 11.9) * 4.
        )

        # Cardiovascular
        map = df[map]
        # sofa[map >= 70] += 0
        # sofa[map < 70] += 1
        sofa += (map < 70) * 1.

        # crea
        creatinine = df[crea]
        # sofa[creatinine < 1.2] += 0
        # sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
        # sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
        # sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
        # sofa[creatinine > 4.9] += 4
        sofa += (
            ((1.2 <= creatinine) & (creatinine <= 1.9)) * 1.
            + ((1.9 < creatinine) & (creatinine <= 3.4)) * 2.
            + ((3.4 < creatinine) & (creatinine <= 4.9)) * 3.
            + (creatinine > 4.9) * 4.
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
        sofa_det = (sofa_det < 0) * 0. + (sofa_det >= 0) * sofa_det
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


class Normalizer(TransformerMixin, BaseEstimator):
    """
    Performs normalization (z-scoring) of columns.
    """

    def __init__(self, patient_ids, suffix=None):
        """
        Args:
        - patient_ids: Patient ids that should be used for computing
          normalization statistics
        - suffix: when provided all columns having this suffix are normalized.
            otherwise, all but the excluded columns are normalized.
        """

        self.patient_ids = patient_ids
        self.suffix = suffix
        self.stats = None

    def _drop_columns(self, df, cols_to_drop):
        """ Utiliy function, to select available columns to
            drop (the list can reach over different feature sets)
        """
        drop_cols = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=drop_cols)
        return df, drop_cols

    def _compute_stats(self, df):
        patients = df.loc[self.patient_ids]
        means, stds = dask.compute(patients.mean(), patients.std())
        self.stats = {
            'means': means,
            'stds': stds
        }

    def _apply_normalization(self, df):
        assert self.stats is not None
        return (df - self.stats['means']) / self.stats['stds']

    def fit(self, df, labels=None):
        if type(self.suffix) == str:
            suffix = [self.suffix]
        else:
            suffix = self.suffix
        self.columns = [col for col in df.columns if any(
            [s in col for s in suffix])
        ]
        self._compute_stats(df[self.columns])
        return self

    def transform(self, df):
        normalized = self._apply_normalization(df[self.columns])
        return df.assign(**{
            col: normalized[col]
            for col in normalized.columns
        })


class WaveletFeatures(DaskIDTransformer):
    """ Computes wavelet scattering up to given time per time series channel """

    def __init__(self, T=32, J=2, Q=1, output_size=32, suffix='_locf', **kwargs):
        """
        Running 1D wavelet scattering.
        Inputs:
            - T: time steps = samples = support (needs to be power of 2)
            - J: scale
            - Q: number of wavelets per octave
            - output_size: 32 (hard coded, as I didn't find an implemented way
                to precompute it, even the docu says, that one dim is "roughly$
                portional" to some function of the input params).
        """
        super().__init__(**kwargs)
        # we process the raw time series measurements
        self.T = T
        self.col_suffix = suffix
        self.scatter = Scattering1D(J, T, Q)
        self.output_size = output_size

    def fit(self, df, labels=None):
        return self

    def pre_transform(self, ddf):
        drop_cols = [x for x in ddf.columns if self.col_suffix not in x]
        return ddf.drop(drop_cols, axis=1).rename(lambda col: col[:-len(self.col_suffix)])

    def post_transform(self, input_ddf, transformed_ddf):
        return dd.multi.concat([input_ddf, transformed_ddf], axis=1)

    def transform_id(self, inputs):
        """ process invididual patient """
        # pad time series with T-1 0s (for getting online features of fixed window size)
        inputs = self._pad_df(inputs, n_pad=self.T-1)
        #inputs.reset_index(drop=True, inplace=True)

        wavelets = self._compute_wavelets(inputs)

        assert inputs.shape[0] == wavelets.shape[0]
        wavelets.index = inputs.index
        return wavelets

    def _pad_df(self, df, n_pad, value=0):
        """ util function to pad <n_pad> zeros at the start of the df 
            and impute 0s at the remaining nans (of lcof imputation). 
        """
        x = np.concatenate([np.zeros([n_pad, df.shape[1]]), df.values])
        return pd.DataFrame(x, columns=df.columns).fillna(0)

    def _compute_wavelets(self, df):
        """ Loop over all columns, compute column-wise wavelet features
             and create wavelet output columns"""
        col_dfs = []
        for col in df.columns:
            col_df = self._process_column(df, col)
            col_dfs.append(col_df)
        return pd.concat(col_dfs, axis=1)

    def _process_column(self, df, col):
        """ Processing a single column. """
        out_cols = [col + f'_wavelet_{i}' for i in range(self.output_size)]
        col_df = pd.DataFrame(columns=out_cols)
        # we go over input column and write result to col_df, which is a
        # df gathering all features from the current col
        df[col].rolling(window=self.T).apply(
            self._rolling_function, kwargs={'df': col_df})
        return col_df

    def _rolling_function(self, window, df):
        """actually compute wavelet scattering in a rolling window """
        df.loc[window.index.min()] = self.scatter(window.values).flatten()
        return 1  # rolling funcs need to return index..
