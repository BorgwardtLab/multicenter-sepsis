from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import dask
import dask.dataframe as dd
import numpy as np

from kymatio.numpy import Scattering1D
import iisignature as iis
from src.evaluation.physionet2019_score import compute_prediction_utility


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

    def pre_transform(self, ddf):
        return ddf

    def post_transform(self, input_ddf, transformed_ddf):
        return transformed_ddf

    def transform(self, dask_df):
        """ Parallelized transform."""
        result = self.pre_transform(dask_df)\
            .groupby(self.vm('id'), group_keys=False) \
            .apply(self.transform_id)

        return self.post_transform(dask_df, result)


class LookbackFeatures(DaskIDTransformer):
    """
    Simple statistical features including moments over a tunable look-back window.
    """

    def __init__(self, stats=None, windows=[4, 8, 16],
                 suffices=['_raw', '_derived'],  **kwargs):
        """ takes dictionary of stats (keys) and corresponding look-back windows (values)
            to compute each stat for each time series variable. 
            Below a default stats dictionary of look-back 5 hours is implemented.
        """
        super().__init__(**kwargs)
        self.col_suffices = suffices  # apply look-back stats to time series columns
        self.stats = stats if stats is not None else [
            'min', 'max', 'mean', 'median', 'var']
        self.windows = windows

    def pre_transform(self, ddf):
        used_cols = [col for col in ddf.columns if any(
            [s in col for s in self.col_suffices])]

        def remove_suffix(col):
            for suffix in self.col_suffices:
                if col.endswith(suffix):
                    return col[:-len(suffix)]
        return ddf[used_cols].rename(columns=remove_suffix)

    def post_transform(self, input_ddf, transformed_ddf):
        return dd.multi.concat([input_ddf, transformed_ddf], axis=1)

    def transform_id(self, df):
        features = []
        for window in self.windows:
            rolling_window = df.rolling(
                window, min_periods=0)
            for stat in self.stats:
                feature_col = getattr(rolling_window, stat)()
                feature_col.rename(
                    columns=lambda col: col + f'_{stat}_{window}_hours',
                    inplace=True
                )
                features.append(feature_col)

        return pd.concat(features, axis=1)


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
    """Computes wavelet scattering up to given time per time series channel."""

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
        self.J = J
        self.Q = Q
        self.col_suffix = suffix
        self.scatter = Scattering1D(J, T, Q)
        self.output_size = output_size

    def fit(self, df, labels=None):
        return self

    def pre_transform(self, ddf):
        drop_cols = [x for x in ddf.columns if self.col_suffix not in x]
        return ddf.drop(drop_cols, axis=1).rename(columns=lambda col: col[:-len(self.col_suffix)])

    def post_transform(self, input_ddf, transformed_ddf):
        return dd.multi.concat([input_ddf, transformed_ddf], axis=1)

    def transform_id(self, inputs):
        """ process invididual patient """
        # pad time series with T-1 0s (for getting online features of fixed window size)
        input_values = self._pad_df(inputs, n_pad=self.T-1).values
        wavelets = self._compute_wavelets(input_values)
        wavelet_columns = self._build_wavelet_column_names(inputs.columns)
        out = pd.DataFrame(index=inputs.index,
                           columns=wavelet_columns, data=wavelets)
        return out

    def _pad_df(self, df, n_pad, value=0):
        """ util function to pad <n_pad> zeros at the start of the df
            and impute 0s at the remaining nans (of lcof imputation).
        """
        x = np.concatenate([np.zeros([n_pad, df.shape[1]]), df.values])
        return pd.DataFrame(x, columns=df.columns).fillna(0)

    def _compute_wavelets(self, input_values):
        # Should have shape n_tp x n_cols, shapelet computations expects tp to be
        # be the last dimension and previous dimensions to be batches
        input_values = np.transpose(input_values)  # Now n_cols x n_tp
        # Now we want a rolling window over the values, we can use stride
        # tricks for that
        sliding_view = np.lib.stride_tricks.sliding_window_view(
            input_values, self.T, axis=1)  # n_cols x n_tp x self.T
        wavelets = self.scatter(sliding_view)  # n_cols x n_tp x wf1 x wf2
        # Flatten wavelet dimension
        wavelets = np.reshape(wavelets, wavelets.shape[:2] + (-1, ))
        # n_cols x n_tp x n_wl
        # Now we carefully need to permute the axis, such that our wavelet
        # features and their derived columns match
        wavelets = np.transpose(wavelets, [1, 0, 2])  # n_tp x n_col x n_wl
        # Flatten wavelet features into columns, we should then have the
        # following pattern: f1_wl1, f1_wl2, f1_wl3 ... f2_wl1, f2_wl2,...
        wavelets = np.reshape(
            wavelets, (wavelets.shape[0], wavelets.shape[1]*wavelets.shape[2]))
        return wavelets

    def _build_wavelet_column_names(self, column_names):
        # Build column names, a bit unusual, but should work
        columns = np.array(column_names)[:, None] + \
            np.array([
                f'_wavelet_{i}' for i in range(self.output_size)])[None, :]
        columns = np.ravel(columns)
        return columns


class SignatureFeatures(DaskIDTransformer):
    """ Computes signature features for a given look back window """

    def __init__(self, look_back=7, order=3, suffices=['_locf', '_derived'], **kwargs):
        """
        Inputs:
            - n_jobs: for parallelization on the patient level
            - lock_back: how many hours to look into the past
                (default 7 following Morril et al.)
            - order: signature truncation level of univariate signature features
                --> for multivariate signatures (of groups of channels), we use order-1
        """
        super().__init__(**kwargs)
        # we process the raw time series measurements
        self.order = order
        self.suffices = suffices
        self.look_back = look_back
        # self.columns = extended_ts_columns
        column_dict = {  # concept variables
            'vitals':  ['hr', 'o2sat', 'temp', 'map', 'resp', 'fio2'],
            'labs': ['ph', 'lact', 'wbc', 'plt', 'ptt'],
            'scores': ['ShockIndex', 'bun/cr', 'SOFA', 'SepticShock', 'MEWS']
        }
        self.column_dict = self._add_suffices(column_dict)
        # self.output_size = iis.siglength(2, self.order) # channel + time = 2
        # self.output_size_all = iis.siglength(len(self.columns) + 1, self.order)

    def fit(self, df, labels=None):
        return self

    def pre_transform(self, ddf):
        drop_cols = [
            col for col in ddf.columns
            if not any([suffix in col for suffix in self.suffices])
        ]
        return ddf.drop(drop_cols, axis=1)

    def post_transform(self, input_ddf, transformed_ddf):
        return dd.multi.concat([input_ddf, transformed_ddf], axis=1)

    def transform_id(self, df):
        """ process invididual patient """
        # pad time series with look_back size many 0s (for getting online features of fixed window size)
        inputs = self._pad_df(df, n_pad=self.look_back)
        inputs['path'] = np.arange(inputs.shape[0])

        # channel-wise signatures
        # signatures = self._compute_signatures(inputs)

        # multivariable signatures over variable groups
        mv_signatures = self._compute_mv_signatures(inputs)
        mv_signatures.index = df.index
        return mv_signatures

    def _add_suffices(self, col_dict):
        out = {}
        for key in col_dict.keys():
            if 'scores' in key:
                columns = [col + '_derived' for col in col_dict[key]]
            else:
                columns = [col + '_locf' for col in col_dict[key]]
            out[key] = columns
        return out

    def _pad_df(self, df, n_pad, value=0):
        """ util function to pad <n_pad> zeros at the start of the df
        """
        x = np.concatenate([np.zeros([n_pad, df.shape[1]]), df.values])
        return pd.DataFrame(x, columns=df.columns).fillna(0)

    # multivariate / multi channel signatures:
    def _compute_mv_signatures(self, df):
        """ Loop over columns groups compute multivariate signature features
             and create output columns"""
        col_dfs = []
        for group, columns in self.column_dict.items():
            col_df = self._process_group(df[['path'] + columns])
            col_df.rename(
                columns=lambda col: f'{group}_signature_{col}', inplace=True)
            col_dfs.append(col_df)
        out = pd.concat(col_dfs, axis=1)
        return out

    def _process_group(self, df):
        """ Processing a group of columns. """
        group_output_size = iis.siglength(len(df.columns), self.order)

        sliding_window_view = np.lib.stride_tricks.sliding_window_view(
            df.values, self.look_back+1, axis=0)
        sliding_window_view = np.transpose(sliding_window_view, [0, 2, 1])
        signature = iis.sig(sliding_window_view, self.order)
        return pd.DataFrame(
            data=signature, index=df.index[self.look_back:], columns=np.arange(group_output_size))


class CalculateUtilityScores(DaskIDTransformer):
    """Calculate utility scores from patient.

    Inspired by Morill et al. [1], this transformer calculates the
    utility target U(1) - U(0) of a patient.  It can either function
    as a passthrough class that stores data internally or as a
    transformer class that extends a given data frame.

    [1]: http://www.cinc.org/archives/2019/pdf/CinC2019-014.pdf
    """

    def __init__(
        self,
        label='sep3',
        score_name='utility',
        shift=0,
        **kwargs
    ):
        """Create new instance of class.

        Parameters
        ----------
        label : str
            Indicates which column to use for the sepsis label.

        score_name : str
            Indicates the name of the column that will contain the
            calculated utility score. If `passthrough` is set, the
            column name will only be used in the result data frame
            instead of being used as a new column for the *input*.

        shift : int
            Number of hours to shift the sepsis label into the future.
            This makes it possible to compensate for label propagation
            if need be.

        **kwargs:
            Optional keyword arguments that will be passed to the
            parent class.
        """
        super().__init__(**kwargs)

        self.label = label
        self.score_name = score_name
        self.shift = shift

    def pre_transform(self, ddf):
        return ddf[self.label]

    def post_transform(self, input_ddf, transformed_ddf):
        return dd.multi.concat([input_ddf, transformed_ddf], axis=1)

    def transform_id(self, labels):
        """Calculate utility score differences for each patient."""
        n = len(labels)

        zeros = compute_prediction_utility(
            labels.values,
            np.zeros(shape=n),
            shift_labels=self.shift,
            return_all_scores=True
        )

        ones = compute_prediction_utility(
            labels.values,
            np.ones(shape=n),
            shift_labels=self.shift,
            return_all_scores=True
        )

        scores = pd.DataFrame(
            index=labels.index,
            data=ones - zeros,
            columns=[self.score_name]
        )

        return scores


class InvalidTimesFiltration(TransformerMixin, BaseEstimator):
    """Remove invalid rows.

    This transform removes invalid time steps right before training
    / prediction phase (final step of preprocessing).
        - time steps before ICU admission (i.e. negative timestamps)
        - time steps with less than <thres> many observations.
    """

    def __init__(self, vm, thres=1, suffix='_raw'):
        self.vm = vm
        self.thres = thres
        self.col_suffix = suffix

    def fit(self, df, labels=None):
        return self

    def _remove_pre_icu(self, ddf):
        time = self.vm('time')
        return ddf[ddf[time] >= 0]

    def _remove_too_few_observations(self, ddf, columns):
        """Remove rows with too few observations.

        In rare cases it is possible that Lookbackfeatures leak zeros into
        invalid nan rows (which makes time handling easier) additionally drop
        those rows by identifying nan labels.
        """
        ind_to_keep = (~ddf[columns].isnull()).sum(axis=1) >= self.thres
        # sanity check to prevent lookbackfeatures 0s to mess up nan rows
        ind_labels = (~ddf[self.vm('label')].isnull())
        return ddf[ind_to_keep & ind_labels]

    def transform(self, ddf):
        columns = [x for x in ddf.columns if self.col_suffix in x]
        ddf = self._remove_pre_icu(ddf)
        ddf = self._remove_too_few_observations(ddf, columns)
        return ddf
