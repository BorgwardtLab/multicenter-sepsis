from sklearn.base import TransformerMixin
import pandas as pd
import dask.dataframe as dd
import numpy as np

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


class BoolToFloat(TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, ddf):
        bool_cols = [col for col in ddf.columns if ddf[col].dtype == bool]
        ddf[bool_cols] = ddf[bool_cols].astype(float)
        return ddf


class ApplyOnNormalized(TransformerMixin):
    """Apply transforms to a normalized version of the data."""

    def __init__(self, normalizer, transforms):
        self.normalizer = normalizer
        self.transforms = transforms

    def fit(self, X):
        self.normalizer.fit(X)
        for transform in self.transforms:
            transform.fit(X)
        return self

    def transform(self, X):
        X_transformed = self.normalizer.transform(X)
        normalized_cols = X_transformed.columns
        for transform in self.transforms:
            X_transformed = transform.transform(X_transformed)
        return dd.multi.concat(
            [X, X_transformed.drop(columns=normalized_cols)], axis=1)


class PatientPartitioning(TransformerMixin):
    """Create a partitioning where each patient is only in one partition."""

    def __init__(self, max_rows_per_partition):
        """Initialize patient partitioning transform.

        Args:
            max_rows_per_partition: Maximum number of rows per partition.
               Ensure this is larger than the longest expected patient!
        """
        super().__init__()
        self.max_rows_per_partition = max_rows_per_partition

    def fit(self, X):
        return self

    def transform(self, X: dd.DataFrame):
        n_obs_per_patient = X.index.value_counts(sort=False).compute()
        n_obs_per_patient.sort_index(inplace=True)
        divisions = []
        cur_count = 0
        for patient_id, n_obs in n_obs_per_patient.iteritems():
            if (len(divisions) == 0) or (cur_count + n_obs > self.max_rows_per_partition and cur_count != 0):
                # Either first partition or we cannot add more to the current
                # partition
                divisions.append(patient_id)
                cur_count = n_obs
            else:
                cur_count += n_obs

        if cur_count != 0:
            divisions.append(n_obs_per_patient.index[-1])
        return X.repartition(divisions=divisions)


class DaskPersist(TransformerMixin):
    """Transform to persist a dask dataframe, i.e. to hold it in memory."""

    def fit(self, X):
        return self

    def transform(self, X: dd.DataFrame):
        return X.persist()


class DaskRepartition(TransformerMixin):
    """Repartition dask dataframe."""

    def __init__(self, **kwargs):
        """Initialize repartition transformer.

        Args:
            kwargs: Parameters to pass to dask.dataframe.repartition
        """
        self.kwargs = kwargs

    def fit(self, X):
        return self

    def transform(self, X: dd.DataFrame):
        return X.repartition(**self.kwargs)


class DaskIDTransformer(TransformerMixin):
    """
    Dask-based Parallelized Base class when performing transformations over ids. The child class requires to have a transform_id method.
    """

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            pass
            #print('Warning: Class must take a transform_id method')
            # raise TypeError
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self

    def pre_transform(self, ddf):
        return ddf

    def post_transform(self, input_ddf, transformed_ddf):
        return transformed_ddf

    def transform(self, dask_df):
        """ Parallelized transform."""
        result = self.pre_transform(dask_df) \
            .groupby(dask_df.index.name, sort=False, group_keys=False) \
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

    # def pre_transform(self, ddf):
    #     used_cols = [col for col in ddf.columns if any(
    #         [s in col for s in self.col_suffices])]

    #     def remove_suffix(col):
    #         for suffix in self.col_suffices:
    #             if col.endswith(suffix):
    #                 return col[:-len(suffix)]
    #     return ddf[used_cols].rename(columns=remove_suffix)

    # def post_transform(self, input_ddf, transformed_ddf):
    #     return dd.multi.concat([input_ddf, transformed_ddf], axis=1)

    def transform_id(self, input_df):
        used_cols = [col for col in input_df.columns if any(
            [s in col for s in self.col_suffices])]

        def remove_suffix(col):
            for suffix in self.col_suffices:
                if col.endswith(suffix):
                    return col[:-len(suffix)]
        df = input_df[used_cols].rename(columns=remove_suffix)
        features = [input_df]
        for window in self.windows:
            rolling_window = df.rolling(
                window, min_periods=1)
            for stat in self.stats:
                feature_col = getattr(rolling_window, stat)()
                feature_col.rename(
                    columns=lambda col: col + f'_{stat}_{window}_hours',
                    inplace=True
                )
                features.append(feature_col)

        return pd.concat(features, axis=1)


class MeasurementCounterandIndicators(TransformerMixin):
    """ Adds a count of the number of measurements up to the given timepoint. """

    def __init__(self, suffix='_raw'):
        # we only count raw time series measurements
        self.col_suffix = suffix  # apply look-back stats to time series columns

    def fit(self, df, labels=None):
        return self

    def transform(self, ddf: dd.DataFrame):
        # Make a counts frame
        cols = [x for x in ddf.columns if self.col_suffix in x]

        def counter_and_indicators(df):
            indicators = (
                ~(df[cols].isna())).astype(np.uint16)
            counts = indicators \
                .groupby(indicators.index.name, sort=False, group_keys=True) \
                .cumsum()
            indicators = indicators.rename(
                columns=lambda col: col[:-len(self.col_suffix)]+'_indicator')
            indicators = 1-indicators
            counts = counts.rename(
                columns=lambda col: col[:-len(self.col_suffix)]+'_count')
            return pd.concat([df, indicators, counts], axis=1)

        return ddf.map_partitions(counter_and_indicators)


class DerivedFeatures(TransformerMixin):
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
        temp = vm("temp")
        hr = vm("hr")
        resp = vm("resp")
        pco2 = vm("pco2")
        wbc = vm("wbc")

        # Calculate score:
        temp_data = (df[temp] > 38) | (df[temp] < 36) * 1
        temp_data.name = "temp"
        hr_data = (df[hr] > 90) * 1
        hr_data.name = "hr"
        paco2_data = ((df[resp] > 20) | (df[pco2] < 32)) * 1
        paco2_data.name = "rr.paco2"
        wbc_data = ((df[wbc] < 4) | (df[wbc] > 12)) * 1
        wbc_data.name = "wbc_"

        # df_sirs = dd.multi.concat([temp_data, hr_data, paco2_data, wbc_data], axis=1)

        # Sum each row, if >= 2 then mark as SIRS
        sirs = ((temp_data + hr_data + paco2_data + wbc_data) >= 2) * 1.0

        # Leave the binary and the path sirs
        # sirs_df = dd.multi.concat([sirs, df_sirs.sum(axis=1)], axis=1)
        # sirs_df.columns = ["SIRS", "SIRS_path"]

        return sirs

    def mews_score(self, df):
        # mews = np.zeros(shape=df.shape[0])
        # determine variables:
        vm = self.vm
        sbp = vm("sbp")
        hr = vm("hr")
        resp = vm("resp")
        temp = vm("temp")

        # sbp
        sbp = df[sbp]
        hr = df[hr]
        resp = df[resp]
        temp = df[temp]
        mews = (
            (sbp <= 70) * 3.0
            + ((70 < sbp) & (sbp <= 80)) * 2.0
            + ((80 < sbp) & (sbp <= 100)) * 1.0
            + (sbp >= 200) * 2.0
            + ((40 < hr) & (hr <= 50)) * 1.0
            + ((100 < hr) & (hr <= 110)) * 1.0
            + ((110 < hr) & (hr < 130)) * 2.0
            + (hr >= 130) * 3.0
            + (resp < 9) * 2.0
            + ((15 < resp) & (resp <= 20)) * 1.0
            + ((20 < resp) & (resp < 30)) * 2.0
            + (resp >= 30) * 3.0
            + (temp < 35) * 2.0
            + (temp >= 38.5) * 2.0
        )
        return mews

    def qSOFA(self, df):
        vm = self.vm
        resp = vm("resp")
        sbp = vm("sbp")

        # qsofa = np.zeros(shape=df.shape[0])
        # qsofa[df[resp].values >= 22] += 1
        # qsofa[df[sbp].values <= 100] += 1
        return (df[resp] >= 22) * 1.0 + (df[sbp] <= 100) * 1.0

    def SOFA(self, df):
        vm = self.vm
        plt = vm("plt")
        bili = vm("bili")
        map = vm("map")
        crea = vm("crea")

        # sofa = np.zeros(shape=df.shape[0])

        platelets = df[plt]
        bilirubin = df[bili]
        map = df[map]
        creatinine = df[crea]
        sofa = (
            # Coagulation
            ((100 <= platelets) & (platelets < 150)) * 1.0
            + ((50 <= platelets) & (platelets < 100)) * 2.0
            + ((20 <= platelets) & (platelets < 50)) * 3.0
            + (platelets < 20) * 4.0
            # Liver
            + ((1.2 <= bilirubin) & (bilirubin <= 1.9)) * 1.0
            + ((1.9 < bilirubin) & (bilirubin <= 5.9)) * 2.0
            + ((5.9 < bilirubin) & (bilirubin <= 11.9)) * 3.0
            + (bilirubin > 11.9) * 4.0
            # Cardiovascular
            + (map < 70) * 1.0
            # crea
            + ((1.2 <= creatinine) & (creatinine <= 1.9)) * 1.0
            + ((1.9 < creatinine) & (creatinine <= 3.4)) * 2.0
            + ((3.4 < creatinine) & (creatinine <= 4.9)) * 3.0
            + (creatinine > 4.9) * 4.0
        )

        return sofa

    def SOFA_deterioration(self, s):
        def check_24hr_deterioration(s):
            """ Check the max deterioration over the last 24 hours, if >= 2 then mark as a 1"""
            min_prev_23_hrs = s.rolling(
                24, min_periods=1).apply(np.nanmin, raw=True)
            return s - min_prev_23_hrs

        sofa_det = s.groupby(s.index.name, sort=False).apply(
            check_24hr_deterioration)
        # Remove negative sofa values
        sofa_det = (sofa_det < 0) * 0.0 + (sofa_det >= 0) * sofa_det
        return sofa_det

    def septic_shock(self, df):
        vm = self.vm
        map = vm("map")
        lact = vm("lact")

        # shock = np.zeros(shape=df.shape[0])
        # shock[df[map].values < 65] += 1
        # shock[df[lact].values > 2] += 1
        return (df[map] < 65) * 1 + (df[lact] > 2) * 1

    def transform(self, ddf):
        vm = self.vm

        def derived_features(df):
            """Function to apply to each partition.

            This reduces the number of dask tasks that need to be created
            significantly.
            """
            hr = vm("hr")
            sbp = vm("sbp")
            bun = vm("bun")
            crea = vm("crea")
            po2 = vm("po2")
            fio2 = vm("fio2")
            plt = vm("plt")
            map = vm("map")
            bili = vm("bili")

            sofa = self.SOFA(df[[plt, map, crea, bili]])

            return df.assign(
                **{
                    # Ratios:
                    "ShockIndex_derived": df[hr] / df[sbp],
                    "bun/cr_derived": df[bun] / df[crea],
                    "po2/fio2_derived": df[po2] / df[fio2],
                    # shouldnt it be PaO2/Fi ratio?
                    # SOFA
                    "SOFA_derived": sofa,
                    "SOFAdeterioration_derived": self.SOFA_deterioration(sofa),
                    "qSOFA_derived": self.qSOFA(df),
                    "SepticShock_derived": self.septic_shock(df),
                    # Other scores
                    "MEWS_derived": self.mews_score(df),
                    "SIRS_derived": self.sirs_criteria(df),
                }
            )

        return ddf.map_partitions(derived_features)


class DropColumns(TransformerMixin):
    """
    Drop column group via shared suffix.
    """
    def __init__(self, suffix='_locf'):
        self.suffix = suffix  
    def fit(self, df, labels=None):
        return self
    def get_drop_cols(self, cols):
        return [x for x in cols if self.suffix in x]
    def transform(self, df):
        columns = self.get_drop_cols(df.columns)
        df = df.drop(columns, axis=1)
        return df



class Normalizer(TransformerMixin):
    """Performs normalization (z-scoring) of columns."""

    def __init__(self, patient_ids, suffix=None, drop_cols=None, assign_values=True, eps=0.01):
        """
        Args:
        - patient_ids: Patient ids that should be used for computing
          normalization statistics
        - suffix: when provided all columns having this suffix are normalized.
            otherwise, all but the excluded columns are normalized.
        - drop_cols: optional list of columns to drop; must be valid
          columns of the input data frame
        - assign_values: Assign the transformed values to the input dataframe.
          If set to false will solely return the normalized columns.
        - eps: epsilon threshold to reject spurious stds() too close to zero, 
            these stds are then replaced with 1 for robuster z-scoring.
        """
        self.patient_ids = patient_ids
        self.suffix = suffix
        self.drop_cols = drop_cols
        self.stats = None
        self.assign_values = assign_values
        self.eps = eps

    def _drop_columns(self, df):
        """ Utility function, to select available columns to
            drop (the list can reach over different feature sets)
        """
        if self.suffix is not None:
            columns = [col for col in df.columns if any(
                [s in col for s in self.suffix])
            ]
            return df[columns]
        if self.drop_cols is not None:
            return df.drop(columns=self.drop_cols, errors='ignore')
            # ignoring errors useful for downstream loading where we normalize with column subset
        else:
            return df

    def _compute_stats(self, df):
        if self.patient_ids is not None:
            df = df.loc[self.patient_ids]
        self.stats = {
            # We want to persist these. This ensures that when a worker is
            # killed we don't loose the result of this expensive computation.
            'means': df.mean(split_every=5),
            'stds':  df.std(split_every=5).map_partitions(self.robustify_vec, eps=self.eps)
        }

    @staticmethod
    def robustify(x,eps=0.1):
        return 1 if abs(x) < eps else x 

    @staticmethod
    def robustify_vec(series, eps=0.1):
        series[series.abs() < eps] = 1
        return series

    def _apply_normalization(self, df):
        assert self.stats is not None
        return (df - self.stats['means']) / self.stats['stds']

    def fit(self, df, labels=None):
        self._compute_stats(self._drop_columns(df))
        return self

    def transform(self, df):
        normalized = self._apply_normalization(self._drop_columns(df))
        if self.assign_values:
            return df.assign(**{
                col: normalized[col]
                for col in normalized.columns
            })
        else:
            return normalized
        # def normalize(df, means, stds):
        #     data = self._drop_columns(df)
        #     normalized = ((data - means) / stds).astype(float)

        #     if self.assign_values:
        #         return df.assign(**{
        #             col: normalized[col]
        #             for col in normalized.columns
        #         })
        #     else:
        #         return normalized

        # return ddf.map_partitions(
        #     normalize, means=self.stats['means'], stds=self.stats['stds'])




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

    def transform_id(self, inputs):
        """Calculate utility score differences for each patient."""
        labels = inputs[self.label]
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

        return pd.concat([inputs, scores], axis=1)


class InvalidTimesFiltration(TransformerMixin):
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
        def drop_invalid_times(df):
            df = self._remove_pre_icu(df)
            df = self._remove_too_few_observations(df, columns)
            return df
        return ddf.map_partitions(drop_invalid_times)
