"""
Various data loading, filtering and feature transformers, TODO: add copyright notice here! 
"""
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, BaseEstimator
from kymatio.numpy import Scattering1D

from .utils import save_pickle, load_pickle 
from .base import BaseIDTransformer, ParallelBaseIDTransformer, DaskIDTransformer
from .extracted import (ts_columns, columns_not_to_normalize, extended_ts_columns, 
    colums_to_drop, baseline_cols) 

import dask.dataframe as dd

from src import datasets
from src.evaluation.sklearn_utils import nanany
from src.evaluation.physionet2019_score import compute_prediction_utility


class DataframeFromDataloader(TransformerMixin, BaseEstimator):
    """
    Transform method that takes dataset class, loads corresponding iterable dataloader and 
    returns (and saves if requested) the dataset in one large sklearn-ready pd dataframe 
    format with patient and time multi-indices.
    """
    def __init__(self, save=False, dataset_cls=None, data_dir=None, split='train', drop_label=True, n_jobs=10, concat_output=False, custom_path=None):
        self.save = save
        dataset_class = getattr(datasets, dataset_cls)
        self.split = split
        if custom_path:
            #remove last two folders (framework-agnostic) from data_dir path to get to base_dir
            custom_path = os.path.split(os.path.split(data_dir)[0])[0]
        self.dataloader = dataset_class(split=split, as_dict=False, custom_path=custom_path)
        self.data_dir = data_dir #outdir to save raw dataframe
        self.drop_label = drop_label
        self.n_jobs = n_jobs
        self.concat_output = concat_output
 
    def fit(self, df, labels=None):
        return self

    def _load_index_and_prepare(self, index):
        patient_id, df = self.dataloader[index]
        df['id'] = patient_id
        # Idx according to id and time
        df = df.rename(columns={'stay_time': 'time'}) #rename for easier understanding
        df.reset_index(drop=True, inplace=True)
        df.set_index(['id', 'time'], inplace=True)
        df.sort_index(ascending=True, inplace=True)

        #Sanity check: ensure that bool cols are floats (checking all columns each time is costly, so wrote out the identified few)
        bool_cols = ['sep3', 'vaso_ind', 'vent_ind'] 
        df[bool_cols] = df[bool_cols].astype(float) #necessary, as ints would mess with categorical onehot encoder
        
        #Remove few columns which are not used at all (e.g. interventions)
        df = df.drop(columns = colums_to_drop)
        return df

    def transform(self, df):
        """
        Takes dataset_cls (in self), loads iteratable instances and concatenates them to a multi-indexed 
        pandas dataframe which is returned
        """
        # Make the dataframe
        if self.n_jobs == 1: #also use parallel when n_jobs=-1
            #this case prevents us from adding an additional argument for batchsize when using demo dataset
            output = [self._load_index_and_prepare(i) for i in range(len(self.dataloader))]
        else:  
            output = Parallel(n_jobs=self.n_jobs, verbose=1, batch_size=1000)(
                delayed(self._load_index_and_prepare)(i) for i in range(len(self.dataloader)))
        if self.concat_output:
            output = pd.concat(output)
        print('Done with DataframeFromDataloader')
        return output


class CalculateUtilityScores(BaseIDTransformer):
    """Calculate utility scores from patient.

    This transformer calculates the utility score of a patient. It can
    either function as a passthrough class that stores data internally
    or as a transformer class that extends a given data frame.
    """

    def __init__(
        self,
        passthrough=True,
        label='sep3',
        score_name='utility'
    ):
        """Create new instance of class.

        Parameters
        ----------
        passthrough : bool
            If set, does not modify input data. Instead, the scores are
            calculated and stored in the `scores` property of the class,
            which is a stand-alone data frame.

        label : str
            Indicates which column to use for the sepsis label.

        score_name : str
            Indicates the name of the column that will contain the
            calculated utility score. If `passthrough` is set, the
            column name will only be used in the result data frame
            instead of being used as a new column for the *input*.
        """
        self.passthrough = passthrough
        self.label = label
        self.score_name = score_name
        self.df_scores = None

    @property
    def scores(self):
        """Return scores calculated by the class.

        Returns
        -------
        Data frame containing information about the sample/patient ID,
        the time, and the respective utility score. Can be `None` when
        the class was not used before.

        The data frame will share the same index as the original data,
        which is used as the input.
        """
        return self.df_scores

    def transform_id(self, df):
        """Calculate utility score differences for each patient."""
        labels = df[self.label]
        n = len(labels)

        zeros = compute_prediction_utility(
            labels.values,
            np.zeros(shape=n),
            return_all_scores=True
        )

        ones = compute_prediction_utility(
            labels.values,
            np.ones(shape=n),
            return_all_scores=True
        )

        scores = pd.DataFrame(
            index=labels.index,
            data=ones - zeros,
            columns=[self.score_name]
        )

        self.df_scores = scores

        # Check whether passthrough is required. If so, there's nothing
        # to do from our side---we just store the data frame & continue
        # by returning the original data frame.
        if self.passthrough:
            pass

        # Create a new column that stores the score. Some additional
        # sanity checks ensure that we do not do make any mistakes.
        else:
            assert self.score_name not in df.columns, \
                   'Score column name must not exist in data frame.'

            assert df.index.equals(self.df_scores.index), \
                   'Index of original data frame must not deviate.'

            df[self.score_name] = scores[self.score_name]

        print('Done with CalculateUtilityScores')
        return df


class DropLabels(TransformerMixin, BaseEstimator):
    """
    Remove label information, which was required for filtering steps.
    """
    def __init__(self, label='sep3', save=True, data_dir=None, split=None):
        self.label = label
        self.save = save
        self.data_dir = data_dir
        self.split = split

    def fit(self, df, labels=None):
        return self

    def transform(self, df):
        if self.save:
            labels = df[self.label]
            save_pickle(labels, os.path.join(self.data_dir, f'y_{self.split}.pkl'))
        df = df.drop(self.label, axis=1)

        print('Done with DropLabels')
        return df

class DropColumns(TransformerMixin, BaseEstimator):
    """
    Drop and potentially save columns. By default we drop all baseline scores.
    """
    def __init__(self, columns=baseline_cols, label='sep3', save=False, 
                 data_dir=None, split=None):
        self.columns = columns
        self.label = label
        self.save = save
        self.data_dir = data_dir
        self.split = split
        
    def fit(self, df, labels=None):
        return self

    def transform(self, df):
        if self.save:
            cols_to_save = self.columns + [self.label]
            save_pickle(df[cols_to_save], os.path.join(self.data_dir, f'baselines_{self.split}.pkl'))
        df = df.drop(self.columns, axis=1, errors='ignore')

        print('Done with DropColumns')
        return df


class CategoricalOneHotEncoder(TransformerMixin, BaseEstimator):
    """
    Categorical variables are one-hot encoded. 
    """
    def __init__(self):
        pass

    def fit(self, df, labels=None):
        return self

    def transform(self, df):
        categorical_cols = list(set(df.columns) - set(df._get_numeric_data().columns))
        #Sanity check as labels can be boolean
        if 'sep3' in categorical_cols:
            categorical_cols.remove('sep3')

        print(f'Encoding {categorical_cols}')    
        df = pd.get_dummies(df)

        print('Done with Categorical Variable One-hot Encoder..')
        
        print('currently very rare extra cols are removed to harmonize feature set')
        # for gender this is still encoded as both male and female 0
        for col in ['sex_Other', 'sex_Unknown']:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

class PatientFiltration(ParallelBaseIDTransformer):
    """
    Removes patients which do not match inclusion criteria:
    --> sepsis-cases with:
        - onset within the first t_start hours of ICU stay (or before)
        - onset after t_end hours of ICU stay 
        defaults: t_start = 3 (which corresponds to 4 hours due to rounding of the chart times), t_end = 168 (1 week)
    """
    def __init__(self, save=False, data_dir=None, split='train', onset_bounds=(3,168), label='sep3', **kwargs):
        self.save = save
        self.data_dir = data_dir
        self.split = split
        self.onset_bounds = onset_bounds
        self.label = label
        super().__init__(**kwargs)

    def fit(self, df, labels=None):
        return self

    def transform_id(self, df):
        """ It seems like the easiest solution is to quickly load the small label pickle (instead of breaking the pipeline API here)
        """
        label = df[self.label] #load_pickle(os.path.join(self.data_dir, f'y_{self.split}.pkl'))
        is_case = nanany(label)
        if is_case:
            onset = np.nanargmax(label)
            start, end = self.onset_bounds
            if onset <= start or onset > end:
                return None  # This drops the patient
        return df


class CaseFiltrationAfterOnset(ParallelBaseIDTransformer):
    """
    This transform removes case time points after a predefined cut_off time after 
    sepsis onset. This prevents us from making predictions long after sepsis (which
    would be not very useful) while it still allows to punish models that detect sepsis
    only too late.
    Additionally, to ensure that controls cannot be systematically longer than cases,
    we cut controls after cut_off + onset_bounds[1] hours (i.e. after 7d + 24h) 
    """
    def __init__(self, cut_off=24, onset_bounds=(3,168), label='sep3', **kwargs):
        self.cut_off = cut_off
        self.label = label
        self.control_cut_off = cut_off + onset_bounds[1]
        super().__init__(**kwargs)

    def transform_id(self, df):
        """ Patient-level transform
        """
        if df[self.label].sum() > 0:
            #Case:
            onset = np.nanargmax(df[self.label])
            return df[:onset+self.cut_off+1]
        else:
            #Control:
            return df[df.index.levels[1] <= self.control_cut_off] #we assume time to be index.levels[1]

class InvalidTimesFiltration(TransformerMixin, BaseEstimator):
    """
    This transform removes invalid time steps right before training / prediction phase (final step of preprocessing).
        - time steps before ICU admission (i.e. negative timestamps) 
        - time steps with less than <thres> many observations. 
    """
    def __init__(self, thres=1, label='sep3'):
        self.thres = thres
        self.label = label

    def fit(self, df, labels=None):
        return self

    def _remove_pre_icu(self, df):
        return df[df['time'] >= 0]

    def _remove_too_few_observations(self, df, thres):
        """ in rare cases it is possible that Lookbackfeatures leak zeros into invalid nan rows (which makes time handling easier)
            additionally drop those rows by identifying nan labels
        """
        ind_to_keep = (~df[ts_columns].isnull()).sum(axis=1) >= thres
        ind_labels = (~df[self.label].isnull()) #sanity check to prevent lookbackfeatures 0s to mess up nan rows
        ind_to_keep = np.logical_and(ind_to_keep, ind_labels)
        return df[ind_to_keep]

    def _transform_id(self, df):
        """ Patient level transformation
        """
        df = self._remove_pre_icu(df)
        return self._remove_too_few_observations(df, self.thres)

    def transform(self, df):
        """ As this time filtering step also affects labels, for simplicity we load and adjust them too. 
        """
        df = df.groupby('id', group_keys=False).apply(self._transform_id)
        
        print('Done with InvalidTimesFiltration')
        return df

class CarryForwardImputation(DaskIDTransformer):
    """
    First fills in missing values by carrying forward, then fills remaining NaN (at start) with zero. 
    """
    def transform_id(self, df):
        return df.fillna(method='ffill').fillna(0)

class IndicatorImputation(ParallelBaseIDTransformer):
    """
    Adds indicator dimension for every channel to indicate if there was a nan
    IndicatorImputation still requires FillMissing afterwards!
    """
    def __init__(self, n_jobs=4, **kwargs):
        super().__init__(n_jobs=n_jobs, **kwargs)

    def transform_id(self, df):
        cols = ts_columns #we consider all time-series columns (for consistency with pytorch approach)
        invalid_indicators = (df[cols].isnull()).astype(int).add_suffix('_indicator') 
        df = pd.concat([df.fillna(0), invalid_indicators], axis=1)
        return df

class FillMissing(TransformerMixin, BaseEstimator):
    """ Method to fill nan columns, either with zeros, column mean oder median (last two leak from the future if done offline) """
    def __init__(self, method='zero', col_vals=None):
        self.method = method
        self.col_vals = col_vals

    def fit(self, df, labels=None):
        if self.method == 'mean':
            self.col_vals = df.mean().to_dict()
        elif self.method == 'median':
            self.col_vals = df.median().to_dict()
        elif self.method == 'zero':
            self.col_vals = 0
        return self

    def transform(self, df):
        if self.col_vals is not None:
            df = df.fillna(self.col_vals)
        return df


class Normalizer(TransformerMixin, BaseEstimator):
    """
    Performs normalization (z-scoring) of columns which are not explicitly excluded.
    Caches stats of the train split for the remaining splits to use.
    """
    def __init__(self, data_dir, split):
        self.split = split
        self.data_dir = data_dir
        self.drop_cols = columns_not_to_normalize 
        self.normalizer_dir = os.path.join(data_dir, 'normalizer')
        self.normalizer_file = os.path.join(self.normalizer_dir, f'normalizer_stats.pkl')
 
    def _drop_columns(self, df, cols_to_drop):
        """ Utiliy function, to select available columns to 
            drop (the list can reach over different feature sets)
        """
        drop_cols = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=drop_cols)
        return df, drop_cols         
    
    def _compute_stats(self, df):
        return {
            'means': df.mean(),
            'stds': df.std()
        }
    def _apply_normalizer(self, df, stats):
        means = stats['means']
        stds = stats['stds']
        return (df - means) / stds 
  
    def fit(self, df, labels=None):
        if self.split == 'train':
            df, _ = self._drop_columns(df, self.drop_cols)
            self.stats = self._compute_stats(df)
            os.makedirs(self.normalizer_dir, exist_ok=True)
            save_pickle(self.stats, self.normalizer_file)
        else:
            try:
                self.stats = load_pickle(self.normalizer_file) 
            except:
                raise ValueError('Normalization file not found. Compute normalization for train split first!') 
            #TODO: assert that cols in loaded stats are the same as all_cols \ drop_cols
        return self
    
    def transform(self, df):
        df_to_normalize, remaining_cols = self._drop_columns(df, self.drop_cols)
        df_normalized = self._apply_normalizer(df_to_normalize, self.stats)
        df_out = pd.concat([df_normalized, df[remaining_cols]], axis=1) 
        return df_out


class LookbackFeatures(DaskIDTransformer):
    """
    Simple statistical features including moments over a tunable look-back window.
    """
    def __init__(self, stats=None, windows = [4, 8, 16],**kwargs):
        """ takes dictionary of stats (keys) and corresponding look-back windows (values)
            to compute each stat for each time series variable. 
            Below a default stats dictionary of look-back 5 hours is implemented.
        """
        self.cols = extended_ts_columns #apply look-back stats to time series columns
        if stats is None:
            keys = ['min', 'max', 'mean', 'median', 'var']
            stats = []
            for key in keys:
                for window in windows:
                    stats.append( (key, window) )
        self.stats = stats
        super().__init__(**kwargs)

    def _compute_stat(self, df, stat, window):
        """ Computes current statistic over look-back window for all time series variables
            and returns renamed df
        """
        # first get the function to compute the statistic:
        # determine available columns (which could be a subset of the
        # predefined cols, depending on the setup):
        used_cols = [col for col in self.cols if col in df.columns]
        grouped = df[used_cols].rolling(window, min_periods=0)
        stats = getattr(grouped, stat)().fillna(0)
        # the first window is  computed in an expanding fashion. Still certain
        # stats (e.g. var) leave nan in the start, replace it with 0s here. 
        # rename the features by the statistic:
        stats = stats.add_suffix(f'_{stat}_{window}_hours') 

        return stats

    def transform_id(self, df):
        #compute all statistics in stats dictionary:
        features = [df]
        for stat, window in self.stats:
            feature = self._compute_stat(df, stat, window)
            features.append(feature)
        df_out = pd.concat(features, axis=1)
        return df_out

#TODO adjust variable names!
class DerivedFeatures(TransformerMixin, BaseEstimator):
    """
    Adds any derived features thought to be useful
        - Shock Index: hr/sbp
        - Bun/crea ratio: Bun/crea
        - Hepatic SOFA: Bilirubin SOFA score

    # Can add renal and neruologic sofa
    """
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    @staticmethod
    def sirs_criteria(df):
        # Create a dataframe that stores true false for each category
        df_sirs = pd.DataFrame(index=df.index, columns=['temp', 'hr', 'rr.paco2', 'wbc_'])
        df_sirs['temp'] = ((df['temp'] > 38) | (df['temp'] < 36))
        df_sirs['hr'] = df['hr'] > 90
        df_sirs['rr.paco2'] = ((df['resp'] > 20) | (df['pco2'] < 32))
        #TODO: wbc is not available anymore!
        df_sirs['wbc_'] = ((df['wbc'] < 4) | (df['wbc'] > 12))

        # Sum each row, if >= 2 then mar as SIRS
        sirs = pd.to_numeric((df_sirs.sum(axis=1) >= 2) * 1)

        # Leave the binary and the path sirs
        sirs_df = pd.concat([sirs, df_sirs.sum(axis=1)], axis=1)
        sirs_df.columns = ['SIRS', 'SIRS_path']

        return sirs_df

    @staticmethod
    def mews_score(df):
        mews = np.zeros(shape=df.shape[0])

        # sbp
        sbp = df['sbp'].values
        mews[sbp <= 70] += 3
        mews[(70 < sbp) & (sbp <= 80)] += 2
        mews[(80 < sbp) & (sbp <= 100)] += 1
        mews[sbp >= 200] += 2

        # hr
        hr = df['hr'].values
        mews[hr < 40] += 2
        mews[(40 < hr) & (hr <= 50)] += 1
        mews[(100 < hr) & (hr <= 110)] += 1
        mews[(110 < hr) & (hr < 130)] += 2
        mews[hr >= 130] += 3

        # resp
        resp = df['resp'].values
        mews[resp < 9] += 2
        mews[(15 < resp) & (resp <= 20)] += 1
        mews[(20 < resp) & (resp < 30)] += 2
        mews[resp >= 30] += 3

        # temp
        temp = df['temp'].values
        mews[temp < 35] += 2
        mews[(temp >= 35) & (temp < 38.5 ) ] += 0
        mews[temp >= 38.5] += 2
        
        return mews

    @staticmethod
    def qSOFA(df):
        qsofa = np.zeros(shape=df.shape[0])
        qsofa[df['resp'].values >= 22] += 1
        qsofa[df['sbp'].values <= 100] += 1
        return qsofa

    @staticmethod
    def SOFA(df):
        sofa = np.zeros(shape=df.shape[0])

        # Coagulation
        platelets = df['plt'].values
        sofa[platelets >= 150] += 0
        sofa[(100 <= platelets) & (platelets < 150)] += 1
        sofa[(50 <= platelets) & (platelets < 100)] += 2
        sofa[(20 <= platelets) & (platelets < 50)] += 3
        sofa[platelets < 20] += 4

        # Liver
        bilirubin = df['bili'].values
        sofa[bilirubin < 1.2] += 0
        sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
        sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
        sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
        sofa[bilirubin > 11.9] += 4

        # Cardiovascular
        map = df['map'].values
        sofa[map >= 70] += 0
        sofa[map < 70] += 1

        # crea
        creatinine = df['crea'].values
        sofa[creatinine < 1.2] += 0
        sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
        sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
        sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
        sofa[creatinine > 4.9] += 4

        return sofa

    @staticmethod
    def SOFA_max_24(s):
        """ Get the max value of the SOFA score over the prev 24 hrs """
        def find_24_hr_max(s):
            prev_24_hrs = pd.concat([s.shift(i) for i in range(24)], axis=1).values[:, ::-1]
            return pd.Series(index=s.index, data=np.nanmax(prev_24_hrs, axis=1))
        sofa_24 = s.groupby('id').apply(find_24_hr_max)
        return sofa_24

    @staticmethod
    def SOFA_deterioration_new(s):
        def check_24hr_deterioration(s):
            """ Check the max deterioration over the last 24 hours, if >= 2 then mark as a 1"""
            prev_24_hrs = pd.concat([s.shift(i) for i in range(24)], axis=1).values[:, ::-1]

            def max_deteriorate(arr):
                return np.nanmin([arr[i] - np.nanmax(arr[i+1:]) for i in range(arr.shape[-1]-1)])

            tfr_hr_min = np.apply_along_axis(max_deteriorate, 1, prev_24_hrs)
            return pd.Series(index=s.index, data=tfr_hr_min)
        sofa_det = s.groupby('id').apply(check_24hr_deterioration)
        return sofa_det

    @staticmethod
    def SOFA_deterioration(s):
        def check_24hr_deterioration(s):
            """ Check the max deterioration over the last 24 hours, if >= 2 then mark as a 1"""
            prev_23_hrs = pd.concat([s.shift(i) for i in range(1, 24)], axis=1).values
            tfr_hr_min = np.nanmin(prev_23_hrs, axis=1)
            return pd.Series(index=s.index, data=(s.values - tfr_hr_min))
        sofa_det = s.groupby('id').apply(check_24hr_deterioration)
        sofa_det[sofa_det < 0] = 0
        sofa_det = sofa_det
        return sofa_det

    @staticmethod
    def septic_shock(df):
        shock = np.zeros(shape=df.shape[0])
        shock[df['map'].values < 65] += 1
        shock[df['lact'].values > 2] += 1
        return shock

    def transform(self, df):
        # Compute things
        df['ShockIndex'] = df['hr'].values / df['sbp'].values
        df['bun/cr'] = df['bun'].values / df['crea'].values
        df['po2/fio2'] = df['po2'].values / df['fio2'].values #shouldnt it be PaO2/Fi ratio?

        # SOFA
        df['SOFA'] = self.SOFA(df[['plt', 'map', 'crea', 'bili']])
        df['SOFA_deterioration'] = self.SOFA_deterioration(df['SOFA'])
        #df['sofa_max_24hrs'] = self.SOFA_max_24(df['SOFA'])
        df['qSOFA'] = self.qSOFA(df)
        # df['SOFA_24hrmaxdet'] = self.SOFA_deterioration(df['SOFA_max_24hrs'])
        # df['SOFA_deterioration_new'] = self.SOFA_deterioration_new(df['SOFA_max_24hrs'])
        df['SepticShock'] = self.septic_shock(df)

        # Other scores
        sirs_df = self.sirs_criteria(df)
        df['MEWS'] = self.mews_score(df)
        df['SIRS'] = sirs_df['SIRS']
        return df

class MeasurementCounter(DaskIDTransformer):
    """ Adds a count of the number of measurements up to the given timepoint. """
    def __init__(self, n_jobs=4, **kwargs):
        super().__init__(n_jobs=n_jobs, **kwargs)
        # we only count raw time series measurements 
        self.columns = ts_columns

    def fit(self, df, labels=None):
        return self

    def transform_id(self, df):
        # Make a counts frame
        counts = deepcopy(df)
        counts.drop([x for x in df.columns if x not in self.columns], axis=1, inplace=True)

        # Turn any floats into counts
        for col in self.columns:
            counts[col][~counts[col].isna()] = 1
        counts = counts.replace(np.nan, 0)

        # Get the counts for each person
        counts = counts.cumsum() 

        # Rename
        counts.columns = [x + '_count' for x in counts.columns]

        return pd.concat([df, counts], axis=1)

class WaveletFeatures(ParallelBaseIDTransformer):
    """ Computes wavelet scattering up to given time per time series channel """
    def __init__(self, n_jobs=4, T=32, J=2, Q=1, output_size=32, **kwargs):
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
        super().__init__(n_jobs=n_jobs, **kwargs)
        # we process the raw time series measurements 
        self.T = T
        self.columns = ts_columns
        self.scatter = Scattering1D(J, T, Q) 
        self.output_size = output_size

    def fit(self, df, labels=None):
        return self

    def transform_id(self, df):
        """ process invididual patient """  
        inputs = deepcopy(df)
        inputs.drop([x for x in df.columns if x not in self.columns], axis=1, inplace=True)

        # pad time series with T-1 0s (for getting online features of fixed window size)
        inputs = self._pad_df(inputs, n_pad=self.T-1)
        #inputs.reset_index(drop=True, inplace=True)
        
        wavelets = self._compute_wavelets(inputs)
        
        assert df.shape[0] == wavelets.shape[0]
        wavelets.index = df.index

        return pd.concat([df, wavelets], axis=1)

    def _pad_df(self, df, n_pad, value=0):
        """ util function to pad <n_pad> zeros at the start of the df 
        """
        x = np.concatenate([np.zeros([n_pad,df.shape[1]]), df.values]) 
        return pd.DataFrame(x, columns=df.columns)

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
        df[col].rolling(window=self.T).apply(self._rolling_function, kwargs={'df': col_df})
        return col_df

    def _rolling_function(self, window, df):
        """actually compute wavelet scattering in a rolling window """
        df.loc[window.index.min()] = self.scatter(window.values).flatten()
        return 1 # rolling funcs need to return index..  
    
