"""
Various data loading, filtering and feature transformers. 

One class, DerivedFeatures, was kindly provided by James Morrill,
source: https://github.com/jambo6/physionet_sepsis_challenge_2019/blob/master/src/data/transformers.py
Notably, we modified these derived features to account for our larger variable set.
 
"""
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, BaseEstimator
from kymatio.numpy import Scattering1D
import iisignature as iis 

from .utils import save_pickle, load_pickle 
from .base import BaseIDTransformer, ParallelBaseIDTransformer, DaskIDTransformer
from .extracted import (ts_columns, columns_not_to_normalize, extended_ts_columns, 
    colums_to_drop, baseline_cols, vital_columns, lab_columns_chemistry, 
    lab_columns_organs, lab_columns_hemo, scores_and_indicators_columns ) 

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
    def __init__(self, dataset_cls=None, data_dir=None, split='train', n_jobs=10, concat_output=False, custom_path=None):
        dataset_class = getattr(datasets, dataset_cls)
        self.split = split
        if custom_path:
            #remove last two folders (framework-agnostic) from data_dir path to get to base_dir
            custom_path = os.path.split(os.path.split(data_dir)[0])[0]
        self.dataloader = dataset_class(split=split, as_dict=False, custom_path=custom_path)
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

class DataframeFromParquet(TransformerMixin, BaseEstimator):
    """
    This transform allows to load data directly from parquet to 
    a pandas dataframe.
    """
    def __init__(self, path, vm=None):
        """
        Args:
            - path: path to parquet data file
            - vm: variable mapping object
        """
        self.path = path
        self.vm = vm

    def fit(self, df, labels=None):
        return self

    def transform(self, df=None):
        df = pd.read_parquet(self.path, engine='pyarrow', columns=self.vm.core_set) 

        # convert bools to float
        bool_cols = [col for col in df.columns if df[col].dtype == bool]        
        df[bool_cols] = df[bool_cols].astype(float)
        vm = self.vm
        df.set_index([vm('id'), vm('time')], inplace=True)
        
        df.sort_index(ascending=True, inplace=True)

        return df 

class CalculateUtilityScores(ParallelBaseIDTransformer):
    """Calculate utility scores from patient.

    Inspired by Morill et al. [1], this transformer calculates the
    utility target U(1) - U(0) of a patient.  It can either function
    as a passthrough class that stores data internally or as a
    transformer class that extends a given data frame.

    [1]: http://www.cinc.org/archives/2019/pdf/CinC2019-014.pdf
    """

    def __init__(
        self,
        passthrough=True,
        label='sep3',
        score_name='utility',
        shift=0,
        **kwargs
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

        )score_name : str
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

        self.passthrough = passthrough
        self.label = label
        self.score_name = score_name
        self.shift = shift

        # Internal data frame with scores; can be used later on for
        # other purposes.
        self.df_scores = None

        # Indicated for superclass usage: will automatically concatenate
        # all outputs instead of delivering them for all patients
        # individually.
        self.concat_output = True

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

        return df


class DropColumns(TransformerMixin, BaseEstimator):
    """
    Drop and potentially save columns. By default we drop all baseline scores.
    """
    def __init__(self, columns=baseline_cols, label='sep3', time='time', save=False, 
                 data_dir=None, split=None):
        self.columns = columns
        self.label = label
        self.time = time
        self.save = save
        self.data_dir = data_dir
        self.split = split
        
    def fit(self, df, labels=None):
        return self

    def transform(self, df):
        if self.save:
            cols_to_save = self.columns + [self.label, self.time]
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
    def __init__(self, thres=1, vm=None, suffix='_raw'):
        self.thres = thres
        self.label = vm('label')
        self.col_suffix = suffix

    def fit(self, df, labels=None):
        return self

    def _remove_pre_icu(self, df):
        return df[df['time'] >= 0]

    def _remove_too_few_observations(self, df, thres):
        """ in rare cases it is possible that Lookbackfeatures leak zeros into invalid nan rows (which makes time handling easier)
            additionally drop those rows by identifying nan labels
        """
        ind_to_keep = (~df[self.columns].isnull()).sum(axis=1) >= thres
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
        self.columns = [x for x in df.columns if self.col_suffix in x]

        df = df.groupby(self.vm('id'), group_keys=False).apply(self._transform_id)
        
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
    def __init__(self, n_jobs=4, suffix='_raw', imputation_value=None, **kwargs):
        """
        - imputation_value: (optional) value to impute at missing locations
        """
        self.col_suffix = suffix
        self.imputation_value = imputation_value
         
        super().__init__(n_jobs=n_jobs, **kwargs)

    def transform_id(self, df):
        cols = [x for x in df.columns if self.col_suffix in x]
        used_df = df[cols]
        used_df.columns = strip_cols(used_df.columns, [self.col_suffix])
        invalid_indicators = (used_df.isnull()).astype(int).add_suffix('_indicator')
        val = self.imputation_value
        if val:
            df = df.fillna(val) 
        df = pd.concat([df, invalid_indicators], axis=1)
        return df

class Normalizer(TransformerMixin, BaseEstimator):
    """
    Performs normalization (z-scoring) of columns.
    """
    def __init__(self, split_info, split='dev', config_dir='config', 
                 suffix=None, vm=None, save=False):
        """
        Args:
        - split_info: dict containing split information of the current dataset
        - split: which split to use for computing the normalization stats:
            [dev, train_0, train_1, train_2, train_3, train_4] where 'dev' represents
            the entire development data, and trainX the train part of 5 
            train/val splits of dev.
        - config_dir: directory for storing normalization stats 
        - suffix: when provided all columns having this suffix are normalized.
            otherwise, all but the excluded columns are normalized.
        - vm: variable mapping object, required when no suffix is provided 
            to determine exlucded columns.
        - save: flag, whether fitted stats are written out to json. 
        Note that column selection via suffix has precedence over column exclusion
        (via vm) in case both are provided. 
        """
                 
        self.split_info = split_info
        self.split = split
        self.config_dir = config_dir
        self.suffix = suffix
        self.vm = vm
        self.save = save
        self._check_signature_and_set_mode()
        self.ids = self._get_split_ids(split_info, split)
        #self.normalizer_dir = os.path.join(data_dir, 'normalizer')
        #self.normalizer_file = os.path.join(self.normalizer_dir, f'normalizer_stats.pkl')

    def _check_signature_and_set_mode(self):
        """ check that either suffix or vm is provided and split is among valid splits."""
        try:
            assert any([self.suffix, self.vm])
        except:
            raise ValueError('Either suffix or variable mapping must be provided!')
        if self.suffix: 
            self.mode = 'suffix'
        elif self.vm:
            self.mode = 'exclusion'
        assert self.split in ['dev', 'train_0', 'train_1', 'train_2', 'train_3', 'train_4']
        return self

    def _get_split_ids(self, info, name):
        if name == 'dev':
            ids = info['dev']['total'] 
        else:
            prefix, count = name.split('_')
            ids = info['dev'][f'split_{count}'][prefix] 
        return np.array(ids) 
  
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
        if self.mode == 'suffix':
            suffix = self.suffix
            if type(suffix) == str:
                suffix = [suffix]
            self.columns = [col for col in df.columns if any(
                [s in col for s in suffix] )
            ]
        elif self.mode == 'exclusion':
            vm = self.vm
            drop_cols = vm.all_cat('baseline')
            drop_cols.extend(
                [ vm(var) for var in ['label', 'sex', 'time']]
            )
            self.columns = [col for col in df.columns if col not in drop_cols]

        self.stats = self._compute_stats(df[self.columns])
        if self.save:
            raise NotImplementedError('TODO: implement saving of normalizer')
        return self
    
    def transform(self, df):
        if not self.stats:
            raise NotImplementedError('TODO: load from saved normalizer file')
        df[self.columns] = self._apply_normalizer(df[self.columns], self.stats) 
        #df_to_normalize, remaining_cols = self._drop_columns(df, self.drop_cols)
        #df_normalized = self._apply_normalizer(df_to_normalize, self.stats)
        #df_out = pd.concat([df_normalized, df[remaining_cols]], axis=1) 
        return df


class LookbackFeatures(DaskIDTransformer):
    """
    Simple statistical features including moments over a tunable look-back window.
    """
    def __init__(self, stats=None, windows = [4, 8, 16], 
        suffices=['_raw', '_derived'], vm=None,  **kwargs):
        """ takes dictionary of stats (keys) and corresponding look-back windows (values)
            to compute each stat for each time series variable. 
            Below a default stats dictionary of look-back 5 hours is implemented.
        """
        self.col_suffices = suffices  #apply look-back stats to time series columns
        if stats is None:
            keys = ['min', 'max', 'mean', 'median', 'var']
            stats = []
            for key in keys:
                for window in windows:
                    stats.append( (key, window) )
        self.stats = stats
        super().__init__(vm, **kwargs)

    def _compute_stat(self, df, stat, window):
        """ Computes current statistic over look-back window for all time series variables
            and returns renamed df
        """
        # first get the function to compute the statistic:
        # determine available columns (which could be a subset of the
        # predefined cols, depending on the setup):
        used_cols = [col for col in df.columns if any(
            [s in col for s in self.col_suffices] )]

        grouped = df[used_cols].rolling(window, min_periods=0)
        stats = getattr(grouped, stat)()
        # the first window is  computed in an expanding fashion. Note that certain
        # stats (e.g. var) leave nan in the start!
        # first strip away old suffix:
        stats.columns = strip_cols(stats.columns, self.col_suffices) 
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
        df_sirs = pd.DataFrame(index=df.index, columns=['temp', 'hr', 'rr.paco2', 'wbc_'])

        # determine variables:
        vm = self.vm
        temp = vm('temp')
        hr = vm('hr')
        resp = vm('resp')
        pco2 = vm('pco2')
        wbc = vm('wbc')
 
        # Calculate score: 
        df_sirs['temp'] = ((df[temp] > 38) | (df[temp] < 36))
        df_sirs[hr] = df[hr] > 90
        df_sirs['rr.paco2'] = ((df[resp] > 20) | (df[pco2] < 32))
        df_sirs['wbc_'] = ((df[wbc] < 4) | (df[wbc] > 12))

        # Sum each row, if >= 2 then mar as SIRS
        sirs = pd.to_numeric((df_sirs.sum(axis=1) >= 2) * 1)

        # Leave the binary and the path sirs
        sirs_df = pd.concat([sirs, df_sirs.sum(axis=1)], axis=1)
        sirs_df.columns = ['SIRS', 'SIRS_path']

        return sirs_df

    def mews_score(self, df):
        mews = np.zeros(shape=df.shape[0])
        # determine variables:
        vm = self.vm
        sbp = vm('sbp') 
        hr = vm('hr')
        resp = vm('resp')
        temp = vm('temp') 
 
        # sbp
        sbp = df[sbp].values
        mews[sbp <= 70] += 3
        mews[(70 < sbp) & (sbp <= 80)] += 2
        mews[(80 < sbp) & (sbp <= 100)] += 1
        mews[sbp >= 200] += 2

        # hr
        hr = df[hr].values
        mews[hr < 40] += 2
        mews[(40 < hr) & (hr <= 50)] += 1
        mews[(100 < hr) & (hr <= 110)] += 1
        mews[(110 < hr) & (hr < 130)] += 2
        mews[hr >= 130] += 3

        # resp
        resp = df[resp].values
        mews[resp < 9] += 2
        mews[(15 < resp) & (resp <= 20)] += 1
        mews[(20 < resp) & (resp < 30)] += 2
        mews[resp >= 30] += 3

        # temp
        temp = df[temp].values
        mews[temp < 35] += 2
        mews[(temp >= 35) & (temp < 38.5 ) ] += 0
        mews[temp >= 38.5] += 2
        
        return mews

    def qSOFA(self, df):
        vm = self.vm
        resp = vm('resp')
        sbp = vm('sbp')

        qsofa = np.zeros(shape=df.shape[0])
        qsofa[df[resp].values >= 22] += 1
        qsofa[df[sbp].values <= 100] += 1
        return qsofa

    def SOFA(self, df):
        vm = self.vm
        plt = vm('plt')
        bili = vm('bili')
        map = vm('map')
        crea = vm('crea')
 
        sofa = np.zeros(shape=df.shape[0])
        
        # Coagulation
        platelets = df[plt].values
        sofa[platelets >= 150] += 0
        sofa[(100 <= platelets) & (platelets < 150)] += 1
        sofa[(50 <= platelets) & (platelets < 100)] += 2
        sofa[(20 <= platelets) & (platelets < 50)] += 3
        sofa[platelets < 20] += 4

        # Liver
        bilirubin = df[bili].values
        sofa[bilirubin < 1.2] += 0
        sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
        sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
        sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
        sofa[bilirubin > 11.9] += 4

        # Cardiovascular
        map = df[map].values
        sofa[map >= 70] += 0
        sofa[map < 70] += 1

        # crea
        creatinine = df[crea].values
        sofa[creatinine < 1.2] += 0
        sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
        sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
        sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
        sofa[creatinine > 4.9] += 4

        return sofa

    def SOFA_deterioration(self, s):
        def check_24hr_deterioration(s):
            """ Check the max deterioration over the last 24 hours, if >= 2 then mark as a 1"""
            prev_23_hrs = pd.concat([s.shift(i) for i in range(1, 24)], axis=1).values
            tfr_hr_min = np.nanmin(prev_23_hrs, axis=1)
            return pd.Series(index=s.index, data=(s.values - tfr_hr_min))
        sofa_det = s.groupby(
            self.vm('id')).apply(check_24hr_deterioration)
        sofa_det[sofa_det < 0] = 0
        sofa_det = sofa_det
        return sofa_det

    def septic_shock(self, df):
        vm = self.vm
        map = vm('map')
        lact = vm('lact')
        shock = np.zeros(shape=df.shape[0])
        shock[df[map].values < 65] += 1
        shock[df[lact].values > 2] += 1
        return shock

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
        df['ShockIndex_derived'] = df[hr].values / df[sbp].values
        df['bun/cr_derived'] = df[bun].values / df[crea].values
        df['po2/fio2_dervied'] = df[po2].values / df[fio2].values #shouldnt it be PaO2/Fi ratio?

        # SOFA
        df['SOFA_derived'] = self.SOFA(df[[plt, map, crea, bili]])
        df['SOFA_deterioration_derived'] = self.SOFA_deterioration(df['SOFA_derived'])
        df['qSOFA_derived'] = self.qSOFA(df)
        df['SepticShock_derived'] = self.septic_shock(df)

        # Other scores
        sirs_df = self.sirs_criteria(df)
        df['MEWS_derived'] = self.mews_score(df)
        df['SIRS_derived'] = sirs_df['SIRS']
        return df

class MeasurementCounter(DaskIDTransformer):
    """ Adds a count of the number of measurements up to the given timepoint. """
    def __init__(self, vm=None, suffix='_raw', **kwargs):
        super().__init__(vm=vm, **kwargs)
        # we only count raw time series measurements 
        self.col_suffix = suffix #apply look-back stats to time series columns

    def fit(self, df, labels=None):
        return self

    def transform_id(self, df):
        # Make a counts frame
        counts = deepcopy(df)
        drop_cols = [x for x in df.columns if not self.col_suffix in x] 
        counts.drop(drop_cols, axis=1, inplace=True)

        # Turn any floats into counts
        for col in counts.columns:
            counts[col][~counts[col].isna()] = 1
        counts = counts.replace(np.nan, 0)

        # Get the counts for each person
        counts = counts.cumsum() 

        # Rename
        cols = strip_cols(counts.columns, [self.col_suffix])
        counts.columns = [x + '_count' for x in cols]

        return pd.concat([df, counts], axis=1)

class WaveletFeatures(ParallelBaseIDTransformer):
    """ Computes wavelet scattering up to given time per time series channel """
    def __init__(self, n_jobs=4, T=32, J=2, Q=1, output_size=32, suffix='_locf', **kwargs):
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
        self.col_suffix = suffix 
        self.scatter = Scattering1D(J, T, Q) 
        self.output_size = output_size

    def fit(self, df, labels=None):
        return self

    def transform_id(self, df):
        """ process invididual patient """  
        inputs = deepcopy(df)
        drop_cols = [x for x in df.columns if self.col_suffix not in x]
        inputs.drop(drop_cols, axis=1, inplace=True)

        # pad time series with T-1 0s (for getting online features of fixed window size)
        inputs = self._pad_df(inputs, n_pad=self.T-1)
        #inputs.reset_index(drop=True, inplace=True)
        
        wavelets = self._compute_wavelets(inputs)
        
        assert df.shape[0] == wavelets.shape[0]
        wavelets.index = df.index

        return pd.concat([df, wavelets], axis=1)

    def _pad_df(self, df, n_pad, value=0):
        """ util function to pad <n_pad> zeros at the start of the df 
            and impute 0s at the remaining nans (of lcof imputation). 
        """
        x = np.concatenate([np.zeros([n_pad,df.shape[1]]), df.values]) 
        return pd.DataFrame(x, columns=df.columns).fillna(0)

    def _compute_wavelets(self, df):
        """ Loop over all columns, compute column-wise wavelet features
             and create wavelet output columns"""
        col_dfs = []
        df.columns = strip_cols(df.columns, [self.col_suffix])
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
  

class SignatureFeatures(ParallelBaseIDTransformer):
    """ Computes signature features for a given look back window """
    def __init__(self, n_jobs=4, look_back=7, order=3, 
            suffices=['_locf', '_derived'], **kwargs):
        """
        Inputs:
            - n_jobs: for parallelization on the patient level
            - lock_back: how many hours to look into the past 
                (default 7 following Morril et al.)
            - order: signature truncation level of univariate signature features
                --> for multivariate signatures (of groups of channels), we use order-1 
        """
        n_jobs=1
        super().__init__(n_jobs=n_jobs, **kwargs)
        # we process the raw time series measurements 
        self.order = order
        self.suffices = suffices
        self.look_back = look_back
        #self.columns = extended_ts_columns
        column_dict = { #concept variables
            'vitals':  ['hr', 'o2sat', 'temp', 'map', 'resp', 'fio2'],
            'labs': ['ph', 'lact', 'wbc', 'plt', 'ptt'],
            'scores': ['ShockIndex', 'bun/cr', 'SOFA', 'SepticShock', 'MEWS'] 
        }
        self.column_dict = self._add_suffices(column_dict)
        #self.output_size = iis.siglength(2, self.order) # channel + time = 2
        #self.output_size_all = iis.siglength(len(self.columns) + 1, self.order)    

    def fit(self, df, labels=None):
        return self

    def transform_id(self, df):
        """ process invididual patient """  
        inputs = deepcopy(df)
        drop_cols = [col for col in df.columns if not any(
            [suffix in col for suffix in self.suffices ] )]
        inputs.drop(drop_cols, axis=1, inplace=True)

        # pad time series with look_back size many 0s (for getting online features of fixed window size)
        inputs = self._pad_df(inputs, n_pad=self.look_back)
        
        #channel-wise signatures
        #signatures = self._compute_signatures(inputs)
        
        #multivariable signatures over variable groups
        mv_signatures = self._compute_mv_signatures(inputs) 
 
        #assert df.shape[0] == signatures.shape[0]
        #signatures.index = df.index
        
        assert df.shape[0] == mv_signatures.shape[0]
        mv_signatures.index = df.index

        return pd.concat([df, mv_signatures], axis=1) #mv_signatures

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
        x = np.concatenate([np.zeros([n_pad,df.shape[1]]), df.values]) 
        return pd.DataFrame(x, columns=df.columns).fillna(0)

    def _to_path(self, X):
        """ Convert single, evenly spaced time series to path by adding time axis
            X: np array (n time steps, d dimensions)
        """
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        n = X.shape[0]
        steps = np.arange(n).reshape(-1,1)
        path = np.concatenate((steps, X), axis=1)
        return path
    
    # univariate / single channel signatures
    def _compute_signatures(self, df):
        """ Loop over all columns, compute column-wise signature features
             and create signature output columns"""
        col_dfs = []
        for col in df.columns:
            col_df = self._process_column(df, col)
            col_dfs.append(col_df)
        return pd.concat(col_dfs, axis=1)
        
    def _process_column(self, df, col):
        """ Processing a single column. """
        out_cols = [col + f'_signature_{i}' for i in range(self.output_size)]
        col_df = pd.DataFrame(columns=out_cols)
        # we go over input column and write result to col_df, which is a 
        # df gathering all features from the current col
        df[col].rolling(window = self.look_back+1 ).apply(self._rolling_function, 
            kwargs={'df': col_df, 'order': self.order})
        return col_df

    def _rolling_function(self, window, df, order):
        """compute univariate signatures of a pd.rolling window """
        path = self._to_path(window.values)
        df.loc[window.index.min()] = iis.sig(path, order)
        return 1 # rolling funcs need to return index.. 

    # multivariate / multi channel signatures:
    def _compute_mv_signatures(self, df):
        """ Loop over columns groups compute multivariate signature features
             and create output columns"""
        col_dfs = []
        for group, columns in self.column_dict.items():
            col_df = self._process_group(df, group, columns)
            col_dfs.append(col_df)
        out = pd.concat(col_dfs, axis=1)
        return out[self.look_back:]

    def _process_group(self, df, group, columns):
        """ Processing a group of columns. """
        group_output_size = iis.siglength(len(columns)+1, self.order)
        out_cols = [group + f'_signature_{i}' for i in range(group_output_size)]
        #col_df = pd.DataFrame(columns=out_cols)
        # we go over input column and write result to col_df, which is a 
        # df gathering all features from the current col
        #df[columns].rolling(window = self.look_back+1, axis=1).apply(self._rolling_function, 
        #    kwargs={'df': col_df, 'order': self.order })
        func_kwargs = {'order': self.order } 
        col_df = self._mv_rolling(df = df[columns],
                                  cols = out_cols, 
                                  window = self.look_back+1,
                                  func = self._mv_rolling_function,
                                  func_kwargs = func_kwargs
        )
                                    
        return col_df

    def _mv_rolling(self, df, cols, window, func, func_kwargs={}):
        """
        Custom rolling operation for functions that 
        require access to multiple columns.
        Args:
        - df: dataframe to roll over
        - cols: list of output columns 
        - window: window length (similar as pd.rolling)
        - func: rolling function object (e.g. mean, max etc)
        - func_kwargs: arguments to function 
        """
        col_df = df.copy()
        col_df = col_df.reindex(columns = cols)
        
        for i in np.arange(window-1, len(df)):
            start = i - window + 1
            end = i + 1
            col_df.iloc[i,:] = func(df.iloc[start:end, :], **func_kwargs)
        return col_df
    
    def _mv_rolling_function(self, window, order):
        """compute univariate signatures of a custom rolling window """
        path = self._to_path(window.values)
        return iis.sig(path, order)


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

