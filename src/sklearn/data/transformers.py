"""
Various data loading, filtering and feature transformers, TODO: add copyright notice here! 
"""
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, BaseEstimator

from utils import save_pickle, load_pickle 
from base import BaseIDTransformer, ParallelBaseIDTransformer
from extracted import columns_with_nans, ts_columns, extended_ts_columns ,columns_not_to_normalize

import sys
sys.path.append(os.getcwd())
from src import datasets

class DataframeFromDataloader(TransformerMixin, BaseEstimator):
    """
    Transform method that takes dataset class, loads corresponding iterable dataloader and 
    returns (and saves if requested) the dataset in one large sklearn-ready pd dataframe 
    format with patient and time multi-indices.
    """
    def __init__(self, save=False, dataset_cls=None, data_dir=None, split='train', drop_label=True, n_jobs=10, concat_output=False):
        self.save = save
        dataset_class = getattr(datasets, dataset_cls)
        self.split = split
        self.dataloader = dataset_class(split=split, as_dict=False)
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
        df = df.rename(columns={'ICULOS': 'time'}) #rename for easier understanding
        df.reset_index(drop=True, inplace=True)
        df.set_index(['id', 'time'], inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df

    def transform(self, df):
        """
        Takes dataset_cls (in self), loads iteratable instances and concatenates them to a multi-indexed 
        pandas dataframe which is returned
        """
        # Make the dataframe
        output = Parallel(n_jobs=self.n_jobs, verbose=1, batch_size=1000)(
            delayed(self._load_index_and_prepare)(i) for i in range(len(self.dataloader)))

        if self.concat_output:
            output = pd.concat(output)
        return output

        # Get values and labels
        if 'SepsisLabel' in df_idxed.columns:
            if self.drop_label:
                df_values, labels = df_idxed.drop('SepsisLabel', axis=1), df_idxed['SepsisLabel']
            else:
                df_values, labels = df_idxed, df_idxed['SepsisLabel']
        else:
            df_values = df_idxed

        # Save if specified
        if self.save is not False:
            os.makedirs(self.data_dir, exist_ok=True)
            save_pickle(labels, os.path.join(self.data_dir, f'raw_y_{self.split}.pkl'))
            save_pickle(df_values, os.path.join(self.data_dir, f'raw_data_{self.split}.pkl')) 

        print('Done with DataframeFromDataloader')
        return df_values

class DropLabels(TransformerMixin, BaseEstimator):
    """
    Remove label information, which was required for filtering steps.
    """
    def __init__(self, label='SepsisLabel', save=True, data_dir=None, split=None):
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

class PatientFiltration(ParallelBaseIDTransformer):
    """
    Removes patients which do not match inclusion criteria:
    --> sepsis-cases with:
        - onset within the first t_start hours of ICU stay (or before)
        - onset after t_end hours of ICU stay 
        defaults: t_start = 3 (which corresponds to 4 hours due to rounding of the chart times), t_end = 168 (1 week)
    """
    def __init__(self, save=False, data_dir=None, split='train', onset_bounds=(3,168), label='SepsisLabel', **kwargs):
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
        onset = np.argmax(label)
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
    """
    def __init__(self, cut_off=24, label='SepsisLabel', **kwargs):
        self.cut_off = cut_off
        self.label = label
        super().__init__(**kwargs)

    def transform_id(self, df):
        """ Patient-level transform
        """
        if df[self.label].sum() > 0:
            #Case:
            onset = df[self.label].argmax()
            return df[:onset+self.cut_off+1]
        else:
            #Control:
            return df


class InvalidTimesFiltration(TransformerMixin, BaseEstimator):
    """
    This transform removes invalid time steps right before training / prediction phase (final step of preprocessing).
        - time steps before ICU admission (i.e. negative timestamps) 
        - time steps with less than <thres> many observations. 
    """
    def __init__(self, save=True, data_dir=None, split=None, thres=1, label='SepsisLabel'):
        self.save = save
        self.data_dir = data_dir
        self.split = split
        self.thres = thres
        self.label = label

    def fit(self, df, labels=None):
        return self

    def _remove_pre_icu(self, df):
        return df[df.index.get_level_values('time') >=0]
    
    def _remove_too_few_observations(self, df, thres):
        ind_to_keep = (~df[ts_columns].isnull()).sum(axis=1) >= thres
        return df[ind_to_keep]
    
    def _transform_id(self, df):
        """ Patient level transformation
        """
        df = self._remove_pre_icu(df)
        return self._remove_too_few_observations(df, self.thres) 
     
    def transform(self, df):
        """ As this time filtering step also affects labels, for simplicity we load and adjust them too. 
        """
        labels = load_pickle(os.path.join(self.data_dir, f'y_{self.split}.pkl'))    
        assert len(labels) == len(df)
        df['SepsisLabel'] = labels #we temporarily add the labels to consistently remove time steps.
        
        df = df.groupby('id', as_index=False).apply(self._transform_id)
        #groupby can create None indices, drop them:
        if None in df.index.names:
            print('None in indices, dropping it')
            df.index = df.index.droplevel(None)
  
        if self.save:
            save_pickle(df['SepsisLabel'], os.path.join(self.data_dir, f'y_{self.split}.pkl'))
        df = df.drop('SepsisLabel', axis=1) #after filtering time steps drop labels again 
        print('Done with InvalidTimesFiltration')
        return df

class CarryForwardImputation(ParallelBaseIDTransformer):
    """
    First fills in missing values by carrying forward, then fills backwards. The backwards method takes care of the
    NaN values at the start that cannot be filled by a forward fill.
    """
    def transform_id(self, df):
        return df.fillna(method='ffill')

class IndicatorImputation(ParallelBaseIDTransformer):
    """
    Adds indicator dimension for every channel to indicate if there was a nan
    """
    def __init__(self, n_jobs=4):
        super().__init__(n_jobs=n_jobs)

    def transform_id(self, df):
        cols = ts_columns #we consider all time-series columns (for consistency with pytorch approach)
        valid_indicators = (~df[cols].isnull()).astype(int).add_suffix('_indicator') 
        df = pd.concat([df, valid_indicators], axis=1)
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


class LookbackFeatures(ParallelBaseIDTransformer):
    """ 
    Simple statistical features including moments over a tunable look-back window. 
    """
    def __init__(self, stats=None, **kwargs):
        """ takes dictionary of stats (keys) and corresponding look-back windows (values) 
            to compute each stat for each time series variable. 
            Below a default stats dictionary of look-back 5 hours is implemented.
        """
        self.cols = extended_ts_columns #apply look-back stats to time series columns
        if stats is None:
            keys = ['min','max', 'mean', 'median','var']
            windows = [4,8,16] #5
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
        #first get the function to compute the statistic:
        func = getattr(pd.DataFrame, stat)
        
        #determine available columns (which could be a subset of the predefined cols, depending on the setup):
        used_cols = [col for col in self.cols if col in df.columns]  
        stats = df[used_cols].rolling(window, min_periods=0).apply(func).fillna(0) #min_periods=0 ensures, that
        #the first window is  computed in an expanding fashion. Still certain stats (e.g. var) leave nan 
        #in the start, replace it with 0s here. 
        
        #rename the features by the statistic:
        stats = stats.add_suffix(f'_{stat}_{window}_hours') 
        
        return stats
 
    def transform_id(self, df):
        #compute all statistics in stats dictionary:
        
        #features = [df]
        features = [df]
        for stat, window in self.stats:
            feature = self._compute_stat(df, stat, window)
            features.append(feature)
        df_out = pd.concat(features, axis=1)
        return df_out

class DerivedFeatures(TransformerMixin, BaseEstimator):
    """
    Adds any derived features thought to be useful
        - Shock Index: HR/SBP
        - Bun/Creatinine ratio: Bun/Creatinine
        - Hepatic SOFA: Bilirubin SOFA score

    # Can add renal and neruologic sofa
    """
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    @staticmethod
    def hepatic_sofa(df):
        """ Updates a hepatic sofa score """
        hepatic = np.zeros(shape=df.shape[0])

        # Bili
        bilirubin = df['Bilirubin_total'].values
        hepatic[bilirubin < 1.2] += 0
        hepatic[(bilirubin >= 1.2) & (bilirubin < 1.9)] += 1
        hepatic[(df['Bilirubin_total'] >= 1.9) & (bilirubin < 5.9)] += 2
        hepatic[(bilirubin >= 5.9) & (bilirubin < 11.9)] += 3
        hepatic[(bilirubin >= 11.9)] += 4

        # MAP
        hepatic[df['MAP'].values < 70] += 1

        # Creatinine
        creatinine = df['Creatinine'].values
        hepatic[(creatinine >= 1.2) & (creatinine < 1.9)] += 1
        hepatic[(creatinine >= 1.9) & (creatinine < 3.4)] += 2
        hepatic[(creatinine >= 3.5) & (creatinine < 4.9)] += 3
        hepatic[(creatinine >= 4.9)] += 4

        # Platelets
        platelets = df['Platelets'].values
        hepatic[(platelets >= 100) & (platelets < 150)] += 1
        hepatic[(platelets >= 50) & (platelets < 100)] += 2
        hepatic[(platelets >= 20) & (platelets < 49)] += 3
        hepatic[(platelets < 20)] += 4

        return hepatic

    @staticmethod
    def sirs_criteria(df):
        # Create a dataframe that stores true false for each category
        df_sirs = pd.DataFrame(index=df.index, columns=['temp', 'hr', 'rr.paco2', 'wbc'])
        df_sirs['temp'] = ((df['Temp'] > 38) | (df['Temp'] < 36))
        df_sirs['hr'] = df['HR'] > 90
        df_sirs['rr.paco2'] = ((df['Resp'] > 20) | (df['PaCO2'] < 32))
        df_sirs['wbc'] = ((df['WBC'] < 4) | (df['WBC'] > 12))

        # Sum each row, if >= 2 then mar as SIRS
        sirs = pd.to_numeric((df_sirs.sum(axis=1) >= 2) * 1)

        # Leave the binary and the path sirs
        sirs_df = pd.concat([sirs, df_sirs.sum(axis=1)], axis=1)
        sirs_df.columns = ['SIRS', 'SIRS_path']

        return sirs_df

    @staticmethod
    def mews_score(df):
        mews = np.zeros(shape=df.shape[0])

        # SBP
        sbp = df['SBP'].values
        mews[sbp <= 70] += 3
        mews[(70 < sbp) & (sbp <= 80)] += 2
        mews[(80 < sbp) & (sbp <= 100)] += 1
        mews[sbp >= 200] += 2

        # HR
        hr = df['HR'].values
        mews[hr < 40] += 2
        mews[(40 < hr) & (hr <= 50)] += 1
        mews[(100 < hr) & (hr <= 110)] += 1
        mews[(110 < hr) & (hr < 130)] += 2
        mews[hr >= 130] += 3

        # Resp
        resp = df['Resp'].values
        mews[resp < 9] += 2
        mews[(15 < resp) & (resp <= 20)] += 1
        mews[(20 < resp) & (resp < 30)] += 2
        mews[resp >= 30] += 3

        return mews

    @staticmethod
    def qSOFA(df):
        qsofa = np.zeros(shape=df.shape[0])
        qsofa[df['Resp'].values >= 22] += 1
        qsofa[df['SBP'].values <= 100] += 1
        return qsofa

    @staticmethod
    def SOFA(df):
        sofa = np.zeros(shape=df.shape[0])

        # Coagulation
        platelets = df['Platelets'].values
        sofa[platelets >= 150] += 0
        sofa[(100 <= platelets) & (platelets < 150)] += 1
        sofa[(50 <= platelets) & (platelets < 100)] += 2
        sofa[(20 <= platelets) & (platelets < 50)] += 3
        sofa[platelets < 20] += 4

        # Liver
        bilirubin = df['Bilirubin_total'].values
        sofa[bilirubin < 1.2] += 0
        sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
        sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
        sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
        sofa[bilirubin > 11.9] += 4

        # Cardiovascular
        map = df['MAP'].values
        sofa[map >= 70] += 0
        sofa[map < 70] += 1

        # Creatinine
        creatinine = df['Creatinine'].values
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
        shock[df['MAP'].values < 65] += 1
        shock[df['Lactate'].values > 2] += 1
        return shock

    def transform(self, df):
        # Compute things
        df['ShockIndex'] = df['HR'].values / df['SBP'].values
        df['BUN/CR'] = df['BUN'].values / df['Creatinine'].values
        df['O2Sat/FiO2'] = df['O2Sat'].values / df['FiO2'].values #shouldnt it be PaO2/Fi ratio?

        # SOFA
        df['SOFA'] = self.SOFA(df[['Platelets', 'MAP', 'Creatinine', 'Bilirubin_total']])
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


class AddRecordingCount(BaseEstimator, TransformerMixin):
    """ Adds a count of the number of entries up to the given timepoint. """
    def __init__(self, last_only=False):
        self.columns = columns_with_nans

    def fit(self, df, labels=None):
        return self

    def transform_id(self, df):
        return df.cumsum()

    def transform(self, df):
        # Make a counts frame
        counts = deepcopy(df)
        counts.drop([x for x in df.columns if x not in self.columns], axis=1, inplace=True)

        # Turn any floats into counts
        for col in self.columns:
            counts[col][~counts[col].isna()] = 1
        counts = counts.replace(np.nan, 0)

        # Get the counts for each person
        counts = counts.groupby('id').apply(self.transform_id)

        # Rename
        counts.columns = [x + '_count' for x in counts.columns]

        return pd.concat([df, counts], axis=1)


def make_eventual_labels(labels):

    def make_one(s):
        return pd.Series(index=s.index, data=s.max())

    return labels.groupby('id').apply(make_one)


if __name__ == '__main__':
    CreateDataframe()  

#    df = load_pickle(DATA_DIR + '/interim/munged/df.pickle')
#    labels_binary = load_pickle(DATA_DIR + '/processed/labels/original.pickle')
#    evn = make_eventual_labels(labels_binary)
#    save_pickle(evn, DATA_DIR + '/processed/labels/eventual_sepsis.pickle')
