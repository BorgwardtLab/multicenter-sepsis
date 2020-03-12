"""
Various data transformers, TODO: add copyright notice here! 
"""
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from utils import save_pickle, load_pickle 
from base import BaseIDTransformer
from extracted import columns_with_nans, ts_columns

from IPython import embed

class CreateDataframe(TransformerMixin, BaseEstimator):
    """
    Transform method takes either a filename or folder and creates a dataframe of the file or all psv files contained
    in the folder
    available splits:
        - train, validation, test
    """
    def __init__(self, save=False, data_dir=None, split='train', split_file='../../../datasets/physionet2019/data/split_info.pkl'):
        self.save = save
        self.data_dir = data_dir
        self.split = split
        split_info = load_pickle(split_file)
        self.split_ids = split_info['split_0'][split] 

    def _file_to_id(self, string):
        """ Convert string of psv file to integer id
        """
        return int(string.rstrip('.psv').lstrip('p'))
    
    def _id_to_file(self, pat_id, pad_until=6):
        """ Convert patient id (integer) to file string with zeros padded """
        return 'p' + str(pat_id).zfill(pad_until) + '.psv'

    def _extract_split(self, fnames):
        """ Return list of fnames (to load) which are contained in predefined split ids
        """
        used_files = []
        for fname in fnames:
            tail, head = os.path.split(fname)
            curr_id = self._file_to_id(head)
            if curr_id in self.split_ids:
                used_files.append(fname) 
        return used_files

    def fit(self, df, y=None):
        return self
    
    @staticmethod
    def psv_to_dataframe(fname):
        """ Transforms a single psv file into a dataframe with an id column"""
        df = pd.read_csv(fname, sep='|')
        df['id'] = int(fname.split('.psv')[0].split('p')[-1])

        return df

    def transform(self, location):
        """
        Given either a location of psv files or single psv file, transforms into dataframes indexed with time and id

        :param location: either a folder containing psv files or a single psv file
        :return: df of all psv files indexed by [id, time]
        """

        # If location is a directory, make with all files, else make with a single file
        if isinstance(location, list):
            fnames = [l + '/' + x for l in location for x in os.listdir(l)]
        elif not location.endswith('.psv'):
            fnames = [location + '/' + x for x in os.listdir(location)]
        else:
            fnames = [location]
        
        fnames_old = fnames
        fnames = self._extract_split(fnames)

        # Make the dataframe
        df = pd.concat([self.psv_to_dataframe(fname) for fname in fnames])

        # Idx according to id and time
        df.index.name = 'time'
        df_idxed = df.reset_index().set_index(['id', 'time']).sort_index(ascending=True)

        # Get values and labels
        if 'SepsisLabel' in df_idxed.columns:
            df_values, labels = df_idxed.drop('SepsisLabel', axis=1), df_idxed['SepsisLabel']
        else:
            df_values = df_idxed

        # Save if specified
        if self.save is not False:
            os.makedirs(data_dir, exist_ok=True)
            save_pickle(labels, os.path.join(data_dir, f'y_{split}.pkl'))
            save_pickle(df_values, os.path.join(data_dir, f'raw_data_{split}.pkl')) 

        return df_values


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


class RemoveExtremeValues(BaseEstimator, TransformerMixin):
    def __init__(self, quantile):
        self.quantile = quantile
        self.cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
                     'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                     'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                     'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                     'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                     'Fibrinogen', 'Platelets']

    def fit(self, df, labels=None):
        self.percentiles_1 = np.nanpercentile(df[self.cols], self.quantile, axis=0)
        self.percentiles_2 = np.nanpercentile(df[self.cols], 1 - self.quantile, axis=0)
        return self

    def transform(self, df):
        # Drop derived cols
        df.drop(['HepaticSOFA', 'SIRS', 'SIRS_path', 'MEWS', 'qSOFA', 'SOFA', 'SepticShock'], axis=1, inplace=True)

        # Remove extreme vals
        for i, col in enumerate(self.cols):
            p1, p2 = self.percentiles_1[i], self.percentiles_2[i]
            mask = (min(p1, p2) > df[col]) | (max(p1, p2) < df[col])
            df[mask][col] = np.nan

        # Redo ffill and derived feature calcs
        df.fillna(method='ffill', inplace=True)
        df = DerivedFeatures().transform(df)

        return df


class CarryForwardImputation(BaseIDTransformer):
    """
    First fills in missing values by carrying forward, then fills backwards. The backwards method takes care of the
    NaN values at the start that cannot be filled by a forward fill.
    """
    def transform_id(self, df):
        return df.fillna(method='ffill')

class IndicatorImputation(BaseIDTransformer):
    """
    Adds indicator dimension for every channel to indicate if there was a nan
    """
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
        shock[df['Lactate'].values < 2] += 1
        return shock

    def transform(self, df):
        # Compute things
        df['ShockIndex'] = df['HR'].values / df['SBP'].values
        df['BUN/CR'] = df['BUN'].values / df['Creatinine'].values
        df['SaO2/FiO2'] = df['SaO2'].values / df['FiO2'].values

        # SOFA
        df['SOFA'] = self.SOFA(df[['Platelets', 'MAP', 'Creatinine', 'Bilirubin_total']])
        df['SOFA_deterioration'] = self.SOFA_deterioration(df['SOFA'])
        # df['SOFA_max_24hrs'] = self.SOFA_max_24(df['SOFA'])
        # df['HepaticSOFA'] = self.hepatic_sofa(df)
        # df['qSOFA'] = self.qSOFA(df)
        # df['SOFA_24hrmaxdet'] = self.SOFA_deterioration(df['SOFA_max_24hrs'])
        # df['SOFA_deterioration_new'] = self.SOFA_deterioration_new(df['SOFA_max_24hrs'])
        # df['SepticShock'] = self.septic_shock(df)

        # Other scores
        # sirs_df = self.sirs_criteria(df)
        # df['MEWS'] = self.mews_score(df)
        # df['SIRS'] = sirs_df['SIRS']
        # df['SIRS_path'] = sirs_df['SIRS_path']
        return df


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