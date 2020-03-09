"""
Various data transformers 
"""
from copy import deepcopy
import os
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from utils import save_pickle, load_pickle 

class CreateDataframe(TransformerMixin, BaseEstimator):
    """
    Transform method takes either a filename or folder and creates a dataframe of the file or all psv files contained
    in the folder
    """
    def __init__(self, save=False, data_dir=None):
        self.save = save
        self.data_dir = data_dir

    def fit(self, df, y=None):
        return self

    @staticmethod
    def psv_to_dataframe(fname):
        """ Transforms a single psv file into a dataframe with an id column"""
        df = pd.read_csv(fname, sep='|')
        df['id'] = int(fname.split('.psv')[0].split('p')[-1])

        # Add the folder as a column
        folder = fname.split('/')[-2]
        df['hospital'] = folder

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

        # Make the dataframe
        df = pd.concat([self.psv_to_dataframe(fname) for fname in fnames])

        # Change hospital to a numeric col
        hospitals = df['hospital'].unique()
        remap = range(1, len(hospitals) + 1)
        df['hospital'] = df['hospital'].replace(hospitals, remap)

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
            save_pickle(labels, self.data_dir + '/labels/original.pickle')
            save_pickle(df_values, self.data_dir + '/from_raw/df.pickle')

        return df_values



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
