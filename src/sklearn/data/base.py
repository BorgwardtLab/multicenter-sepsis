"""Base Classes for Dataset Transformers, TODO: add copyright notice here!""" 

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from joblib import Parallel, delayed

class BaseIDTransformer(TransformerMixin, BaseEstimator):
    """
    Base class when performing transformations over ids. One must implement a transform_id method.
    """
    def __init__(self):
        pass

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            df_transformed = df.groupby(['id'], as_index=False).apply(self.transform_id)
        elif isinstance(df, pd.Series):
            df_transformed = df.groupby(['id']).apply(self.transform_id)

        # Sometimes creates a None level
        if None in df_transformed.index.names:
            print('None in indices, dropping it')
            df_transformed.index = df_transformed.index.droplevel(None)

        return df_transformed



class ParallelBaseIDTransformer(TransformerMixin, BaseEstimator):
    """
    Parallelized Base class when performing transformations over ids. The child class requires to have a transform_id method.
    """
    def __init__(self, n_jobs=4):
        self.n_jobs=n_jobs

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self
        
    def _transform_func(self, df):
        """
        Actually performing the transform (via transform_id method of child class)
        """
        if isinstance(df, pd.DataFrame):
            df_transformed = df.groupby(['id'], as_index=False).apply(self.transform_id)
        elif isinstance(df, pd.Series):
            df_transformed = df.groupby(['id']).apply(self.transform_id)
        # Sometimes creates a None level
        if None in df_transformed.index.names:
            print('None in indices, dropping it')
            df_transformed.index = df_transformed.index.droplevel(None)
        return df_transformed
 
    def transform(self, df):
        """ Parallelized transform
        """
        #we assume that instance ids are on the first level of the df multi-indices (id, time) 
        ids = df.index.levels[0].tolist() #gather all instance ids 
        dfs = Parallel(n_jobs=self.n_jobs)(delayed(self._transform_func)(df.loc[[patient_id]]) for patient_id in ids) 
        return pd.concat(dfs)
