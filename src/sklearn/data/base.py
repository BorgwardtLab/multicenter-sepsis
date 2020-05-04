"""Base Classes for Dataset Transformers, TODO: add copyright notice here!""" 

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from joblib import Parallel, delayed
import dask
from dask.distributed import Client
import dask.bag as db

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
    def __init__(self, n_jobs=4, concat_output=False):
        self.n_jobs = n_jobs
        self.concat_output = concat_output

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self

    def transform(self, df_or_list):
        """ Parallelized transform
        """
        if isinstance(df_or_list, list):
            # Remove Nones
            df_or_list = list(filter(lambda x: x is not None, df_or_list))
            n = len(df_or_list)

            def get_instance(index):
                return df_or_list[index]

        elif isinstance(df_or_list, pd.DataFrame):
            #we assume that instance ids are on the first level of the df multi-indices (id, time) 
            ids = df_or_list.index.levels[0].tolist() #gather all instance ids 
            n = len(ids)

            def get_instance(index):
                return df_or_list.loc[[ids[index]]]

        else:
            raise ValueError('Unknown input: {}'.format(type(df_or_list)))

        # Use multiprocessing as we can then share memory via fork.
        output = Parallel(n_jobs=self.n_jobs, batch_size=100, max_nbytes=None, verbose=1)(
            delayed(self.transform_id)(get_instance(i)) for i in range(n))

        if self.concat_output:
            output = pd.concat(output)

        print('Done with', self.__class__.__name__)
        return output

class DaskIDTransformer(TransformerMixin, BaseEstimator):
    """
    Dask-based Parallelized Base class when performing transformations over ids. The child class requires to have a transform_id method.
    """
    def __init__(self, n_jobs=4, concat_output=False):
        self.n_jobs = n_jobs
        self.concat_output = concat_output
        self.client = Client(n_workers=n_jobs)

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self
   
    def transform(self, df_or_list):
        """ Parallelized transform
        """
        if isinstance(df_or_list, list):
            # Remove Nones
            df_or_list = list(filter(lambda x: x is not None, df_or_list))
            n = len(df_or_list)

            def get_instance(index):
                return df_or_list[index]

        elif isinstance(df_or_list, pd.DataFrame):
            #we assume that instance ids are on the first level of the df multi-indices (id, time) 
            ids = df_or_list.index.levels[0].tolist() #gather all instance ids 
            n = len(ids)

            def get_instance(index):
                return df_or_list.loc[[ids[index]]]

        else:
            raise ValueError('Unknown input: {}'.format(type(df_or_list)))
        
        #VERSION A:
        #def transform_wrapper(transform_func, get_instance):
        #    """
        #    Wrapper to let the transform_id function work on padient ids, instead of patient dataframes)
        #    """
        #    def wrapped(index):
        #        return transform_func( get_instance(index))
        #    return wrapped
        #wrapped_transform = transform_wrapper(self.transform_id, get_instance)
        ## Use dask bag to create batches containing patients (a patient-wise split would result in 130k tasks, while naive chunking would split a patient in separate chunks) 
        #bag = db.from_sequence(range(n), npartitions=5)
        #bag = bag.map(wrapped_transform) 
        #VERSION B:
        bag = db.from_sequence([get_instance(i) for i in range(n)], npartitions=5)
        bag = bag.map(self.transform_id)
        bag = self.client.persist(bag)
        output = bag.compute() 
 
        # Use multiprocessing as we can then share memory via fork.
        #output = Parallel(n_jobs=self.n_jobs, batch_size=100, max_nbytes=None, verbose=1)(
        #    delayed(self.transform_id)(get_instance(i)) for i in range(n))

        if self.concat_output:
            output = pd.concat(output)

        print('Done with', self.__class__.__name__)
        return output
