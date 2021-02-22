"""Base Classes for Dataset Transformers, TODO: add copyright notice here!""" 

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from joblib import Parallel, delayed

class BaseIDTransformer(TransformerMixin, BaseEstimator):
    """
    Base class when performing transformations over ids. One must implement a transform_id method.
    """
    def __init__(self, vm):
        self.vm = vm

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            df_transformed = df.groupby([self.vm('id')], as_index=False).apply(self.transform_id)
        elif isinstance(df, pd.Series):
            df_transformed = df.groupby([self.vm('id')]).apply(self.transform_id)

        # Sometimes creates a None level
        if None in df_transformed.index.names:
            print('None in indices, dropping it')
            df_transformed.index = df_transformed.index.droplevel(None)

        return df_transformed



class ParallelBaseIDTransformer(TransformerMixin, BaseEstimator):
    """
    Parallelized Base class when performing transformations over ids. 
    The child class requires to have a transform_id method.
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

class ChunkedTransformer(TransformerMixin, BaseEstimator):
    """
    Parallelized Base class when performing transformations over ids. 
    The child class requires to have a transform_id method.
    """
    def __init__(self, n_jobs=4, chunk_size=10, **kwargs):
        """
        Args:
        - n_jobs: number of jobs for parallelism
        - chunks_size: number of patients in a chunk that is processed
            sequentially by one parallelized process. 
        """
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size

    def __init_subclass__(cls, *args, **kwargs):
        if not hasattr(cls, 'transform_id'):
            raise TypeError('Class must take a transform_id method')
        return super().__init_subclass__(*args, **kwargs)

    def fit(self, df, y=None):
        return self

    def transform(self, df_or_list):
        """ Parallelized transform
        """
        chunk_size = self.chunk_size

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
        chunk_size = min(chunk_size, n) #ensuring that we have n_chunks >=1
        n_chunks = int(np.ceil(n / chunk_size)) 
        index_chunks = np.array_split(np.arange(n), n_chunks) 

        # Use multiprocessing as we can then share memory via fork.
        output = Parallel(n_jobs=self.n_jobs, batch_size=100, max_nbytes=None, verbose=1)(
            delayed(self.transform_chunk)(get_instance, index_chunks[i]) for i in range(n_chunks))

        output = pd.concat(output)

        print('Done with', self.__class__.__name__)
        return output
    
    def transform_chunk(self, id_fn, index_chunk):
        """
        Apply transform_id sequentially to chunk of ids.
        - id_fn: get_instance function
        - index_chunk: chunk of indices to apply transform_id
        """
        out = [ self.transform_id( 
            id_fn(i) ) for i in index_chunk
        ]
        return pd.concat(out)


class DaskIDTransformer(TransformerMixin, BaseEstimator):
    """
    Dask-based Parallelized Base class when performing transformations over ids. The child class requires to have a transform_id method.
    """
    def __init__(self, vm, **kwargs):
        print('Got unused args:', kwargs)
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
        result = dask_df.groupby( self.vm('id'),
             group_keys=False).apply(self.transform_id)
        print('Done with', self.__class__.__name__)
        return result
