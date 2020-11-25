import numpy as np
import os
import pickle
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from .utils import save_pickle, load_pickle 

class ChallengeFeatureSubsetter(TransformerMixin, BaseEstimator):
    """
    This class acts as a transform before running a sklearn clf in order to
    restrict the clf to features that can be derived from the physionet challenge
    data.
    #feature_flag: changing the feature names will affect this class! 
    """
    def __init__(self, prefix='datasets/physionet2019/data/sklearn/processed', 
            split='train', feature_suffices=['ind','pt', 'dir']): 
        imputed_path = os.path.join(prefix, f'X_features_{split}.pkl')
        not_imputed_path = os.path.join(prefix, f'X_features_no_imp_{split}.pkl')
        drop_features_path = os.path.join(prefix, f'drop_features_{split}.pkl')
        if not os.path.exists(drop_features_path): 
            #all features including imputations
            self.Xi = load_pickle(imputed_path)
            self.Xn = load_pickle(not_imputed_path) 
            
            self.drop_vars = self._get_drop_vars(self.Xn)
            self.drop_features = self._get_features_from_drop_vars(feature_suffices)
            with open(drop_features_path, 'wb') as f:
                pickle.dump(self.drop_features, f)
        else:
            self.drop_features = load_pickle(drop_features_path)
 
    def _get_drop_vars(self, df):
        nan_sum = df.isnull().sum()
        max_nan = nan_sum.max()
        #get vars with only/max nans to drop
        return list(df.columns[nan_sum == max_nan])
    
    def _get_features_from_drop_vars(self, feature_suffices):
        """
        Function to retrieve all features that are not derived from challenge variable set
        by checking for columns in the imputed feature matrix.
        We have to handle edge cases, e.g. _ind_ and _pt_ appear in variable names which makes
        our separation of words with '_' harder.
        """
        drop_features = []
        cols = list(self.Xi.columns)
        
        for col in cols:
            split = col.split('_')
            long_feature_name = False
            for d in self.drop_vars:
                if len(split) > 1:
                    if any([l in split[1] for l in feature_suffices]):
                        # e.g. vent_ind is the variable name with a suffix _ind
                        feature = '_'.join(split[:2])
                        long_feature_name = True
                if not long_feature_name:
                    feature = split[0]
                if (d == feature) and col not in drop_features:
                    drop_features.append(col)  
        return drop_features
 
    def fit(self, df, labels=None):
        return self
 
    def transform(self, df):
        return df.drop(columns=self.drop_features, errors='ignore')

    def __call__(self, df):
        return self.transform(df)


class FeatureSubsetter():
    """ Choosing feature subset (which is available from
        physionet challenge dataset.
    """ 
    def __init__(self, feature_set):
        if feature_set == 'challenge':
            self.transform = ChallengeFeatureSubsetter()
        elif feature_set == 'all':
            self.transform = lambda x: x
        else:
            raise ValueError(f'feature set {feature_set} not among valid feature sets: [all, challenge]')

    def __call__(self, instance):
        return self.transform(instance) 

