""" Baseline model """

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

def ranges(key):
    d = { 'sofa': [0, 24], 'sirs': [0, 4],
          'news': [0, 21], 'mews': [0, 15],
          'qsofa':[0, 3] }
    return d[key]

class BaselineModel(BaseEstimator):
    def __init__(self, column='sofa', theta=0.5):
        """
        This baseline model takes the raw values of clinically used scores
        and treats them as prediction scores (additionally a threshold for prediction
        can be tuned). 
        - column: which data column is used to predict
        - theta: which threshold is used to binarize prediction
        """ 
        self.column = column
        self.theta = theta
        self.classes_ = np.array([0., 1.])
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        scores = self._forward(X)
        preds = (scores >= self.theta).astype(float)
        #preds = (scores < self.theta).astype(float)
        #preds['1'] = (scores >= self.theta).astype(float)
        #preds = preds.rename(columns={self.column: '0'})
        preds = preds.values.flatten()
        return preds
 
    def predict_proba(self, X):
        scores = self._forward(X)
        scores = pd.concat([1-scores, scores], axis=1)
        return scores.values 

    def decision_function(self, X):
        return self._forward(X)

    def _forward(self, X):
        baseline = X[[self.column]]
        p = self._to_prob(baseline)
        # forward filling last observerd score 
        p = p.ffill().fillna(0)
        return p
   
    def _to_prob(self, x):
        """
        convert raw values in array or df 
        to probability scores between [0,1]
        """
        min_, max_ = ranges(self.column) 
        x_ = (x - min_) / max_
        return x_

