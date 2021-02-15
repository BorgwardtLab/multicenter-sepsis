"""
Classes for dynamically mapping variables. 
"""
import json
import pandas as pd
import numpy as np

class VariableMapping:
    """
    This class allows for a variable mapping based on a variable json file.
    By fixing the input column, to change the variable names one can simply 
    add a category to the variable file and a corresponding new column here 
    as output_col.
    
    """
    def __init__(self, variable_file='config/variables.json', input_col='concept', output_col='name'):
        """
        Arguments:
            - variable_file: path to json file containing all variable mappings
            - input_col: column in resulting df which is used to query variable name
            - output_col: column in resulting df which is used to return variable name  
        """
        self.input_col = input_col
        self.output_col = output_col
        self.var_df = self._load_and_process(variable_file) 

    def _load_json(self, fpath):
        with open(fpath, 'r') as f:
            return json.load(f) 

    def _formatting(self, x):
        """
        utility function to reformat the variables json which has
        entries encoded as lists and missing values as empty dicts.
        When applying to pandas df, the entries are subsetted from
        lists and proper nans are put in place.
        """
        res = x.copy()
        for i, x_ in enumerate(x):
            if type(x_) == list:
                res[i] = x_[0]
            elif type(x_) == dict:
                res[i] = np.nan
        return res

    def _load_and_process(self, fpath):
        """
        This funtions loads the variable file and formats it 
        into a comprehensible pd dataframe.
        """
        d = self._load_json(fpath)
        df = pd.DataFrame(d)
        df = df.apply(self._formatting) 
        return df 
        
    def __call__(self, x):
        """
        map input variable string x to output variable
        """
        df = self.var_df
        return df[df[self.input_col] == x][self.output_col]
    
    def get_category(self, category, category_col='category'):
        """
        returns a list of all variables which are part of the provided 
        category (e.g. 'vitals') as defined by the column category_col.
        """
        df = self.var_df
        return df[df[category_col] == category][self.output_col].tolist() 
 
