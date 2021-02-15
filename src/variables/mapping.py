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
    def __init__(self, variable_file='config/variables.json', 
                 input_col='concept', output_col='name',
                 keys={'id': 'stay_id', 'time':'stay_time', 'label': 'sep3'}):
        """
        Arguments:
            - variable_file: path to json file containing all variable mappings
            - input_col: column in resulting df which is used to query variable name
            - output_col: column in resulting df which is used to return variable name 
            - keys: dict of default variables to use in the core set 
        """
        self.input_col = input_col
        self.output_col = output_col
        self.var_df = self._load_and_process(variable_file) 
        self.keys = keys
        self.suffix = None 

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
        map input variable string x to output variable.
        If suffix is avaible, the input is mapped to <output_suffix>.
        """
        if x in self.keys.keys():
            output_var = self.keys[x]
        else:
            df = self.var_df
            output_var = df[df[self.input_col] == x][self.output_col].values[0]
            if self.suffix:
                cat = self.check_cat(output_var, variable_col=self.output_col)
                if cat not in ['static', 'baseline']:
                    output_var = '_'.join([output_var, self.suffix])
        return output_var 
    
    def all_cat(self, category, category_col='category'):
        """
        returns a list of all variables which are part of the provided 
        category (e.g. 'vitals') as defined by the column category_col.
        """
        df = self.var_df
        return df[df[category_col] == category][self.output_col].tolist() 
    
    def check_cat(self, variable, variable_col='name', category_col='category'):
        """
        returns the category (in terms of category_col) of the provided 
        variable and variable_col.
        """
        df = self.var_df
        return df[df[variable_col] == variable][category_col].values[0] #to return string 

    @property
    def core_set(self, suffices=['raw', 'locf']):
        """
        Array of core variables to use.
        """
        df = self.var_df
        df = df[~df[self.output_col].isnull()] # we are only interested in those cols
        result = list(self.keys.values()) 
        variables = df[self.output_col] 
        for i, suffix in enumerate(suffices):
            for v in variables:
                cat = self.check_cat(v) 
                if cat in ['static', 'baseline']: #hack as cat is technically a series 
                    if i==0:
                        result.append(v)
                else: #add suffix
                    v = '_'.join([v, suffix])
                    result.append(v)
        return result
 
