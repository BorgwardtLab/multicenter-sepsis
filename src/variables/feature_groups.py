import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import pathlib
from .mapping import VariableMapping
from src.sklearn.loading import SplitInfo, ParquetLoader

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


class ColumnFilter:
    """ Class that handles filtering of column names including:
            - physionet subset of features
            - small vs large feature set
    """
    def __init__(self, 
                 path=f'datasets/mimic_demo/data/parquet/features'
                ):
        """
        Arguments:
        - path: path to example feature parquet dump that indicates data format
        
        """
        pl = ParquetLoader(path, form='pyarrow')
        table = pl.load()
        self.columns = table.column_names
        self.groups = self.columns_to_groups(self.columns)
        self.physionet_prefixes = self.get_physionet_prefixes()
    
    def columns_to_groups(self, cols):
        """ maps columns to feature groups
            for now skipping, i.e. ignoring baseline columns    
            and label columns (sep3 and utility)
         """
        columns = []
        skipped = VM_DEFAULT.all_cat('baseline')
        skipped.extend(
            [ VM_DEFAULT(x) for x in ['label','utility']]
        )
        for col in cols:
            # feature categories are indicated as _featuregroup
            s = col.split('_') 
            if len(s) == 1: # variable without category 
                if col not in skipped:
                    columns.append(col)
            else:
                prefix = '_' + s[1] # feature cat is second entry
            if prefix not in columns:
                columns.append(prefix)
        return sorted(columns)
        
    def groups_to_columns(self, groups):
        """ maps feature groups to columns whereas the image of the mapping
            is assumed to be self.columns.
            We assume that for a given group all members can be found
            by checking for the group suffix via endswith().
        """
        def contained(col, columns):
            """ first check if col is contained in columns (as a static)
                then check if any column of columns is contained in col 
                e.g. whethr a suffix '_indicator' is in col
            """
            suffix_columns = [c for c in columns if '_' in c]
            if col in columns: 
                #statics are directly available without suffices
                return True
            elif any([c in col for c in suffix_columns]):
                return True
            else:
                return False
 
        used_cols = [ col for col in self.columns 
            if contained(col, groups) 
        ]
        return used_cols

    def feature_set(self, name='large', groups=False):
        """
        Choose columns for large and small feature set
        - name: which feature set [large, small]
        - groups: return feature groups instead of all columns
        """
        if name == 'large':
            used_groups = self.groups
        elif name == 'small':
            used_groups  = [ 
             '_count',
             '_derived',
             '_id',
             '_indicator',
             '_raw',
             '_time',
             'height',
             'weight']
             #'_var',
             #'_wavelet',
             #'_signature', 
             #'_locf',
             #'_max',
             #'_mean',
             #'_median',
             #'_min',
        else:
            raise ValueError('No valid feature set name provied [large, small]') 
        if groups:
            return used_groups 
        else:
            return self.groups_to_columns(used_groups)
    
    def get_physionet_prefixes(self):
        """
        we highlight non-statics with underscore: <name>_ 
        for more robust greping of correct prefixes (as there
        could be duplicates, e.g. greping for kalium 'k' returns 
        false positive columns without k_
        """ 
        df = VM_DEFAULT.var_df  
        statics = VM_DEFAULT.all_cat('static') 
        prefixes = df[(~df['challenge'].isnull()) & (~df['name'].isnull())]
        prefixes = prefixes['name'].tolist() 
        return [ p + '_' if p not in statics else p
            for p in prefixes
        ] 
 
    def physionet_set(self, columns=None, feature_set='large'):
        """
        map set of columns to the corresponding physionet subset
        --> acts on explicit columns, not feature groups.
        """
        if not columns:
            columns = self.columns
        # groups we need to add:
        groups = [ 
             '_derived',
             '_id',
             '_time',
        ]
        if feature_set == 'large':
            # as these features are grouped over columns and don't
            # have a variable prefix, they need to be explictly added.
            groups.append('_signature')
        group_cols = self.groups_to_columns(groups)
        prefix_cols = [ col for col in columns if any(
            [col.startswith(p) for p in self.physionet_prefixes])
        ]
        total_cols =  sorted(group_cols + prefix_cols)
        # sanity check as there were duplicates at some stage:
        u,c = np.unique(total_cols, return_counts=True)
        assert len(u[c>1]) == 0
        return total_cols

class ColumnFilterLight(ColumnFilter):
    """Column filter which uses an existing list of columns."""

    def __init__(self, dataset_columns):
        self.columns = dataset_columns
        self.groups = self.columns_to_groups(self.columns)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        help='name of dataset to use', 
                        default='mimic_demo')
    parser.add_argument('--out-file', 
                        help='name of output file', 
                        default='config/features.json')
    args = parser.parse_args()
    dataset_name = args.dataset 
    path = f'datasets/{dataset_name}/data/parquet/features'

    feature_sets = ['large', 'small']
    variable_sets = ['full', 'physionet']

    cf = ColumnFilter(path=path)
    result = {x: {} for x in variable_sets}
    for feature_set in feature_sets:
        for variable_set in variable_sets:
            output = {}
            groups = cf.feature_set(name=feature_set, groups=True)
            columns = cf.feature_set(name=feature_set, groups=False) 
            if variable_set == 'physionet':
                # reduce columns and then adjust groups:
                columns = cf.physionet_set(columns, feature_set=feature_set) 
                groups = cf.columns_to_groups(columns)     
            output['groups'] = groups; output['columns'] = columns 
            result[variable_set][feature_set] = output 
    with open(args.out_file, 'w') as f:
        json.dump(result, f, indent=4) 
