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
                 path=f'datasets/mimic_demo/data/parquet/features_middle'
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

    def groups_to_columns_and_drop_cats(self, groups, drop_cats, suffices):
        """concatenates groups_to_columns with a dropping of categories for given suffices"""
        used_cols = self.groups_to_columns(groups)
        if drop_cats is not None:
            used_cols = self.drop_categories(used_cols, drop_cats, suffices)
        return used_cols
 
    def drop_categories(self, cols, drop_cats, suffices=['_raw']):
        """ drop variables of given suffices belonging to categories in drop_cats"""
        def drop_per_suffix(cols, drop_cats, suffix):
            drop_vars = [] #which variables do we want to drop
            for cat in drop_cats:
                drop_vars += VM_DEFAULT.all_cat(cat)
            for col in cols:
                if col.endswith(suffix):
                    if any([col.split('_')[0] in drop_vars]):
                        # this is more robust than 'startswith()' as variables may contain
                        # other var names at the start
                        cols.remove(col)
            return cols
 
        all_cats = ['vitals', 'chemistry', 'organs', 'hemo'] # all cats we consider here for dropping
        assert all([x in all_cats for x in drop_cats])
        for suffix in suffices:
            cols = drop_per_suffix(cols, drop_cats, suffix)

        return cols

    def feature_set(self, name='large', groups=False, drop_cats=None, suffices=None):
        """
        Choose columns for large and small feature set
        - name: which feature set [large, small]
        - groups: return feature groups instead of all columns
        - drop_cats: optional list of which variable categories to drop (vitals, ..)
            as specified in the variable mapping VM_DEFAULT
        - suffices: optional list of suffices to apply the drop_cats step to (raw, counts) etc
        """
         # 
        if name == 'large':
            raise NotImplementedError("""For large feature_set, prepro pipeline must 
                be run w/ large set and path at init set""")
            #used_groups = self.groups.copy()
        elif name == 'middle':
            used_groups = self.groups.copy()
            drop_groups = ['_wavelet', '_signature']
            for group in drop_groups:
                if group in used_groups:
                    used_groups.remove(group)
        elif name == 'middle2':
            raise NotImplementedError("""Currently this feature set expects large 
                feature set as default, however this needs first to be run in preprocessing 
                and specified as init path""")
            used_groups = self.groups.copy()
            drop_groups = ['_wavelet']
            for group in drop_groups:
                if group in used_groups:
                    used_groups.remove(group)
        elif name == 'small':
            used_groups  = [ 
             '_count',
             '_derived',
             '_id',
             '_indicator',
             '_raw',
             '_time',
             '_static'
            ]
        elif name == 'raw':
            used_groups  = [ 
             '_id',
             '_raw',
             '_time',
             '_static'
            ]
        elif name == 'counts':
            used_groups = [
            '_id',
            '_time',
            '_count',
            '_static' #statics currently can't be left out at this stage 
            ]
        elif name == 'locf':
            used_groups = [
            '_id',
            '_time',
            '_locf',
            '_static' #statics currently can't be left out at this stage 
            ]
        elif name == 'raw_vitals':
            used_groups  = [ 
             '_id',
             '_raw',
             '_time',
             '_static'
            ]
            drop_cats = ['chemistry', 'organs', 'hemo']
            suffices = ['_raw']
        elif name == 'counts_vitals':
            used_groups = [
            '_id',
            '_time',
            '_count',
            '_static' #statics currently can't be left out at this stage 
            ]
            drop_cats = ['chemistry', 'organs', 'hemo']
            suffices = ['_count']
        elif name == 'raw_labs':
            used_groups  = [ 
             '_id',
             '_raw',
             '_time',
             '_static'
            ]
            drop_cats = ['vitals']
            suffices = ['_raw']
        elif name == 'counts_labs':
            used_groups = [
            '_id',
            '_time',
            '_count',
            '_static' #statics currently can't be left out at this stage 
            ]
            drop_cats = ['vitals']
            suffices = ['_count']
        else:
            raise ValueError('No valid feature set name provied') 
        if groups:
            return used_groups 
        else:
            return self.groups_to_columns_and_drop_cats(used_groups, drop_cats)
    
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
 
    def physionet_set(self, columns=None, feature_set=None):
        """
        map set of columns to the corresponding physionet subset
        --> acts on explicit columns, not feature groups.
        if columns are not provided explicitly, feature_set is used
        as the set of columns. Only specify columns XOR feature_set!
        Arguments:
        - columns: list of columns to subset for variables available in 
            physionet data
        - feature_set: one of ['small','middle','large'], specifies 3 groups
            of feature groups
        - groups: return column groups instead of columns
        """
        c_none = columns is None
        f_none = feature_set is None
        assert c_none ^ f_none
 
        if not columns:
            columns = self.feature_set(name=feature_set)
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
        self.physionet_prefixes = self.get_physionet_prefixes()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', 
                        help='path to features', 
                        default='datasets/dataset_name/data/parquet/features_middle')
    #parser.add_argument('--dataset', 
    #                    help='name of dataset to use', 
    #                    default='mimic_demo')
    parser.add_argument('--out-file', 
                        help='name of output file', 
                        default='config/features.json')
    args = parser.parse_args()
    #dataset_name = args.dataset 
    #path = f'datasets/{dataset_name}/data/parquet/features'
    path = args.input_path
    # Defensive step:
    if 'middle' not in path:
        raise ValueError(f'we currently assume the middle feature set, however it is not provided!')

    feature_sets = ['small', 'middle', 'raw'] #run large prepro before creating its feature groups
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
                columns = cf.physionet_set(feature_set=feature_set) 
                groups = cf.columns_to_groups(columns)     
            output['groups'] = groups; output['columns'] = columns 
            result[variable_set][feature_set] = output 
    with open(args.out_file, 'w') as f:
        json.dump(result, f, indent=4) 
