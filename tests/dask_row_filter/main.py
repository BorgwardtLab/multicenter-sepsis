import dask.dataframe as dd


path = 'datasets/downloads/mimic_demo_0.3.0.parquet'

# read full df:
df = dd.read_parquet(path, engine='pyarrow').compute()

# generate subset of ids:
ids = df['stay_id'].unique()[:5]

# filter:
filt = [('stay_id', 'in', ids)]

# read subset of ids:
df2 = dd.read_parquet(path, engine='pyarrow', filters=filt)
