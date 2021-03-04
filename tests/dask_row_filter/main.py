import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq

from IPython import embed

path = 'datasets/downloads/mimic_demo_0.3.0.parquet'

# read full df:
df = dd.read_parquet(path, engine='pyarrow').compute()

# generate subset of ids:
ids = tuple(df['stay_id'].unique()[:5])

# filter:
filt = [('stay_id', 'in', ids)]

# read subset of ids:
df2 = dd.read_parquet(path, engine='pyarrow', filters=filt).compute()

df3 = pd.read_parquet(path, engine='pyarrow', filters=filt)

dataset = pq.ParquetDataset(path, use_legacy_dataset=False, filters=filt)
df4 = dataset.read().to_pandas()

embed()
