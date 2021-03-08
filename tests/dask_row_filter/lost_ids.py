import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq

from IPython import embed

def load(path, filt=None):
    return dd.read_parquet(path, engine='pyarrow-dataset', filters=filt).compute()

def load_with_index(path, cols=None):
    return dd.read_parquet(
                path, 
                engine='pyarrow',   
                index='stay_id',
                columns=cols).compute()

if __name__ == '__main__':
    
    dataset = 'mimic'
    path = f'datasets/downloads/{dataset}_0.3.0.parquet'
    path2 = f'datasets/{dataset}/data/parquet/features_no_ITF'
    path3 = f'datasets/{dataset}/data/parquet/features_ITF_set_index'
    #path4 = f'datasets/{dataset}/data/parquet/features_pyarrow_dataset'

    ## read full df:
    #df = dd.read_parquet(path, engine='pyarrow').compute()
    #
    ## generate subset of ids:
    #ids = tuple(df['stay_id'].unique()[:5])

    ids = (243329, 246529, 211143, 217289, 284941, 235421, 296222, 220704, 237736, 221247, 203903)
    # filter:
    filt = [('stay_id', 'in', ids )]

    # read subset of ids:

    cols = ['stay_id', 'stay_time', 'sep3']
    #df = dd.read_parquet(path, engine='pyarrow-dataset', columns=cols) #, #filters=filt 
    #ind1 = df['stay_id'].unique().compute()
    #print(len(ind1))
    #
    #df2 = dd.read_parquet(path2, engine='pyarrow-dataset', columns=cols[1:]) # [1:] for eicu
    #ind2 = df2.index.unique().compute()
    #print(len(ind2))
    #
    #difference = set(ind1).difference(ind2)
    #print('Difference between raw data and features (set_index)')
    #print(difference)
    #
    ## si = ITF + set_index
    #df_is = dd.read_parquet(path3, engine='pyarrow-dataset', columns=cols[1:])
    #ind3 = df_is.index.unique().compute()
    #print(len(ind3))

    # Original, raw data:
    df = load(path, filt=filt)
    #df = dd.read_parquet(path, engine='pyarrow-dataset', filters=filt).compute()
    # Processed, without InvalidTimesFiltration (ITF) 
    df2 = load(path2, filt=filt)
    #df2 = dd.read_parquet(path2, engine='pyarrow-dataset', filters=filt).compute()
    # Processed, with ITF and with set_index
    df3 = load(path3, filt=filt)
    #df3 = dd.read_parquet(path3, engine='pyarrow-dataset', filters=filt).compute() 

    df4 = load_with_index(path2, cols=cols[1:])
    #df4 = dd.read_parquet(path3, engine='pyarrow-dataset', filters=filt).compute() 

    embed();sys.exit()


    
    df = dd.read_parquet(path3, engine='pyarrow-dataset', columns=cols[1:])
    ind3 = df.index.unique().compute()
    print(len(ind3))

    d1 = set(ind1).difference(ind3)
    print(f'Difference between raw data and features_test (repartition): {d1}')

    df = dd.read_parquet(path4, engine='pyarrow-dataset', columns=cols[1:])
    ind4 = df.index.unique().compute()
    print(len(ind3))

    d2 = set(ind1).difference(ind4)
    print(f'Difference between raw data and features_test (no repartition): {d2}')


    #df3 = pd.read_parquet(path, engine='pyarrow', filters=filt)
    #
    #dataset = pq.ParquetDataset(path, use_legacy_dataset=False, filters=filt)
    #df4 = dataset.read().to_pandas()

    embed()
