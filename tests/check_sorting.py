from src.sklearn.loading import ParquetLoader
from IPython import embed

if __name__ == '__main__':
    path = 'datasets/{}/data/parquet/features'
    datasets = ['mimic_demo', 'mimic', 'hirid', 'physionet2019', 'aumc', 'eicu']  
    
    for dataset in datasets:
        pl = ParquetLoader(path.format(dataset), 'dask')
        df = pl.load(columns=['stay_time']).compute()
        print(f'Testing sorting of {dataset}...')
        min_diff = df.groupby('stay_id')['stay_time'].diff().min()
        if min_diff < 1:
            print('FAILED!')
            embed()
        else: 
            print(f'OK.')
