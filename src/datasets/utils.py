
def get_file_mapping(version):
    return {    'mimic_demo': f'mimic_demo-{version}.parquet',
                'mimic': f'mimic-{version}.parquet',
                'hirid': f'hirid-{version}.parquet',
                'eicu': f'eicu-{version}.parquet',
                'aumc': f'aumc-{version}.parquet',
                'physionet2019': f'physionet2019-{version}.parquet',


    }

