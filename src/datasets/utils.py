
def get_file_mapping(version):
    return {    'demo': f'mimic_demo_{version}.parquet',
                'mimic': f'mimic_{version}.parquet',
                'hirid': f'hirid_{version}.parquet',
                'eicu': f'eicu_{version}.parquet',
                'aumc': f'aumc_{version}.parquet',
                'physionet2019': f'physionet2019_{version}.parquet',


    }

