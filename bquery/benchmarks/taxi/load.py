import urllib
import glob
import pandas as pd
import dask.dataframe as dd
from bquery import ctable
import os

def download_data(workdir, year_list=None, month_list=None, force=False):
    # wget https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-01.csv
    year_list = year_list or [2015]
    month_list = month_list or range(1, 13)
    result = list()
    for year in year_list:
        for month in month_list:
            filename  = 'yellow_tripdata_{0}-{1:02d}.csv'.format(year, month)
            url = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{filename}'
            output_file = os.path.join(workdir, filename)
            print("url: {0}".format(url))
            print("output file: {0}".format(output_file))
            if not os.path.exists(output_file) or force:
                urllib.request.urlretrieve(url, output_file)
            result.append(output_file)
    return result


def load_file(filename):
    print("loading: {0}".format(filename))
    
    import_df = pd.read_csv(filename)

    # lower columns because of input inconsistencies
    import_df.columns = [x.lower() for x in import_df.columns]
    import_df.columns = [x.strip() for x in import_df.columns]
    import_df.columns = [x.replace('tpep_', '') for x in import_df.columns]

    import_df['nr_rides'] = 1

    import_df['pickup_date'] = import_df['pickup_datetime'].str[0:10]
    import_df['pickup_date'] = import_df['pickup_date'].str.replace('-', '')
    import_df['pickup_year'] = import_df['pickup_date'].str[0:4].astype(int)
    import_df['pickup_yearmonth'] = import_df['pickup_date'].str[0:6].astype(int)
    import_df['pickup_month'] = import_df['pickup_date'].str[4:6].astype(int)
    import_df['pickup_date'] = import_df['pickup_date'].astype(int)
    import_df['pickup_time'] = import_df['pickup_datetime'].str[11:]
    import_df['pickup_time'] = import_df['pickup_time'].str.replace(':', '')
    import_df['pickup_hour'] = import_df['pickup_time'].str[0:2].astype(int)
    import_df['pickup_time'] = import_df['pickup_time'].astype(int)
    del import_df['pickup_datetime']

    import_df['dropoff_date'] = import_df['dropoff_datetime'].str[0:10]
    import_df['dropoff_date'] = import_df['dropoff_date'].str.replace('-', '')
    import_df['dropoff_year'] = import_df['dropoff_date'].str[0:4].astype(int)
    import_df['dropoff_yearmonth'] = import_df['dropoff_date'].str[0:6].astype(int)
    import_df['dropoff_month'] = import_df['dropoff_date'].str[4:6].astype(int)
    import_df['dropoff_date'] = import_df['dropoff_date'].astype(int)
    import_df['dropoff_time'] = import_df['dropoff_datetime'].str[11:]
    import_df['dropoff_time'] = import_df['dropoff_time'].str.replace(':', '')
    import_df['dropoff_hour'] = import_df['dropoff_time'].str[0:2].astype(int)
    import_df['dropoff_time'] = import_df['dropoff_time'].astype(int)
    del import_df['dropoff_datetime']
    
    return import_df
    
def create_bcolz(workdir):
    csv_search_pattern = os.path.join(workdir, "yellow_tripdata_*.csv")
    print("csv search pattern: {0}".format(csv_search_pattern))
    file_list = sorted(glob.glob(csv_search_pattern))
    if not file_list:
        raise ValueError('No Files Found')

    expected_len = len(file_list) * 50000000

    rootdir=os.path.join(workdir, 'bcolz')

    for i, filename in enumerate(file_list):

        import_df = load_file(filename)
        
        if i == 0:
            import_ct = ctable.fromdataframe(import_df, rootdir=rootdir, expectedlen=expected_len, mode='w')
            del import_df
        else:
            temp_ct = ctable.fromdataframe(import_df)
            import_ct.append(temp_ct)
            del temp_ct
            del import_df

    print("Flushing ...")
    import_ct.flush()

    print("Ctable ...")
    import_ct = ctable(rootdir=rootdir, mode='a')

    print("Cache factor ...")

    import_ct.cache_factor([
        'dropoff_date',
        'dropoff_hour',
        'dropoff_latitude',
        'dropoff_longitude',
        'dropoff_month',
        'dropoff_time',
        'dropoff_year',
        'dropoff_yearmonth',
        'payment_type',
        'pickup_date',
        'pickup_hour',
        'pickup_latitude',
        'pickup_longitude',
        'pickup_month',
        'pickup_time',
        'pickup_year',
        'pickup_yearmonth',
        'ratecodeid',
        'store_and_fwd_flag',
        'vendorid'])


def create_bcolz_chunks(workdir):
    print(workdir)
    csv_search_pattern = os.path.join(workdir, "yellow_tripdata_*.csv")
    print("csv search pattern: {0}".format(csv_search_pattern))
    file_list = sorted(glob.glob(csv_search_pattern))
    if not file_list:
        raise ValueError('No Files Found')

    for i, filename in enumerate(file_list):
        rootdir = os.path.join(workdir, 'bcolz_chunks', 'taxi_{0}'.format(i))
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        import_df = load_file(filename)
        import_ct = ctable.fromdataframe(import_df, rootdir=rootdir, expectedlen=len(import_df), mode='w')
        del import_df

        import_ct.flush()

        import_ct = ctable(rootdir=rootdir, mode='a')

        import_ct.cache_factor([
            'dropoff_date',
            'dropoff_hour',
            'dropoff_latitude',
            'dropoff_longitude',
            'dropoff_month',
            'dropoff_time',
            'dropoff_year',
            'dropoff_yearmonth',
            'payment_type',
            'pickup_date',
            'pickup_hour',
            'pickup_latitude',
            'pickup_longitude',
            'pickup_month',
            'pickup_time',
            'pickup_year',
            'pickup_yearmonth',
            'ratecodeid',
            'store_and_fwd_flag',
            'vendorid'])

        
def create_parquet(workdir):
    """
    Create parquet
    
    """
    csv_search_pattern = os.path.join(workdir, "yellow_tripdata_*.csv")
    print("csv search pattern: {0}".format(csv_search_pattern))
    file_list = sorted(glob.glob(csv_search_pattern))
    if not file_list:
        raise ValueError('No Files Found')
        
    rootdir = os.path.join(workdir, 'parquet')
    if not os.path.exists(rootdir):
            os.makedirs(rootdir)
            
    measure_list = ['extra',
                'fare_amount',
                'improvement_surcharge',
                'mta_tax',
                'nr_rides',
                'passenger_count',
                'tip_amount',
                'tolls_amount',
                'total_amount',
                'trip_distance']
            
    # csv file
    output_dir_list = list()
    for i, csv_file in enumerate(file_list):
        import_df = load_file(csv_file)
        ddf = dd.from_pandas(import_df, chunksize=400000)
        #ddf.to_parquet(rootdir, engine='pyarrow', partition_on=['pickup_year', 'pickup_date'])
        output_dir = os.path.join(rootdir, str(i))
        ddf.to_parquet(output_dir, engine='fastparquet')
        output_dir_list.append(output_dir)
    return output_dir_list
        
        
    