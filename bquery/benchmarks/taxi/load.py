import urllib
import glob
import pandas as pd
from bquery import ctable
import bquery
import bcolz

# do not forget to install numexpr
workdir = '/home/carst/Desktop/taxi/'

# wget https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-01.csv
for year in [2015]:
    for month in range(1, 13):
        filename = 'yellow_tripdata_' + str(year) + '-' + ('0' + str(month))[-2:] + '.csv'
        url = 'https://storage.googleapis.com/tlc-trip-data/' + str(year) + '/' + filename
        print(url)
        urllib.urlretrieve(url, workdir + '/' + filename)

file_list = sorted(glob.glob(workdir + 'yellow_tripdata_*.csv'))
if not file_list:
    raise ValueError('No Files Found')

expected_len = len(file_list) * 50000000

for i, filename in enumerate(file_list):

    print(filename)

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

    if i == 0:
        import_ct = ctable.fromdataframe(import_df, rootdir=workdir + 'taxi', expectedlen=expected_len, mode='w')
        del import_df
    else:
        temp_ct = ctable.fromdataframe(import_df)
        import_ct.append(temp_ct)
        del temp_ct
        del import_import_ct = ctable.fromdataframe(import_df, rootdir=workdir + 'taxi', expectedlen=expected_len,
                                                    mode='w')
        df

import_ct.flush()

import_ct = ctable(rootdir=workdir + 'taxi', mode='a')

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


bquery.set_nthreads(bquery.ncores)

import_ct.groupby(['pickup_yearmonth'], ['nr_rides'])
%timeit import_ct.groupby(['pickup_yearmonth'], ['nr_rides'])

# In [22]: %timeit import_ct.groupby(['pickup_yearmonth'], ['nr_rides'])
# 1 loop, best of 3: 6.91 s per loop

 %time nyc2015.payment_type.value_counts().compute()

import_ct.groupby(['payment_type'], ['nr_rides'])
%time import_ct.groupby(['payment_type'], ['nr_rides'])


In [26]: %time import_ct.groupby(['payment_type'], ['nr_rides'])
CPU times: user 11.3 s, sys: 284 ms, total: 11.5 s
Wall time: 7.41 s

compared to dask winine m3.2xlarge nodes on EC2. These have eight cores and 30GB of RAM each
>>> %time nyc2015.payment_type.value_counts().compute()
CPU times: user 132 ms, sys: 0 ns, total: 132 ms
Wall time: 558 ms

13.8 times slower but without parallelization and only 84mb resident + 400mb virtual memory usage
