#!/usr/bin/env python
"""
This script implements a data analysis job for a Telco dataset.

The dataset is obtained from API providers (), the main libraries used for the
analysis are GraphLab and Pandas.
"""

__author__ = "Dennis Muiruri"
__version__ = "1.0.0"
__status__ = "Development"
__maintainer__ = "Dennis Muiuri"

import pandas as pd
import numpy as np
import graphlab as gp
import graphlab.aggregate as agg
import requests as req
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir, rename, chdir
from datetime import datetime as dt


def rename_files(path):
    """
    Rename files in batch manner.

    path: location of files to be renamed
    """
    for fn in listdir(path):
        if fn.endswith('txt'):
            rename(path + fn, path + fn.replace('txt', 'csv'))
            print 'Renamed {} to {}\n'.format(fn, fn.replace('txt', 'csv'))


def process_text_files(path_txt, path_csv, sep, cols=[]):
    """
    Process txt files by reading them with pandas and saving them to csv.

    path_txt: Path to read text files
    path_csv: Path to save csv files
    sep: column seperator character
    cols = Names to use for columns ['sq_id', 'prov', 'timeint', 'sq_to_prov']

    Graphlab loads csv files much faster than pandas and can scale to a
    distributed storage.

    analysis.process_text_files('/Users/muiruri/Downloads/mi_prov/',
                                '../data/mi_provinces/', cols=cols_prov)
    """
    # TODO: quite slow processing, parellize the file reads and write, maybe
    # reading and re-writing the whole file is not needed
    # GraphLab's SFrame.read_csv API does not allow passing of column names at
    # the time of reading the csv as pandas does.
    rename_files(path_txt)

    for file_name in listdir(path_txt):
        if file_name.endswith('csv'):
            print 'Processing file {}\n'.format(file_name)
            df = pd.read_csv(path_txt + file_name, sep=sep, header=None,
                             names=cols, index_col=0)
            df.to_csv(path_csv + file_name)


def get_data_from_csv(path, header=True):
    """
    Get data for analysis.

    path: A path string to a directory containing csv rile

    returns a GraphLab SFrame containing data.
    """
    return gp.SFrame.read_csv(path, header=header)


def agg_by_time_stamp(sf, groupingcol, ops={}):
    """
    Aggregate a given SFrame by the given set of operations.

    sf: A graphlab SFrame of data
    groupingcol: a string indicating the groupby column name
    ops: A dict of operations as defined in graphlab API
    Returns a pandas DataFrame

    The data is recorded in intervals of 10 mins
    """
    # TODO Refactor and remove the code to convert to a dataframe
    res = sf.groupby(key_columns=groupingcol, operations=ops)
    # Convert into a pandas dataframe better handling of timestamps
    res['utc'] = res['time'].apply(lambda x: dt.utcfromtimestamp(x/1e3).
                                   strftime('%Y-%m-%d %H:%M:%S'))
    sf_df = res.to_dataframe()
    sf_df.set_index('utc', inplace=True)
    sf_df.index = pd.to_datetime(sf_df.index)
    sf_df.drop(columns='time', inplace=True)
    return sf_df


def agg_traffic_24_hours(sfdf):
    """
    Agg data according to recording hour.

    This means calculating the mean of the data over a given hour, e.g. 23:00
    taking the mean of all data recorded at 23:00 each day so that it is
    possible to compare traffic on per hour basis or the recording time.
    """
    opsv2 = {'SMSsIn': agg.MEAN('smsin_tot'),
             'SMSsout': agg.MEAN('smsout_tot'),
             'CallsIn': agg.MEAN('callin_tot'),
             'CallsOut': agg.MEAN('callout_tot'),
             'WebTraff': agg.MEAN('web_tot')
             }

    sfdf['hour'] = sfdf['time'].apply(lambda x: dt.utcfromtimestamp(x/1e3).
                                      strftime('%H:%M:%S'))
    sf = gp.SFrame(data=sfdf[['callin_tot', 'callout_tot', 'smsin_tot',
                   'web_tot', 'smsout_tot', 'hour']])
    sf_grouped = sf.groupby('hour', operations=opsv2)
    sfdf = sf_grouped.sort('hour').to_dataframe().set_index('hour')
    sfdf.index = pd.to_datetime(sfdf.index)
    sfdf = sfdf.resample('H').sum()
    return np.log(sfdf.aggregate('sum', axis='columns'))


def agg_traffic_hourly(sf):
    """
    Aggregate data hourly.
    """
    ops = {'smsin_tot': agg.SUM('smsin'),
           'smsout_tot': agg.SUM('smsout'),
           'callin_tot': agg.SUM('callin'),
           'callout_tot': agg.SUM('callout'),
           'web_tot': agg.SUM('web')
           }

    # sfdf = agg_by_time_stamp(sf, groupingcol, ops)
    # sfdf = sfdf.resample('H').sum()

    res = sf.groupby(key_columns='time', operations=ops)
    # Convert into a pandas dataframe better handling of timestamps
    res['utc'] = res['time'].apply(lambda x: dt.utcfromtimestamp(x/1e3).
                                   strftime('%Y-%m-%d %H:%M:%S'))
    sf_df = res.to_dataframe()
    sf_df.set_index('utc', inplace=True)
    sf_df.index = pd.to_datetime(sf_df.index)
    sf_df = sf_df.resample('H').sum()
    sf_df.drop(columns='time', inplace=True)

    return sf_df


def get_congested_day(df):
    """
    Get the most congested day.

    Aggregate all traffic (Calls in, calls out, SMSs in, SMSs out, web)
    on a day basis to identify the most congested day
    """
    df_daily = df.resample('D').sum()
    dtotal = df_daily.sum(axis=1)
    return dtotal[dtotal == [dtotal.agg(max)]]


def get_most_called_province(sf):
    """Get the province most contacted."""
    ops = {'sq_to_prov_total': agg.SUM('sq_to_prov')}
    temp = sf.groupby('prov', operations=ops)
    # temp[temp['sq_to_prov_tot'] == temp['sq_to_prov_tot'].max()]['prov'][0]
    return temp.sort('sq_to_prov_total', ascending=False)[1:6]


def get_data_api(api_id, api_key):
    """
    Get data through the API

    Returns a GraphLab SFrame
    """
    done = False
    limit = 50000
    url = 'https://api.dandelion.eu/datagems/v2/SpazioDati/social-pulse-milano/data?'
    params = '$limit={}&$offset={}&$app_id={}&$app_key={}'  # 269290
    offset = 0
    df = pd.DataFrame()
    while not done:
        print('A new round offset: {} and limit: {}\n'.format(offset, limit))
        try:
            r = req.get(url + params.format(limit, offset, api_id, api_key))
        except ValueError:
            print 'Could not process the get request'
        data = r.json()
        df = df.append(pd.DataFrame(data['items']))
        offset += limit
        if not len(data['items']) > 0:
            done = True
    return gp.SFrame(data=df)


def tweeting_language_popularity(sf):
    """Get language popularity."""
    return sf.groupby('language', operations={'tweets': agg.COUNT()}
                      ).sort('tweets', ascending=False)


def plot_distributions(sfdf):
    """
    Plot distributions comparing Dec and Nov agg_traffic_24_hours.
    sfdf: is a pandas DataFrame
    """
    calls_nov = sfdf['26/11/2013':'28/11/2013'][['callin_tot', 'callout_tot']]
    web_nov = sfdf['26/11/2013':'28/11/2013'][['web_tot']]
    calls_dec = sfdf['24/12/2013':'26/12/2013'][['callin_tot', 'callout_tot']]
    web_dec = sfdf['24/12/2013':'26/12/2013'][['web_tot']]

    sns.set(style="white", color_codes=True)  # palette="muted"
    f, axes = plt.subplots(2, 2, figsize=(7, 7))  # sharex=True
    sns.despine(left=True)
    sns.distplot(np.log(calls_nov.agg('sum', axis='columns')),
                 color="m", ax=axes[0, 0], label='Nov')
    sns.distplot(np.log(calls_dec.agg('sum', axis='columns')),
                 color="m", ax=axes[0, 1], label='Dec')
    sns.distplot(np.log(web_nov), color="g", ax=axes[1, 0],
                 label='Nov Web Traffic')
    plt.legend()
    sns.distplot(np.log(web_dec), color="r", ax=axes[1, 1],
                 label='Dec Web Traffic')
    # plt.setp(axes)
    plt.tight_layout()
    plt.legend()
    plt.savefig('./distributions.png')


def process_sensors(path_sensors):
    """
    Process sensors data.
    path_sensors: path to a sensors file

    There are different sensors in various locations and in some cases more
    than one sensor measuring the same weather parameter.
    """
    sens = get_data_from_csv(path_sensors)
    st = sens.groupby(key_columns='street', operations={'count': agg.count()})
    return st


def get_all_weather_data(path_weather, path_sensor):
    """
    Get all recorded sensor data in one stable

    path_weather: path to weather data
    path_sensor: path to sensor data

    Returns a pandas DataFrame
    The weather recording data and sensor information is given in two seperate
    files, where the weather data contains measurements and sensor id, sensor
    data contains sensor information.
    """
    weather = get_data_from_csv(path_weather)  # '../data/mi_weather/'
    sensors = get_data_from_csv(path_sensor)  # '../data/mi_sensors/'
    ws = weather.join(sensors, on='sensor', how='left')

    ind = ws[ws['sensor'] == ws['sensor'][0]]['time']
    resdf = pd.DataFrame(index=pd.to_datetime(ind))

    for sens in sensors['sensor']:
        wdf = ws[ws['sensor'] == sens].sort('time')[['time', 'obs']]
        name = sensors[sensors['sensor'] == sens]['senstype'][0]
        df = wdf.to_dataframe().rename(columns={'obs': sens}).set_index('time')
        resdf = resdf.join(df, how='outer')  # contains NaN values
    return resdf


def weather_stations(sensor_path):
    """
    Get the weather stations and the amount of sensors in each station.
    """
    sens = get_data_from_csv(sensor_path)  # '../data/mi_sensors/'
    st = sens.groupby(key_columns='street', operations={'count': agg.COUNT()})
    return st


def get_sensors_in_a_given_station(sensor_path, station):
    """
    Get sensors located in a given weather station.
    """
    sens_d = get_data_from_csv(sensor_path)  # '../data/mi_sensors/'
    return sens_d[sens_d['street'] == station]


def calculate_weather_traffic_corr(sf, station, path_weather, path_sensor):
    """
    Calculate the correlation between weather and telco traffic.

    sf: traffic SFrame
    """
    # df.agg("mean", axis="columns") # multiple sensors in same wstation
    tot_traffic = agg_traffic_hourly(sf).aggregate('sum', axis='columns')
    data = pd.DataFrame(tot_traffic)
    sens = get_sensors_in_a_given_station(path_sensor, station)
    # sensors = sensor_sf['sensor']
    weather = get_all_weather_data(path_weather, path_sensor)
    for sensor in sens['sensor']:
        data = data.join(weather[sensor], how='outer')
    data.dropna(inplace=True)

    # Renaming results
    names = sens.to_dataframe().set_index('sensor').to_dict()
    colnames = {sensor: names['senstype'][sensor] for sensor in sens['sensor']}
    data.rename(columns=colnames, inplace=True)
    return data.corr()


if __name__ == '__main__':
    # Question 1):

    # process_text_files()
    sf_milano = get_data('../data/mi_telco_all')
    sf_trentino = get_data('')  # TODO different cities
    operations = {'smsin_tot': agg.SUM('smsin'),
                  'smsout_tot': agg.SUM('smsout'),
                  'callin_tot': agg.SUM('callin'),
                  'callout_tot': agg.SUM('callout'),
                  'web_tot': agg.SUM('web')
                  }
    mi_df = agg_by_time_stamp(sf_milano, time, ops=operations)
    mi_res = get_congested_day(mi_df)
    mi_res_hourly = agg_traffic_24_hours(mi_df)
    # tr_df = agg_by_time_stamp(sf_milano, ops=operations)
    # tr_res = get_congested_day(mi_df)
    print 'Aggregated time slots {}\n'.format(mi_res)

    # Question 2):
    mi_prov_sf = get_data('../data/mi_provinces')
    tr_prov_sf = get_data('')
    mi_to_prov = get_most_called_province(mi_prov_sf)
    tr_to_prov = get_most_called_province(tr_prov_sf)
    print 'Most called province from Milano {}'.format(mi_to_prov)

    # Question 3):
    api_id = 'xxxx'
    api_key = 'xxxx'
    mi_tweet_sf = get_data_api(api_id, api_key)
    res = tweeting_language_popularity(sf)
    print 'Top five language used in tweeting {}'.format(res[:5])

    # Question 4):
    sfdf_dist = agg_traffic_24_hours(sf_milano)  # confirm if correct func
    plot_distributions(sfdf)

    # Question 5):

    calculate_weather_traffic_corr(sf,
                                   'Milano - P.zza  Zavattari',  # Wth Station
                                   '../data/mi_weather/',
                                   '../data/mi_sensors/')
