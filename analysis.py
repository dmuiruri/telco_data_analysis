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
import requests
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


def process_text_files(path_txt, path_csv, cols=[]):
    """
    Process txt files by reading them with pandas and saving them to csv.

    path_txt: Path to read text files
    path_csv: Path to save csv files
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
        print 'Processing file {}\n'.format(file_name)
        df = pd.read_csv(path_txt + file_name, sep='\t', header=None,
                         names=cols, index_col=0)
        df.to_csv(path_csv)


def get_data(path):
    """Get data for analysis."""
    return gp.SFrame.read_csv(path)


def agg_by_time_slots(sf, groupingcol, ops={}):
    """
    Aggregate a given SFrame by the given set of operations.

    sf: A graphlab SFrame of data
    groupingcol: a string indicating the groupby column name
    ops: A dict of operations as defined in graphlab API
    Returns a pandas DataFrame

    The data is recorded in intervals of 10 mins, aggregate them to
    a daily frequency.
    """
    # TODO Refactor and remove the code to convert to a dataframe
    res = sf.groupby(key_columns=groupingcol, operations=ops)
    # Convert into a pandas dataframe better handling of timestamps
    res['utc'] = res['time_col'].apply(lambda x: dt.utcfromtimestamp(x/1e3).
                                       strftime('%Y-%m-%d %H:%M:%S'))
    # df['utc'] = [dt.utcfromtimestamp(i/1e3).strftime('%Y-%m-%d %H:%M:%S')
    #              for i in df['time']]
    df = res.to_dataframe()
    df.set_index('utc', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.drop(columns='time', inplace=True)
    return df


# def convert_sf_to_df(sf, time_col):
#     """
#     Convert an SFrame into a dataframe and assign an index(time).
#
#     sf: The SFrame to be converted
#     time_col
#     """
#     sf['utc'] = sf['time_col'].apply(lambda x: dt.utcfromtimestamp(x/1e3).
#                                      strftime('%Y-%m-%d %H:%M:%S'))


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

    Returns a json dataset
    """
    url = 'https://api.dandelion.eu/datagems/v2/SpazioDati/social-pulse-milano/data?$limit=269290&$offset=0&$app_id={}&$app_key={}'.format(API_ID, API_KEY)
    r = requests.get(url)
    return r.json()


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
    mi_df = agg_by_time_slots(sf_milano, ops=operations)
    mi_res = get_congested_day(mi_df)
    # tr_df = agg_by_time_slots(sf_milano, ops=operations)
    # tr_res = get_congested_day(mi_df)
    print 'Aggregated time slots {}\n'.format(mi_res)

    # Question 2):
    mi_prov_sf = get_data('../data/mi_provinces')
    tr_prov_sf = get_data('')
    mi_to_prov = get_most_called_province(mi_prov_sf)
    tr_to_prov = get_most_called_province(tr_prov_sf)
    print 'Most called province from Milano {}'.format(mi_to_prov)

    # Question 3):
