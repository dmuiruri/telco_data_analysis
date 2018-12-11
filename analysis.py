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
from os import listdir
from datetime import datetime as dt


def process_text_files():
    """
    Process txt files by reading them with pandas and saving them to csv.

    Graphlab loads csv files much faster than pandas.
    """
    cols = ['sq_id', 'time', 'ccode', 'smsin',
            'smsout', 'callin', 'callout', 'web']
    url = 'dec_full/'  # 'sms-call-internet-mi-2013-{}-{}.txt'
    for file_name in listdir('./dec_full/'):
        print 'Processing file {}\n'.format(file_name)
        df = pd.read_csv(url + file_name, sep='\t', header=None, names=cols,
                         index_col=0)
        df.to_csv('data_csv/' + file_name.replace('txt', 'csv'))


def get_data(path):
    """Get data for analysis."""
    return gp.SFrame.read_csv(path)


def agg_by_time_slots(sf, ops={}):
    """
    Aggregate a given SFrame by the given set of operations.

    sf: A graphlab SFrame of data
    ops: A dict of operations as defined in graphlab API
    Returns a pandas DataFrame

    The data is recorded in intervals of 10 mins, aggregate them to
    a daily frequency.
    """

    print 'Aggregating the SFrame...'
    res = sf.groupby(key_columns='time', operations=ops)
    # Convert into a pandas dataframe better handling of timestamps
    print 'converting to dataframe...'
    df = res.to_dataframe()
    df['utc'] = [dt.utcfromtimestamp(i/1e3).strftime('%Y-%m-%d %H:%M:%S')
                 for i in df['time']]
    df.set_index('utc', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.drop(columns='time', inplace=True)
    return df


def get_congested_day(df):
    """
    Get the most congested day.

    Aggregate all traffic (Calls in, calls out, SMSs in, SMSs out, web)
    on a day basis to identify the most congested day
    """
    df_daily = df.resample('D').sum()
    dtotal = df_daily.sum(axis=1)
    return dtotal[dtotal == [dtotal.agg(max)]]


if __name__ == '__main__':
    sf_milano = get_data('../data/mi_telco_all')  # TODO different cities
    sf_trentino = get_data()
    operations = {'smsin_tot': agg.SUM('smsin'),
                  'smsout_tot': agg.SUM('smsout'),
                  'callin_tot': agg.SUM('callin'),
                  'callout_tot': agg.SUM('callout'),
                  'web_tot': agg.SUM('web'),
                  }
    # process_text_files()
    sfmi = agg_by_time_slots(sf_milano, ops=operations)
    print 'Aggregated time slots {}\n'.format(df.head())
