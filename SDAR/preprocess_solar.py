# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:17:40 2020

@author: 18096
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import math
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prep_data(data, covariates, data_start, train=True, name='power', lag=2):
    # print("train: ", train)
    x_data = data[name].values    # modify it!!!
    time_len = x_data.shape[0]
    input_size = window_size - stride_size
    windows = (time_len - window_size + 1)
    x_input = np.zeros((windows, window_size, lag + num_covariates + 1), dtype='float32')
    label = np.zeros((windows, window_size), dtype='float32')
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for i in range(windows):
        if train:
            window_start = i + data_start
        else:
            window_start = i
        window_end = window_start + window_size
        for j in range(lag):
            x_input[count, (j+1):, j] = x_data[window_start:window_end - j - 1]
        x_input[count, :, lag:lag + num_covariates] = covariates[window_start:window_end, :]  # two lagged terms
        x_input[count, :, lag+num_covariates] = data['power'].values[window_start:window_end]
        # x_input[count, :, -1] = series
        label[count, :] = x_data[window_start:window_end]
        nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
        count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix + 'data_' + save_name, x_input)
    np.save(prefix + 'label_' + save_name, label)



def gen_covariates(raw, num_covariates):
    times = raw.index
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.hour
        covariates[i, 1] = input_time.month
    covariates[:, 2] = np.array(raw['ssrd'])  # relative humidity at 1000 m (r)
    # covariates[:, 3] = np.array(raw['cs_ghi'])  # ghi in ideal clear sky (cs_ghi)
    covariates[:, 3] = np.array(raw['2t'])  # total cloud cover (tcc)
    covariates[:, 4] = np.array(raw['tcc'])  # temperature at 2 m (2t)
    covariates[:, 5] = np.array(raw['rh']) # surface solar rad down (ssrd)
    covariates[:, 6] = np.array(raw['strd']) # surface thermal rad down (strd)
    for i in range(2, num_covariates):
        covariates[:, i] = stats.zscore(covariates[:, i])
    return covariates[:, :num_covariates]



def calResi(df, terms):
    power = df['power'].values
    spec = np.fft.fft(power[:8760] - power[:8760].mean())
    fit = np.zeros(power.size)
    t = np.arange(power.size)
    for term in terms:
        w = 2 * np.pi * term / 8760
        fit = fit + 2 * (np.real(spec[term]) * np.cos(w * t) -
                         np.imag(spec[term]) * np.sin(w * t))
    fit = fit / 8760 + power[:8760].mean()
    # reSpec = np.zeros(8760,dtype=complex)
    # reSpec[terms] = spec[terms]
    # reSpec[8760-terms] = spec[8760-terms]
    # rePower = np.real(np.fft.ifft(reSpec)) + power[:8760].mean()
    return fit


if __name__ == '__main__':
    global save_path
    name = 'Zone1.csv'
    save_name = 'Zone1'
    window_size = 44
    stride_size = 4
    num_covariates = 7
    train_start = '2012-04-01 01:00:00'
    train_end = '2013-04-01 00:00:00'
    test_start = '2013-03-27 01:00:00'  # need additional 5 days as given info
    test_end = '2013-07-01 00:00:00'
    pred_days = 5
    given_days = 5
    num_series = 1

    save_path = os.path.join('data', save_name)
    csv_path = os.path.join(save_path, name)

    data_frame = pd.read_csv(csv_path, sep=",", index_col=0, parse_dates=True)
    data_frame.fillna(0, inplace=True)

    fit = calResi(data_frame,np.array((1,2,364,365,366,729,730,731)))
    # fit = calResi(data_frame, np.array((1, 2, 3, 363, 364, 365, 366, 367, 728, 729, 730, 731, 732)))
    # fit = calResi(data_frame, np.array((1, 365, 730)))
    data_frame['resi'] = data_frame['power'] - fit

    # remove extreme values from power data to prepare for the next transformations
    data_frame['power'][data_frame['power']>=1] = 0.9999
    data_frame['power'][data_frame['power']<=0] = 0.0001

    data_frame['logistic'] = -np.log(1 / data_frame['power'] - 1)
    # data_frame['logistic'][np.isposinf(data_frame['logistic'])] = 6.9
    # data_frame['logistic'][np.isneginf(data_frame['logistic'])] = -6.9

    data_frame['abssig'] = (2*data_frame['power']-1)/(1-np.abs(2*data_frame['power']-1))

    df = data_frame[data_frame.index.hour < 8]

    data_start = 0  # find first nonzero value in each time series

    covariates = gen_covariates(df[train_start:test_end], num_covariates)
    # df['npower'] = df['power']*1000/df['cs_ghi']
    train_data = df[train_start:train_end]
    test_data = df[test_start:test_end]

    prep_data(train_data, covariates, data_start, name='power', lag=3)
    prep_data(test_data, covariates, data_start, train=False, name='power', lag=3)
