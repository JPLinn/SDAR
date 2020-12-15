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
import heapq
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prep_data(x_data, covars, usage='train', lag=2,
              window_size=44, data_name='Zone1'):
    num_covars = covars.shape[1]
    time_len = x_data.shape[0]
    # input_size = window_size - stride_size
    windows = time_len - window_size + 1
    x_input = np.zeros((windows, window_size, lag + num_covars + 1),
                       dtype='float32')
    label = np.zeros((windows, window_size), dtype='float32')
    count = 0
    for i in range(windows):
        window_start = i
        window_end = window_start + window_size
        for j in range(lag):
            x_input[count, (j + 1):, j] = x_data[
                                          window_start:window_end - j - 1]
        x_input[count, :, lag:lag + num_covars] = \
            covars[window_start:window_end, :]
        # x_input[count, :, -1] = series
        label[count, :] = x_data[window_start:window_end]
        count += 1

    save_path = os.path.join('data', data_name)
    prefix = os.path.join(save_path, usage + '_')
    np.save(prefix + 'data_' + data_name, x_input)
    np.save(prefix + 'label_' + data_name, label)


def gen_covars(raw, ids_covar, fit_len):
    times = raw.index
    covars = np.zeros((times.shape[0], len(ids_covar)))
    raw_value = raw.values
    count = 0
    for id in ids_covar:
        if id == -1:
            covars[:, count] = times.hour
            count = count + 1
        elif id == 0:
            covars[:, count] = times.month
            count = count + 1
        else:
            mean = raw_value[:fit_len, id].mean()
            std = raw_value[:fit_len, id].std()
            covars[:, count] = (raw_value[:, id] - mean)/std

            # covars[:, count] = raw.values[:, id]
            count = count + 1
    return covars


def cal_fourier_resi(raw, terms):
    spec = np.fft.fft(raw[:8760] - raw[:8760].mean())
    fit = np.zeros(raw.size)
    t = np.arange(raw.size)
    for term in terms:
        w = 2 * np.pi * term / 8760
        fit = fit + 2 * (np.real(spec[term]) * np.cos(w * t) -
                         np.imag(spec[term]) * np.sin(w * t))
    fit = fit / 8760 + raw[:8760].mean()
    return raw - fit


def cal_fourier_fit_adaptive(raw, terms):
    source = raw[:8760]
    spec = np.fft.fft(source - source.mean())
    fit = np.zeros(len(raw))
    t = np.arange(len(raw))
    # magSpec = np.power(np.abs(spec), 2)
    # if isinstance(num_terms, int):
    #     terms = heapq.nlargest(num_terms, range(len(magSpec)), magSpec.__getitem__)
    #     terms = np.array(terms)
    #     terms = terms[terms < source.size / 2]
    # elif isinstance(num_terms, list):
    #     for id in num_terms:
    #         precise = source.size/id
    #         roun = round(precise)
    #         terms = terms + [roun-1, roun, roun+1, roun-2 if roun > precise else roun + 2]
    for term in terms:
        w = 2 * np.pi * term / source.size
        fit = fit + 2 * (np.real(spec[term]) * np.cos(w * t) -
                         np.imag(spec[term]) * np.sin(w * t))
    fit = fit / source.size + source.mean()
    return fit


def prepare_data(source='Zone1', format='power', ids_covars=list(range(-1, 13)),
                 lag_num=3, window_size=[44, 44], terms=[1, 365, 730]):
    data_path = os.path.join('data', source)
    csv_path = os.path.join(data_path, source + '.csv')

    df = pd.read_csv(csv_path, sep=',', index_col=0, parse_dates=True)
    df.fillna(0, inplace=True)

    df['power'][df['power'] >= 1] = 0.9999
    df['power'][df['power'] <= 0] = 0.0001

    df['logistic'] = -np.log(1 / df['power'] - 1)
    df['abssig'] = (2 * df['power'] - 1) / (1 - np.abs(2 * df['power'] - 1))

    train_start = '2012-04-01 01:00:00'
    train_end = '2013-08-01 00:00:00'
    vali_start = '2013-07-27 01:00:00'  # need additional 5 days as given info
    vali_end = '2013-12-01 00:00:00'
    test_start = '2013-11-26 01:00:00'
    test_end = '2014-07-01 00:00:00'

    fit = cal_fourier_fit_adaptive(df['power'].values, terms=terms)
    df['fourier'] = df['power'] - fit
    tdf = df[df.index.hour < 8][train_start:test_end]
    covars = gen_covars(tdf, ids_covars, tdf[train_start:train_end].shape[0])

    train = tdf[train_start:train_end][format].values
    prep_data(train, covars[:train.size], usage='train',
              lag=lag_num, window_size=window_size[0])

    vali = tdf[vali_start:vali_end][format].values
    prep_data(vali, covars[train.size-40:][:vali.size],
              usage='vali', lag=lag_num, window_size=window_size[1])

    test = tdf[test_start:test_end][format].values
    prep_data(test, covars[-test.size:], usage='test',
              lag=lag_num, window_size=window_size[1])


if __name__ == '__main__':
    covars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, 0]
    # covars = [4, 5, 8, 9, 10, 13, 14]
    # covars = [13, 14]
    prepare_data('Zone1', format='power', ids_covars=covars, terms=[1, 2, 364, 365, 366, 729, 730, 731])
