import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import math
import random
from tqdm import trange
import heapq

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats


def prep_data(data, covars, train=True, name='power', lag=2):
    # print("train: ", train)
    x_data = data[name].values  # modify it!!!
    num_covars = covars.shape[1]
    time_len = x_data.shape[0]
    input_size = window_size - stride_size
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
        x_input[count, :, lag + num_covars] = \
            data['power'].values[window_start:window_end]
        # x_input[count, :, -1] = series
        label[count, :] = x_data[window_start:window_end]
        count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix + 'data_' + data_dir, x_input)
    np.save(prefix + 'label_' + data_dir, label)


def gen_covars(raw, ids_covar):
    times = raw.index
    covars = np.zeros((times.shape[0], len(ids_covar)))
    count = 0
    for id in ids_covar:
        if id == 13:
            covars[:, count] = times.hour
            count = count + 1
        elif id == 14:
            covars[:, count] = times.month
            count = count + 1
        else:
            covars[:, count] = stats.zscore(raw.values[:, id])
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

def cal_fourier_resi_adaptive(source, dest, num_terms):
    spec = np.fft.fft(source - source.mean())
    fit = np.zeros(source.size + dest.size)
    t = np.arange(source.size + dest.size)
    magSpec = np.power(np.abs(spec),2)
    terms = heapq.nlargest(num_terms, range(len(magSpec)), magSpec.__getitem__)
    terms = np.array(terms)
    terms = terms[terms < 5000]
    for term in terms:
        w = 2 * np.pi * term / source.size
        fit = fit + 2 * (np.real(spec[term]) * np.cos(w * t) -
                         np.imag(spec[term]) * np.sin(w * t))
    fit = fit / source.size + source.mean()
    return dest - fit[source.size:]


if __name__ == '__main__':
    global save_path
    name = 'Zone1.csv'
    data_dir = 'Zone1'
    window_size = 44
    stride_size = 4
    num_covars = 7
    pred_days = 5

    data_path = os.path.join('data', data_dir)
    csv_path = os.path.join(data_path, name)

    df = pd.read_csv(csv_path, sep=',', index_col=0, parse_dates=True)
    df.fillna(0, inplace=True)

    df['power'][df['power'] >= 1] = 0.9999
    df['power'][df['power'] <= 0] = 0.0001

    df['logistic'] = -np.log(1 / df['power'] - 1)
    df['abssig'] = (2 * df['power'] - 1) / (1 - np.abs(2 * df['power'] - 1))

    fourier_terms = [1, 2, 364, 365, 366, 729, 730, 731]
    df['fourier_resi'] = np.zeros(df.shape[0])

    # df = df[df.index.hour < 8]  # truncate 24 hours to 8 hours

    ids_covars = [13, 14]

    ref_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype='int')
    ref_months = np.append(ref_months, ref_months)
    train_start = datetime.strptime('2012-04-01 01:00:00', '%Y-%m-%d %H:%M:%S')
    train_end = datetime.strptime('2013-04-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    val_start = train_end - timedelta(hours=pred_days * 24 - 1)  # 24h per day
    val_end = train_end + timedelta(
        days=int(np.sum(ref_months[train_end.month-1:train_end.month+2])))
    for i in range(6):
        save_path = os.path.join(data_path, 'ts' + str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df['fourier_resi'][train_start:val_end] = cal_fourier_resi(
            df[train_start:val_end]['power'].values, fourier_terms)
        tdf = df[df.index.hour < 8][train_start:val_end]
        covars = gen_covars(tdf, ids_covars)
        prep_data(tdf[train_start:train_end], covars, name='fourier_resi', lag=3)
        prep_data(tdf[val_start:val_end], covars, train=False,
                  name='fourier_resi', lag=3)
        delta_t = timedelta(days=int(np.sum(ref_months[train_end.month-1])))
        train_start += delta_t
        train_end += delta_t
        val_start = train_end - timedelta(hours=pred_days*24-1)
        val_end += timedelta(days=int(ref_months[val_end.month-1]))
