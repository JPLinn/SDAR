import os
import sys
import json
import logging
import argparse
import multiprocessing
from copy import copy
from itertools import product
from subprocess import check_call

import numpy as np
import utils

import pickle
import time

logger = logging.getLogger('DeepAR.Searcher')

utils.set_logger('param_search.log')

PYTHON = sys.executable
gpu_ids: list
param_template: utils.Params
args: argparse.ArgumentParser
search_params: dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Zone1', help='Dataset name')
parser.add_argument('--data-dir', default='data', help='Directory containing the dataset')
parser.add_argument('--model-name', default='param_search', help='Parent directory for all jobs')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--gpu-ids', nargs='+', default=[0,1], type=int, help='GPU ids')
parser.add_argument('--sampling', help='Whether to do ancestral sampling during evaluation', default=True)
parser.add_argument('--job-dir', default='2', help='the directory of specific job')

def launch_training_job(search_range):
    """Launch training of the model with a set of hyper-parameters in parent_dir/job_name
    Args:
        search_range: one combination of the params to search
    """

    search_range = search_range[0]
    params = {k: search_params[k][search_range[idx]] for idx, k in enumerate(sorted(search_params.keys()))}
    model_param_list = '-'.join('_'.join((k, f'{v:.2f}')) for k, v in params.items())
    model_param = copy(param_template)
    for k, v in params.items():
        setattr(model_param, k, v)

    pool_id, job_idx = multiprocessing.Process()._identity
    gpu_id = gpu_ids[pool_id % 2 ]

    logger.info(f'Worker {pool_id} running {job_idx} using GPU {gpu_id}')

    # Create a new folder in parent_dir with unique_name 'job_name'
    model_name = os.path.join(model_dir, args.job_dir, model_param_list)
    model_input = os.path.join(args.model_name, args.job_dir, model_param_list)
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    # Write parameters in json file
    json_path = os.path.join(model_name, 'params.json')
    model_param.save(json_path)
    logger.info(f'Params saved to: {json_path}')

    # Launch training with this config
    cmd = f'{PYTHON} train.py ' \
        f'--model-name={model_input} ' \
        f'--dataset={args.dataset} ' \
        f'--data-folder={args.data_dir} '
    if args.sampling:
        cmd += ' --sampling'
    if args.relative_metrics:
        cmd += ' --relative-metrics'

    logger.info(cmd)
    check_call(cmd, shell=True, env={'CUDA_VISIBLE_DEVICES': str(gpu_id),
                                     'OMP_NUM_THREADS': '4'})


def start_pool(project_list, processes):

    pool = multiprocessing.Pool(processes)
    pool.map(launch_training_job, [(i, ) for i in project_list])
    # for i in project_list:
    #     pool.apply(launch_training_job, args=((i,)))

def main():
    # Load the 'reference' parameters from parent_dir json file
    global param_template, gpu_ids, args, search_params, model_dir

    time_start = time.time()

    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_file = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_file), f'No json configuration file found at {args.json}'
    param_template = utils.Params(json_file)

    gpu_ids = args.gpu_ids
    logger.info(f'Running on GPU: {gpu_ids}')

    # Perform hypersearch over parameters listed below
    search_params = {
        'lstm_dropout': np.arange(0, 0.501, 0.1, dtype=np.float32).tolist(),
        'lstm_hidden_dim': np.arange(16, 31, 2, dtype=np.int).tolist()
        # 'num_spline': np.arange(5,66,10,dtype=int).tolist()
    }

    keys = sorted(search_params.keys())
    search_range = list(product(*[[*range(len(search_params[i]))] for i in keys]))

    start_pool(search_range, len(gpu_ids)*4)

    results = np.empty((6,len(search_range)))  # 6 is the number of metrics
    count = 0
    for i in search_range:
        params = {k: search_params[k][i[idx]] for idx, k in enumerate(sorted(search_params.keys()))}
        model_param_list = '-'.join('_'.join((k, f'{v:.2f}')) for k, v in params.items())
        model_name = os.path.join(model_dir, model_param_list)
        json_path = os.path.join(model_name, 'metrics_test_best_weights.json')
        with open(json_path) as f:
            temp = json.load(f)
            results[:, count] = np.array(list(temp.values()))
            count += 1

    save_name = os.path.join(model_dir, '2.1__' + '-'.join(k for k in search_params.keys()))
    with open(save_name, 'wb') as f:
        pickle.dump(search_params, f)
        pickle.dump(results, f)

    time_end = time.time()

    print('time cost:', (time_end - time_start)/60, 'min')
    # X, Y = np.meshgrid(search_params['lstm_dropout'], search_params['lstm_hidden_dim'])
    # crps = results[0,:].reshape(-1,len(search_params['lstm_hidden_dim'])).T
    # fig = plt.figure()
    # ax3 = plt.axes(projection='3d')
    # ax3.plot_surface(X, Y, crps)
    # plt.show()

if __name__ == '__main__':
    main()
