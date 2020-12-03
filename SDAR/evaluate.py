import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import pickle
import utils
import model.net_normal as net
from dataloader import *

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Zone1', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='epoch_18',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def transform(samples, labels, trans):
    if trans == 'logistic':
        samples = 1 / (1 + torch.exp(-samples))
        labels = 1 / (1 + torch.exp(-labels))
    elif trans == 'abssig':
        samples = (samples/(1+torch.abs(samples))+1)/2.
        labels = (labels/(1+torch.abs(labels))+1)/2.
    return samples, labels


def evaluate(model, loss_fn, test_loader, params, plot_num=5, sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
        plot_batch = np.random.randint(len(test_loader) - 1)

        summary_metric = {}
        raw_metrics = utils.init_metrics(sample=sample)

        # Test_loader:
        # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        # id_batch ([batch_size]): one integer denoting the time series id;
        # v ([batch_size, 2]): scaling factor for each window;
        # labels ([batch_size, train_window]): z_{1:T}.
        for i, (test_batch, id_batch, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
            id_batch = id_batch.unsqueeze(0).to(params.device)
            labels = labels.to(torch.float32).to(params.device)
            batch_size = test_batch.shape[1]
            input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device)  # scaled
            input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device)  # scaled
            hidden = model.init_hidden(batch_size)
            cell = model.init_cell(batch_size)

            for t in range(params.test_predict_start):
                # if z_t is missing, replace it by output mu from the last time step
                # zero_index = (test_batch[t, :, 0] == 0)
                # if t > 0 and torch.sum(zero_index) > 0:
                #     test_batch[t, zero_index, 0] = mu[zero_index]
                if params.spline:
                    u, beta, gama, hidden, cell = \
                        model(test_batch[t].unsqueeze(0),
                              id_batch, hidden, cell)
                else:
                    mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0),
                                                    id_batch, hidden, cell)

            if sample:
                samples, sample_mu, _ = \
                    model.test(test_batch, id_batch, hidden, cell,
                               sampling=True)
                if params.trans:
                    samples, labels = transform(samples, labels, params.trans)
                raw_metrics = \
                    utils.update_metrics(raw_metrics, sample_mu, labels,
                                         params.test_predict_start, samples,
                                         relative=params.relative_metrics)
            else:
                print("hello")
        summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
        strings = '\nCRPS: ' + str(summary_metric['crps']) + '\nRMSE: ' + str(
            summary_metric['RMSE']) + '\nrou90: ' + str(summary_metric['rou90']) + \
                  '\nrou50: ' + str(summary_metric['rou50']) + '\nsharp50: ' + str(
            summary_metric['sharp'][:4]) + '\nsharp90: ' + str(summary_metric['sharp'][4:]) + \
                  '\nrc: ' + str(summary_metric['rc'][:, -1])
        summary_metric['crps'] = summary_metric['crps'].mean()
        logger.info('- Full test metrics: ' + strings)
    ss_metric = {}
    ss_metric['crps'] = summary_metric['crps']
    ss_metric['rou90'] = summary_metric['rou90']
    ss_metric['rou50'] = summary_metric['rou50']
    ss_metric['sharp50'] = summary_metric['sharp'][:4].mean()
    ss_metric['sharp90'] = summary_metric['sharp'][4:].mean()
    ss_metric['rc'] = summary_metric['rc'][:, -1].mean()
    # ss_metric['test_loss'] = summary_metric['test_loss']
    return ss_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):
    x = np.arange(window_size)
    nrows = 21
    ncols = 1


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')

    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
