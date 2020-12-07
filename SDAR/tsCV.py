import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
# import model.net_beta as net
import model.net_normal as net
# import model.net_cauchy as net
from evaluate import evaluate
from dataloader import *

import multiprocessing
import copy

logger = logging.getLogger('DeepAR.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Zone1', help='Name of the dataset')
parser.add_argument('--data-folder', default='data',
                    help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model',
                    help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true',
                    help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true',
                    help='Whether to sample during evaluation', default=True)
parser.add_argument('--save-best', action='store_true',
                    help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing '
                         'weights to reload before training')  # 'best' or
# 'epoch_#'
parser.add_argument('--trans', default=None,
                    help='Whether to pre-transform data')


def stabilityTest(model: nn.Module, loss_fn, test_loader: DataLoader,
                  params: utils.Params, epoch: int, id: int,
                  result_dict: dict, num=10):
    utils.load_checkpoint(
        params.model_dir + '/epoch_' + str(epoch - 1) + '.pth.tar', model)
    logger.info(
        'Epochs run out! Now start testing how stable the best epoch (%d) is '
        '!' % (epoch))
    crps_results = np.zeros(num)
    rou50_results = np.zeros(num)
    rou90_results = np.zeros(num)
    rc_results = np.zeros(num)
    sharp90_results = np.zeros(num)
    sharp50_results = np.zeros(num)
    for k in range(num):
        logger.info('Experiment %d' % (k + 1))
        results = evaluate(model, loss_fn, test_loader, params)
        crps_results[k] = results['crps']
        rou50_results[k] = results['rou50']
        rou90_results[k] = results['rou90']
        rc_results[k] = results['rc']
        sharp50_results[k] = results['sharp50']
        sharp90_results[k] = results['sharp90']
    result_dict[id] = [crps_results.mean(), crps_results.std()]
    # logger.info('The MEAN and STDERR of metrics (epoch %d) are:\n' % (epoch) +
    #             'CRPS: %.4f %.4f\n' % (
    #             crps_results.mean(), crps_results.std()) +
    #             'ROU50: %.4f %.4f\n' % (
    #             rou50_results.mean(), rou50_results.std()) +
    #             'ROu90: %.4f %.4f\n' % (
    #             rou90_results.mean(), rou90_results.std()) +
    #             'RC: %.4f %.4f\n' % (rc_results.mean(), rc_results.std()) +
    #             'SHARP50: %.4f %.4f\n' % (
    #             sharp50_results.mean(), sharp50_results.std()) +
    #             'SHARP90: %.4f %.4f\n' % (
    #             sharp90_results.mean(), sharp90_results.std()))


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep,
        and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{
    # 1:T}, note that z_0 = 0;
    # idx ([batch_size]): one integer denoting the time series id;
    # labels_batch ([batch_size, train_window]): z_{1:T}.
    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(
            params.device)  # not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(
            params.device)  # not scaled
        idx = idx.unsqueeze(0).to(params.device)

        loss = torch.zeros(1, device=params.device)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.train_window):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                train_batch[t, zero_index, 0] = p[zero_index]
            p, gama, hidden, cell = model(train_batch[t].unsqueeze_(0).clone(),
                                          idx, hidden, cell)
            loss += loss_fn(p, gama, labels_batch[t])

        loss.backward()
        optimizer.step()
        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss
        '''
        if i % 1000 == 0:
            test_metrics = evaluate(model, loss_fn, test_loader, params, 
            epoch, sample=args.sampling)
            model.train()
            logger.info(f'train_loss: {loss}')
        if i == 0:
            logger.info(f'train_loss: {loss}')
        '''
    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim, loss_fn,
                       params: utils.Params,
                       id: int,
                       return_dict: dict) -> None:
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep,
        and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (
        without its extension .pth.tar)
    """
    best_test_CRPS = (int(0), float('inf'))
    train_len = len(train_loader)
    CRPS_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))
    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] \
            = train(model,
                    optimizer,
                    loss_fn,
                    train_loader,
                    test_loader,
                    params,
                    epoch)
        test_metrics = evaluate(model, loss_fn, test_loader, params, epoch,
                                sample=args.sampling)
        CRPS_summary[epoch] = test_metrics['crps']
        is_best = CRPS_summary[epoch] <= best_test_CRPS[1]

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              checkpoint=params.model_dir)

        if is_best:
            logger.info('- Found new best CRPS')
            best_test_CRPS = (epoch + 1, CRPS_summary[epoch])
            best_json_path = os.path.join(params.model_dir,
                                          'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best CRPS is: %.5f of epoch %d' % (
        best_test_CRPS[1], best_test_CRPS[0]))

        utils.plot_all_epoch(CRPS_summary[:epoch + 1], args.dataset + '_CRPS',
                             params.plot_dir)
        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len],
                             args.dataset + '_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir,
                                      'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)

    if args.save_best:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = args.search_params.split(',')
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best CRPS: ' + str(best_test_CRPS) + '\n')
        logger.info(print_params)
        logger.info(f'Best CRPS: {best_test_CRPS}')
        f.close()
        utils.plot_all_epoch(CRPS_summary, print_params + '_CRPS',
                             location=params.plot_dir)
        utils.plot_all_epoch(loss_summary, print_params + '_loss',
                             location=params.plot_dir)

    stabilityTest(model, loss_fn, test_loader, params, best_test_CRPS[0],
                  id, return_dict)


def start_train(model: nn.Module, params: utils.Params,
                id: int, return_dict):
    train_set = TrainDataset(params.data_dir, args.dataset, params.num_class)
    test_set = TestDataset(params.data_dir, args.dataset, params.num_class)
    train_loader = DataLoader(train_set, batch_size=params.batch_size,
                              num_workers=0)  # modify 4 to 0
    test_loader = DataLoader(test_set, batch_size=params.predict_batch,
                             sampler=RandomSampler(test_set),
                             num_workers=0)  # modify 4 to 0

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function
    loss_fn = net.loss_fn_crps

    utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    logger.info('Staring training')
    # Train the model
    train_and_evaluate(model, train_loader, test_loader, optimizer, loss_fn,
                       params, id, return_dict)



if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(
        json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    pool = multiprocessing.Pool(6)
    for i in range(6):
        tparams = copy.copy(params)
        tparams.model_dir = os.path.join(model_dir, 'ts' + str(i))
        tparams.plot_dir = os.path.join(tparams.model_dir, 'figures')
        tparams.trans = None
        tparams.spline = False
        tparams.data_dir = os.path.join(data_dir, 'ts' + str(i))
        try:
            os.makedirs(tparams.plot_dir)
        except FileExistsError:
            pass

        tparams.device = torch.device('cuda:' + str(i%2))
        model = net.Net(tparams).cuda(tparams.device)
        print('the %d th process'%(i))
        pool.apply_async(start_train, args=(model, tparams, i, return_dict))

    pool.close()
    pool.join()
    return_dict = dict(return_dict)
    print(return_dict)
    logger.info(str(return_dict))
    crps_mean = 0
    for i in range(6):
        crps_mean += return_dict[i][0]
    crps_mean /= 6
    print('\nMean CRPS is %.4f'%crps_mean)
    logger.info('\nMean CRPS is %.4f'%crps_mean)


