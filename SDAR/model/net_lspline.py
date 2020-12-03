# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:52:22 2020

@author: 18096
"""

'''Defines the neural network, loss function and metrics'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import properscoring as ps
from statsmodels.distributions.empirical_distribution import ECDF

logger = logging.getLogger('DeepAR.Net')


class Net(nn.Module):
    def __init__(self, params):
        '''
        We define a recurrent network that predicts the future values
        of a time-dependent variable based on past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim)

        self.lstm = nn.LSTM(input_size=1 + params.cov_dim +
                                       params.embedding_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommanded by
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()

        self.spline_pre_u = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, params.num_spline)
        self.spline_pre_beta = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, params.num_spline)
        self.spline_gama = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, 1)
        self.spline_u = nn.Softmax(dim=1)
        self.spline_beta = nn.Softplus()

    def forward(self, x, idx, hidden, cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM c from time step t
        '''
        onehot_embed = self.embedding(idx)
        # TODO: is it possible to do this only once per
        #  window instead of per step?
        lstm_input = torch.cat((x, onehot_embed), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = \
            hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)

        #### settings of beta distribution
        pre_u = self.spline_pre_u(hidden_permute)
        spline_u = self.spline_u(pre_u)  # sigmax to make sure Σu equals to 1
        pre_beta = self.spline_pre_beta(hidden_permute)
        spline_beta = self.spline_beta(pre_beta)
        # softplus to make sure the intercept is positive
        spline_gama = self.spline_gama(hidden_permute)

        return torch.squeeze(spline_u), torch.squeeze(spline_beta), \
               torch.squeeze(spline_gama), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size,
                           self.params.lstm_hidden_dim,
                           device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size,
                           self.params.lstm_hidden_dim,
                           device=self.params.device)

    def test(self, x, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size,
                                  self.params.predict_steps,
                                  device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    u_de, beta_ge, gama_de, decoder_hidden, decoder_cell = \
                        self(x[self.params.predict_start + t].unsqueeze(0),
                        id_batch, decoder_hidden, decoder_cell)
                    uniform = torch.distributions.uniform.Uniform(
                        torch.tensor([0.0], device='cuda:0'),
                        torch.tensor([1.0], device='cuda:0'))
                    pred = torch.squeeze(uniform.sample([batch_size]))
                    b = beta_ge - torch.nn.functional.pad(
                        beta_ge, (1, 0), 'constant', 0)[:, :-1]
                    d = torch.cumsum(
                        torch.nn.functional.pad(u_de, (1, 0), 'constant', 0),
                        dim=1)
                    d = d[:, :-1]
                    bd = b*d
                    ind = d[:, 1:].permute(1,0) < pred
                    ind = ind.permute(1,0)
                    k = torch.sum(ind*beta_ge[:,1:], dim=1) + beta_ge[:,0]
                    pred = \
                        (gama_de + torch.sum(ind*bd[:,1:], dim=1) + bd[:,0])/k
                    samples[j, :, t] = pred
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.mean(samples, dim=0)  # mean or median ?
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        # else:
        #     decoder_hidden = hidden
        #     decoder_cell = cell
        #     sample_p = torch.zeros(batch_size, self.params.predict_steps,
        #                            device=self.params.device)
        #     sample_gama = torch.zeros(batch_size, self.params.predict_steps,
        #                               device=self.params.device)
        #     for t in range(self.params.predict_steps):
        #         p_de, gama_de, decoder_hidden, decoder_cell = self(
        #             x[self.params.predict_start + t].unsqueeze(0),
        #             id_batch, decoder_hidden, decoder_cell)
        #         sample_p[:, t] = p_de
        #         sample_gama[:, t] = gama_de
        #         if t < (self.params.predict_steps - 1):
        #             x[self.params.predict_start + t + 1, :, 0] = p_de
        #     return sample_p, sample_gama


def loss_fn_crps(u: Variable, beta: Variable, gama: Variable, labels: Variable):
    zero_index = (labels != 0)
    knots = torch.cumsum(u*beta,dim=1).permute(1,0) + gama
    knots = knots[:-1,:]
    ind = knots < labels
    ind = ind.permute(1,0)
    b = beta - torch.nn.functional.pad(beta,(1,0),'constant',0)[:,:-1]
    d = torch.cumsum(torch.nn.functional.pad(u,(1,0),'constant',0),dim=1)
    d = d[:,:-1]
    bd = b*d
    denom = torch.sum(ind*b[:,1:],dim=1) + b[:,0]
    pnom = torch.sum(ind*bd[:,1:],dim=1) + bd[:,0]
    nom = labels - gama + pnom
    alpha_cu = nom/denom

    max_ad = d + (alpha_cu > d.permute(1,0)).permute(1,0) * \
             (alpha_cu - d.permute(1,0)).permute(1,0)
    crps = (2*alpha_cu - 1) * labels + (1 - 2*alpha_cu) * gama + \
           torch.sum(b*((1-d*d*d)/3 - d + max_ad * (2*d - max_ad)), dim=1)
    crps = torch.mean(crps)
    return crps


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        # summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, torch.sum(zero_index).item()]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    # zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu - labels), (mu - labels)),
                     dim=0).cpu().detach().numpy()
    # if relative:
    #     return [diff, torch.sum(zero_index).item(), torch.sum(
    #     zero_index).item()]
    # else:
    #     summation = torch.sum(torch.abs(labels[zero_index])).item()
    #     if summation == 0:
    #         logger.error('summation denominator error! ')
    #     return [diff, summation, torch.sum(zero_index).item()]
    diff = np.append(diff, labels.shape[0])
    return diff


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor,
                 relative=False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[
                           0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[
                labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[
                    labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]


def accuracy_CRPS(samples: torch.Tensor, labels: torch.Tensor):
    samples_permute = samples.permute(1, 2, 0)
    numerator = ps.crps_ensemble(labels.cpu().detach().numpy(),
                                 samples_permute.cpu().detach().numpy()).sum(
        axis=0)
    denominator = labels.shape[0]
    return np.append(numerator, denominator)


def accuracy_SHA(samples: torch.Tensor):
    samples = samples.cpu().detach().numpy()
    q5 = np.percentile(samples, 5, axis=0)
    q95 = np.percentile(samples, 95, axis=0)
    sharp90 = (q95 - q5).sum(axis=0)

    q25 = np.percentile(samples, 25, axis=0)
    q75 = np.percentile(samples, 75, axis=0)
    sharp50 = (q75 - q25).sum(axis=0)

    return np.append(sharp50, np.append(sharp90, samples.shape[1]))


def accuracy_RC(samples: torch.Tensor, labels: torch.Tensor):
    empi_freq = np.zeros((4, 10))
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    for i in range(samples.shape[1]):
        for j in range(samples.shape[2]):
            ecdf = ECDF(samples[:, i, j])
            prob_ob = ecdf(labels[i, j])
            if prob_ob < 0.5:
                empi_freq[j, int(100 * (0.5 - prob_ob) // 5):] += 1
            else:
                empi_freq[j, int(100 * (prob_ob - 0.5) // 5):] += 1
    empi_freq[:, -1] = labels.size / 4
    return empi_freq


def accuracy_RH(samples: torch.Tensor, labels: torch.Tensor):
    return (torch.sum(samples < labels, dim=0).cpu().detach().numpy()) // 10 + 1


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor,
                  relative=False):
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    samples[:, mask] = 0.

    pred_samples = samples.shape[0]
    rou_th = math.floor(pred_samples * rou)

    samples = np.sort(samples, axis=0)
    rou_pred = samples[rou_th]

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(
        abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)

    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result
