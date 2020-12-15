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
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.params = params
        # self.embedding = nn.Embedding(params.num_class, params.embedding_dim)

        self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
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
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()

        self.distribution_pre_theta = \
            nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_pre_reci_n = \
            nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_theta = nn.Sigmoid()
        self.distribution_reci_n = nn.Sigmoid()

    def forward(self, x, idx, hidden, cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        # onehot_embed = self.embedding(idx)  # TODO: is it possible to do this only once per window instead of per step?
        # lstm_input = torch.cat((x, onehot_embed), dim=2)
        # output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        while torch.isnan(hidden).sum() != 0:  # detect possible inf values
            print('nihao')
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)

        #### settings of beta distribution
        pre_theta = self.distribution_pre_theta(hidden_permute)
        theta = self.distribution_theta(pre_theta)
        # sigmoid to make sure theta value in [0, 1]
        pre_reci_n = self.distribution_pre_reci_n(hidden_permute)
        reci_n = self.distribution_reci_n(pre_reci_n)
        # sigmoid to make sure reci_n value in [0, 1]


        return torch.squeeze(theta), torch.squeeze(reci_n), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def test(self, x, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps,
                                  device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    theta_de, reci_n_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                       id_batch, decoder_hidden, decoder_cell)
                    uniform = torch.distributions.uniform.Uniform(
                        torch.tensor([0.0], device=self.params.device),
                        torch.tensor([1.0], device=self.params.device))
                    pred_cdf = torch.squeeze(uniform.sample([batch_size]))
                    pred = torch.zeros_like(pred_cdf)
                    ind = pred_cdf < theta_de
                    pred[ind] = theta_de[ind] * torch.pow(
                        pred_cdf[ind]/theta_de[ind], reci_n_de[ind])
                    pred[~ind] = \
                        1 - (1 - theta_de[~ind]) * \
                        torch.pow((1 - pred_cdf[~ind]) / (1 - theta_de[~ind]),
                                  reci_n_de[~ind])
                    samples[j, :, t] = pred
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.mean(samples, dim=0)  # mean or median ?
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_p = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_gama = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                p_de, gama_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                   id_batch, decoder_hidden, decoder_cell)
                sample_p[:, t] = p_de
                sample_gama[:, t] = gama_de
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = p_de
            return sample_p, sample_gama


def loss_fn(theta: Variable, reci_n: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    # zero_index = (labels != 0)
    ind = labels < theta
    n = torch.reciprocal(reci_n)
    adj_labels = torch.zeros_like(labels)
    adj_theta = torch.zeros_like(theta)
    adj_theta[ind] = theta[ind]
    adj_theta[~ind] = 1 - theta[~ind]
    adj_labels[ind] = labels[ind]
    adj_labels[~ind] = 1 - labels[~ind]
    likelihood = -torch.log(reci_n) + (n-1)*(torch.log(adj_labels)-torch.log(adj_theta))
    x = -torch.sum(likelihood)
    if torch.isnan(x):
        print('likelihood:')
        print(likelihood.cpu().detach().numpy())
        print('theta:')
        print(theta.cpu().detach().numpy())
        print('reci_n:')
        print(reci_n.cpu().detach().numpy())
        print('label:')
        print(labels.cpu().detach().numpy())
    return -torch.mean(likelihood)

def loss_fn_rou(x0: Variable, gama: Variable, labels: Variable):
    zero_index = (labels != 0)
    rou50_score = torch.mean(torch.abs(labels - x0)).item()
    distribution = torch.distributions.cauchy.Cauchy(x0[zero_index], gama[zero_index])
    rou75_diff = labels - distribution.icdf(torch.tensor(0.75))
    rou75_score = 2 * (0.75 * torch.sum(rou75_diff[rou75_diff > 0]) - 0.25 * torch.sum(
        rou75_diff[rou75_diff < 0])) / labels.numel()
    rou25_diff = labels - distribution.icdf(torch.tensor(.25))
    rou25_score = 2 * (0.25 * torch.sum(rou25_diff[rou25_diff > 0]) - 0.75 * torch.sum(
        rou25_diff[rou25_diff < 0])) / labels.numel()
    rou_score = (rou75_score + rou25_score + rou50_score) / 3
    return rou_score

def loss_fn_crps(theta: Variable, reci_n: Variable, labels: Variable):
    n = torch.reciprocal(reci_n)
    ind = labels < theta
    const_term = labels - theta
    const_term[ind] = -const_term[ind]
    quad_term = (torch.pow(theta, 3) + torch.pow(1-theta, 3))/(2*n+1)
    adj_labels = torch.zeros_like(labels)
    adj_theta = torch.zeros_like(theta)
    adj_theta[~ind] = 1 - theta[~ind]
    adj_theta[ind] = theta[ind]
    adj_labels[~ind] = 1 - labels[~ind]
    adj_labels[ind] = labels[ind]
    linear_term = 2*(torch.pow(adj_labels, n+1)/(torch.pow(adj_theta + 0.0001, n-1))
                     - torch.pow(adj_labels, 2))/(n+1)
    # linear_term = 2*(torch.pow(adj_labels, 2) - torch.pow(adj_labels, 2))/(n+1)
    crps = const_term + quad_term + linear_term
    return torch.mean(crps)








