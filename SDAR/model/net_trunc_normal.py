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

        # self.distribution_prep = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        # self.distribution_pregama = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        # self.distribution_p = nn.Sigmoid()
        # self.distribution_gama = nn.Softplus()
        self.distribution_mu = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_pre_sigma = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()

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
        # onehot_embed = self.embedding(idx)
        # lstm_input = torch.cat((x, onehot_embed), dim=2)
        # output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)

        #### settings of beta distribution
        # pre_p = self.distribution_prep(hidden_permute)
        # p = self.distribution_p(pre_p) # sigmoid to make sure p value in [0, 1]
        # pre_gama = self.distribution_pregama(hidden_permute)
        # gama = self.distribution_gama(pre_gama)  # softplus to make sure standard deviation is positive

        mu = self.distribution_mu(hidden_permute)
        pre_sigma = self.distribution_pre_sigma(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)

        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

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
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                       id_batch, decoder_hidden, decoder_cell)
                    # gama_de[(p_de * gama_de < 1) & (gama_de < 2)] = 3
                    normal = torch.distributions.normal.Normal(mu_de, sigma_de)
                    uniform = torch.distributions.uniform.Uniform(torch.zeros_like(mu_de), torch.ones_like(mu_de))
                    pred_t = uniform.sample()
                    phi0 = normal.cdf(torch.tensor(0.0, device=mu_de.device))
                    phi1 = normal.cdf(torch.tensor(1.0, device=mu_de.device))
                    pred = np.sqrt(2) * sigma_de * torch.erfinv(2*(pred_t*(phi1 - phi0) + phi0) - 1) + mu_de
                    # pred = normal.sample()  # not scaled
                    while torch.isinf(pred).sum() != 0:  # detect possible inf values
                        ind = torch.isinf(pred)
                        pred_t = uniform.sample()
                        pred[ind] = np.sqrt(2)*sigma_de[ind] * \
                                    torch.erfinv(2*(pred_t[ind]*(phi1[ind]-phi0[ind])+phi0[ind])-1) + mu_de[ind]
                    samples[j, :, t] = pred
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.mean(samples, dim=0)  # mean or median ?
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                   id_batch, decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de
                sample_sigma[:, t] = sigma_de
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma


def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(mu[zero_index],
                                                 sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index]) - \
                 torch.log(1e-10 + distribution.cdf(torch.tensor(1.0,device=mu.device)) -
                           distribution.cdf(torch.tensor(0,device=mu.device)))
    x = -torch.mean(likelihood)
    if torch.isnan(x):
        print('likelihood:')
        print(likelihood.cpu().detach().numpy())
        print('gama:')
        print(sigma.cpu().detach().numpy())
        print('p:')
        print(mu.cpu().detach().numpy())
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

def loss_fn_crps(x0: Variable, sigma: Variable, labels: Variable):
    zeros_index = (labels != 0)
    norm_labels = (labels - x0)/sigma
    normal = torch.distributions.normal.Normal(
        torch.zeros_like(x0), torch.ones_like(sigma))
    crps = sigma*(norm_labels*(2*normal.cdf(norm_labels)-1) +
                  2*torch.exp(normal.log_prob(norm_labels))-1/np.sqrt(np.pi))
    return crps.mean()







