# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from .layers import MultiHeadGraphAttention, GraphConvolution

import torch.nn.init as init
from torch.autograd import Variable
from math import exp

import pdb


class RWalk(object):
    def __init__(self, model, dataloader, epsilon, prev_guards=[None]):
        self.model = model
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.update_ewc_parameter = 0.4
        # extract model parameters and store in dictionary
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        # initialize the guidance matrix
        self._means = {}

        self.previous_guards_list = prev_guards

        # Generate Fisher (F) Information Matrix
        self._precision_matrices = self._calculate_importance_ewc()

        self._n_p_prev, self._n_omega = self._calculate_importance()
        self.W, self.p_old = self._init_()

    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}

        # if self.dataloader is not None:
        if False:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad and p.grad is not None:
                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                    W = getattr(self.model, '{}_W'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W / (1.0 / 2.0 * self._precision_matrices[n] * p_change**2 + self.epsilon)
                    try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = 0.5 * omega + 0.5 * omega_add
                    n_omega[n] = omega_new
                    n_p_prev[n] = p_current

                    # Store these new values in the model
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                    self.model.register_buffer('{}_SI_omega'.format(n), omega_new)

        else:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad and p.grad is not None:
                    n_p_prev[n] = p.detach().clone()
                    n_omega[n] = p.detach().clone().zero_()
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())
        return n_p_prev, n_omega

    def _calculate_importance_ewc(self):
        precision_matrices = {}
        for n, p in self.params.items():
            # initialize Fisher (F) matrix（all fill zero）
            n = n.replace('.', '__')
            precision_matrices[n] = p.clone().detach().fill_(0)
            for i in range(len(self.previous_guards_list)):
                if self.previous_guards_list[i]:
                    precision_matrices[n] += self.previous_guards_list[i][n]

        self.model.eval()
        if self.dataloader is not None:
            number_data = len(self.dataloader)
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    n = n.replace('.', '__')
                    precision_matrices[n].data *= (1 - self.update_ewc_parameter)
            for data in self.dataloader:
                self.model.zero_grad()
                loss, _ = self.model(data)
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        n = n.replace('.', '__')
                        precision_matrices[n].data += self.update_ewc_parameter * p.grad.data ** 2 / number_data

            precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad and p.grad is not None:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]
                # Generate regularization term  _loss by omega and Fisher Matrix
                _loss = (omega + self._precision_matrices[n]) * (p - prev_values) ** 2
                loss += _loss.sum()

        return loss

    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer('{}_W'.format(n), self.W[n])
                self.p_old[n] = p.detach().clone()
        return


class MAS(object):
    def __init__(self, model: nn.Module, dataloader, prev_guards=[None], ent_w_img=None):
        self.model = model
        self.dataloader = dataloader
        # extract all parameters in models
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        # pdb.set_trace()
        # initialize parameters
        self.p_old = {}
        # self.device = device
        self.ent_w_img = ent_w_img
        # save previous guards
        self.previous_guards_list = prev_guards

        # generate Omega(Ω) matrix for MAS
        self._precision_matrices = self.calculate_importance()

        # keep the old parameter in self.p_old
        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach()

    def calculate_importance(self):
        precision_matrices = {}
        # initialize Omega(Ω) matrix（all filled zero）
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)
            for i in range(len(self.previous_guards_list)):
                if self.previous_guards_list[i]:
                    precision_matrices[n] += self.previous_guards_list[i][n]

        self.model.eval()
        if self.dataloader is not None:
            num_data = 1
            num_p = 0
            avb_p = set()
            # for batch in self.dataloader:
            self.model.zero_grad()
            # output = self.model(data[0].cuda())
            # generate Omega(Ω) matrix for MAS.

            joint_emb_fz, weight_norm = self.model.joint_emb_generat()
            joint_emb_fz = joint_emb_fz[self.ent_w_img, :]
            weight_norm = weight_norm[self.ent_w_img, :]

            op1 = joint_emb_fz.pow(2)
            op2 = weight_norm.pow(2)
            loss1 = torch.sum(op1, dim=1)
            loss2 = torch.sum(op2, dim=1)
            loss = loss1.mean() + loss2.mean()

            loss.backward()
            ################################################################
            for name, param in self.model.named_parameters():
                # pdb.set_trace()
                if param.requires_grad and param.grad is not None:
                    num_p += 1
                    avb_p.add(name)
                    precision_matrices[name].data += param.grad.abs() / num_data
            torch.cuda.empty_cache()
            print(f"{num_p/num_data} param update:")
            print(f"{avb_p}")

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                _loss = self._precision_matrices[name] * (param - self.p_old[name]) ** 2

                loss += _loss.sum()
                # if _loss.sum() > 0 and "img_fc.weight" in name:
                #     pdb.set_trace()
        return loss

    def update(self, model):
        # do nothing
        return

# https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py


class AutomaticWeightedLoss(nn.Module):
    # '''
    # automatically weighted multi-task loss
    # Params£º
    #     num: int£¬the number of loss
    #     x: multi-task loss
    # Examples£º
    #     loss1=1
    #     loss2=2
    #     awl = AutomaticWeightedLoss(2)
    #     loss_sum = awl(loss1, loss2)
    # '''
    def __init__(self, num=2, args=None):
        super(AutomaticWeightedLoss, self).__init__()
        # if args is None or args.use_awl:
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        # else:
        # params = torch.ones(num, requires_grad=False)
        # self.params = torch.nn.Parameter(params, requires_grad=False)

    def forward(self, loss_list):
        loss_sum = 0
        # for i, loss in enumerate(x):
        for i in range(len(loss_list)):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss_list[i] + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)

            x = gat_layer(x, adj)

            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


""" vanilla GCN """


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class VAE(nn.Module):
    def __init__(self, embedding_size, hidden_size_list: list, mid_hidden):
        super(VAE, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size_list = hidden_size_list
        self.mid_hidden = mid_hidden

        self.enc_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden * 2]
        self.dec_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden]

        self.encoder = nn.ModuleList(
            [nn.Linear(self.enc_feature_size_list[i], self.enc_feature_size_list[i + 1]) for i in
             range(len(self.enc_feature_size_list) - 1)])
        self.decoder = nn.ModuleList(
            [nn.Linear(self.dec_feature_size_list[i], self.dec_feature_size_list[i - 1]) for i in
             range(len(self.dec_feature_size_list) - 1, 0, -1)])

    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            x = self.encoder[i](x)
            if i != len(self.encoder) - 1:
                # x=self.bn(x)
                x = F.relu(x)
        return x

    def decode(self, x):
        for i, layer in enumerate(self.decoder):
            x = self.decoder[i](x)
            # if i != len(self.decoder) - 1:
            #     x=self.bn(x)
            x = F.relu(x)
        return x

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # x = x.to(used_device)

        # import pdb
        # pdb.set_trace()

        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var

        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        hidden_norm = mu * torch.exp(sigma) ** 0.5  # var => std
        # autoencoder
        # hidden = torch.randn_like(sigma) + mu  # var => std
        # hidden_norm = mu  # var => std
        # VAE
        # hidden = self.reparameterize(mu, sigma)
        # hidden_norm = hidden
        # hidden_norm = hidden

        x_hat = self.decode(hidden)
        kl_div = 0.5 * torch.sum(torch.exp(sigma) + torch.pow(mu, 2) - 1 - sigma) / (x.shape[0] * x.shape[1])
        return hidden, hidden_norm, x_hat, kl_div

    def get_hidden(self, x):
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        return hidden


'''
In this class, input_shape is the shape of the input data, 
num_steps is the number of diffusion steps to perform, 
num_filters and num_hidden represent the number of filters and hidden units to use in the generator and encoder networks respectively.
fractal_steps controls how many times we should repeat our encodings at different diffusion steps.

The _diffuse function performs the stochastic diffusion process for a given input tensor x at a given time t.

In the forward method, we first encode the input x to a latent vector z using the encoder network. Then we repeat the encodings for fractal_steps times.

We then perform num_steps Gaussian diffusion steps, where for each step, we first diffuse the latent state z using our _diffuse function, perform the generator network for zand get the prediction z, and pass it through the Relu activation function. The resulting output is reshaped to be the same shape as the input data and returned as the final output.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch


class GDiffusionModel(nn.Module):
    def __init__(self, input_shape, num_hidden, num_steps=100, num_filters=32, fractal_steps=1):
        super(GDiffusionModel, self).__init__()

        self.num_steps = num_steps
        self.fractal_steps = fractal_steps
        self.input_shape = input_shape

        # Define the learned variance and shape parameters
        self.alpha = nn.Parameter(torch.ones(input_shape))
        self.sigma = nn.Parameter(torch.ones(input_shape))

        # Define the generator network
        self.generator = nn.Sequential(
            nn.Linear(num_hidden, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, input_shape)
        )

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, num_hidden)
        )

    def _diffuse(self, x, t):
        # Implements the stochastic diffusion process for the given input tensor x
        alpha_t = torch.exp(-0.5 * self.alpha * (t / self.num_steps))
        sigma_t = torch.exp(0.5 * self.sigma * (t / self.num_steps))
        noise = torch.randn_like(x)

        return x * alpha_t + (sigma_t * noise)

    def forward(self, x):
        # Implements the forward pass through the GDM
        z = self.encoder(x)
        z = z.repeat(self.fractal_steps, 1)
        for i in range(self.num_steps - 1, -1, -1):
            z = self._diffuse(z, i)
            if i == 0:
                break
            z = self.generator(z)
            z = F.relu(z)
        return z.view(-1, *self.input_shape)


def kl_divergence(mu, logvar):
    # total_kld = 0.5 * torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar) / (x.shape[0] * x.shape[1])
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
