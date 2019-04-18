import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from models import PropNet
from data import load_data, normalize, denormalize
from data import construct_fully_connected_rel, construct_Cradle_rel, construct_Rope_rel
from utils import to_var, count_parameters, to_np, Tee
from utils import render_Cradle, render_Rope, render_Box

parser = argparse.ArgumentParser()
parser.add_argument('--pn_mode', default='full')
parser.add_argument('--pstep', type=int, default=3)
parser.add_argument('--pstep_encode', type=int, default=-1)
parser.add_argument('--pstep_decode', type=int, default=-1)
parser.add_argument('--epoch', type=int, default=-1)
parser.add_argument('--iter', type=int, default=-1)

parser.add_argument('--st_idx', type=int, default=0)
parser.add_argument('--ed_idx', type=int, default=10)

parser.add_argument('--n_particle', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--dt', type=float, default=1. / 50.)

parser.add_argument('--nf_relation', type=int, default=150)
parser.add_argument('--nf_particle', type=int, default=100)
parser.add_argument('--nf_effect', type=int, default=100)
parser.add_argument('--agg_method', default='sum', help='the method for aggregating the particle representations, sum|mean')

parser.add_argument('--env', default='')
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval_type', default='rollout', help='valid|rollout')

parser.add_argument('--len_seq', type=int, default=2)
parser.add_argument('--history_window', type=int, default=1)
parser.add_argument('--scheduler_factor', type=float, default=0.8)

parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# action:
parser.add_argument('--action_dim', type=int, default=0)

# relation:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()


if args.env == 'Cradle':
    args.pn_mode = 'full'
    args.time_step = 1000
    args.n_particle = 5
    args.dt = 0.001
    data_names = ['attrs', 'states']
    lim = 400

    # attr [ball, anchor]
    args.attr_dim = 2

    # state [x, y, xdot, ydot]
    args.state_dim = 4
    args.position_dim = 2

    # relation [None] - placeholder
    args.relation_dim = 1

    args.outf = 'dump_Cradle/' + args.outf
    args.evalf = 'dump_Cradle/' + args.evalf

elif args.env == 'Rope':
    args.pn_mode = 'full'
    args.time_step = 100
    args.n_particle = 15
    args.dt = 1./50.
    data_names = ['attrs', 'states', 'actions']
    lim = [0, 600, 0, 600]

    # attr [moving, fixed, radius]
    args.attr_dim = 3

    # state [x, y, xdot, ydot]
    args.state_dim = 4
    args.position_dim = 2

    # action [act_x, act_y]
    args.action_dim = 2

    # relation [collision, onehop, bihop]
    args.relation_dim = 3

    args.outf = 'dump_Rope/' + args.outf
    args.evalf = 'dump_Rope/' + args.evalf

elif args.env == 'Box':
    args.pn_mode = 'partial'
    args.time_step = 100
    args.n_particle = 20
    args.dt = 1./50.
    data_names = ['states', 'actions', 'vis']
    lim = [-600, 600, -15, 400]

    # attr [None] - placeholder
    args.attr_dim = 0

    # state [x, y, angle, xdot, ydot, angledot]
    args.state_dim = 6
    args.position_dim = 3

    # action [act_x, act_xdot]
    args.action_dim = 2

    # relation [None] - placeholder
    args.relation_dim = 1

    args.pstep_encode = 2
    args.pstep_decode = 1
    args.history_window = 5
    args.len_seq = 10

    args.outf = 'dump_Box/' + args.outf
    args.evalf = 'dump_Box/' + args.evalf

else:
    raise AssertionError("Unsupported env")

# make names for log dir and data dir
args.outf = args.outf + '_' + args.env
args.evalf = args.evalf + '_' + args.env
if args.env == 'Box':
    args.outf += '_pstep_' + str(args.pstep_encode) + '_' + str(args.pstep_decode)
    args.outf += '_hisWindow_' + str(args.history_window)
    args.outf += '_lenSeq_' + str(args.len_seq)
    args.evalf += '_pstep_' + str(args.pstep_encode) + '_' + str(args.pstep_decode)
    args.evalf += '_hisWindow_' + str(args.history_window)
    args.evalf += '_lenSeq_' + str(args.len_seq)
else:
    args.outf += '_pstep_' + str(args.pstep)
    args.evalf += '_pstep_' + str(args.pstep)

args.dataf = 'data/' + args.dataf + '_' + args.env
os.system('mkdir -p ' + args.evalf)

log_path = os.path.join(args.evalf, 'log.txt')
tee = Tee(log_path, 'w')
print(args)

# load stat
print("Loading stored stat from %s" % args.dataf)
stat_path = os.path.join(args.dataf, 'stat.h5')
stat = load_data(data_names, stat_path)

# use_gpu
use_gpu = torch.cuda.is_available()

# define model network
model = PropNet(args, residual=True, use_gpu=use_gpu)

# print model #params
print("model #params: %d" % count_parameters(model))

# load pretrained checkpoint
if args.epoch == -1:
    model_path = os.path.join(args.outf, 'net_best.pth')
else:
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

print("Loading saved ckp from %s" % model_path)
model.load_state_dict(torch.load(model_path))
model.eval()

if use_gpu:
    model.cuda()


# for Cradle and Rope, preprocess relations
if args.env == 'Cradle':
    Rr, Rs, Ra = construct_Cradle_rel(args.n_particle, args.relation_dim, use_gpu)
elif args.env == 'Rope':
    Rr, Rs, Ra = construct_Rope_rel(args.n_particle, args.relation_dim, use_gpu)

for idx in range(args.st_idx, args.ed_idx):

    print("Rollout %d / %d" % (idx, args.ed_idx))

    # ground truth
    for step in range(args.time_step):
        data_path = os.path.join(args.dataf, 'valid', str(idx), str(step) + '.h5')
        data_nxt_path = os.path.join(args.dataf, 'valid', str(idx), str(step + 1) + '.h5')

        data = load_data(data_names, data_path)
        data_nxt = load_data(data_names, data_path)

        if args.env == 'Cradle':
            attrs, states = data
            if step == 0:
                states_acc = states
                attrs_gt = np.zeros((args.time_step, attrs.shape[0], attrs.shape[1]))
                states_gt = np.zeros((args.time_step, states.shape[0], states.shape[1]))
                states_pred = np.zeros((args.time_step - 1, states.shape[0], states.shape[1]))
            else:
                d = args.position_dim
                states_acc[:, :d] = states_acc[:, :d] + states[:, d:] * args.dt
                states_acc[:, d:] = states[:, d:]

            attrs_gt[step] = attrs
            states_gt[step] = states_acc

        elif args.env == 'Rope':
            attrs, states, actions = data
            if step == 0:
                states_acc = states
                attrs_gt = np.zeros((args.time_step, attrs.shape[0], attrs.shape[1]))
                states_gt = np.zeros((args.time_step, states.shape[0], states.shape[1]))
                states_pred = np.zeros((args.time_step - 1, states.shape[0], states.shape[1]))
                actions_gt = np.zeros((args.time_step, actions.shape[0], actions.shape[1]))
            else:
                d = args.position_dim
                states_acc[:, :d] = states_acc[:, :d] + states[:, d:] * args.dt
                states_acc[:, d:] = states[:, d:]

            attrs_gt[step] = attrs
            states_gt[step] = states_acc
            actions_gt[step] = actions

        elif args.env == 'Box':
            states, actions, vis = data
            if step == 0:
                states_gt = np.zeros((args.time_step, states.shape[0], states.shape[1]))
                states_pred = np.zeros((args.time_step - 1, states.shape[0], states.shape[1]))
                actions_gt = np.zeros((args.time_step, actions.shape[0], actions.shape[1]))
                vis_gt = np.zeros((args.time_step, vis.shape[0]))

            states_gt[step] = states
            actions_gt[step] = actions
            vis_gt[step] = vis

        else:
            raise AssertionError("Unsupported env %s" % args.env)


    if args.env == 'Cradle':
        state_cur = states_gt[0].copy()

        for step in range(args.time_step - 1):
            states_pred[step] = state_cur.copy()

            data = normalize([attrs_gt[step], state_cur], stat)
            attr = to_var(data[0], use_gpu)[None, :, :]
            state = to_var(data[1], use_gpu)[None, :, :]

            Rr_batch = Rr[None, :, :]
            Rs_batch = Rs[None, :, :]
            Ra_batch = Ra[None, :, :]

            with torch.set_grad_enabled(False):
                pred = model([attr, state, Rr_batch, Rs_batch, Ra_batch], args.pstep)

            d = args.position_dim
            label = normalize([states_gt[step + 1]], [stat[1]])[0][:, d:]
            label = to_var(label, use_gpu)[None, :, :]

            loss = F.l1_loss(pred, label)

            print("roll step %d: loss: %.6f" % (step, loss.item()))

            d = args.position_dim
            if args.eval_type == 'rollout':
                state[:, :, d:] = pred
            elif args.eval_type == 'valid':
                state[:, :, d:] = label

            state = denormalize([state.data.cpu().numpy()], [stat[1]])[0]
            state_cur[:, :d] = state_cur[:, :d] + state[0, :, d:] * args.dt
            state_cur[:, d:] = state[0, :, d:]

        render_Cradle(args.evalf, 'eval_Cradle_%d' % idx, lim, states_pred, states_gt=states_gt, video=True, image=True)

    elif args.env == 'Rope':
        state_cur = states_gt[0].copy()

        for step in range(args.time_step - 1):
            states_pred[step] = state_cur.copy()

            data = normalize([attrs_gt[step].copy(), state_cur, actions_gt[step]], stat)
            attr = to_var(data[0], use_gpu)[None, :, :]
            state = to_var(data[1], use_gpu)[None, :, :]
            action = to_var(data[2], use_gpu)[None, :, :]

            Rr_batch = Rr[None, :, :]
            Rs_batch = Rs[None, :, :]
            Ra_batch = Ra[None, :, :]

            with torch.set_grad_enabled(False):
                pred = model([attr, state, Rr_batch, Rs_batch, Ra_batch], args.pstep, action=action)

            d = args.position_dim
            label = normalize([states_gt[step + 1]], [stat[1]])[0][:, d:]
            label = to_var(label, use_gpu)[None, :, :]

            loss = F.l1_loss(pred, label)

            print("roll step %d: loss: %.6f" % (step, loss.item()))

            d = args.position_dim
            if args.eval_type == 'rollout':
                state[:, :, d:] = pred
            elif args.eval_type == 'valid':
                state[:, :, d:] = label

            state = denormalize([state.data.cpu().numpy()], [stat[1]])[0]
            state_cur[:, :d] = state_cur[:, :d] + state[0, :, d:] * args.dt
            state_cur[:, d:] = state[0, :, d:]

        render_Rope(args.evalf, 'eval_Rope_%d' % idx, lim, attrs_gt[0], states_pred, states_gt=states_gt, video=True, image=True)

    elif args.env in ['Box']:
        latents = []
        encodes = []
        actions = []
        datas = []
        Rrs, Rss, Ras = [], [], []
        for step in range(args.time_step - 1):
            state, action = normalize([states_gt[step], actions_gt[step]], stat[:2])

            state = torch.FloatTensor(state[np.newaxis, vis_gt[step].astype(np.bool)])
            action = torch.FloatTensor(action[np.newaxis, ...])
            Rr_idx, Rs_idx, values, Ra = construct_fully_connected_rel(state.size(1), args.relation_dim)
            d = [state, action, Rr_idx, Rs_idx, values, Ra]
            datas.append([x.cuda() if use_gpu else x for x in d])
            state, action, Rr_idx, Rs_idx, values, Ra = datas[step]

            n_vis = state.size(1)

            Rr = torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([n_vis, Ra.size(0)]))
            Rs = torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([n_vis, Ra.size(0)]))
            Ra = Ra[None, :, :] # add batch dimension

            with torch.set_grad_enabled(False):
                encode = model.encode([state, Rr, Rs, Ra], args.pstep_encode)
                latent = model.to_latent(encode)

            Rrs.append(Rr)
            Rss.append(Rs)
            Ras.append(Ra)
            encodes.append(encode)
            latents.append(latent)
            actions.append(action)

        # decode loss
        losses_encode = []
        for step in range(args.time_step - 1):
            d = [encodes[step], Rrs[step], Rss[step], Ras[step]]
            with torch.set_grad_enabled(False):
                decode = model.decode(d, args.pstep_decode)
            decode_gt = datas[step][0]
            losses_encode.append(F.l1_loss(decode, decode_gt).item())

            states_pred[step] = states_gt[step]
            states_pred[step][vis_gt[step].astype(np.bool)] = \
                    denormalize([decode.data.cpu().numpy()], [stat[0]])[0]

        render_Box(
            args.evalf, 'eval_Box_%d' % idx, lim,
            states_pred, actions_gt, vis_gt, states_gt=states_gt, vis_gt=vis_gt,
            video=True, image=False)

        for step in range(args.time_step - 1):
            loss_roll = 0.

            # forward loss
            latent_roll = torch.cat(latents[step:step+args.history_window], 2)
            action_roll = torch.cat(actions[step:step+args.history_window], 2)
            for i in range(step+args.history_window, min(args.time_step - 1, step+args.len_seq)):
                with torch.set_grad_enabled(False):
                    latent_pred = model.rollout(latent_roll, action_roll)

                assert latent_pred.size(0) == 1
                assert latent_pred.size(1) == 1
                assert latent_pred.size(2) == args.nf_effect
                loss_roll += torch.abs(latent_pred - latents[i]).sum() / args.nf_effect

                if args.eval_type == 'rollout' and i == step + args.history_window:
                    latents[i] = latent_pred

                latent_roll = torch.cat([
                    latent_roll[:, :, args.nf_effect:], latent_pred], 2)
                action_roll = torch.cat([
                    action_roll[:, :, args.action_dim:], actions[i]], 2)

            loss_encode = np.sum(losses_encode[step:step + args.len_seq]) / args.len_seq
            loss_roll /= (args.len_seq - args.history_window)
            loss = loss_encode + loss_roll * 0.3

            print("roll step %d: loss: %.6f, encode: %.6f, roll: %.6f" % (step, loss, loss_encode, loss_roll))

