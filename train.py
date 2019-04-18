import os
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import PropNet
from data import PhysicsDataset, collate_fn
from data import construct_fully_connected_rel, construct_Cradle_rel, construct_Rope_rel

from utils import count_parameters, to_var, Tee, AverageMeter
from progressbar import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('--pn_mode', default='full', help='full|partial, whether to use full state or partial observation')
parser.add_argument('--pstep', type=int, default=3, help="propagation step")
parser.add_argument('--pstep_encode', type=int, default=-1, help='propagation step for encoding, used for partial')
parser.add_argument('--pstep_decode', type=int, default=-1, help='propagation step for decoding, used for partial')

parser.add_argument('--n_rollout', type=int, default=0, help="number of rollout")
parser.add_argument('--n_particle', type=int, default=0, help="number of particles")
parser.add_argument('--time_step', type=int, default=0, help="time step per rollout")
parser.add_argument('--dt', type=float, default=1./50., help="delta t between adjacent time step")

parser.add_argument('--nf_relation', type=int, default=150, help="dim of hidden layer of relation encoder")
parser.add_argument('--nf_particle', type=int, default=100, help="dim of hidden layer of object encoder")
parser.add_argument('--nf_effect', type=int, default=100, help="dim of propagting effect")
parser.add_argument('--agg_method', default='sum', help='the method for aggregating the particle representations, sum|mean')

parser.add_argument('--env', default='Rope', help="name of environment, Cradle|Rope|Box")
parser.add_argument('--outf', default='files', help="name of log dir")
parser.add_argument('--dataf', default='data', help="name of data dir")
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gen_data', type=int, default=0, help="whether to generate new data")
parser.add_argument('--gen_stat', type=int, default=0, help='whether to rengenerate statistics data')
parser.add_argument('--train_valid_ratio', type=float, default=0.85, help="percentage of training data")

parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--update_per_iter', type=int, default=-1, help="update the network params every x iter")
parser.add_argument('--log_per_iter', type=int, default=-1, help="print log every x iterations")
parser.add_argument('--ckp_per_iter', type=int, default=-1, help="save checkpoint every x iterations")
parser.add_argument('--eval', type=int, default=0, help="used for debugging")
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--history_window', type=int, default=1, help='history window used for partial')
parser.add_argument('--len_seq', type=int, default=2, help='train rollout length')
parser.add_argument('--scheduler_factor', type=float, default=0.8)
parser.add_argument('--scheduler_patience', type=float, default=0)

parser.add_argument('--resume_epoch', type=int, default=-1)
parser.add_argument('--resume_iter', type=int, default=-1)

parser.add_argument('--verbose_data', type=int, default=0, help="print debug information during data loading")
parser.add_argument('--verbose_model', type=int, default=0, help="print debug information during model forwarding")

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
    args.n_rollout = 2000
    args.time_step = 1000
    args.n_particle = 5
    args.dt = 0.001

    # attr [ball, anchor]
    args.attr_dim = 2

    # state [x, y, xdot, ydot]
    args.state_dim = 4
    args.position_dim = 2

    # relation [None] - placeholder
    args.relation_dim = 1

    args.batch_size = 32
    args.log_per_iter = 5000
    args.ckp_per_iter = 50000
    args.update_per_iter = 1
    args.len_seq = 2
    args.scheduler_patience = 2
    args.outf = 'dump_Cradle/' + args.outf

elif args.env == 'Rope':
    args.pn_mode = 'full'
    args.n_rollout = 5000
    args.time_step = 100
    args.n_particle = 15
    args.dt = 1./50.

    # attr [moving, fixed, radius]
    args.attr_dim = 3

    # state [x, y, xdot, ydot]
    args.state_dim = 4
    args.position_dim = 2

    # action [act_x, act_y]
    args.action_dim = 2

    # relation [collision, onehop, bihop]
    args.relation_dim = 3

    args.batch_size = 32
    args.log_per_iter = 2000
    args.ckp_per_iter = 10000
    args.update_per_iter = 1
    args.len_seq = 2
    args.scheduler_patience = 2
    args.outf = 'dump_Rope/' + args.outf

elif args.env == 'Box':
    args.pn_mode = 'partial'
    args.n_rollout = 5000
    args.time_step = 100
    args.n_particle = 20
    args.dt = 1./50.

    # attr [None] - placeholder
    args.attr_dim = 0

    # state [x, y, angle, xdot, ydot, angledot]
    args.state_dim = 6
    args.position_dim = 3

    # action [act_x, act_xdot]
    args.action_dim = 2

    # relation [None] - placeholder
    args.relation_dim = 1

    args.batch_size = 1
    args.log_per_iter = 200
    args.ckp_per_iter = 1000
    args.update_per_iter = 2
    args.pstep_encode = 2
    args.pstep_decode = 1
    args.history_window = 5
    args.len_seq = 10
    args.scheduler_patience = 2
    args.outf = 'dump_Box/' + args.outf

else:
    raise AssertionError("Unsupported env")

# make names for log dir and data dir
args.outf = args.outf + '_' + args.env
if args.env == 'Box':
    args.outf += '_pstep_' + str(args.pstep_encode) + '_' + str(args.pstep_decode)
    args.outf += '_hisWindow_' + str(args.history_window)
    args.outf += '_lenSeq_' + str(args.len_seq)
else:
    args.outf += '_pstep_' + str(args.pstep)

args.dataf = 'data/' + args.dataf + '_' + args.env
os.system('mkdir -p ' + args.outf)
os.system('mkdir -p ' + args.dataf)

# generate data
datasets = {phase: PhysicsDataset(args, phase) for phase in ['train', 'valid']}
for phase in ['train', 'valid']:
    if args.gen_data:
        datasets[phase].gen_data()
    else:
        datasets[phase].load_data()


use_gpu = torch.cuda.is_available()

if args.pn_mode == 'full':
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers)
        for x in ['train', 'valid']}
elif args.pn_mode == 'partial':
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=collate_fn)
        for x in ['train', 'valid']}


# define model network
model = PropNet(args, residual=True, use_gpu=use_gpu)

# print model #params
print("model #params: %d" % count_parameters(model))

# if resume from a pretrained checkpoint
if args.resume_epoch >= 0:
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
    print("Loading saved ckp from %s" % model_path)
    model.load_state_dict(torch.load(model_path))

# criterion
criterionMSE = nn.MSELoss()

# optimizer
params = model.parameters()
optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(
    optimizer, 'min',
    factor=args.scheduler_factor,
    patience=args.scheduler_patience,
    verbose=True)

if use_gpu:
    model = model.cuda()
    criterionMSE = criterionMSE.cuda()

st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf

log_fout = open(os.path.join(args.outf, 'log_st_epoch_%d.txt' % st_epoch), 'w')
tee = Tee(os.path.join(args.outf, 'train.log'), 'w')
print(args)


# for Cradle and Rope, preprocess relations
if args.env == 'Cradle':
    Rr, Rs, Ra = construct_Cradle_rel(args.n_particle, args.relation_dim, use_gpu)
elif args.env == 'Rope':
    Rr, Rs, Ra = construct_Rope_rel(args.n_particle, args.relation_dim, use_gpu)


for epoch in range(st_epoch, args.n_epoch):

    phases = ['train', 'valid'] if args.eval == 0 else ['valid']

    for phase in phases:

        model.train(phase == 'train')

        meter_loss = AverageMeter()

        if args.pn_mode == 'partial':
            meter_encode_loss = AverageMeter()
            meter_roll_loss = AverageMeter()

        bar = ProgressBar(max_value=len(dataloaders[phase]))

        loader = dataloaders[phase]

        for i, data in bar(enumerate(loader)):

            with torch.set_grad_enabled(phase == 'train'):

                if args.pn_mode == 'full':
                    if args.env == 'Cradle':
                        assert len(data) == 3
                        attr, state, label = [x.cuda() if use_gpu else x for x in data]

                        bs = attr.size(0)
                        Rr_batch = Rr[None, :, :].repeat(bs, 1, 1)
                        Rs_batch = Rs[None, :, :].repeat(bs, 1, 1)
                        Ra_batch = Ra[None, :, :].repeat(bs, 1, 1)

                        pred = model([attr, state, Rr_batch, Rs_batch, Ra_batch], args.pstep)

                    elif args.env == 'Rope':
                        assert len(data) == 4
                        attr, state, action, label = [x.cuda() if use_gpu else x for x in data]

                        bs = attr.size(0)
                        Rr_batch = Rr[None, :, :].repeat(bs, 1, 1)
                        Rs_batch = Rs[None, :, :].repeat(bs, 1, 1)
                        Ra_batch = Ra[None, :, :].repeat(bs, 1, 1)

                        pred = model([attr, state, Rr_batch, Rs_batch, Ra_batch], args.pstep, action=action)

                    # print('label shape', label.size())
                    loss = F.l1_loss(pred, label)

                elif args.pn_mode == 'partial':
                    assert len(data) == args.len_seq
                    bs = 1

                    encodes = []
                    latents = []
                    actions = []
                    Rrs = []
                    Rss = []
                    Ras = []

                    loss_encode = 0.
                    loss_roll = 0.

                    for step in range(args.len_seq):
                        data[step] = [x.cuda() if use_gpu else x for x in data[step]]
                        state, action, Rr_idx, Rs_idx, values, Ra = data[step]

                        n_vis = state.size(1)

                        Rr = torch.sparse.FloatTensor(
                            Rr_idx, values, torch.Size([n_vis, Ra.size(0)]))
                        Rs = torch.sparse.FloatTensor(
                            Rs_idx, values, torch.Size([n_vis, Ra.size(0)]))
                        Ra = Ra[None, :, :] # add batch dimension

                        encode = model.encode([state, Rr, Rs, Ra], args.pstep_encode)
                        latent = model.to_latent(encode)

                        Rrs.append(Rr)
                        Rss.append(Rs)
                        Ras.append(Ra)
                        encodes.append(encode)
                        latents.append(latent)
                        actions.append(action)

                    # add decode loss
                    for step in range(args.len_seq):
                        d = [encodes[step], Rrs[step], Rss[step], Ras[step]]
                        decode = model.decode(d, args.pstep_decode)
                        states_gt = data[step][0]
                        loss_encode += F.l1_loss(decode, states_gt)

                    # add forward loss
                    latent_roll = torch.cat(latents[:args.history_window], 2)
                    action_roll = torch.cat(actions[:args.history_window], 2)
                    for step in range(args.history_window, args.len_seq):
                        latent_pred = model.rollout(latent_roll, action_roll)
                        assert latent_pred.size(0) == 1
                        assert latent_pred.size(1) == 1
                        assert latent_pred.size(2) == args.nf_effect
                        loss_roll += torch.abs(latent_pred - latents[step]).sum() / args.nf_effect

                        latent_roll = torch.cat([
                            latent_roll[:, :, args.nf_effect:], latent_pred], 2)
                        action_roll = torch.cat([
                            action_roll[:, :, args.action_dim:], actions[step]], 2)

                    loss_encode /= args.len_seq
                    loss_roll /= (args.len_seq - args.history_window)
                    loss = loss_encode + loss_roll * 0.3

                    meter_encode_loss.update(loss_encode.item(), n=bs)
                    meter_roll_loss.update(loss_roll.item(), n=bs)

            '''prediction loss'''
            meter_loss.update(loss.item(), n=bs)

            if phase == 'train':
                if i % args.update_per_iter == 0:
                    # update parameters every args.update_per_iter
                    if i != 0:
                        loss_acc /= args.update_per_iter
                        optimizer.zero_grad()
                        loss_acc.backward()
                        optimizer.step()
                    loss_acc = loss
                else:
                    loss_acc += loss

            if i % args.log_per_iter == 0:
                if args.pn_mode == 'full':
                    log = '%s [%d/%d][%d/%d] Loss: %.6f (%.6f)' % (
                        phase, epoch, args.n_epoch, i, len(loader), loss.item(), meter_loss.avg)

                elif args.pn_mode == 'partial':
                    log = '%s [%d/%d][%d/%d] Loss: %.6f (%.6f), encode: %.6f (%.6f), roll: %.6f (%.6f)' % (
                        phase, epoch, args.n_epoch, i, len(loader), loss.item(), meter_loss.avg,
                        loss_encode.item(), meter_encode_loss.avg, loss_roll.item(), meter_roll_loss.avg)

                print()
                print(log)
                log_fout.write(log + '\n')
                log_fout.flush()

            if phase == 'train' and i % args.ckp_per_iter == 0:
                torch.save(model.state_dict(), '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

        log = '%s [%d/%d] Loss: %.4f, Best valid: %.4f' % (phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss)
        print(log)
        log_fout.write(log + '\n')
        log_fout.flush()

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg
                torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))


log_fout.close()
