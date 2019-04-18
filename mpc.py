import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
from math import sin, cos, radians, pi
import matplotlib.pyplot as plt
import multiprocessing as mp
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import PropNet
from data import load_data, normalize, denormalize
from data import construct_fully_connected_rel, construct_Rope_rel
from utils import count_parameters, to_var, to_np, Tee, ChamferLoss, rand_float
from utils import render_Rope, render_Box

from physics_engine import RopeEngine, BoxEngine


parser = argparse.ArgumentParser()
parser.add_argument('--pn_mode', default='full')
parser.add_argument('--pstep', type=int, default=3)
parser.add_argument('--pstep_encode', type=int, default=-1)
parser.add_argument('--pstep_decode', type=int, default=-1)
parser.add_argument('--epoch', type=int, default=-1)
parser.add_argument('--iter', type=int, default=-1)

parser.add_argument('--n_particle', type=int, default=0)
parser.add_argument('--roll_step', type=int, default=0)
parser.add_argument('--dt', type=float, default=1. / 50.)

parser.add_argument('--nf_relation', type=int, default=150)
parser.add_argument('--nf_particle', type=int, default=100)
parser.add_argument('--nf_effect', type=int, default=100)
parser.add_argument('--agg_method', default='sum', help='the method for aggregating the particle representations, sum|mean')

parser.add_argument('--env', default='')
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--mpcf', default='mpc')

parser.add_argument('--len_seq', type=int, default=2)
parser.add_argument('--history_window', type=int, default=1)

parser.add_argument('--lr', type=float, default=3.)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--optim_iter_init', type=int, default=100)
parser.add_argument('--optim_iter', type=int, default=10)

parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)

parser.add_argument('--act_scale_min', type=float, default=-np.inf)
parser.add_argument('--act_scale_max', type=float, default=np.inf)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# action:
parser.add_argument('--action_dim', type=int, default=0)

# relation:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()



if args.env == 'Rope':
    args.pn_mode = 'full'
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
    args.mpcf = 'dump_Rope/' + args.mpcf

    args.act_scale_min = -20.
    args.act_scale_max = 20.

elif args.env == 'Box':
    args.pn_mode = 'partial'
    args.n_particle = 20
    args.dt = 1./50.
    data_names = ['states', 'actions', 'vis']
    lim = [-600, 600, -15, 400]

    # attr [none] - placeholder
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
    args.mpcf = 'dump_Box/' + args.mpcf

    args.act_scale_min = -600.
    args.act_scale_max = 100.

else:
    raise AssertionError("Unsupported env")


# make names for log dir and data dir
args.outf = args.outf + '_' + args.env
args.mpcf = args.mpcf + '_' + args.env
if args.env == 'Box':
    args.outf += '_pstep_' + str(args.pstep_encode) + '_' + str(args.pstep_decode)
    args.outf += '_hisWindow_' + str(args.history_window)
    args.outf += '_lenSeq_' + str(args.len_seq)
    args.mpcf += '_pstep_' + str(args.pstep_encode) + '_' + str(args.pstep_decode)
    args.mpcf += '_hisWindow_' + str(args.history_window)
    args.mpcf += '_lenSeq_' + str(args.len_seq)
else:
    args.outf += '_pstep_' + str(args.pstep)
    args.mpcf += '_pstep_' + str(args.pstep)

args.dataf = 'data/' + args.dataf + '_' + args.env
os.system('mkdir -p ' + args.mpcf)

log_path = os.path.join(args.mpcf, 'log.txt')
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


# for Rope, preprocess relations
if args.env == 'Rope':
    Rr, Rs, Ra = construct_Rope_rel(15, args.relation_dim, use_gpu)
    Rr_batch = Rr[None, :, :]
    Rs_batch = Rs[None, :, :]
    Ra_batch = Ra[None, :, :]


def gen_attr_Rope(engine, args):
    # construct attributes
    n_particle = args.n_particle
    attr = np.zeros((n_particle + 2, args.attr_dim))
    attr[0, 1] = 1
    attr[1:n_particle, 0] = 1
    attr[n_particle:, 1] = 1
    attr[:n_particle, 2] = engine.radius
    attr[n_particle:, 2] = engine.c_radius
    assert np.sum(attr[:, 0]) == 14
    assert np.sum(attr[:, 1]) == 3
    return attr


def generate_Rope_goal(args, video=True, image=False):
    engine_goal = RopeEngine(args.dt, args.state_dim, args.action_dim)
    scene_ckp = engine_goal.reset_scene(args.n_particle)

    act_scale = 12

    states_rec = np.zeros((args.roll_step, args.n_particle + 2, args.state_dim))
    act = np.zeros((args.n_particle, args.action_dim))

    for t in range(args.roll_step):
        f = np.zeros(args.action_dim)
        for k in range(args.n_particle):
            f += (np.random.rand(args.action_dim) * 2 - 1) * act_scale
            act[k] = f

        states_rec[t, :args.n_particle] = engine_goal.get_state()
        states_rec[t, args.n_particle:, :args.position_dim] = engine_goal.c_positions

        engine_goal.set_action(action=act)
        engine_goal.step()
        # print('t', t, engine_goal.get_states())

    state_goal = np.zeros((args.n_particle + 2, args.state_dim))
    state_goal[:args.n_particle] = engine_goal.get_state()
    state_goal[args.n_particle:, :args.position_dim] = engine_goal.c_positions
    assert state_goal.shape[0] == args.n_particle + 2
    assert state_goal.shape[1] == args.state_dim
    print('states_goal', state_goal.shape)

    attr = gen_attr_Rope(engine_goal, args)
    render_Rope(args.mpcf, 'mpc_Rope_goal', lim, attr, states_rec, video=video, image=image)

    return state_goal, scene_ckp


def generate_Box_goal(args, video=True, image=False):
    engine_goal = BoxEngine(args.dt, args.state_dim, args.action_dim)
    scene_ckp = engine_goal.reset_scene(args.n_particle)

    states_rec = np.zeros((args.roll_step, args.n_particle, args.state_dim))
    actions_rec = np.zeros((args.roll_step, 1, args.action_dim))
    viss_rec = np.zeros((args.roll_step, args.n_particle))

    for t in range(args.roll_step):
        engine_goal.set_action(rand_float(-600., 100.))

        states_rec[t] = engine_goal.get_state()
        actions_rec[t] = engine_goal.get_action()
        viss_rec[t] = engine_goal.get_vis(states_rec[t])

        engine_goal.step()

    state_goal_full = engine_goal.get_state()
    vis_goal_full = engine_goal.get_vis(state_goal_full)
    assert state_goal_full.shape[0] == args.n_particle
    assert state_goal_full.shape[1] == args.state_dim
    print('states_goal_full', state_goal_full.shape)

    render_Box(args.mpcf, 'mpc_Box_goal', lim, states_rec, actions_rec, viss_rec, video=video, image=image)

    return state_goal_full[vis_goal_full], state_goal_full, vis_goal_full, scene_ckp


# map state to latent space, used for partially observable environments
def encode_partial(state, var=False):
    if state.dtype == np.float:
        state = normalize([state], [stat[0]])[0]
    else:
        state = normalize([state], [stat[0]], var=True)[0]

    Rr_idx, Rs_idx, values, Ra = construct_fully_connected_rel(state.size(1), args.relation_dim)
    d = [state, Rr_idx, Rs_idx, values, Ra]
    d = [x.cuda() if use_gpu else x for x in d]
    state, Rr_idx, Rs_idx, values, Ra = d

    Rr = torch.sparse.FloatTensor(
        Rr_idx, values, torch.Size([state.size(1), Ra.size(0)]))
    Rs = torch.sparse.FloatTensor(
        Rs_idx, values, torch.Size([state.size(1), Ra.size(0)]))
    Ra = Ra[None, :, :]

    with torch.set_grad_enabled(var):
        encode = model.encode([state, Rr, Rs, Ra], args.pstep_encode)
        latent = model.to_latent(encode)

    return latent


# generate init and goal positions
print("Prepare initial and goal configurations")
if args.env == 'Rope':
    state_goal, scene_ckp = generate_Rope_goal(args)
    state_goal_v = to_var(state_goal, use_gpu=use_gpu)[None, :, :]

    engine = RopeEngine(args.dt, args.state_dim, args.action_dim)
    engine.reset_scene(args.n_particle, ckp=scene_ckp)

    # construct attributes
    attr = gen_attr_Rope(engine, args)

    # normalize attr and change to torch variable
    attr_normalized = normalize([attr], [stat[0]])[0]
    attr_normalized = to_var(attr_normalized, use_gpu)[None, :, :]

    states_roll = np.zeros((args.roll_step, args.n_particle + 2, args.state_dim))
    actions_roll = np.zeros((args.roll_step, args.n_particle + 2, args.action_dim))

    control_v = to_var(actions_roll, use_gpu, requires_grad=True)

elif args.env == 'Box':
    state_goal, state_goal_full, vis_goal_full, scene_ckp = generate_Box_goal(args)
    state_goal_v = to_var(state_goal, use_gpu=use_gpu)[None, :, :]
    latent_goal = encode_partial(state_goal_v)

    engine = BoxEngine(args.dt, args.state_dim, args.action_dim)
    ckp = engine.reset_scene(args.n_particle, ckp=scene_ckp)

    states_roll = np.zeros((args.roll_step, args.n_particle, args.state_dim))
    actions_roll = np.zeros((args.roll_step, 1, args.action_dim))
    viss_roll = np.zeros((args.roll_step, args.n_particle))

    # !!! need tuning
    sample_force = rand_float(-600., -450.)
    sample_force = -300.
    sample_force = -600.
    control = np.ones(args.roll_step) * sample_force
    control_v = to_var(control, use_gpu, requires_grad=True)

else:
    raise AssertionError("Unsupported env")


criterionMSE = nn.MSELoss()
criterionChamfer = ChamferLoss()
optimizer = optim.Adam([control_v], lr=args.lr, betas=(args.beta1, 0.999))

for step in range(args.roll_step):

    print("Step: %d / %d" % (step, args.roll_step))


    if args.env == 'Rope':
        states_roll[step, :args.n_particle] = engine.get_state()
        states_roll[step, args.n_particle:, :args.position_dim] = engine.c_positions
        state_cur = states_roll[step]

    elif args.env == 'Box':
        states_roll[step] = engine.get_state()
        state_cur = states_roll[step]
        viss_roll[step] = engine.get_vis(state_cur)

        state_cur_v = state_cur[viss_roll[step].astype(np.bool)][np.newaxis, :, :]
        state_cur_v = to_var(state_cur_v, use_gpu)

        latent_cur = encode_partial(state_cur_v)
        assert latent_cur.size(0) == 1
        assert latent_cur.size(1) == 1
        assert latent_cur.size(2) == args.nf_effect

        # print('latent_cur size', latent_cur.size())

        if step == 0:
            action_cur = to_var(engine.get_action()[np.newaxis, :, :], use_gpu)
            latents = torch.cat([latent_cur] * args.history_window, 2)
            actions = torch.cat([action_cur] * args.history_window, 2)
        else:
            latents = torch.cat([latents[:, :, args.nf_effect:], latent_cur], 2)

    optim_iter = args.optim_iter_init if step == 0 else args.optim_iter

    for i in range(optim_iter):

        # rollout and calculate distance to goal
        if args.env == 'Rope':
            state_cur_v = to_var(state_cur, use_gpu)[None, :, :]

            for j in range(step, args.roll_step):
                action_cur = control_v[j:j+1]
                action_cur_normalized = normalize([action_cur], [stat[2]], var=True)[0]
                state_cur_normalized = normalize([state_cur_v], [stat[1]], var=True)[0]

                # print(attr_normalized.size(), state_cur_normalized.size(), action_cur_normalized.size())
                with torch.set_grad_enabled(True):
                    pred = model([attr_normalized, state_cur_normalized, Rr_batch, Rs_batch, Ra_batch],
                                 args.pstep, action=action_cur_normalized)

                stat_vel = stat[1][args.position_dim:, :]
                pred = denormalize([pred], [stat_vel], var=True)[0]
                assert pred.requires_grad
                state_cur_v[:, :, :args.position_dim] = state_cur_v[:, :, :args.position_dim] + pred * args.dt
                state_cur_v[:, :, args.position_dim:] = pred

            loss = criterionMSE(
                state_cur_v[:, :args.n_particle, :args.position_dim],
                state_goal_v[:, :args.n_particle, :args.position_dim])

        elif args.env == 'Box':
            # print('actions', actions)
            # print('latents', latents)
            actions_cur = actions.clone()
            latents_cur = latents.clone()

            for j in range(step, args.roll_step):
                action_cur = torch.cat([actions_cur[0, 0, -2:-1] + control_v[j:j+1] * args.dt, control_v[j:j+1]])
                actions_cur = torch.cat([actions_cur[:, :, args.action_dim:], action_cur[None, None, :]], 2)
                actions_cur_normalized = normalize([actions_cur], [stat[1]], var=True)[0]

                with torch.set_grad_enabled(True):
                    latent_pred = model.rollout(latents_cur, actions_cur_normalized)

                latents_cur = torch.cat([latents_cur[:, :, args.nf_particle:], latent_pred], 2)
                assert latents_cur.requires_grad

            loss = criterionMSE(latent_pred, latent_goal)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        # print('control', control_v)
        # print('control.grad', control_v.grad)
        optimizer.step()

        # clamp the control input
        if args.env == 'Rope':
            control_v.data[:, :args.n_particle-2].clamp_(min=0., max=0.)
            control_v.data[:, args.n_particle:].clamp_(min=0., max=0.)

            control_v.data[:, args.n_particle-2:args.n_particle].clamp_(
                min=args.act_scale_min, max=args.act_scale_max)

        elif args.env == 'Box':
            control_v.data.clamp_(min=args.act_scale_min, max=args.act_scale_max)

        if i % 5 == 0:
            print("  Iter %d / %d: Loss %.6f" % (i, optim_iter, np.sqrt(loss.item())))

    engine.set_action(to_np(control_v[step]))
    actions_roll[step] = engine.get_action()

    if args.env == 'Box':
        actions = torch.cat([
            actions[:, :, args.action_dim:],
            to_var(actions_roll[step], use_gpu)[None, :, :]], 2)

    engine.step()


# calculate distance and render
if args.env == 'Rope':
    state_init = states_roll[0]
    state_cur = states_roll[args.roll_step - 1]
    print("L2 Distance Init:", np.sqrt(criterionMSE(
        to_var(state_init[:args.n_particle, :args.position_dim], use_gpu),
        state_goal_v[:, :args.n_particle, :args.position_dim]).item()))
    print("L2 Distance Final:", np.sqrt(criterionMSE(
        to_var(state_cur[:args.n_particle, :args.position_dim], use_gpu),
        state_goal_v[:, :args.n_particle, :args.position_dim]).item()))

    states_goal = state_goal[np.newaxis, ...].repeat(args.roll_step, 0)

    render_Rope(args.mpcf, 'mpc_Rope', lim, attr, states_roll, actions=actions_roll,
                states_gt=states_goal, video=True, image=True)

elif args.env == 'Box':
    ob_init = states_roll[0, viss_roll[0].astype(np.bool)]
    ob_cur = states_roll[args.roll_step - 1, viss_roll[args.roll_step-1].astype(np.bool)]
    ob_goal = state_goal
    print("ChamferLoss Init:", criterionChamfer(
        to_var(ob_init, use_gpu), to_var(ob_goal, use_gpu)).item())
    print("ChamferLoss Final:", criterionChamfer(
        to_var(ob_cur, use_gpu), to_var(ob_goal, use_gpu)).item())

    states_goal = state_goal_full[np.newaxis, ...].repeat(args.roll_step, 0)
    viss_gt = vis_goal_full.astype(np.bool)[np.newaxis, ...].repeat(args.roll_step, 0)

    render_Box(args.mpcf, 'mpc_Box', lim, states_roll, actions_roll, viss_roll,
               states_gt=states_goal, vis_gt=viss_gt, video=True, image=True)

