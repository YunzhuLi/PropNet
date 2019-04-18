import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py

import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from physics_engine import CradleEngine, RopeEngine, BoxEngine
from utils import rand_float, init_stat, combine_stat, load_data, store_data
from progressbar import ProgressBar


def collate_fn(data):
    return data[0]


def construct_Cradle_rel(n_ball, relation_dim, use_gpu):
    rel = np.zeros((2 * n_ball + 2 * (n_ball - 1), 2))
    # relation between balls and anchors
    for i in range(n_ball):
        rel[i * 2, 0] = i
        rel[i * 2, 1] = i + n_ball
        rel[i * 2 + 1, 0] = i + n_ball
        rel[i * 2 + 1, 1] = i
    # relation between balls
    for i in range(n_ball - 1):
        rel[i * 2 + 2 * n_ball, 0] = i
        rel[i * 2 + 2 * n_ball, 1] = i + 1
        rel[i * 2 + 2 * n_ball + 1, 0] = i + 1
        rel[i * 2 + 2 * n_ball + 1, 1] = i

    n_rel = rel.shape[0]
    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    Ra = torch.FloatTensor(np.zeros((n_rel, relation_dim)))

    values = torch.ones(n_rel)

    Rr = torch.sparse.FloatTensor(
        Rr_idx, values, torch.Size([2 * n_ball, Ra.size(0)])).to_dense()
    Rs = torch.sparse.FloatTensor(
        Rs_idx, values, torch.Size([2 * n_ball, Ra.size(0)])).to_dense()

    if use_gpu:
        Rr, Rs, Ra = Rr.cuda(), Rs.cuda(), Ra.cuda()

    return Rr, Rs, Ra


def construct_Rope_rel(n_ball, relation_dim, use_gpu):
    rel = np.zeros((n_ball * 2 + n_ball * n_ball + (n_ball - 1) * 2 + (n_ball - 2) * 2, 2))
    n_rel = rel.shape[0]
    Ra = np.zeros((n_rel, relation_dim))

    # relation between rope and balls
    st_idx = 0
    for i in range(n_ball):
        rel[i * 2, 0] = i
        rel[i * 2, 1] = n_ball
        rel[i * 2 + 1, 0] = i
        rel[i * 2 + 1, 1] = n_ball + 1
    ed_idx = st_idx + n_ball * 2
    Ra[st_idx:ed_idx, 0] = 1
    st_idx = ed_idx

    # relation between rope particles
    rel[st_idx:st_idx + n_ball * n_ball, 0] = np.repeat(np.arange(n_ball), n_ball)
    rel[st_idx:st_idx + n_ball * n_ball, 1] = np.tile(np.arange(n_ball), n_ball)
    ed_idx = st_idx + n_ball * n_ball
    Ra[st_idx:ed_idx, 0] = 1
    st_idx = ed_idx

    # relation onehop
    for i in range(n_ball - 1):
        rel[st_idx + i * 2, 0] = i
        rel[st_idx + i * 2, 1] = i + 1
        rel[st_idx + i * 2 + 1, 0] = i + 1
        rel[st_idx + i * 2 + 1, 1] = i
    ed_idx = st_idx + (n_ball - 1) * 2
    Ra[st_idx:ed_idx, 1] = 1
    st_idx = ed_idx

    # relation bihop
    for i in range(n_ball - 2):
        rel[st_idx + i * 2, 0] = i
        rel[st_idx + i * 2, 1] = i + 2
        rel[st_idx + i * 2 + 1, 0] = i + 2
        rel[st_idx + i * 2 + 1, 1] = i
    ed_idx = st_idx + (n_ball - 2) * 2
    Ra[st_idx:ed_idx, 2] = 1

    assert (np.sum(Ra, 1) == np.ones(n_rel)).all()
    assert (np.sum(Ra, 0) == np.array([n_ball * 2 + n_ball * n_ball, (n_ball - 1) * 2, (n_ball - 2) * 2])).all()

    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    Ra = torch.FloatTensor(Ra)

    values = torch.ones(n_rel)

    Rr = torch.sparse.FloatTensor(
        Rr_idx, values, torch.Size([2 + n_ball, Ra.size(0)])).to_dense()
    Rs = torch.sparse.FloatTensor(
        Rs_idx, values, torch.Size([2 + n_ball, Ra.size(0)])).to_dense()

    if use_gpu:
        Rr, Rs, Ra = Rr.cuda(), Rs.cuda(), Ra.cuda()

    return Rr, Rs, Ra


def construct_fully_connected_rel(size, relation_dim):
    rel = np.zeros((size**2, 2))
    rel[:, 0] = np.repeat(np.arange(size), size)
    rel[:, 1] = np.tile(np.arange(size), size)

    n_rel = rel.shape[0]
    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    Ra = torch.FloatTensor(np.zeros((n_rel, relation_dim)))

    values = torch.ones(n_rel)

    return Rr_idx, Rs_idx, values, Ra


def gen_Cradle(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_particle, n_rollout, time_step = info['n_particle'], info['n_rollout'], info['time_step']
    dt, args = info['dt'], info['args']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim    # ball, anchor
    state_dim = args.state_dim  # x, y, xdot, ydot
    assert attr_dim == 2
    assert state_dim == 4

    lim = 300
    attr_dim = 2
    state_dim = 4
    relation_dim = 4

    stats = [init_stat(attr_dim), init_stat(state_dim)]

    engine = CradleEngine(dt)

    n_objects = n_particle * 2 # add the same number of anchor points
    attrs = np.zeros((n_rollout, time_step, n_objects, attr_dim))
    states = np.zeros((n_rollout, time_step, n_objects, state_dim))

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        theta = rand_float(0, 90)
        engine.reset_scene(n_particle, theta)

        for j in range(time_step):
            states[i, j] = engine.get_state()
            if j > 0:
                states[i, j, :, 2:] = (states[i, j, :, :2] - states[i, j - 1, :, :2]) / dt

            attrs[i, j, :n_particle, 0] = 1    # balls
            attrs[i, j, n_particle:, 1] = 1    # anchors

            data = [attrs[i, j], states[i, j]]
            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        datas = [attrs[i].astype(np.float64), states[i].astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats


def gen_Rope(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, n_particle, time_step = info['n_rollout'], info['n_particle'], info['time_step']
    dt, args = info['dt'], info['args']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim        # fixed, moving, radius
    state_dim = args.state_dim      # x, y, xdot, ydot
    action_dim = args.action_dim    # xddot, yddot
    assert attr_dim == 3
    assert state_dim == 4
    assert action_dim == 2

    act_scale = 15

    # attr, state, action
    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    engine = RopeEngine(dt, state_dim, action_dim)

    attrs = np.zeros((n_rollout, time_step, n_particle + 2, attr_dim))
    states = np.zeros((n_rollout, time_step, n_particle + 2, state_dim))
    actions =  np.zeros((n_rollout, time_step, n_particle + 2, action_dim))

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        engine.reset_scene(n_particle)

        act = np.zeros((n_particle, action_dim))
        for j in range(time_step):

            f = np.zeros(action_dim)
            for k in range(n_particle):
                f += (np.random.rand(action_dim) * 2 - 1) * act_scale
                act[k] = f

            engine.set_action(action=act)

            state = engine.get_state()
            action = engine.get_action()

            states[i, j, :n_particle] = state
            states[i, j, n_particle:, :2] = engine.c_positions
            actions[i, j, :n_particle] = action

            # reset velocity
            if j > 0:
                states[i, j, :, 2:] = (states[i, j, :, :2] - states[i, j - 1, :, :2]) / dt

            # attrs: [1, 0] => moving; [0, 1] => fixed
            n_obj = attrs.shape[2]
            attr = np.zeros((n_obj, attr_dim))
            attr[0, 1] = 1              # the first ball is fixed
            attr[1:n_particle, 0] = 1   # the rest of the balls is free to move
            attr[n_particle:, 1] = 1  # the cylinders are fixed
            attr[:n_particle, 2] = engine.radius
            attr[n_particle:, 2] = engine.c_radius
            # assert np.sum(attr[:, 0]) == 14
            assert np.sum(attr[:, 1]) == 3
            attrs[i, j] = attr

            data = [attr, states[i, j], actions[i, j]]

            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        datas = [attrs[i].astype(np.float64), states[i].astype(np.float64), actions[i].astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats


def gen_Box(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, n_particle, time_step = info['n_rollout'], info['n_particle'], info['time_step']
    dt, args = info['dt'], info['args']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    state_dim = args.state_dim      # x, y, angle, xdot, ydot, angledot
    action_dim = args.action_dim    # x, xdot
    assert state_dim == 6
    assert action_dim == 2

    stats = [init_stat(state_dim), init_stat(action_dim)]

    engine = BoxEngine(dt, state_dim, action_dim)

    states = np.zeros((n_rollout, time_step, n_particle, state_dim))
    actions = np.zeros((n_rollout, time_step, 1, action_dim))
    viss = np.zeros((n_rollout, time_step, n_particle))

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        engine.reset_scene(n_particle)

        for j in range(time_step):
            engine.set_action(rand_float(-600., 100.))

            states[i, j] = engine.get_state()
            actions[i, j] = engine.get_action()
            viss[i, j] = engine.get_vis(states[i, j])

            if j > 0:
                states[i, j, :, 3:] = (states[i, j, :, :3] - states[i, j - 1, :, :3]) / dt
                actions[i, j, :, 1] = (actions[i, j, :, 0] - actions[i, j - 1, :, 0]) / dt

            data = [states[i, j], actions[i, j], viss[i, j]]

            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        datas = [states[i].astype(np.float64), actions[i].astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats


def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            data_shape = data[i].size()
            stat_dim = stat[i].shape[0]
            n_rep = int(data_shape[-1] / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(data_shape)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.

            data_shape = data[i].shape
            stat_dim = stat[i].shape[0]
            n_rep = int(data_shape[-1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape(data_shape)

    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data


class PhysicsDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
        self.stat = None

        os.system('mkdir -p ' + self.data_dir)

        if args.env in ['Cradle']:
            self.data_names = ['attrs', 'states']
        elif args.env in ['Rope']:
            self.data_names = ['attrs', 'states', 'actions']
        elif args.env in ['Box']:
            self.data_names = ['states', 'actions', 'vis']
        else:
            raise AssertionError("Unknown env")

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

        self.T = self.args.len_seq

    def load_data(self):
        env = self.args.env
        if env in ['Cradle', 'Rope']:
            self.stat = load_data(self.data_names, self.stat_path)
        elif env in ['Box']:
            self.stat = load_data(self.data_names[:2], self.stat_path)

    def gen_data(self):
        # if the data hasn't been generated, generate the data
        n_rollout, n_particle = self.n_rollout, self.args.n_particle
        time_step, dt = self.args.time_step, self.args.dt

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'data_names': self.data_names,
                    'n_particle': n_particle,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'args': self.args}

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)

        env = self.args.env

        if env == 'Cradle':
            data = pool.map(gen_Cradle, infos)
        elif env == 'Rope':
            data = pool.map(gen_Rope, infos)
        elif env == 'Box':
            data = pool.map(gen_Box, infos)
        else:
            raise AssertionError("Unknown env")

        print("Training data generated, warpping up stats ...")

        if self.phase == 'train' and self.args.gen_stat:
            if env in ['Cradle']:
                self.stat = [init_stat(self.args.attr_dim),
                             init_stat(self.args.state_dim)]
            elif env in ['Rope']:
                self.stat = [init_stat(self.args.attr_dim),
                             init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]
            elif env in ['Box']:
                self.stat = [init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]

            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])

            store_data(self.data_names[:len(self.stat)], self.stat, self.stat_path)

        else:
            print("Loading stat from %s ..." % self.stat_path)

            if env in ['Cradle', 'Rope']:
                self.stat = load_data(self.data_names, self.stat_path)
            elif env in ['Box']:
                self.stat = load_data(self.data_names[:2], self.stat_path)

    def __len__(self):
        return self.n_rollout * (self.args.time_step - self.T)

    def __getitem__(self, idx):
        args = self.args
        idx_rollout = idx // (args.time_step - self.T + 1)
        idx_timestep = idx % (args.time_step - self.T + 1)

        if args.pn_mode == 'full':
            assert self.T == 2
            data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
            data = normalize(load_data(self.data_names, data_path), self.stat)

            data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')
            label = normalize(load_data(self.data_names, data_path), self.stat)[1][:, 2:]

            data.append(label)

            for i in range(len(data)):
                data[i] = torch.FloatTensor(data[i])

            return tuple(data)

        elif args.pn_mode == 'partial':
            seq_data = []
            for t in range(self.T):
                data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + t) + '.h5')
                data = load_data(self.data_names, data_path)
                data = normalize(data, self.stat)

                # filter unobservable boxes
                state, action, vis = data
                state = state[vis.astype(np.bool)]

                state = torch.FloatTensor(state[np.newaxis, ...])
                action = torch.FloatTensor(action[np.newaxis, ...])

                Rr_idx, Rs_idx, values, Ra = construct_fully_connected_rel(state.size(1), args.relation_dim)
                data = [state, action, Rr_idx, Rs_idx, values, Ra]

                seq_data.append(data)

            return seq_data

        else:
            raise AssertionError("Unsupported pn_mode %s" % args.pn_mode)

