import argparse
import time
import numpy as np
import cv2
import copy
import matplotlib
import socket
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import h5py
import os

import torch
from torch.autograd import Variable
import sys


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def norm(x, p=2):
    return np.power(np.sum(x ** p), 1. / p)


def var_norm(x):
    return torch.sqrt((x ** 2).sum()).item()


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + \
                   (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor), requires_grad=requires_grad)


def to_np(x):
    return x.detach().cpu().numpy()


def render_Cradle(des_dir, filename, lim, states, states_gt=None, video=False, image=False):

    color = ['r', 'b', 'g', 'k', 'y', 'm', 'c']

    if video:
        video_name = os.path.join(des_dir, filename) + '.avi'
        print('Render video ' + video_name)
        os.system('mkdir -p ' + des_dir)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(video_name, fourcc, 25, (640, 480))

    if image:
        image_name = os.path.join(des_dir, filename)
        print('Render image ' + image_name)
        os.system('mkdir -p ' + image_name)

    boxcolor = (0.5, 0.5, 0.5)
    edgecolor = None

    num_frame = states.shape[0]
    num_obj = states.shape[1]

    for i in range(num_frame):
        if i % 5 != 0:
            continue

        fig, ax = plt.subplots(1)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        boxes = []
        circles = []
        circles_gt = []
        circle_colors = []

        num_balls = num_obj // 2

        for j in range(num_balls):
            s0 = states[i, j]
            s1 = states[i, j + num_balls]
            plt.plot([s0[0], s1[0]], [s0[1], s1[1]], 'r-', lw=1)

            if states_gt is not None:
                s0 = states_gt[i, j]
                s1 = states_gt[i, j + num_balls]
                plt.plot([s0[0], s1[0]], [s0[1], s1[1]], 'r-', lw=1, alpha=0.5)

        for j in range(num_balls):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=25)
            circles.append(circle)
            circle_colors.append('b')

            if states_gt is not None:
                circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=25)
                circles_gt.append(circle)

        for j in range(num_balls, num_obj):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=5)
            circles.append(circle)
            circle_colors.append('r')

            if states_gt is not None:
                circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=5)
                circles_gt.append(circle)

        pc = PatchCollection(circles, facecolor=circle_colors, edgecolor=edgecolor)
        ax.add_collection(pc)
        if states_gt is not None:
            pc = PatchCollection(circles_gt, facecolor=circle_colors, edgecolor=edgecolor, alpha=0.5)
            ax.add_collection(pc)
        ax.set_aspect('equal')

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if video:
            out.write(frame)

        if image:
            cv2.imwrite(os.path.join(des_dir, filename, '%d.jpg' % i), frame)

        plt.close()

    if video:
        out.release()


def render_Rope(des_dir, filename, lim, attrs, states, actions=None, states_gt=None,
                count_down=True, video=False, image=False):

    color = ['r', 'b', 'g', 'k', 'y', 'm', 'c']

    if video:
        video_name = os.path.join(des_dir, filename) + '.avi'
        print('Render video ' + video_name)
        os.system('mkdir -p ' + des_dir)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(video_name, fourcc, 25, (640, 480))

    if image:
        image_name = os.path.join(des_dir, filename)
        print('Render image ' + image_name)
        os.system('mkdir -p ' + image_name)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    time_step = states.shape[0]
    n_particle = states.shape[1]
    n_b = np.sum(attrs[:, 0] == 1) + 1

    for i in range(time_step):
        fig, ax = plt.subplots(1)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])

        if actions is not None:
            assert (actions[i, :-4] == 0).all()
            assert (actions[i, -2:] == 0).all()
            F = (actions[i, -4] + actions[i, -3]) / 2.
            normF = norm(F)
            Fx = F / normF * np.sqrt(np.sqrt(normF)) * 15.
            st = (states[i, -4, :2] + states[i, -3, :2]) / 2. + F / normF * 20.
            ax.arrow(st[0], st[1], Fx[0], Fx[1], fc='Orange', ec='Orange', width=6., head_width=30., head_length=30.)

        circles = []
        circles_gt = []
        circles_color = []

        for j in range(n_particle):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=attrs[j, 2])
            circles.append(circle)

            if states_gt is not None:
                circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=attrs[j, 2])
                circles_gt.append(circle)

            if j < n_b:
                circles_color.append('r')
            else:
                circles_color.append('b')

            if 0 < j < n_b:
                if states_gt is not None:
                    plt.plot([states_gt[i, j, 0], states_gt[i, j - 1, 0]],
                             [states_gt[i, j, 1], states_gt[i, j - 1, 1]], 'r-', lw=1, alpha=0.3)

                plt.plot([states[i, j, 0], states[i, j - 1, 0]],
                         [states[i, j, 1], states[i, j - 1, 1]], 'r-', lw=1, alpha=0.5)

        pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1)
        ax.add_collection(pc)

        if states_gt is not None:
            pc = PatchCollection(circles_gt, facecolor=circles_color, linewidth=0, alpha=0.5)
            ax.add_collection(pc)

        ax.set_aspect('equal')

        if count_down:
            plt.text(0, 500, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        write_times = 25 if i == time_step - 1 else 1

        if video:
            for j in range(write_times):
                out.write(frame)

        if image:
            cv2.imwrite(os.path.join(des_dir, filename, '%d.jpg' % i), frame)

        plt.close()

    if video:
        out.release()


def render_Box(des_dir, filename, lim, states, actions, vis, states_gt=None, vis_gt=None,
               count_down=True, video=False, image=False):

    color = ['r', 'b', 'g', 'k', 'y', 'm', 'c']

    if video:
        video_name = os.path.join(des_dir, filename) + '.avi'
        print('Render video ' + video_name)
        os.system('mkdir -p ' + des_dir)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(video_name, fourcc, 25, (640, 480))

    if image:
        image_name = os.path.join(des_dir, filename)
        print('Render image ' + image_name)
        os.system('mkdir -p ' + image_name)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    wallcolor = (0.5, 0.5, 0.5)
    edgecolor = None

    time_step = states.shape[0]
    n_particle = states.shape[1]

    for i in range(time_step):
        fig, ax = plt.subplots(1)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])

        boxes = []
        box_colors = []

        boxes_goal = []
        box_colors_goal = []

        radius = 15 - 0.5

        for j in range(n_particle):
            t = mpl.transforms.Affine2D().rotate_deg_around(
                states[i, j, 0], states[i, j, 1], np.rad2deg(states[i, j, 2]))

            box = Rectangle((states[i, j, 0] - radius, states[i, j, 1] - radius),
                            radius*2, radius*2, transform=t)
            boxes.append(box)

            if vis[i, j]:
                box_colors.append('tomato')
            else:
                box_colors.append('royalblue')

            if states_gt is not None and vis_gt is not None:
                t = mpl.transforms.Affine2D().rotate_deg_around(
                    states_gt[i, j, 0], states_gt[i, j, 1], np.rad2deg(states_gt[i, j, 2]))

                box = Rectangle(
                    (states_gt[i, j, 0] - radius, states_gt[i, j, 1] - radius),
                    radius*2, radius*2, transform=t)
                boxes_goal.append(box)

                if vis_gt[i, j]:
                    box_colors_goal.append('tomato')
                else:
                    box_colors_goal.append('royalblue')

        boxes.append(Rectangle((-600, -10), 1120, 20))
        box_colors.append(wallcolor)
        if actions is not None:
            boxes.append(Rectangle((actions[i, 0, 0] - 10, 10), 20, 300))
            box_colors.append(wallcolor)

        pc = PatchCollection(boxes, facecolor=box_colors, edgecolor='k', linewidth=0.3)
        ax.add_collection(pc)

        if states_gt is not None and vis_gt is not None:
            pc = PatchCollection(boxes_goal, facecolor=box_colors_goal, edgecolor='k', alpha=0.4)
            ax.add_collection(pc)

        ax.set_aspect('equal')

        if count_down:
            plt.text(-600, 300, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        write_times = 25 if i == time_step - 1 else 1

        if video:
            for j in range(write_times):
                out.write(frame)

        if image:
            cv2.imwrite(os.path.join(des_dir, filename, '%d.jpg' % i), frame)

        plt.close()

    if video:
        out.release()


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)   # x: [M, N, D]
        x = x.transpose(0, 1)           # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)   # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)
