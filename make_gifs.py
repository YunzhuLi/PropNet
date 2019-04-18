import os
import cv2
import numpy as np
import imageio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='')
parser.add_argument('--env', default='Box')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--time_step', type=int, default=100)

args = parser.parse_args()

if args.env == 'Cradle':
    args.src_dir = 'dump_Cradle/eval_Cradle_pstep_3/eval_Cradle_%d' % args.idx
elif args.env == 'Rope':
    args.src_dir = 'dump_Rope/mpc_Rope_pstep_3/mpc_Rope_%d' % args.idx
elif args.env == 'Box':
    args.src_dir = 'dump_Box/mpc_Box_pstep_2_1_hisWindow_5_lenSeq_10/mpc_Box_%d' % args.idx


images = []
for i in range(args.time_step):

    if args.env == 'Cradle':
        filename = os.path.join(args.src_dir, '%d.jpg' % (i * 5))
        img = cv2.imread(filename)[160:320, 120:520][:, :, ::-1]
    elif args.env == 'Rope':
        filename = os.path.join(args.src_dir, '%d.jpg' % i)
        img = cv2.imread(filename)[60:420, 90:550][:, :, ::-1]
        img = cv2.resize(img, (317, 248), interpolation=cv2.INTER_AREA)
    elif args.env == 'Box':
        filename = os.path.join(args.src_dir, '%d.jpg' % i)
        img = cv2.imread(filename)[160:340, 30:610][:, :, ::-1]
        img = cv2.resize(img, (400, 125), interpolation=cv2.INTER_AREA)

    print(filename, img.shape)

    write_times = 25 if i == args.time_step - 1 else 1

    for j in range(write_times):
        images.append(img)

imageio.mimsave(args.src_dir + '.gif', images, duration=0.04)

