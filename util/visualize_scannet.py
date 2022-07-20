'''
Visualization
Written by Li Jiang
'''

import random
import numpy as np
# import mayavi.mlab as mlab
import open3d as o3d
import os, glob, argparse
import torch
from operator import itemgetter

from .visualize_utils import COLOR20, COLOR40, SCANNET_CLASS_COLOR as CLASS_COLOR, \
    SCANNET_SEMANTIC_NAMES as SEMANTIC_NAMES, \
    SCANNET_DA_SEMANTIC_NAMES as DA_SEMANTIC_NAMES, \
    SCANNET_SEMANTIC_IDX2NAME as SEMANTIC_IDX2NAME


def visualize_o3d(xyz, color):
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz[(label >= 0) & (label != 255)])
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color / 255.0)
    o3d.visualization.draw_geometries([pcd])


def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, opt.room_split + '_group', opt.room_name + '.pth')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    if opt.room_split == 'test':
        xyz, rgb = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label, sp, _ = torch.load(input_file)
    rgb = (rgb + 1) * 127.5

    if (opt.task == 'semantic_gt'):
        assert opt.room_split != 'test'
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[(label >= 0) & (label != opt.ignore_label)] = np.array(itemgetter(*SEMANTIC_NAMES[label[(label >= 0) & (label != opt.ignore_label)]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'semantic_pred'):
        # assert opt.room_split != 'train'
        # semantic_file = os.path.join(opt.result_root, opt.room_split, 'semantic', opt.room_name + '.npy')
        semantic_file = os.path.join(opt.result_root, opt.room_name + '.npy')
        if os.path.exists(semantic_file):
            assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
            label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        else:
            semantic_file = os.path.join(opt.result_root, opt.room_name + '.txt')
            assert os.path.isfile(semantic_file)
            label_pred = np.loadtxt(semantic_file).reshape(-1).astype(np.int64)

        if opt.da:
            label_pred[label_pred == 255] = len(DA_SEMANTIC_NAMES) - 1
            label_pred_rgb = np.array(itemgetter(*DA_SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        else:
            label_pred[label_pred == 255] = len(SEMANTIC_NAMES) - 1
            label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    if opt.room_split != 'test':
        # print(label.shape, rgb.shape)
        sem_valid = (label != opt.ignore_label)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to the input dataset files', default='./data/scannetv2')
    parser.add_argument('--result_root', help='path to the predicted results')
    parser.add_argument('--room_split', help='train / val / test', default='val')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred', default='input')
    parser.add_argument('--ignore_label', default=255)
    parser.add_argument('--da', default=False, action='store_true')
    opt = parser.parse_args()

    rooms = sorted(os.listdir(opt.result_root))
    for (i, r) in enumerate(rooms):
        print(i, r)
        opt.room_name = r.split('.')[0]
        print(opt.room_name)

        xyz, rgb = get_coords_color(opt)
        visualize_o3d(xyz, rgb)
