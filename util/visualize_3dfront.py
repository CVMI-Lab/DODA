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
    input_file = os.path.join(opt.data_root, 'density1250', opt.room_name + '.npy')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    data = np.load(input_file, allow_pickle=True)
    xyz, rgb, label, inst_label = data[..., :3], data[..., 3:6], data[..., 6], data[..., 7]

    rgb = (rgb + 1) * 127.5

    import json
    label_mapper_file = 'dataset/class_mapper/3dfront_2_scannet.json'
    with open(label_mapper_file, 'r') as fin:
        info = json.load(fin)
        # Map relevant classes to {0,1,...,19}, and ignored classes to -100
    class_names = info['classes']
    src_classes = info['src']
    # tar_classes = info['tar']
    remapper = np.ones(256, dtype=np.int64) * (255)
    for l0 in src_classes:
        remapper[int(l0)] = class_names.index(src_classes[l0])
    label = remapper[label.astype(np.int64)]
    label[label == opt.ignore_label] = len(DA_SEMANTIC_NAMES) - 1

    if (opt.task == 'semantic_gt'):
        assert opt.room_split != 'test'
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[(label != opt.ignore_label)] = np.array(itemgetter(*DA_SEMANTIC_NAMES[label[(label != opt.ignore_label)]])(CLASS_COLOR))
        # label_rgb[(label >= 0) & (label != opt.ignore_label)] = np.array(itemgetter(*SEMANTIC_NAMES[label[(label >= 0) & (label != opt.ignore_label)]])(CLASS_COLOR))
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

        label_pred[label_pred == 255] = len(DA_SEMANTIC_NAMES) - 1
        label_pred_rgb = np.array(itemgetter(*DA_SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    if opt.room_split != 'test':
        # print(label.shape, rgb.shape)
        sem_valid = (label != opt.ignore_label)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to the input dataset files', default='./data/3dfront')
    parser.add_argument('--result_root', help='path to the predicted results')
    parser.add_argument('--room_split', help='train / val / test', default='val')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred', default='input')
    parser.add_argument('--ignore_label', default=255)
    opt = parser.parse_args()

    with open(os.path.join(opt.data_root, 'train_list.txt'), 'r') as fin:
        files = fin.readlines()
    rooms = [f.strip() for f in files]
    for (i, r) in enumerate(rooms):
        print(i, r)
        opt.room_name = r.split('.')[0]
        print(opt.room_name)

        xyz, rgb = get_coords_color(opt)

        visualize_o3d(xyz, rgb)