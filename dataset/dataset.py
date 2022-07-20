'''
ScanNet v2 Dataloader (Modified from PointGroup Dataloader)
Written by Runyu Ding
'''

import open3d as o3d
import os, sys
import torch
import json
import numpy as np
from functools import reduce

sys.path.append('../')

from .augmentor.data_augmentor import DataAugmentor
from lib.pointgroup_ops.functions import pointgroup_ops


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, class_names, batch_size, split='training', training=True, logger=None):
        self.data_root = cfg.DATA_ROOT
        self.dataset = cfg.DATASET
        self.class_names = class_names
        self.batch_size = batch_size
        self.logger = logger
        self.split = split
        self.training = training
        self.voxel_scale = cfg.DATA_PROCESSOR.voxel_scale
        self.max_npoint = cfg.DATA_PROCESSOR.max_npoint
        self.full_scale = cfg.DATA_PROCESSOR.full_scale
        self.point_range = cfg.DATA_PROCESSOR.point_range
        self.voxel_mode = cfg.DATA_PROCESSOR.voxel_mode
        self.class_mapper_file = cfg.get('CLASS_MAPPER_FILE', None)
        self.class_mapper, common_class_names = self.load_mapper_file(self.class_mapper_file)
        if common_class_names is not None:
            self.class_names = common_class_names
        self.ignore_label = cfg.DATA_CLASS.ignore_label

        self.pseudo_labels_dir = None

        self.aug = cfg.DATA_AUG
        self.augmentor = DataAugmentor(
            self.aug, self.dataset, self.class_names, self.ignore_label, self.voxel_scale, self.voxel_mode,
            self.full_scale, self.point_range, self.max_npoint
        )
        self.cache = cfg.DATA_PROCESSOR.cache
        self.downsampling_scale = cfg.DATA_PROCESSOR.get('downsampling_scale', 1)

    def get_data_list(self):
        return self.data_list

    @staticmethod
    def load_mapper_file(map_file):
        if map_file is not None:
            with open(map_file, 'r') as fin:
                info = json.load(fin)
            class_names = info['classes']
            src_classes = info['src']
            remapper = np.ones(256, dtype=np.int64) * (255)
            for l0 in src_classes:
                remapper[int(l0)] = class_names.index(src_classes[l0])
            return remapper, class_names
        else:
            return None, None

    @staticmethod
    def filter_by_index(e_list, idx):
        filtered_e_list = list()
        for e in e_list:
            filtered_e_list.append(e[idx])
        return filtered_e_list

    @staticmethod
    def subsample(xyz, label, ds_scale):
        subsample_idx = np.random.choice(xyz.shape[0], xyz.shape[0], replace=False)[:int(xyz.shape[0] / ds_scale)]
        subsample_idx.sort()
        return subsample_idx

    def set_pseudo_labels_dir(self, pseudo_labels_dir):
        if os.path.exists(pseudo_labels_dir):
            self.pseudo_labels_dir = pseudo_labels_dir
        else:
            raise ValueError('pseudo label path {} doesn\'t exist.'.format(pseudo_labels_dir))

    def load_pseudo_labels(self, data_name):
        with open(str(self.pseudo_labels_dir / 'txt' / (data_name + '.txt')), 'r') as fin:
            labels = np.loadtxt(fin, dtype=np.int64).reshape(-1)
        return labels

    def set_training_mode(self, training):
        self.training = training

    def get_downsampling_scale(self):
        return self.downsampling_scale

    def set_downsampling_scale(self, ds):
        self.downsampling_scale = ds

    def crop_to_regions(self, xyz_all, idx):
        npoint = xyz_all.shape[0]
        crop_masks = []
        if (npoint > 6000000):
            # print('{}: {}'.format(self.data_list[idx].split('/')[-1].split('.')[0], npoint))
            xyz_max = xyz_all.max(0)
            xyz_min = xyz_all.min(0)
            x_mid = (xyz_max[0] + xyz_min[0]) / 2.0
            y_mid = (xyz_max[1] + xyz_min[1]) / 2.0
            crop_mask_1 = (xyz_all[:, 0] > x_mid - 0.5) * (xyz_all[:, 1] > y_mid - 0.5)
            crop_mask_2 = (xyz_all[:, 0] > x_mid - 0.5) * (xyz_all[:, 1] < y_mid + 0.5)
            crop_mask_3 = (xyz_all[:, 0] < x_mid + 0.5) * (xyz_all[:, 1] > y_mid - 0.5)
            crop_mask_4 = (xyz_all[:, 0] < x_mid + 0.5) * (xyz_all[:, 1] < y_mid + 0.5)
            crop_masks = [crop_mask_1, crop_mask_2, crop_mask_3, crop_mask_4]
        return crop_masks

    def __getitem__(self, item):
        raise NotImplementedError

    def load_data(self, index):
        raise NotImplementedError

    def collate_fn(self, items):
        locs = []
        locs_float = []
        feats = []
        labels = []
        batch_offsets = [0]
        ids = []
        selected_idx = []
        mix_idx = []
        mask1 = []  # for mixup dataset
        mask2 = []
        tar_tail_splits = []
        tar_splits_class_ratio = []

        for i, item in enumerate(items):
            xyz, xyz_mid, label, idx, *others = item

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_mid))
            feats.append(torch.from_numpy(xyz_mid))
            labels.append(torch.from_numpy(label))
            ids.append(idx)

            if len(others) > 0:
                if 'selected_idx' in others[0]:
                    selected_idx.append(torch.from_numpy(others[0]['selected_idx']))
                if 'mix_idx' in others[0]:
                    mix_idx.append(others[0]['mix_idx'])
                if 'mask1' in others[0]:
                    mask1.append(torch.from_numpy(others[0]['mask1']))
                if 'mask2' in others[0]:
                    mask2.append(torch.from_numpy(others[0]['mask2']))
                if 'tar_tail_splits' in others[0]:
                    tar_tail_splits.extend(others[0]['tar_tail_splits'])
                if 'tar_splits_class_ratio' in others[0]:
                    tar_splits_class_ratio.append(others[0]['tar_splits_class_ratio'])

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        locs = torch.cat(locs, 0)                                     # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)       # float (N, 3)
        feats = torch.cat(feats, 0) .to(torch.float32)                # float (N, C)
        labels = torch.cat(labels, 0).long()                          # long (N)
        if len(selected_idx) > 0:
            selected_idx = torch.cat(selected_idx, 0)
        if len(mask1) > 0:
            mask1 = torch.cat(mask1, 0)
        if len(mask2) > 0:
            mask2 = torch.cat(mask2, 0)
        if len(tar_splits_class_ratio) > 0:
            tar_splits_class_ratio = reduce(lambda x, y: x + y, tar_splits_class_ratio)

        try:
            spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)
        except Exception:
            print(locs.shape)
            spatial_shape = np.array([self.full_scale[0]] * 3, dtype=np.int64)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.voxel_mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map, 'locs_float': locs_float,
                'feats': feats, 'labels': labels, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'id': ids,
                'selected_idx': selected_idx, 'mix_idx': mix_idx, 'mask1': mask1, 'mask2': mask2,
                'tar_tail_splits': tar_tail_splits, 'tar_splits_class_ratio': tar_splits_class_ratio}

    def test_collate_fn(self, items):
        locs = []
        locs_float = []
        feats = []
        labels = []
        batch_offsets = [0]
        ids = []

        for i, item in enumerate(items):
            xyz, xyz_mid, label, idx, *others = item

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_mid))
            feats.append(torch.from_numpy(xyz_mid))
            labels.append(torch.from_numpy(label))
            ids.append(idx)

        # merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0).to(torch.float32)  # float (N, C)
        labels = torch.cat(labels, 0).long()                          # long (N)
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)
        ids = torch.tensor(ids, dtype=torch.int)  # int (B)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.voxel_mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'offsets': batch_offsets,
                'spatial_shape': spatial_shape, 'id': ids}

    def collate_batch(self, items):
        if not self.training:
            return self.test_collate_fn(items)
        else:
            return self.collate_fn(items)
