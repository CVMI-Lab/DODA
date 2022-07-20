"""
Modified from OpenPCDet data_augmentor
"""

import os
import random
import numpy as np
from functools import partial
import torch

from . import augmentor_utils
import dataset.dataset as dataset


class Queue(object):
    def __init__(self, size):
        assert size > 0
        self.size = size
        self.queue = [None] * self.size
        self.ptr = 0
        self.cur_size = 0
        self.got = 0

    def update_queue(self, items):
        if len(items) == 0:
            return
        items = items[:self.size]  # update maximum self.size items
        new_ptr = self.ptr + len(items)
        self.queue[self.ptr: min(new_ptr, self.size)] = items[:min(new_ptr, self.size) - self.ptr]
        self.queue[:new_ptr - min(new_ptr, self.size)] = items[min(new_ptr, self.size) - self.ptr:]
        self.cur_size = min(self.cur_size + len(items), self.size)
        self.ptr = new_ptr % self.size

    def get_item(self, n):
        if self.cur_size == 0:
            return []
        n = min(n, self.cur_size)
        items = random.sample(self.queue[:self.cur_size], n)
        self.got += n
        return items


class SplitSampler(object):
    def __init__(self, cfg):
        self.total_size = cfg.size
        self.num_c = cfg.num_class

    def init_finish(self):
        return hasattr(self, 'class_ratio')

    def init_class_ratio(self, cfg):
        self.class_ratio = cfg.class_ratio
        mask = self.class_ratio > 0
        self.inverse_class_ratio = np.where(mask, 1.0 / (cfg.class_ratio + 10e-10), 10e-10)
        self.tail_class_ratio = np.sort(-self.inverse_class_ratio)[:self.num_c]
        self.tail_class_ratio /= (self.tail_class_ratio.sum() + 10e-10)
        self.tail_class_idx = np.argsort(-self.inverse_class_ratio)[:self.num_c]
        self.queues = []
        self.init_queue()

    def update_cfg(self, cfg):
        cfg.class_ratio = self.class_ratio
        cfg.class_thres = np.ones_like(cfg.class_ratio)
        cfg.class_thres[self.tail_class_idx] = self.class_ratio[self.tail_class_idx]
        cfg.tail_class_idx = self.tail_class_idx

    def init_queue(self):
        for c in range(self.num_c):
            size = max(1, int(self.total_size * self.tail_class_ratio[c]))
            self.queues.append(Queue(size))

    def update(self, items):
        if not self.init_finish():
            raise ValueError('Split sampler is not inited!')
        assert len(items) == self.num_c
        for c in range(self.num_c):
            self.queues[c].update_queue(items[c])

    def get_split(self, n):
        if not self.init_finish():
            raise ValueError('Split sampler is not inited!')
        if n == 0:
            return []
        item_c = np.random.choice(self.num_c, n, p=self.tail_class_ratio)
        items = []
        for c in item_c:
            items.extend(self.queues[c].get_item(1))
        return items

    def update_class_ratio(self, class_ratio):
        if class_ratio.max() > 0.0:
            inverse_class_ratio = 1.0 / (class_ratio + 10e-1)
            inverse_class_ratio /= inverse_class_ratio.sum()
            self.tail_class_ratio = 0.999 * self.tail_class_ratio + 0.001 * inverse_class_ratio

    def load_sampler(self, path):
        buffer = torch.load(path)
        self.queues = buffer['queues']
        self.class_ratio = buffer['class_ratio']
        self.inverse_class_ratio = buffer['inverse_class_ratio']
        self.tail_class_ratio = buffer['tail_class_ratio']
        self.tail_class_idx = buffer['tail_class_idx']

    def save_sampler(self, path):
        torch.save(
            {'queues': self.queues, 'class_ratio': self.class_ratio, 'inverse_class_ratio': self.inverse_class_ratio,
             'tail_class_ratio': self.tail_class_ratio, 'tail_class_idx': self.tail_class_idx}, path
        )


class DataAugmentor(object):
    def __init__(self, aug_cfg, dataset_name, class_names, ignore_label, voxel_scale, voxel_mode, full_scale,
                 point_range, max_npoint):
        self.cfg = aug_cfg
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.voxel_scale = voxel_scale
        self.voxel_mode = voxel_mode
        self.full_scale = full_scale
        self.point_range = point_range
        self.max_npoint = max_npoint
        self.ignore_label = ignore_label

        self.init_augmentor_queue(self.cfg.aug_list)

        if 'tacm' in self.cfg and self.cfg.tacm.enabled:
            self.split_sampler = SplitSampler(self.cfg.tacm.cuboid_queue)

    def init_augmentor_queue(self, aug_list):
        self.data_augmentor_queue = []
        for aug in aug_list:
            cur_augmentor = partial(getattr(self, aug), cfg=self.cfg[aug] if aug in self.cfg else None)
            self.data_augmentor_queue.append(cur_augmentor)

    def init_split_sampler(self):
        self.split_sampler = SplitSampler(self.cfg.tacm.cuboid_queue)

    def forward(self, data_dict):
        data_dict['valid'] = True
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict

    def check_func(self, key):
        return augmentor_utils.check_key(key) and augmentor_utils.check_p(key)

    def check_data(self, data_dict):
        return ('valid' not in data_dict) or data_dict['valid']

    def set_total_epochs(self, epoch):
        self.total_epochs = epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def filter_by_index(e_list, idx):
        return augmentor_utils.filter_by_index(e_list, idx)

    @staticmethod
    def update_data_dict(data_dict, idx):
        if 'xyz_middle' in data_dict:
            data_dict['xyz_middle'] = data_dict['xyz_middle'][idx]
        if 'xyz' in data_dict:
            data_dict['xyz'] = data_dict['xyz'][idx]
        if 'label' in data_dict:
            data_dict['label'] = data_dict['label'][idx]
        return data_dict

    # for independently call
    def elastic(self, cfg=None, data_dict=None):
        data_dict['xyz'] = data_dict['xyz_middle'] * self.voxel_scale
        if self.check_func(cfg) and self.check_data(data_dict):
            try:
                for (gran_fac, mag_fac) in cfg.value:
                    data_dict['xyz'] = augmentor_utils.elastic(
                        data_dict['xyz'], gran_fac * self.voxel_scale // 50, mag_fac * self.voxel_scale / 50
                    )
                if cfg.apply_to_feat:
                    data_dict['xyz_middle'] = data_dict['xyz'] / self.voxel_scale
            except Exception as e:
                data_dict['xyz'] = data_dict['xyz_middle'] * self.voxel_scale

        # offset
        data_dict['xyz'] -= data_dict['xyz'].min(0)
        return data_dict

    def scene_aug(self, cfg=None, data_dict=None):
        if self.check_func(cfg) and self.check_data(data_dict):
            data_dict['xyz_middle'] = augmentor_utils.scene_aug(cfg, data_dict['xyz_middle'])
            if data_dict['xyz_middle'].shape[0] == 0: 
                data_dict['valid'] = False
        return data_dict

    def vss(self, cfg=None, data_dict=None):
        if self.check_func(cfg) and self.check_data(data_dict):
            data_dict['xyz_middle'], selected_idx = augmentor_utils.virtual_scan_simulation(
                cfg, data_dict['xyz_middle'], data_dict['label'], self.class_names,
                ignore_label=self.ignore_label
            )
            data_dict = DataAugmentor.update_data_dict(data_dict, selected_idx)
            if data_dict['xyz_middle'].shape[0] == 0: 
                data_dict['valid'] = False
        return data_dict

    # for independently call
    def tacm(self, cfg=None, data_dict=None):
        if augmentor_utils.check_key(cfg):
            data_dict['xyz_middle'], data_dict['label'], \
            data_dict['others'] = \
                augmentor_utils.tacm(
                    cfg, self.split_sampler, self.dataset_name, self.class_names,
                    [data_dict['xyz_middle1'], data_dict['label1'], {}],
                    [data_dict['xyz_middle2'], data_dict['label2'], {}]
            )

            del data_dict['xyz_middle1']
            del data_dict['xyz_middle2']
            del data_dict['label1']
            del data_dict['label2']
        return data_dict

    def crop(self, cfg=None, data_dict=None):
        data_dict['xyz'], valid_idxs = augmentor_utils.crop(
            data_dict['xyz'], self.full_scale, self.point_range, self.max_npoint
        )
        data_dict = self.update_data_dict(data_dict, valid_idxs)
        if data_dict['xyz_middle'].shape[0] == 0:
            data_dict['valid'] = False
        return data_dict

    def shuffle(self, cfg=None, data_dict=None):
        shuffle_idx = np.random.permutation(data_dict['xyz_middle'].shape[0])
        data_dict = self.update_data_dict(data_dict, shuffle_idx)
        return data_dict

