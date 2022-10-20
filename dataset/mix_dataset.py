import numpy as np
import copy

import torch

from .augmentor.data_augmentor import DataAugmentor

class MixDatasetTemp(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.mix = True
        aug = dataset1.augmentor
        self.augmentor = DataAugmentor(
            aug.cfg, aug.dataset_name, aug.class_names, aug.ignore_label, aug.voxel_scale, aug.voxel_mode,
            aug.full_scale, aug.point_range, aug.max_npoint
        )
        self.augmentor.init_augmentor_queue(['elastic', 'crop', 'shuffle'])

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, item):
       raise NotImplementedError

    def collate_batch(self, items):
        return self.dataset1.collate_batch(items)

    def set_training_mode(self, training):
        self.dataset1.set_training_mode(training)
        self.dataset2.set_training_mode(training)

    def set_pseudo_labels_dir(self, pseudo_labels_dir):
        self.dataset1.set_pseudo_labels_dir(pseudo_labels_dir)

    def get_data_list(self):
        return self.dataset1.get_data_list()

    def get_downsampling_scale(self):
        return self.dataset1.get_downsampling_scale()

    def set_downsampling_scale(self, ds):
        self.dataset1.set_downsampling_scale(ds)

    def forward_aug(self, xyz_middle, label):
        if self.dataset1.training and self.dataset1.aug.enabled:
            data_dict = {'xyz_middle': xyz_middle, 'label': label}
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict['valid']:
                return self.dataset1.__getitem__(np.random.randint(self.__len__()))
            xyz, xyz_middle, label = \
                data_dict['xyz'], data_dict['xyz_middle'], data_dict['label']
        else:
            xyz = xyz_middle * self.voxel_scale
            xyz -= xyz.min(0)
        return xyz, xyz_middle, label, None


class CuboidMixingDataset(MixDatasetTemp):
    def __init__(self, dataset1, dataset2):
        super( CuboidMixingDataset, self).__init__(dataset1, dataset2)

    def __getitem__(self, item):
        if not self.mix or not self.dataset1.training:
            return self.dataset1.__getitem__(item)
        xyz1, xyz_middle1, label1, item1, *others = self.dataset1.__getitem__(item)
        xyz2, xyz_middle2, label2, item2, *others2 = self.dataset2.__getitem__(np.random.randint(self.__len__()))

        data_dict = self.dataset1.augmentor.tacm(
            self.dataset1.aug.tacm,
            {'xyz_middle1': xyz_middle1, 'label1': label1,
             'xyz_middle2': xyz_middle2, 'label2': label2},
        )
        xyz_middle, label = data_dict['xyz_middle'], data_dict['label']
        pc1_mask, pc2_mask = data_dict['others']['pc1_mask'], data_dict['others']['pc2_mask']
        tar_tail_splits = data_dict['others']['tar_tail_splits']
        tar_splits_class_ratio = data_dict['others']['tar_splits_class_ratio']

        xyz, xyz_middle, label, _ = self.forward_aug(xyz_middle, label)
        return xyz, xyz_middle, label, item, \
               {'mask1': pc1_mask, 'mask2': pc2_mask, 'tar_tail_splits': tar_tail_splits,
                'tar_splits_class_ratio': tar_splits_class_ratio}
