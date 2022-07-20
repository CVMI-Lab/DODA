import SharedArray as SA
import os
import numpy as np
import glob
import torch
# sys.path.append('./')
from .dataset import Dataset
from util.common_utils import sa_create, sa_delete


class ScanNetDataset(Dataset):

    def __init__(self, cfg, class_names, batch_size, split='training', training=True, logger=None):
        super(ScanNetDataset, self).__init__(
            cfg, class_names, batch_size, split=split, training=training, logger=logger
        )

        self.data_suffix = cfg.DATA_SPLIT.data_suffix
        self.data_list = sorted(glob.glob(os.path.join(self.data_root, cfg.DATA_SPLIT[split]) + '/*' + self.data_suffix))
        self.split_file = cfg.DATA_SPLIT[split]
        for item in self.data_list:
            if self.cache and not os.path.exists("/dev/shm/{}".format(item.split('/')[-1][:-4] + '_xyz_')):
                if self.split_file.find('test') < 0:
                    xyz, rgb, label, *others = torch.load(item)  # xyzrgb, N*6
                    sa_create("shm://{}".format(item.split('/')[-1][:-4] + '_label_'), label)
                else:
                    xyz, rgb = torch.load(item)
                sa_create("shm://{}".format(item.split('/')[-1][:-4] + '_xyz_'), xyz)
        self.logger.info("Totally {} samples in {} set.".format(len(self.data_list), self.split))
        self.length = None

    def __len__(self):
        return len(self.data_list)


    def load_data(self, index):
        fn = self.data_list[index]
        if self.cache:
            xyz = SA.attach("shm://{}".format(fn.split('/')[-1][:-4] + '_xyz_')).copy()
            if self.split_file.find('test') < 0:
                label = SA.attach("shm://{}".format(fn.split('/')[-1][:-4] + '_label_')).copy()
            else:
                label = np.full(xyz.shape[0], self.ignore_label).astype(np.int64)
        else:
            if self.split_file.find('test') < 0:
                xyz, rgb, label, *others = torch.load(fn)  # .numpy()
            else:
                xyz, rgb = torch.load(fn)
                label = np.full(xyz.shape[0], self.ignore_label)
        # DA common label
        label = self.class_mapper[label.astype(np.int64)] if self.class_mapper is not None else label

        # load pseudo labels
        if self.training and self.pseudo_labels_dir is not None:
            pseudo_label = self.load_pseudo_labels(fn.split('/')[-1][:-4])
            label = pseudo_label
 
        return xyz, label

    def __getitem__(self, item):
        index = item % len(self.data_list)
        xyz, label = self.load_data(index)

        # already done
        # xyz = xyz - xyz.mean(0)
        # rgb = rgb / 127.5 - 1

        # perform augmentations
        if self.training and self.aug.enabled:
            data_dict = {'xyz_middle': xyz, 'label': label}
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict['valid']:
                return self.__getitem__(np.random.randint(self.__len__()))
            xyz, xyz_middle, label = data_dict['xyz'], data_dict['xyz_middle'], data_dict['label']
        else:
            xyz_middle = xyz.copy()
            xyz = xyz_middle * self.voxel_scale
            xyz -= xyz.min(0)

        if self.split_file == 'test':
            return xyz, xyz_middle, index
        else:
            return xyz, xyz_middle, label, index, {}

    def destroy_shm(self):
        if self.cache:
            for item in self.data_list:
                if os.path.exists("/dev/shm/{}".format(item.split('/')[-1][:-4] + '_xyz')):
                    sa_delete("shm://{}".format(item.split('/')[-1][:-4] + '_xyz'))
                    if self.split_file.find('test') < 0:
                        sa_delete("shm://{}".format(item.split('/')[-1][:-4] + '_label'))

