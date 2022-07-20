import SharedArray as SA
import os
import numpy as np
import plyfile
# sys.path.append('./')
from .dataset import Dataset
from util.common_utils import sa_create, sa_delete


def load_data(path, postfix):
    if postfix == '.npy':
        return np.load(path, allow_pickle=True)
    elif postfix == '.ply':
        fn = plyfile.PlyData.read(path)
        return np.array([list(x) for x in fn.elements[0]])  # xyzrgbl, N*7


class Front3dDataset(Dataset):

    def __init__(self, cfg, class_names, batch_size, split='training', training=True, logger=None):
        super(Front3dDataset, self).__init__(
            cfg, class_names, batch_size, split=split, training=training, logger=logger
        )

        self.data_suffix = cfg.DATA_SPLIT.data_suffix
        with open(os.path.normpath(os.path.join(self.data_root, cfg.DATA_SPLIT.split_files[split])), 'r') as fin:
            data_list = fin.readlines()
        self.data_list = [item.strip() for item in data_list]
        for ii, item in enumerate(self.data_list):
            fn = '_'.join(item.split('/'))[:-4]
            if self.cache and not os.path.exists("/dev/shm/{}".format(fn)):
                data = load_data(os.path.join(self.data_root, item[:-4] + self.data_suffix), self.data_suffix)
                sa_create("shm://{}".format(fn), data)
        self.data_idx = np.arange(len(self.data_list))
        self.logger.info("Totally {} samples in {} set.".format(len(self.data_idx), self.split))

    def __len__(self):
        return len(self.data_list)

    def load_data(self, index):
        fn = self.data_list[index]
        if self.cache:
            points = SA.attach("shm://{}".format('_'.join(fn.split('/'))[:-4])).copy()
        else:
            points = load_data(os.path.join(self.data_root, fn)[:-4] + self.data_suffix, self.data_suffix)
        xyz = np.ascontiguousarray(points[:, :3])
        # rgb = np.ascontiguousarray(points[:, 3:6])
        label = np.ascontiguousarray(points[:, 6], dtype=np.int64)

        # DA common label
        label = self.class_mapper[label.astype(np.int64)] if self.class_mapper is not None else label

        # pseudo label
        if self.pseudo_labels_dir is not None:
            label = self.load_pseudo_labels(fn.split('/')[-1][:-4])

        return xyz, label

    def __getitem__(self, item):
        index = self.data_idx[item % len(self.data_idx)]
        xyz, label = self.load_data(index)
        xyz -= xyz.mean(0)

        # subsample.
        subsample_idx = self.subsample(xyz, label, self.downsampling_scale)
        xyz, label = self.filter_by_index([xyz, label], subsample_idx)

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

        if self.training and (xyz.max(0) // 64).min() < 1:
            return self.__getitem__(np.random.randint(self.__len__()))

        return xyz, xyz_middle, label, index

    def destroy_shm(self):
        if self.cache:
            for item in self.data_list:
                fn = '_'.join(item.split('/'))[:-4]
                if os.path.exists("/dev/shm/{}".format(fn)):
                    sa_delete("shm://{}".format(fn))

