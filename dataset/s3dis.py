import SharedArray as SA
import os
import math
import numpy as np
import torch
# sys.path.append('./')
from .dataset import Dataset
from util.common_utils import sa_create, sa_delete
from lib.pointgroup_ops.functions import pointgroup_ops


class S3DISDataset(Dataset):
    def __init__(self, cfg, class_names, batch_size, split='training', training=True, logger=None):
        super(S3DISDataset, self).__init__(
            cfg, class_names, batch_size, split=split, training=training, logger=logger
        )

        data_list = sorted(os.listdir(self.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if cfg.DATA_SPLIT[split] == 'training':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(cfg.DATA_SPLIT.test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(cfg.DATA_SPLIT.test_area) in item]
        for item in self.data_list:
            if self.cache and not os.path.exists("/dev/shm/{}_nn".format(item)):
                data_path = os.path.join(self.data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}_nn".format(item), data)
        self.logger.info("Totally {} samples in {} set.".format(len(self.data_list), self.split))
        self.length = None

    def __len__(self):
        return len(self.data_list)

    def load_data(self, index):
        fn = self.data_list[index]
        if self.cache:
            data = SA.attach("shm://{}_nn".format(fn)).copy()
        else:
            data_path = os.path.join(self.data_root, fn + '.npy')
            data = np.load(data_path)

        xyz_all, label_all = data[:, 0:3], data[:, 6]

        # DA common label
        label_all = self.class_mapper[label_all.astype(np.int64)] if self.class_mapper is not None else label_all

        # pseudo label
        if self.training and self.pseudo_labels_dir is not None:
            pseudo_label_all = self.load_pseudo_labels(fn.split('/')[-1])
            label_all = pseudo_label_all
        return xyz_all, label_all

    def __getitem__(self, item):
        index = item % len(self.data_list)
        xyz_all, label_all = self.load_data(index)
        xyz_all -= xyz_all.mean(0)

        # subsample
        subsample_idx = self.subsample(xyz_all, label_all, self.downsampling_scale)
        xyz, label = self.filter_by_index(
            [xyz_all, label_all], subsample_idx
        )

        # perform augmentations
        if self.training and self.aug.enabled:
            data_dict = {'xyz_middle': xyz, 'label': label}
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict['valid']:
                return self.__getitem__(np.random.randint(self.__len__()))
            xyz, xyz_middle, label = data_dict['xyz'], data_dict['xyz_middle'], data_dict['label']
        else:
            xyz_middle_all = xyz_all.copy()
            xyz_middle = xyz.copy()
            xyz = xyz_middle * self.voxel_scale
            xyz -= xyz.min(0)

        if self.training:
            return xyz, xyz_middle, label, index
        else:
            return xyz, xyz_middle, xyz_all, xyz_middle_all, label, label_all, index

    def destroy_shm(self):
        if self.cache:
            for item in self.data_list:
                if os.path.exists("/dev/shm/{}_nn".format(item)):
                    sa_delete("shm://{}_nn".format(item))

    def test_collate_fn(self, items):
        locs = []
        locs_float = []
        locs_float_all = []
        feats = []
        labels = []
        labels_all = []
        batch_offsets = [0]
        batch_offsets_all = [0]
        ids = []

        for i, item in enumerate(items):
            xyz, xyz_mid, xyz_all, xyz_mid_all, label, label_all, idx = item
            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            batch_offsets_all.append(batch_offsets_all[-1] + xyz_all.shape[0])
            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_mid))
            locs_float_all.append(torch.from_numpy(xyz_mid_all))
            feats.append(torch.from_numpy(xyz_mid))
            labels.append(torch.from_numpy(label))
            labels_all.append(torch.from_numpy(label_all))
            ids.append(idx)

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        batch_offsets_all = torch.tensor(batch_offsets_all, dtype=torch.int)  # int (B+1)
        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        locs_float_all = torch.cat(locs_float_all, 0).to(torch.float32)   # float (N_all, 3)
        feats = torch.cat(feats, 0).to(torch.float32)  # float (N, C)
        labels = torch.cat(labels, 0).long()  # long (N)
        labels_all = torch.cat(labels_all, 0).long()  # long (N)
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)
        ids = torch.tensor(ids, dtype=torch.int)  # int (B)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.voxel_mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map, 'locs_float': locs_float,
                'locs_float_all': locs_float_all, 'id': ids, 'feats': feats, 'labels': labels, 'labels_all': labels_all,
                'offsets': batch_offsets, 'offsets_all': batch_offsets_all, 'spatial_shape': spatial_shape}
