import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from util import common_utils

from .front3d import Front3dDataset
from .s3dis import S3DISDataset
from .scannet import ScanNetDataset
from .mix_dataset import CuboidMixingDataset


__all__ = {
    'front3d': Front3dDataset,
    's3dis': S3DISDataset,
    'scannet': ScanNetDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(
    dataset_cfg, batch_size, dist, workers=4, logger=None, split='training', training=True,
    merge_all_iters_to_one_epoch=False, total_epochs=0, drop_last=False, pin_memory=True, shuffle=False
):

    dataset = __all__[dataset_cfg.DATASET](
        cfg=dataset_cfg,
        class_names=dataset_cfg.DATA_CLASS.class_names,
        batch_size=batch_size,
        split=split,
        training=training,
        logger=logger
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=workers,
        shuffle=shuffle or ((sampler is None) and training), collate_fn=dataset.collate_batch,
        drop_last=drop_last, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler


def build_mix_dataloader(
    dataset_cfg1, dataset_cfg2, batch_size, dist, workers=4, logger=None, split='training', training=True,
    merge_all_iters_to_one_epoch=False, total_epochs=0, drop_last=False, pin_memory=True,
    dataset1=None, dataset2=None
):
    if dataset1 is None:
        dataset1 = __all__[dataset_cfg1.DATASET](
            cfg=dataset_cfg1,
            class_names=dataset_cfg1.DATA_CLASS.class_names,
            batch_size=batch_size,
            split=split,
            training=training,
            logger=logger
        )
    if dataset2 is None:
        dataset2 = __all__[dataset_cfg2.DATASET](
            cfg=dataset_cfg2,
            class_names=dataset_cfg2.DATA_CLASS.class_names,
            batch_size=batch_size,
            split=split,
            training=training,
            logger=logger
        )

    dataset = CuboidMixingDataset(dataset1, dataset2)
    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=drop_last, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler


def get_val_dataset(args, dataset_cfg, dist_train, logger):
    val_data, val_loader, val_sampler = build_dataloader(
        dataset_cfg, args.batch_size, dist_train,
        training=False, workers=args.workers, logger=logger, split='validation', drop_last=False)
    return val_loader, val_sampler


def get_src_train_dataset(cfg, args, dist_train, logger, pin_memory=False):
    # source train data
    src_train_data, src_train_loader, src_train_sampler = build_dataloader(
        cfg.DATA_CONFIG, args.batch_size, dist_train, training=True, workers=args.workers,
        logger=logger, total_epochs=args.epochs, drop_last=True, pin_memory=pin_memory)
    return src_train_data, src_train_loader, src_train_sampler


def get_tar_train_dataset(cfg, args, dist_train, logger, pin_memory=False, src_train_data=None):
    if cfg.DATA_CONFIG_TAR.DATA_AUG.tacm.enabled:  # mix with source
        tar_train_data, tar_train_loader, tar_train_sampler = build_mix_dataloader(
            cfg.DATA_CONFIG_TAR, cfg.DATA_CONFIG, args.batch_size, dist_train, training=True,
            workers=args.workers, logger=logger, total_epochs=args.epochs, drop_last=False, pin_memory=pin_memory,
            dataset2=src_train_data)
    else:
        tar_train_data, tar_train_loader, tar_train_sampler = build_dataloader(
            cfg.DATA_CONFIG_TAR, args.batch_size, dist_train, training=True,
            workers=args.workers, logger=logger, total_epochs=args.epochs, drop_last=False, pin_memory=pin_memory)
    return tar_train_loader, tar_train_sampler


def get_dataset(cfg, args, dist_train, logger, pin_memory=False):
    # source train data
    src_train_data, src_train_loader, src_train_sampler = get_src_train_dataset(
        cfg, args, dist_train, logger, pin_memory=pin_memory
    )
    # target train data
    tar_train_loader, tar_train_sampler = get_tar_train_dataset(
        cfg, args, dist_train, logger, pin_memory=pin_memory, src_train_data=src_train_data
    )
    # target val data
    val_loader, val_sampler = get_val_dataset(args, cfg.DATA_CONFIG_TAR, dist_train, logger)
    return src_train_loader, src_train_sampler, tar_train_loader, tar_train_sampler, val_loader, val_sampler
