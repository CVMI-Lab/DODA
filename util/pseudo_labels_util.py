import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from functools import reduce

import torch
import torch.distributed as tdist

from . import common_utils
from lib.pointops2.functions import pointops2 as pointops


def forward_test_model(batch, model, model_fn, epoch, thres=0.0, with_crop=False):
    with torch.no_grad():
        ret = model_fn(batch, model, epoch, thres=thres, with_crop=with_crop)
    return ret


def generate_pseudo_label_batch(
    cfg, batch, train_loader, model, model_fn_st, epoch, pseudo_labels_dir, thres=0.7
):
    # if os.path.exists(os.path.join(pseudo_labels_dir, 'txt', batch['id'][0].split('/')[-1].split('.')[0] + '.txt'):
    #     return
    num_classes = cfg.COMMON_CLASSES.n_classes
    ret = forward_test_model(
        batch, model, model_fn_st, epoch, thres=thres,
        with_crop=cfg.DATA_CONFIG_TAR.DATA_PROCESSOR.get('crop_to_regions', False)
    )

    pseudo_labels = ret['pseudo_labels_all'] if "pseudo_labels_all" in ret else ret["pseudo_labels"]
    offsets = batch['offsets_all'] if 'offsets_all' in batch else batch['offsets']

    common_utils.save_results(
        pseudo_labels_dir, pseudo_labels.cpu().numpy(), offsets.cpu().numpy(), batch['id'],
        train_loader.dataset.get_data_list(), formats=['txt']
    )
    # compute class ratio
    class_ratio_batch = torch.histc(pseudo_labels, bins=num_classes, min=0, max=num_classes - 1)
    return class_ratio_batch


def generate_pseudo_labels(
    cfg, logger, train_loader, model, model_fn_st, epoch, pseudo_labels_dir, rank=0, thres=0.7, done=True, dist=False
):
    logger.info("******************* Generating Pseudo Labels *********************")
    common_utils.synchronize()
    if cfg.DATA_CONFIG_TAR.DATA_PROCESSOR.get('no_downsample_infer', False):
        ds = train_loader.dataset.get_downsampling_scale()
        train_loader.dataset.set_downsampling_scale(1)
    # if os.path.exists(pseudo_labels_dir / 'done.txt'):
    #     return
    model.eval()
    train_loader.dataset.set_training_mode(False)

    class_ratio = torch.zeros(cfg.COMMON_CLASSES.n_classes).cuda()
    if rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train', dynamic_ncols=True)
    for batch in train_loader:
        class_ratio_batch = generate_pseudo_label_batch(
            cfg, batch, train_loader, model, model_fn_st, epoch, pseudo_labels_dir, thres=thres)
        if dist:
            tdist.all_reduce(class_ratio_batch)
        class_ratio += class_ratio_batch / 1000.0
        if rank == 0:
            pbar.update()
    if rank == 0:
        pbar.close()

    if rank == 0 and done:
        done_flag_path = pseudo_labels_dir / 'done.txt'
        np.savetxt(done_flag_path, np.array([1]))
    common_utils.synchronize()
    if cfg.DATA_CONFIG_TAR.DATA_PROCESSOR.get('no_downsample_infer', False):
        train_loader.dataset.set_downsampling_scale(1)
    return class_ratio.cpu().numpy()


def get_label_confidence(cfg, logger, train_loader, model, test_model_fn_st, epoch, rank, dist=False):
    logger.info("******************* Get Pseudo Label Confidence *********************")
    common_utils.synchronize()
    model.eval()
    train_loader.dataset.set_training_mode(False)
    if cfg.DATA_CONFIG_TAR.DATA_PROCESSOR.get('no_downsample_infer', False):
        ds = train_loader.dataset.get_downsampling_scale()
        train_loader.dataset.set_downsampling_scale(1)
    max_points = 1000000000
    if rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train', dynamic_ncols=True)
    class_confidence = [[] for _ in range(cfg.COMMON_CLASSES.n_classes)]
    class_num = [0 for _ in range(cfg.COMMON_CLASSES.n_classes)]
    for i, batch in enumerate(train_loader):
        with torch.no_grad():
            ret = forward_test_model(batch, model, test_model_fn_st, epoch)
        output = ret['output']
        pseudo_labels = ret['preds']
        output_score = torch.nn.functional.softmax(output, 1).max(1)[0]
        df = pd.DataFrame(np.stack(
            (pseudo_labels.cpu().numpy(), output_score.cpu().numpy()), 1), columns=["key", "val"])
        class_confidence_batch = dict(df.groupby("key").val.apply(pd.Series.tolist))
        class_num_batch = []
        for c in range(cfg.COMMON_CLASSES.n_classes):
            if c not in class_confidence_batch:  # fill class, avoid stuck in gather step
                class_confidence_batch[c] = []
            class_num_batch.append(len(class_confidence_batch[c]))
            random.shuffle(class_confidence_batch[c])
            if dist:
                class_confidence[c].extend(
                    reduce(lambda x, y: x + y, common_utils.all_gather_object(class_confidence_batch[c][:max_points])))
                class_num[c] += \
                    reduce(lambda x, y: x + y, common_utils.all_gather_object(class_num_batch[c])) / 1000.0
                # avoid overflow
            else:
                class_confidence[c].extend(class_confidence_batch[c][:max_points])
                class_num[c] += class_num_batch[c] / 1000.0  # avoid overflow
            class_confidence[c].sort(reverse=True)  # sort confidence
        if rank == 0:
            pbar.update()
    if rank == 0:
        pbar.close()
    if cfg.DATA_CONFIG_TAR.DATA_PROCESSOR.get('no_downsample_infer', False):
        train_loader.dataset.set_downsampling_scale(ds)
    common_utils.synchronize()
    return class_confidence, class_num


def get_thres_per_class_on_thres_ratio(cfg, logger, train_loader, model, test_model_fn_st, epoch, rank, dist=False):
    """given per class thres ratio, get per class thres"""
    class_confidence, _ = get_label_confidence(
        cfg, logger, train_loader, model, test_model_fn_st, epoch, rank, dist=dist
    )
    per_class_thres_list = []
    thres_ratio = cfg.SELF_TRAIN.thres_ratio
    if len(thres_ratio) == 1:  # global thres ratio is assigned
        thres_ratio = thres_ratio * cfg.COMMON_CLASSES.n_classes
    for c in range(len(class_confidence)):
        class_confidence[c].sort(reverse=True)
        try:
            per_class_thres_list.append(
                class_confidence[c][:int(max(1, int(thres_ratio[c] * len(class_confidence[int(c)]))))][-1])
        except IndexError:  # no point is predicted as this class
            per_class_thres_list.append(0.0)
    return per_class_thres_list


def get_perclass_thres(cfg, logger, train_loader, model, model_fn, epoch, rank, dist=False):
    if cfg.SELF_TRAIN.get('global_thres', False):  # global threshold
        thres = cfg.SELF_TRAIN.thres
        if len(thres) == 1:
            thres = thres * cfg.COMMON_CLASSES.n_classes
    else:  # based on per class threshold
        thres = get_thres_per_class_on_thres_ratio(cfg, logger, train_loader, model, model_fn, epoch, rank, dist=dist)
    return thres


def set_pseudo_labels(
    args, cfg, logger, pseudo_labels_dir, train_loader, model, model_fn, epoch, rank=0, dist=False
):
    # generate pseudo labels
    generate = False
    if not os.path.exists(pseudo_labels_dir / 'done.txt'):
        thres = get_perclass_thres(cfg, logger, train_loader, model, model_fn, epoch, rank, dist=dist)
        logger.info('per class thres: {} '.format(thres))
        class_ratio = generate_pseudo_labels(
            cfg, logger, train_loader, model, model_fn, epoch, pseudo_labels_dir, rank, thres,
            done=True, dist=dist
        )
        class_ratio /= (class_ratio.sum() + 10e-10)
        if rank == 0:
            np.savetxt(str(pseudo_labels_dir / 'class_ratio.txt'), class_ratio)
        generate = True
    # set pseudo labels dir
    train_loader.dataset.set_pseudo_labels_dir(pseudo_labels_dir)
    common_utils.synchronize()
    return generate
