import os
import logging
from collections import OrderedDict

import torch

from .common_utils import get_git_commit_id


def build_model(cfg):
    if cfg.MODEL.NAME == 'SparseConvNet':
        from model.unet import SparseConvNet as Model
        from model.unet import model_fn_decorator
    else:
        raise ValueError('{} not supported.'.format(cfg.arch))
    model = Model(cfg)
    return model, model_fn_decorator


def update_checkpoint(checkpoint):
    from collections import OrderedDict
    new_model_stat = OrderedDict()
    if 'state_dict' in checkpoint:
        for key in checkpoint['state_dict']:
            new_model_stat[key.replace('module.','')] = checkpoint['state_dict'][key]
        checkpoint['state_dict'] = new_model_stat
    return checkpoint


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_ckpt(ckpt_path, dist):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu') if dist else None)
    return update_checkpoint(ckpt)


def load_params_from_ckpt(ckpt, dist, model, optimizer=None, logger=logging.getLogger()):

    if os.path.isfile(ckpt):
        checkpoint = get_ckpt(ckpt, dist)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch']))
        logger.info('=> commit id: {}'.format(checkpoint['commit_id']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(ckpt))
    return model, optimizer, checkpoint['epoch']


def load_metric_from_ckpt(ckpt, dist, logger=logging.getLogger()):

    if os.path.isfile(ckpt):
        checkpoint = get_ckpt(ckpt, dist)
        epoch = checkpoint['epoch']
        if 'metric' in checkpoint:
            logger.info("=> loaded best metric '{}' (epoch {})".format(checkpoint['metric'], checkpoint['epoch']))
            metric = checkpoint['metric']
        else:
            logger.info("=> No metric in '{}'".format(ckpt))
            metric = None
    else:
        logger.info("=> no checkpoint found at '{}'".format(ckpt))
    return metric, epoch


def load_params_from_pretrain(
    ckpt, dist, model, logger=logging.getLogger(), strict=True
):

    if os.path.isfile(ckpt):
        checkpoint = get_ckpt(ckpt, dist)
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        logger.info("=> loaded pretrained model '{}' (epoch {})".format(ckpt, checkpoint['epoch']))
        logger.info('=> commit id: {}'.format(checkpoint['commit_id']))
    else:
        logger.info("=> no pretrained model found at '{}'".format(ckpt))
    return model


def save_params(filename, model, optimizer, epoch_log, metric=None):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model_state_to_cpu(model.state_dict())
    torch.save(
        {'epoch': epoch_log, 'state_dict': model_state, 'optimizer': optimizer.state_dict(),
         'commit_id': get_git_commit_id(), 'metric': metric}, filename)
