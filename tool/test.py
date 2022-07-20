import _init_path
import open3d as o3d
import os
import time
import random
import datetime
import numpy as np
import logging
import argparse
import subprocess
import glob
import re
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim
import torch.utils.data


from dataset import build_dataloader
from util import common_utils
from util.common_utils import AverageMeter, intersectionAndUnionGPU, save_results, get_logger
from util.model_utils import load_params_from_pretrain, build_model
from util.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from model.dsnorm import DSNorm, set_ds_target, set_ds_source


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--ckpt', type=str, help='checkpoint to test')
    parser.add_argument('--weight', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--manual_seed', type=int, default=666, help='')
    parser.add_argument('--print_freq', type=int, default=1, help='printing log frequency')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    # parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')

    # parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--save_logit', action='store_true', default=False, help='')
    parser.add_argument('--save_feat', action='store_true', default=False, help='')

    parser.add_argument('--pretrain_not_strict', action='store_true', default=False, help='')

    parser.add_argument('--eval_src', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def update_meter(intersection_meter, union_meter, target_meter, loss_meter, preds, labels, loss, dist_test):
    intersection, union, target = intersectionAndUnionGPU(
        preds, labels, cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG_TAR.DATA_CLASS.ignore_label
    )
    if dist_test:
        dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
    loss_meter.update(loss.item(), labels.size(0))
    return intersection_meter, union_meter, target_meter, loss_meter, accuracy, intersection, union, target


def calc_metrics(intersection_meter, union_meter, target_meter):
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return mIoU, mAcc, allAcc, iou_class, accuracy_class


def test_one_epoch(test_loader, model, model_fn, epoch, dataset_cfg, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info('>>>>>>>>>>>>>>>>>>>> START EVALUATION %d>>>>>>>>>>>>>>>>>>>' % epoch)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], broadcast_buffers=False
        )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()

    if cfg.MODEL.get('dsnorm', False):
        if not args.eval_src:
            model.apply(set_ds_target)
        else:
            model.apply(set_ds_source)
    for i, batch in enumerate(test_loader):
        data_time.update(time.time() - end)

        # forward
        with torch.no_grad():
            ret = model_fn(batch, model, epoch, with_crop=dataset_cfg.DATA_PROCESSOR.get('crop_to_regions', False))
            preds, labels, loss = ret['preds'], ret['labels'], ret['loss']
            logit = ret['output']

        # delete repeated samples
        if dist_test and (i * args.batch_size + (batch['id'].size(0) - 1)) * num_gpus + local_rank + 1> test_loader.dataset.__len__():
            preds, labels = preds[:batch['offsets'][-2]], labels[:batch['offsets'][-2]]
            batch['offsets'] = batch['offsets'][:-1]
            batch['id'] = batch['id'][:-1]

        # save results to file
        if args.save_to_file:
            offsets = batch['offsets_all'] if 'offsets_all' in batch else batch['offsets']
            xyz = batch['locs_float_all'] if 'locs_float_all' in batch else batch['locs_float']
            save_results(
                result_dir / (test_loader.dataset.split + '_' + str(epoch)), preds.clone().cpu().numpy(), offsets.cpu().numpy(),
                batch['id'], test_loader.dataset.data_list, formats=['test'], xyz=xyz.cpu().numpy(), dataset='scannet'
            )

        if args.save_logit:
            offsets = batch['offsets_all'] if 'offsets_all' in batch else batch['offsets']
            save_results(
                result_dir / (test_loader.dataset.split + '_' + str(epoch)) / 'logit', logit.cpu().numpy(),
                offsets.cpu().numpy(), batch['id'], test_loader.dataset.data_list, formats=['npy']
            )

        if args.save_feat:
            offsets = batch['offsets_all'] if 'offsets_all' in batch else batch['offsets']
            save_results(
                result_dir / (test_loader.dataset.split + '_' + str(epoch)) / 'feat', feat.cpu().numpy(),
                offsets.cpu().numpy(), batch['id'], test_loader.dataset.data_list, formats=['npy']
            )

        # update loss
        if dist_test:
            n = preds.size(0)
            loss *= n
            count = labels.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        # update meter
        intersection_meter, union_meter, target_meter, loss_meter, accuracy, _, _, _ = \
            update_meter(intersection_meter, union_meter, target_meter, loss_meter, preds, labels, loss, dist_test)

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(test_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    mIoU, mAcc, allAcc, iou_class, accuracy_class = calc_metrics(intersection_meter, union_meter, target_meter)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    n_classes = cfg.COMMON_CLASSES.n_classes
    class_names = cfg.COMMON_CLASSES.class_names
    for i in range(n_classes):
        logger.info('Class {} : iou/accuracy {:.4f}/{:.4f}.'.format(class_names[i], iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< END EVALUATION <<<<<<<<<<<<<<<<<')
    return {'mIoU_test': mIoU, 'mAcc_test': mAcc, 'allAcc_test': allAcc}



def test(model, model_fn, test_loader, eval_output_dir, ckpt_dir, dist_test, dataset_cfg):

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        writer = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % dataset_cfg.DATA_SPLIT['test'])))

    # load checkpoint
    model = load_params_from_pretrain(
        str(args.ckpt), dist_test, model, logger=logger, strict=not args.pretrain_not_strict)
    model.cuda()
    # start evaluation
    tb_dict = test_one_epoch(
        test_loader, model, model_fn, epoch=args.start_epoch, dataset_cfg=dataset_cfg, dist_test=dist_test,
        result_dir=eval_output_dir
    )

    if cfg.LOCAL_RANK == 0:
        for key, val in tb_dict.items():
            writer.add_scalar(key, val, args.start_epoch)


def main():
    # ==================================== init ==============================================
    global args
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl')
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'
    ckpt_dir = output_dir / 'ckpt'

    dataset_cfg = cfg.DATA_CONFIG_TAR if not args.eval_src else cfg.DATA_CONFIG

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'best'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / dataset_cfg.DATA_SPLIT['test']

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    # eval_output_dir = eval_output_dir / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # log to file
    global logger
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = get_logger(log_file=log_file, rank=cfg.LOCAL_RANK)
    logger.info('*********************************** Start Logging*********************************')
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # ======================================= create network and dataset ==============================
    # network
    model, model_fn_decorator = build_model(cfg)
    model_fn = model_fn_decorator(cfg, args.batch_size, test=True)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif cfg.MODEL.get('dsnorm', False):
        model = DSNorm.convert_dsnorm(model)
    model.cuda()
    # logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    # logger.info(model)

    # dataset
    test_data, test_loader, test_sampler = build_dataloader(
        dataset_cfg, args.batch_size, dist_test,
        training=False, workers=args.workers, logger=logger, split='test')  # cfg.DATA_CONFIG.DATA_SPLIT['test'])

    logger.info('**********************Start testing %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    with torch.no_grad():
        test(model, model_fn, test_loader, eval_output_dir, ckpt_dir, dist_test, dataset_cfg)
    logger.info(' ************************** Clean Shared Memory ***************************')
    test_data.destroy_shm()


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
