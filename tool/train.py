import _init_path
import open3d as o3d
import os
import time
import random
import datetime
import numpy as np
import argparse
import glob
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data

from dataset import get_src_train_dataset, get_val_dataset
from util import common_utils
from util import model_utils
from util.common_utils import AverageMeter, update_meter, calc_metrics, get_logger
from util.model_utils import load_params_from_ckpt, load_params_from_pretrain, load_metric_from_ckpt, save_params
from util.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from model.dsnorm import DSNorm, set_ds_source, set_ds_target


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--weight', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18867, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--reserve_old_ckpt', action='store_true', default=False, help='whether to remove previously saved ckpt')
    parser.add_argument('--manual_seed', type=int, default=None, help='')
    parser.add_argument('--ckpt_save_freq', type=int, default=1, help='number of training epochs')
    # parser.add_argument('--eval_freq', type=int, default=1, help='evaluation frequency')
    parser.add_argument('--print_freq', type=int, default=5, help='printing log frequency')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    # parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def train_epoch(train_loader, model, model_fn, optimizer, scheduler, epoch, rank, dist_train):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        common_utils.adjust_lr(cfg.OPTIMIZATION, optimizer, scheduler, args.epochs, len(train_loader),  epoch, i)

        # forward
        if cfg.MODEL.get('dsnorm', False):
            model.apply(set_ds_source)
        ret = model_fn(batch, model, epoch)
        loss = ret.get('loss', torch.tensor(0).float().cuda())
        labels = ret['labels']
        preds = ret['preds']

        # backward
        optimizer.zero_grad()
        loss.backward()
        if cfg.OPTIMIZATION.get('clip_grad', False) and cfg.OPTIMIZATION.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        # update loss
        if dist_train:
            n = preds.size(0)
            loss *= n
            count = labels.new_tensor([n], dtype=torch.long).cuda()
            dist.all_reduce(loss); dist.all_reduce(count)
            n = count.item()
            loss /= n
        loss_meter.update(loss.item(), labels.size(0))
        # update meter
        intersection_meter, union_meter, target_meter, accuracy, intersection, union, target = \
            update_meter(intersection_meter, union_meter, target_meter, preds, labels, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if (i + 1) % args.print_freq == 0 or i == len(train_loader) - 1:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

        # record to writer
        if rank == 0:
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
            writer.add_scalar('lr', cur_lr, current_iter)

    mIoU, mAcc, allAcc, iou_class, accuracy_class = \
        calc_metrics(intersection_meter, union_meter, target_meter)
    logger.info('Train result at epoch [{}/{}]: mIoU/mPre/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format
        (epoch+1, args.epochs, mIoU,  mAcc, allAcc))
    if rank == 0:
        writer.add_scalar('loss_train', loss_meter.avg, epoch + 1)
        writer.add_scalar('mIoU_train', mIoU, epoch + 1)
        writer.add_scalar('mAcc_train', mAcc, epoch + 1)
        writer.add_scalar('allAcc_train', allAcc, epoch + 1)
    return


def validate_epoch(val_loader, model, model_fn, epoch, rank, dist_train):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    num_gpus = common_utils.get_world_size()
    if cfg.MODEL.get('dsnorm', False):
        model.apply(set_ds_target)
    for i, batch in enumerate(val_loader):
        data_time.update(time.time() - end)

        # forward
        with torch.no_grad():
            ret = model_fn(batch, model, epoch)
            preds, labels, loss = ret['preds'], ret['labels'], ret['loss']
        
        if dist_train and (i * args.batch_size + (batch['id'].size(0) - 1)) * num_gpus + rank + 1> val_loader.dataset.__len__():
            preds, labels = preds[:batch['offsets'][-2]], labels[:batch['offsets'][-2]]
            batch['offsets'] = batch['offsets'][:-1]
            batch['id'] = batch['id'][:-1]

        # update loss
        if dist_train:
            n = preds.size(0)
            loss *= n
            count = labels.new_tensor([n], dtype=torch.long).cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        loss_meter.update(loss.item(), labels.size(0))
        # update meter
        intersection_meter, union_meter, target_meter, accuracy, intersection, union, target = \
            update_meter(intersection_meter, union_meter, target_meter, preds, labels, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    mIoU, mAcc, allAcc, iou_class, accuracy_class = \
        calc_metrics(intersection_meter, union_meter, target_meter)

    logger.info('Val result: mIoU/mPre/mAcc/allPre/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
        mIoU, mAcc, allAcc))
    n_classes = cfg.COMMON_CLASSES.n_classes
    class_names = cfg.COMMON_CLASSES.class_names
    for i in range(n_classes):
        logger.info('Class {} : iou/accuracy {:.4f}/{:.4f}.'.format(
            class_names[i], iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    if rank == 0:
        writer.add_scalar('loss_val', loss_meter.avg, epoch + 1)
        writer.add_scalar('mIoU_val', mIoU, epoch + 1)
        writer.add_scalar('mAcc_val', mAcc, epoch + 1)
        writer.add_scalar('allAcc_val', allAcc, epoch + 1)
    return mIoU


def train(
    model, model_fn, model_fn_test, train_loader, val_loader, optimizer, scheduler, ckpt_dir, rank,
    dist_train=False, train_sampler=None, best_mIoU=None, best_epoch=0
):

    best_mIoU = best_mIoU if best_mIoU is not None else 0.0
    for epoch in range(args.start_epoch, args.epochs):

        if train_sampler is not None:  # compatible for pytorch1.1
            train_sampler.set_epoch(epoch)

        train_epoch(train_loader, model, model_fn, optimizer, scheduler, epoch, rank, dist_train)
        epoch_log = epoch + 1

        if rank == 0 and epoch_log % args.ckpt_save_freq == 0:
            filename = ckpt_dir / ('train_epoch_' + str(epoch_log) + '.pth')
            logger.info('Saving checkpoint to: ' + str(filename))
            save_params(filename, model, optimizer, epoch_log)
            if not args.reserve_old_ckpt:
                try:
                    os.remove(str(ckpt_dir / ('train_epoch_' + str(epoch_log - args.ckpt_save_freq * 2) + '.pth')))
                except Exception:
                    pass

        if cfg.EVALUATION.evaluate and epoch_log % cfg.EVALUATION.eval_freq == 0:
            mIoU_val = validate_epoch(val_loader, model, model_fn_test, epoch, rank, dist_train)
            if rank == 0 and mIoU_val > best_mIoU:
                best_mIoU = mIoU_val
                best_epoch = epoch_log
                filename = ckpt_dir / 'best_train.pth'
                logger.info('Best Model Saving checkpoint to: ' + str(filename))
                save_params(filename, model, optimizer, epoch_log, metric=best_mIoU)
        # scheduler.step()
        logger.info('Best epoch: {}, best mIoU: {}'.format(best_epoch, best_mIoU))


def main():
    # ==================================== init ==============================================
    # import ipdb; ipdb.set_trace()
    global args
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl')
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # log to file
    global logger
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = get_logger(log_file=log_file, rank=cfg.LOCAL_RANK)
    logger.info('*********************************** Start Logging*********************************')
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
    global writer
    writer = SummaryWriter(str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # ================================= create network and dataset ==============================
    # network
    model, model_fn_decorator = model_utils.build_model(cfg)
    model_fn = model_fn_decorator(cfg, args.batch_size)
    model_fn_test = model_fn_decorator(cfg, args.batch_size, test=True)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif cfg.MODEL.get('dsnorm', False):
        model = DSNorm.convert_dsnorm(model)
    model.cuda()
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    # logger.info(model)

    optimizer = common_utils.build_optimizer(cfg.OPTIMIZATION, model)
    scheduler = None

    best_mIoU = None
    best_epoch = 0
    if args.weight:
        model = load_params_from_pretrain(
            args.weight, dist_train, model, logger=logger, strict=not args.pretrain_not_strict
        )
    if args.resume:
        model, optimizer, args.start_epoch = \
            load_params_from_ckpt(args.resume, dist_train, model, optimizer=optimizer, logger=logger)
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*train_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            model, optimizer, args.start_epoch = load_params_from_ckpt(ckpt_list[-1], dist_train, model, optimizer=optimizer, logger=logger)
    best_ckpt = glob.glob(str(ckpt_dir / 'best_train.pth'))
    if len(best_ckpt) > 0:
        best_mIoU, best_epoch = load_metric_from_ckpt(best_ckpt[0], dist_train, logger=logger)

    logger.info('optimizer LR: {}'.format(optimizer.param_groups[0]['lr']))

    if dist_train:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    # dataset
    # source train data
    train_data, train_loader, train_sampler = get_src_train_dataset(
        cfg, args, dist_train, logger, pin_memory=args.pin_memory
    )
    # target val data
    val_loader, val_sampler = get_val_dataset(args, cfg.DATA_CONFIG_TAR, dist_train, logger)

    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train(
        model, model_fn, model_fn_test, train_loader, val_loader, optimizer, scheduler, ckpt_dir,
        cfg.LOCAL_RANK, dist_train=dist_train, train_sampler=train_sampler, best_mIoU=best_mIoU, best_epoch=best_epoch
    )

    logger.info(' ************************** Clean Shared Memory ***************************')
    if cfg.LOCAL_RANK == 0:
        train_data.destroy_shm()
        val_loader.dataset.destroy_shm()


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
