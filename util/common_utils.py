# openpcdet
import open3d as o3d
import logging
import os
import pickle
import random
import subprocess
import logging
from PIL import Image
import SharedArray as SA
from functools import reduce
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class DataReader(object):
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler

    def construct_iter(self):
        self.dataloader_iter = iter(self.dataloader)

    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def read_data(self):
        try:
            return self.dataloader_iter.next()
        except:
            if self.sampler is not None:
                self.sampler.set_epoch(self.cur_epoch)
            self.construct_iter()
            return self.dataloader_iter.next()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:
    Returns:
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_git_commit_id():
    if not os.path.exists('.git'):
        return '0000000'
    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_id = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_id


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cos_learning_rate(optimizer, base_lr, curr_iter, max_iter, warm_iter, hold_base_iter):
    lr = 0.5 * base_lr * (1 + np.cos(np.pi * (curr_iter - warm_iter - hold_base_iter) / \
         float(max_iter - warm_iter - hold_base_iter)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optim_cfg, optimizer, scheduler, total_epochs, total_iters_per_epoch,  epoch, iter):
    # adjust learning rate
    if optim_cfg.lr_decay == 'step':
        step_learning_rate(
            optimizer, optim_cfg.base_lr, epoch - 1, optim_cfg.step_epoch, optim_cfg.multiplier)
    elif optim_cfg.lr_decay == 'poly':
        max_iter = total_iters_per_epoch * total_epochs
        poly_learning_rate(
            optimizer, optim_cfg.base_lr, epoch * total_iters_per_epoch + iter + 1, max_iter)
    elif optim_cfg.lr_decay == 'cos':
        max_iter = total_iters_per_epoch * total_epochs
        cos_learning_rate(
            optimizer, optim_cfg.base_lr, epoch * total_iters_per_epoch + iter + 1, max_iter, 0, 0)
    elif optim_cfg.lr_decay == 'adam_onecycle':
        scheduler.step(epoch * total_iters_per_epoch + iter + 1)
    elif optim_cfg.lr_decay in ['multistep']:
        pass
    else:
        raise NotImplementedError


def build_optimizer(optim_cfg, model):
    if isinstance(model, list):
        params = reduce(
            lambda x, y: chain(
                filter(lambda p: p.requires_grad, x.parameters()), filter(lambda p: p.requires_grad, y.parameters())),
            model)
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())
    if optim_cfg.get('optim', 'sgd') == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=optim_cfg.base_lr, momentum=optim_cfg.momentum, weight_decay=optim_cfg.weight_decay
        )
    elif optim_cfg.get('optim', 'sgd') == 'adam':
        optimizer = torch.optim.Adam(params, lr=optim_cfg.base_lr)
    elif optim_cfg.optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=optim_cfg.base_lr)
    else:
        raise NotImplementedError

    return optimizer


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1).clone()
    target = target.view(-1).clone()
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def update_meter(intersection_meter, union_meter, target_meter, preds, labels, n_classes, ignore_label, dist_train):
    intersection, union, target = intersectionAndUnionGPU(preds, labels, n_classes, ignore_label)
    if dist_train:
        dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
    return intersection_meter, union_meter, target_meter, accuracy, intersection, union, target


def calc_metrics(intersection_meter, union_meter, target_meter):
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return mIoU, mAcc, allAcc, iou_class, accuracy_class


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def sa_create(name, var):
    try:
        x = SA.create(name, var.shape, dtype=var.dtype)
    except FileExistsError:
        return
        # SA.delete(name)
        # x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def sa_delete(name):
    try:
        SA.delete(name)
    except:
        return


def save_results(result_dir, preds, offsets, ids, data_list, formats=['txt']):
    for fmt in formats:
        os.makedirs(os.path.join(result_dir, fmt), exist_ok=True)
    for (i, idx) in enumerate(ids):
        if 'txt' in formats:
            # savetxt
            save_path = os.path.join(result_dir, 'txt', data_list[idx].split('/')[-1].split('.')[0] + '.txt')
            if os.path.exists(save_path):
                continue
            np.savetxt(save_path, preds[offsets[i]: offsets[i + 1]].astype(np.uint8), fmt='%d')


def get_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    console = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s %(process)d] %(message)s"
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data, rank=None):
    """
    Run all_gather on tensors
    Args:
        data: any tensor
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return data

    origin_size = data.size()
    tensor = data.reshape(-1)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    size_list = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    if max_size == 0:
        return data
    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).cuda())
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor[:size]
        data_list.append(buffer)
    # if rank == 0:
    #     import ipdb; ipdb.set_trace(context=10)
    gathered_data = torch.cat(data_list, dim=0)
    gathered_shape = list(origin_size[1:])
    gathered_shape.insert(0, -1)
    gathered_data = gathered_data.reshape(gathered_shape)

    return gathered_data


def all_gather_object(object, rank=None):
    """
    Gathers picklable objects from the whole group and returns a list to each caller.
    Arguments:
        object (Any): Object to be broadcast from the current process. Must be picklable, or the function will fail with an AttributeError.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
    Returns:
        Async work handle, if async_op is set to True.
        List[Any] if not async_op, a list of the objects broadcast by each process. Element i will correspond to the object process i broadcasted.
        None, if not part of the group.
    """

    def _object_to_tensor(obj):
        """
        Helper function to convert an arbitrary picklable python object to a tensor
        Returns a torch.ByteTensor representing the pickled object.
        """
        buf = pickle.dumps(obj)
        storage = torch.ByteStorage.from_buffer(buf)
        byte_tensor = torch.ByteTensor(storage).cuda()
        size = torch.LongTensor([byte_tensor.numel()]).cuda()
        return byte_tensor, size

    def _tensor_to_object(tensor, size):
        """
        Helper function to convert a ByteTensor created by _object_to_tensor back to
        the picklable python object.
        """
        buf = tensor.cpu().numpy().tobytes()[:size]
        return pickle.loads(buf)

    world_size = get_world_size()
    if world_size == 1:
        return object

    local_tensor, local_size = _object_to_tensor(object)

    object_size_list = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    # collect the object sizes from all the tensors
    dist.all_gather(object_size_list, local_size)

    # create the list of output tensors needed for all_gather
    max_object_size = max(object_size_list)
    if max_object_size == 0:
        return object
    tensor_list = [torch.ByteTensor(size=(max_object_size,)).cuda() for _ in range(world_size)]

    # allocate a tensor of max size, and copy byte tensor into it.
    max_size_tensor = torch.ByteTensor(size=(max_object_size,)).cuda()
    max_size_tensor[:local_size[0]] = local_tensor

    dist.all_gather(tensor_list, max_size_tensor)
    # unpickle the tensors back into objects.
    objects = [_tensor_to_object(tensor, object_size.cpu().numpy().item()) for object_size, tensor in zip(object_size_list, tensor_list)]

    return objects


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_softmax_entropy(input):
    p = softmax(input)
    return np.sum(p * np.log(p))
