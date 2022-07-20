import torch
import torch.nn as nn
import spconv
import functools
import sys
import numpy as np
# from torch_scatter import scatter_mean
# sys.path.append('../../')
from util.loss_utils import lovasz_softmax_with_logit
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.pointops2.functions import pointops2 as pointops
from .unet_block import ResidualBlock, VGGBlock, UBlock


class SparseConvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_channel = cfg.MODEL.BACKBONE.in_channel
        mid_channel = cfg.MODEL.BACKBONE.mid_channel
        try:
            n_classes = cfg.COMMON_CLASSES.n_classes
        except:
            n_classes = cfg.DATA_CONFIG.DATA_CLASS.n_classes
        block_reps = cfg.MODEL.BACKBONE.block_reps
        block_residual = cfg.MODEL.BACKBONE.block_residual

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([mid_channel, 2 * mid_channel, 3 * mid_channel, 4 * mid_channel, 5 * mid_channel,
                            6 * mid_channel, 7 * mid_channel], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(mid_channel),
            nn.ReLU()
        )
        self.linear = nn.Linear(mid_channel, n_classes)  # bias(default): True

        # init parameters
        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, input_map, return_mid_feat=False):
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]  # voxel to points

        semantic_scores = self.linear(output_feats)  # (N, nClass), float

        if return_mid_feat:
            return output_feats, semantic_scores
        else:
            return semantic_scores


def test_model_feat(cfg, batch_size, batch, model, epoch):
    # prepare input
    # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
    # 'locs_float': locs_float, 'feats': feats, 'labels': labels,
    # 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'id': id}

    # coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
    voxel_coords = batch['voxel_locs'].cuda(non_blocking=True)  # (M, 1 + 3), long, cuda
    p2v_map = batch['p2v_map'].cuda(non_blocking=True)          # (N), int, cuda
    v2p_map = batch['v2p_map'].cuda(non_blocking=True)          # (M, 1 + maxActive), int, cuda

    coords_float = batch['locs_float'].cuda(non_blocking=True)  # (N, 3), float32, cuda
    feats = batch['feats'].cuda(non_blocking=True)              # (N, C), float32, cuda

    # batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda
    spatial_shape = batch['spatial_shape']

    if cfg.MODEL.BACKBONE.use_xyz:
        feats = torch.cat((feats, coords_float), 1)
    voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.DATA_CONFIG.DATA_PROCESSOR.voxel_mode)
    # (M, C), float, cuda

    input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

    output = model(input_, p2v_map)   # (N, nClass), float
    preds = output.max(1)[1]          # (N), long

    return output, preds


def model_fn_decorator(cfg, batch_size, test=False):

    # criterion
    ignore_label = cfg.DATA_CONFIG.DATA_CLASS.ignore_label
    n_classes = cfg.COMMON_CLASSES.n_classes
    if cfg.OPTIMIZATION.get('loss', 'cross_entropy') == 'cross_entropy':
        semantic_criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).cuda()
        semantic_criterion_weight = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none').cuda()
    elif cfg.OPTIMIZATION.get('loss', 'cross_entropy') == 'lovasz':
        semantic_criterion = lovasz_softmax_with_logit(ignore=ignore_label).cuda()
    else:
        raise NotImplementedError

    def test_model_fn(batch, model, epoch, thres=0.0, with_crop=False):
        # prepare input
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels,
        # 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'id': id}

        batch_size = batch['offsets'].size(0) - 1
        output, preds = test_model_feat(cfg, batch_size, batch, model, epoch)
        weight = torch.zeros(preds.size()).cuda()    # (N), float

        # if threshold
        pseudo_labels = preds.clone()
        thres = thres if isinstance(thres, list) else [thres] * n_classes
        thres = torch.from_numpy(np.array(thres)).cuda()
        softmax_output = torch.nn.functional.softmax(output, dim=1)
        confidence, pseudo_label = torch.max(softmax_output, 1)
        confidence_mask = confidence > thres[pseudo_label]
        pseudo_labels[~confidence_mask] = ignore_label
        weight[confidence_mask] = confidence[confidence_mask]

        if with_crop and batch['offsets'][-1] < batch['offsets_all'][-1]:
            point_idx = pointops.knnquery(
                1, batch['locs_float'].cuda(), batch['locs_float_all'].cuda(), batch['offsets'].cuda(),
                batch['offsets_all'].cuda())[0].reshape(-1).long()  # (N, K)
            # broadcast labels
            output = output[point_idx]
            preds = preds[point_idx]
            weight = weight[point_idx]
            # compute loss
            labels = batch['labels_all'].cuda(non_blocking=True)                      # (N), long, cuda
            loss = semantic_criterion(output, labels)
        else:
            # compute loss
            labels = batch['labels'].cuda(non_blocking=True)                      # (N), long, cuda
            loss = semantic_criterion(output, labels)
        ret = {'loss': loss, 'output': output, 'preds': preds, 'labels': labels, 'weight': weight,
               'pseudo_labels': pseudo_labels}
        return ret

    def model_fn(batch, model, epoch, soft_label=False, loss_weight=None):
        # prepare input
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels,
        # 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'id': id}

        batch_size = batch['offsets'].size(0) - 1
        output, preds = test_model_feat(cfg, batch_size, batch, model, epoch)

        # compute loss
        labels = batch['labels'].cuda(non_blocking=True)                      # (N), long, cuda
        # if cfg.OPTIMIZATION.weighted_loss:
        #     weight = batch['weight'].cuda(non_blocking=True)  # TODO
        #     loss = semantic_criterion_weight(output, labels)
        #     loss = (loss * torch.clamp(weight, min=0.0, max=1.0)).mean()
        if loss_weight is not None:
            assert loss_weight.size(0) == labels.size(0)
            loss = semantic_criterion_weight(output, labels)
            loss = (loss * loss_weight).sum() / (loss_weight.sum() + 10e-10)  # total loss weight: N
            ret = {'loss': loss, 'output': output, 'preds': preds, 'labels': labels}
        elif soft_label and cfg.get('SOFT_LABEL', False) and cfg.SOFT_LABEL.enabled:
            # split to hard label and soft label
            soft_label = batch['soft_labels'].cuda()
            top1_conf, top1_label = soft_label.max(1)
            hard_label_idx = (top1_conf == 1)
            if cfg.SOFT_LABEL.convert_to_hard:
                cum_soft_label = torch.cumsum(soft_label, dim=-1)
                prob = torch.rand(cum_soft_label.size(0)).reshape(-1, 1).cuda()
                hard_label = n_classes - torch.cumsum(cum_soft_label > prob, dim=-1)[..., -1]
                hard_loss, soft_loss = get_hard_and_soft_loss(
                    semantic_criterion, output[hard_label_idx], hard_label[hard_label_idx],
                    semantic_criterion, output[~hard_label_idx], hard_label[~hard_label_idx])
            else:
                if cfg.SOFT_LABEL.thres.enabled:
                    hard_loss, soft_loss = get_hard_and_soft_loss(
                        semantic_criterion, output[hard_label_idx], top1_label[hard_label_idx],
                        semantic_criterion, output[~hard_label_idx], soft_label[~hard_label_idx])
                else:
                    hard_loss = output.sum() * 0.0
                    soft_loss = soft_semantic_criterion(output, soft_label)
            ret = {'hard_loss': hard_loss, 'soft_loss': soft_loss, 'output': output, 'preds': preds, 'labels': labels}
        else:
            loss = semantic_criterion(output, labels)
            ret = {'loss': loss, 'output': output, 'preds': preds, 'labels': labels}
        return ret

    if test:
        return test_model_fn
    else:
        return model_fn
