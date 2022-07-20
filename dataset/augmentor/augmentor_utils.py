import open3d as o3d
import math
import cv2
import numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
import scipy.stats

from lib.pointgroup_ops.functions import pointgroup_ops


def check_key(key):
    exist = key is not None
    if not exist:
        return False
    if isinstance(key, bool):
        enabled = key
    elif isinstance(key, dict):
        enabled = key.get('enabled', True)
    else:
        enabled = True
    return enabled


def check_p(key):
    return (not isinstance(key, dict)) or ('p' not in key) or (np.random.rand() < key['p'])


def filter_by_index(e_list, idx):
    filtered_e_list = list()
    for e in e_list:
        filtered_e_list.append(e[idx])
    return filtered_e_list


def get_instance_ids(dataset_name, class_names):
    if dataset_name == 's3dis' or (dataset_name == 'front3d' and class_names[-1] == 'column'):
        return np.array([2,3,4,7])
    elif dataset_name == 'scannet' or (dataset_name == 'front3d' and class_names[-1] == 'desk'):
        return np.array([2,3,4,5,6,9,10])
    elif dataset_name == 'nyu':
        return np.array([2,3,4,7])
    else:
        raise NotImplementedError


def get_position_ids(dataset_names):
    # floor = 0, ceiling=1, random=2
    if dataset_names == 's3dis':
        return np.array([1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0])
    elif dataset_names == 'scannet':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0])
    elif dataset_names == 'nyu':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1])
    else:
        raise NotImplementedError


# ==== ELASTIC ====
def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(x).max(0).astype(np.int32) // gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag



# ==== SCENE AUG ====
def scene_aug(aug, xyz):
    assert xyz.ndim == 2
    m = np.eye(3)
    if check_key(aug.jitter):
        m += np.random.randn(3, 3) * 0.1
    if check_key(aug.flip) and check_p(aug.flip):
        m[0][0] *= -1  # np.random.randint(0, 2) * 2 - 1  # flip x randomly
    if check_key(aug.rotation) and check_p(aug.rotation):
        theta_x = (np.random.rand() * 2 * math.pi - math.pi) * aug.rotation.value[0]
        theta_y = (np.random.rand() * 2 * math.pi - math.pi) * aug.rotation.value[1]
        theta_z = (np.random.rand() * 2 * math.pi - math.pi) * aug.rotation.value[2]
        Rx = np.array \
            ([[1, 0, 0], [0, math.cos(theta_x), -math.sin(theta_x)], [0, math.sin(theta_x), math.cos(theta_x)]])
        Ry = np.array \
            ([[math.cos(theta_y), 0, math.sin(theta_y)], [0, 1, 0], [-math.sin(theta_y), 0, math.cos(theta_y)]])
        Rz = np.array \
            ([[math.cos(theta_z), math.sin(theta_z), 0], [-math.sin(theta_z), math.cos(theta_z), 0], [0, 0, 1]])
        m = np.matmul(m, Rx.dot(Ry).dot(Rz))
    xyz = np.matmul(xyz, m)
    return xyz


# ==== VIRTUAL SCAN SIMULATION ====
def virtual_scan_simulation(param, xyz, labels, class_names, ignore_label=255):
    # ===== occlusion simulation ======
    selected_idx = occlusion_simulation(param, xyz, labels, class_names, ignore_label=ignore_label)
    # ===== noise simulation ======
    xyz = noise_simulation(param, xyz)
    return xyz, selected_idx


def occlusion_simulation(param, xyz, labels, class_names, ignore_label=255):
    # === initialize to_selected_idx
    whole_idx = np.arange(xyz.shape[0])
    to_select_mask = (labels != ignore_label)
    to_select_idx = whole_idx[to_select_mask]
    _selected_xyz = xyz[to_select_mask]  # delete ignored label
    if _selected_xyz.shape[0] == 0:
        return to_select_mask
    _selected_xyz_c = (_selected_xyz.min(0) + _selected_xyz.max(0)) / 2.0
    _xyz = _selected_xyz - np.array([_selected_xyz_c[0], _selected_xyz_c[1], _selected_xyz.min(0)[2]])
    selected_mask = np.zeros(xyz.shape[0], dtype=bool)

    # ====  get candidate camera positions
    camera_locs = get_camera_candidate_locations(_xyz, labels, to_select_mask, class_names)
    if camera_locs.shape[0] == 0:
        return to_select_mask
    selected_camera = 0
    _xyz_wall = _xyz[labels[to_select_mask] == class_names.index('wall')]
    views = param['value']
    try_times = 0
    while (selected_camera < views):
        # ====  random select camera (view-point)
        idx = np.random.randint(camera_locs.shape[0])
        camera = camera_locs[idx]
        # ==== random select the interest point
        if _xyz_wall.shape[0] > 0:
            interest_point = _xyz_wall[np.random.choice(_xyz_wall.shape[0])]
        else:
            interest_point = np.array([0, 0, 0])
        _camera_f = camera - interest_point
        _xyz_f = _xyz - interest_point
        radius = param['radius']
        # ==== determine the view range
        pcd = o3d.geometry.PointCloud()
        view_range_mask = get_view_range_mask(
            _xyz_f, _camera_f, mode=param['mode'], **{'camera_view': param['camera_view']})
        view_range_idx = to_select_idx[view_range_mask]
        if view_range_mask.sum() < 10:
            try_times += 1
            if try_times > np.maximum(5, views):
                return to_select_mask
            continue
        # ==== determine the visible points
        pcd.points = o3d.utility.Vector3dVector(_xyz_f[view_range_mask])
        _, pt_map = pcd.hidden_point_removal(_camera_f, radius)
        pt_map = np.array(pt_map)
        visible_idx = view_range_idx[pt_map]
        selected_mask[visible_idx] = True
        selected_camera += 1

    return selected_mask


def noise_simulation(param, xyz):
    # ===== noise simulation ======
    if check_key(param.random_jitter) and check_p(param.random_jitter):
        jitter_scale = param.random_jitter.value
        random_noise = (np.random.rand(xyz.shape[0], xyz.shape[1]) - 0.5) * jitter_scale
        xyz += random_noise
    return xyz


def get_camera_candidate_locations(_xyz, labels, to_select_mask, class_names):
    # === voxelize points to voxels
    vox_scale = 10
    height = _xyz[..., 2].max()
    vox_xyz = _xyz[..., :3] * vox_scale
    vox_xyz_min = vox_xyz.min(0)
    vox_xyz -= vox_xyz_min
    vox_xyz = np.concatenate((np.zeros((vox_xyz.shape[0], 1)), vox_xyz), 1).astype(np.int64)
    vox_xyz[..., 3] = 0
    vox_locs, p2v_map, _ = pointgroup_ops.voxelization_idx(torch.from_numpy(vox_xyz).long(), 1, 4)
    vox_locs = vox_locs[..., 1:].cpu().numpy()
    # === get instance voxel coordinates
    if 'ceiling' in class_names:
        vox_inst_locs = vox_locs[p2v_map[(labels[to_select_mask] != class_names.index('floor')) \
            & (labels[to_select_mask] != class_names.index('ceiling'))]]
    else:
        vox_inst_locs = vox_locs[p2v_map[labels[to_select_mask] != class_names.index('floor')]]
    # === get non-occupied floor voxel coordinates
    img = np.zeros(vox_locs.max(0)[:2] + 3, dtype=np.uint8)
    img[(vox_locs[..., 0] + 1), (vox_locs[..., 1] + 1)] = 255
    img[(vox_inst_locs[..., 0] + 1), (vox_inst_locs[..., 1] + 1)] = 0
    # === erosion to shrink boundary
    kernel = np.ones((min(int(vox_scale), int(img.shape[0] / vox_scale)), min(int(vox_scale), int(img.shape[1] / vox_scale))), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    camera_vox_locs = np.where(erosion > 0)
    camera_vox_locs = np.concatenate((camera_vox_locs[0].reshape(-1, 1) - 1, camera_vox_locs[1].reshape(-1, 1) - 1), 1)
    # === recover point coordinate, get camera candidate locations
    camera_locs = (camera_vox_locs + vox_xyz_min[:2]) / vox_scale
    camera_height = np.random.rand() * height / 2.0 + height / 2.0
    camera_locs = np.concatenate((camera_locs, np.full((camera_locs.shape[0], 1), camera_height)), 1)
    return camera_locs


def get_view_range_mask(_xyz_f, _camera_f, mode=4, **kwargs):
    return eval('get_view_range_mask_%s' % mode)(_xyz_f, _camera_f, **kwargs)



def get_view_range_mask_fixed(_xyz_f, _camera_f, **kwargs):
    if _camera_f[2] > 0:
        visible_mask = (_xyz_f[..., 0] * _camera_f[0] + _xyz_f[..., 1] * _camera_f[1] <= (
            _camera_f[0] ** 2 + _camera_f[1] ** 2)) & (_xyz_f[..., 2] < _camera_f[2])
    else:
        visible_mask = (_xyz_f[..., 0] * _camera_f[0] + _xyz_f[..., 1] * _camera_f[1] <= (
            _camera_f[0] ** 2 + _camera_f[1] ** 2)) & (_xyz_f[..., 2] > _camera_f[2])
    return visible_mask


def get_view_range_mask_parallel(_xyz_f, _camera_f, **kwargs):
    camera_view = kwargs['camera_view']
    camera_view_angle = camera_view / 180.0 * np.pi
    pitch_angle = np.arcsin(-_camera_f[2] / (np.linalg.norm(_camera_f) + 10e-10))
    z_upper = np.sqrt(_camera_f[0] ** 2 + _camera_f[1] ** 2) * np.tan(pitch_angle + camera_view_angle / 2.) + \
              _camera_f[2]
    z_lower = np.sqrt(_camera_f[0] ** 2 + _camera_f[1] ** 2) * np.tan(pitch_angle - camera_view_angle / 2.) + \
              _camera_f[2]
    visible_mask = (_xyz_f[..., 0] * _camera_f[0] + _xyz_f[..., 1] * _camera_f[1] <= (
            _camera_f[0] ** 2 + _camera_f[1] ** 2)) & (_xyz_f[..., 2] < z_upper) & (_xyz_f[..., 2] > z_lower)
    return visible_mask


def get_view_range_mask_perspective(_xyz_f, _camera_f, **kwargs):
    camera_view = kwargs['camera_view']
    camera_view_angle = camera_view / 180.0 * np.pi
    pitch_angle = np.arcsin(-_camera_f[2] / (np.linalg.norm(_camera_f) + 10e-10))
    _camera_f_xy = np.sqrt(_camera_f[0] ** 2 + _camera_f[1] ** 2)
    _xyz_f_xy = _xyz_f[..., :2].dot(_camera_f[:2].reshape(2, 1)).reshape(-1) / _camera_f_xy
    z_upper = (_camera_f_xy - _xyz_f_xy) * np.tan(pitch_angle + camera_view_angle / 2.) + \
              _camera_f[2]
    z_lower = (_camera_f_xy - _xyz_f_xy) * np.tan(pitch_angle - camera_view_angle / 2.) + \
              _camera_f[2]
    visible_mask = (_xyz_f[..., 0] * _camera_f[0] + _xyz_f[..., 1] * _camera_f[1] <= (
            _camera_f[0] ** 2 + _camera_f[1] ** 2)) & (_xyz_f[..., 2] < z_upper) & (_xyz_f[..., 2] > z_lower)
    return visible_mask


# ==== TAIL-AWARE CUBOID MIXING ====
def tacm(param, split_sampler, dataset_name, class_names, pc1, pc2):
    # ===== pre-process
    xyz_middle, label,  *others = pc1
    xyz_middle2, label2, *others2 = pc2
    xyz_middle -= (xyz_middle.min(0) + xyz_middle.max(0)) / 2.0
    xyz_middle2 -= (xyz_middle2.min(0) + xyz_middle2.max(0)) / 2.0

    # ===== cuboid split
    split_coord_xyz, split_range = split_space(xyz_middle, param.split)
    split_coord_xyz2, split_range2 = split_space(xyz_middle2, param.split)
    split_idx, split_info = get_split_idx(
        param, xyz_middle, label, split_coord_xyz, split_range, calc_split=True, n_classes=len(class_names)
    )
    split_idx2, _ = get_split_idx(param, xyz_middle2, label2, split_coord_xyz2, split_range2)
    total_splits = param.split[0] * param.split[1] * param.split[2]
    split_status = split_info['split_status']

    # ===== cuboid mixing, 1: source, 0: target, e.g. [0,1,1,0]
    concat = check_p(param)
    if concat:
        concat_seq = (np.random.rand(total_splits) < param.mix_ratio).astype(np.uint8)
    else:
        concat_seq = np.array([0] * total_splits)

    # ===== cuboid permutation. e.g. [2,1,4,3]
    permute = check_p(param.permute_cuboid)
    if permute:
        # target
        split_idx, permuted_cuboid_coord_xyz, _, split_others = permute_cuboid(
            param.permute_cuboid, int(total_splits - concat_seq.sum()), split_idx, split_coord_xyz, split_range,
            xyz=xyz_middle, label=label, split_status=split_status
        )
        split_status = split_others['split_status']
        # source
        split_idx2, permuted_cuboid_coord_xyz2, _, _ = permute_cuboid(
            param.permute_cuboid, int(concat_seq.sum()), split_idx2, split_coord_xyz2, split_range2,
            xyz=xyz_middle2, label=label2
        )
        permuted_cuboid_coord_xyzs = [permuted_cuboid_coord_xyz, permuted_cuboid_coord_xyz2]
    else:
        tar_mapper = np.where(concat_seq == 0, np.cumsum(concat_seq == 0), 0) - 1
        split_idx = tar_mapper[split_idx]
        split_status = split_status[concat_seq == 0]
        src_mapper = np.where(concat_seq == 1, np.cumsum(concat_seq == 1), 0) - 1
        split_idx2 = src_mapper[split_idx2]

    # ===== get target tail-aware cuboids
    tail_cuboids = tail_cuboids_from_sampler(
        param, int(total_splits - concat_seq.sum()), split_status, split_sampler, label=label
    )

    split_idxs = [split_idx, split_idx2]
    split_coord_xyzs = [split_coord_xyz, split_coord_xyz2]
    split_ranges = [split_range, split_range2]
    xyz_middles = [xyz_middle, xyz_middle2]
    masks = [np.zeros(xyz_middle.shape[0], dtype=bool), np.zeros(xyz_middle2.shape[0], dtype=bool)]
    
    # ===== get mixed cuboids
    new_split_coords = []
    new_split_range = []

    concat_seq_tar = concat_seq[concat_seq == 0]
    for i in range(len(tail_cuboids)):
        concat_seq_tar[-i - 1] = 2
    concat_seq[concat_seq == 0] = concat_seq_tar

    ptrs = [0, 0, 0]
    for s in range(total_splits):
        domain = concat_seq[s]
        if domain == 2:  # from tail-aware cuboid sampler
            _split = tail_cuboids[ptrs[domain]]
            _split[..., 0:3] += split_coord_xyzs[0][s] - _split[..., 0:3].max(0)  # use the target domain split_coord
            _split[..., 0:3] = transform_xyz(_split[..., 0:3], param)
            new_split_coords.append(split_coord_xyzs[0][s])
            new_split_range.append(split_ranges[0][s])
        else:
            xyz_idx_s = split_idxs[domain] == ptrs[domain]  # the head of the queue
            if permute:
                xyz_middles[domain][xyz_idx_s] += split_coord_xyzs[domain][s] - permuted_cuboid_coord_xyzs[domain][ptrs[domain]]
            xyz_middles[domain][xyz_idx_s] = transform_xyz(xyz_middles[domain][xyz_idx_s], param)
            masks[domain][xyz_idx_s] = True
            ptrs[domain] += 1
            new_split_coords.append(split_coord_xyzs[domain][s])
            new_split_range.append(split_ranges[domain][s])

    xyz_middle, label = filter_by_index([xyz_middles[0], label], masks[0])
    xyz_middle2, label2 = filter_by_index([xyz_middles[1], label2], masks[1])
    for key in others[0]:
        others[0][key] = others[0][key][masks[0]]
        others2[0][key] = others2[0][key][masks[1]]

    if len(tail_cuboids) > 0:
        tail_cuboids = np.concatenate(tail_cuboids, axis=0)
    else:
        tail_cuboids = np.random.rand(0, 4).astype(xyz_middle.dtype)
    xyz_middle = np.concatenate((xyz_middle, xyz_middle2, tail_cuboids[..., 0:3]), axis=0)
    xyz_middle -= xyz_middle.mean(0)
    label = np.concatenate((label, label2, tail_cuboids[..., 3]), axis=0)
    others_merged = {}
    for key in others[0]:
        others_merged[key] = np.concatenate((others[0][key], others2[0][key]), axis=0)
    others_merged['pc1_mask'] = np.where(np.arange(label.shape[0]) < masks[0].sum(), True, False)
    others_merged['pc2_mask'] = ~others_merged['pc1_mask']
    others_merged['tar_tail_splits'] = split_info['tail_splits']
    if param.cuboid_queue.enabled:
        others_merged['tar_splits_class_ratio'] = \
            np.histogram(tail_cuboids[..., 3], bins=np.arange(len(class_names) + 1))[0][split_sampler.tail_class_idx]
    else:
        others_merged['tar_splits_class_ratio'] = np.zeros(3)
    return xyz_middle, label, others_merged


def get_split_idx(param, xyz, label, split_coord_xyz, split_range, **args):
    split_idx = np.full(xyz.shape[0], 255, dtype=np.int8)
    tail_splits = [[] for _ in range(param.cuboid_queue.num_class)]
    split_status = []
    for s in range(split_coord_xyz.shape[0]):
        xyz_idx_s = xyz_idx_in_split(xyz, split_coord_xyz[s], split_range[s])
        split_idx[xyz_idx_s] = s
        if param.cuboid_queue.enabled and 'calc_split' in args and xyz_idx_s.sum() > 0 and label[xyz_idx_s].min() < 255:
            class_ratio = np.histogram(label[xyz_idx_s], bins=np.arange(args['n_classes'] + 1), density=True)[0]
            status = (class_ratio > param.cuboid_queue.class_thres)[param.cuboid_queue.tail_class_idx]
            split_status.append(np.any(status))
            for i in range(param.cuboid_queue.num_class):
                if status[i]:
                    tail_splits[i].append(np.concatenate((xyz[xyz_idx_s], label[xyz_idx_s].reshape(-1, 1)), axis=-1))
        else:
            split_status.append(False)
    return split_idx, {'tail_splits': tail_splits, 'split_status': np.array(split_status)}


def permute_cuboid(param, n, split_idx, split_coord_xyz, split_range, **args):
    n_split = split_coord_xyz.shape[0]
    permuted_idx = np.random.permutation(np.arange(n_split))
    permuted_cuboid_idx = np.argsort(permuted_idx)[split_idx]
    permuted_cuboid_coord_xyz = split_coord_xyz[permuted_idx][:n]
    permuted_cuboid_range = split_range[permuted_idx][:n]

    if 'split_status' in args:
        args['split_status'] = args['split_status'][permuted_idx][:n]

    return permuted_cuboid_idx, permuted_cuboid_coord_xyz, permuted_cuboid_range, args


def tail_cuboids_from_sampler(param, n, split_status, split_sampler, **args):
    # replace original splits with splits in cuboid_queue
    supp_queues = []
    if param.cuboid_queue.enabled:
        # split_status = get_split_status(param.cuboid_queue.class_thres, args['label'], n, split_idx)
        n_eligible_splits = param.cuboid_queue.num_cuboid
        n_eligible_splits = int((n_eligible_splits // 1) + int(np.random.rand() < n_eligible_splits % 1))
        n_curr_splits = split_status.sum()
        supp_num = min(n, n_eligible_splits) - n_curr_splits
        if supp_num > 0:
            supp_queues = split_sampler.get_split(supp_num)
    return supp_queues


def transform_xyz(xyz, param):
    if xyz.shape[0] > 0:
        mv_direc = - xyz.mean(0)
        xyz += mv_direc * 0.1
    return xyz


# random split
def split_space(xyz, split):
    assert len(split) == 3  # 3 dimensions, [2, 1, 1]
    xyz_min, xyz_max = xyz.min(0), xyz.max(0)
    xyz_range = xyz_max - xyz_min + 0.001
    split_ratio = (1.0 / np.array(split, dtype=np.float32).reshape(3, 1)).tolist()
    split_ratio = [np.cumsum(r * split[i]) for (i, r) in enumerate(split_ratio)]  # [[0.5,1],[1],[1]]
    split_ratio = [np.append(r[:-1] + (np.random.rand() - 0.5) * 0.2, 1.0) for r in split_ratio]  # [[0.44,1],[1],[1]]
    split_ratio_range = [np.append(r[0], np.array(r[1:]) - np.array(r[:-1])) for r in split_ratio]
    total_splits = split[0] * split[1] * split[2]
    split_coord = np.array([
        [split_ratio[0][i // (split[1] * split[2])] * xyz_range[0] + xyz_min[0],
         split_ratio[1][i % (split[1] * split[2]) // split[2]] * xyz_range[1] + xyz_min[1],
         split_ratio[2][i % (split[2])] * xyz_range[2] + xyz_min[2]]
        for i in range(total_splits)])  # (total_splits, 3), [[0.5, 0.4, 0.3], [1.0, 0.4, 0.3]]
    split_range = np.array([
        [split_ratio_range[0][i // (split[1] * split[2])] * xyz_range[0],
         split_ratio_range[1][i % (split[1] * split[2]) // split[2]] * xyz_range[1],
         split_ratio_range[2][i % (split[2])] * xyz_range[2]]
        for i in range(total_splits)])  # (total_splits, 3),
    return split_coord, split_range


def xyz_idx_in_split(xyz, split_max, range):
    return np.all(xyz < split_max, axis=-1) & np.all(xyz >= split_max - range, axis=-1)


# ==== CROP ====
def crop(xyz, full_scale, point_range, max_npoint):
    xyz_offset = xyz.copy()
    valid_idxs = (xyz_offset.min(1) >= 0)
    assert valid_idxs.sum() == xyz.shape[0]
    full_scale = np.array([full_scale[1]] * 3)
    room_range = xyz.max(0) - xyz.min(0)
    curr_scale = room_range[0] * room_range[1] * room_range[2]

    # crop room to fit voxel limit (2e9)
    if curr_scale > point_range:
        crop_scale = math.sqrt(point_range / curr_scale)
        full_scale = np.minimum(
            full_scale, np.array([crop_scale * room_range[0], crop_scale * room_range[1], room_range[2]])
        )
        valid_idxs = (xyz_offset < full_scale).sum(1) == 3

    # crop room to fit point_num limit
    while (valid_idxs.sum() > max_npoint):
        offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
        xyz_offset = xyz + offset
        valid_idxs = valid_idxs & (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
        full_scale[:2] -= 32

    return xyz_offset, valid_idxs
