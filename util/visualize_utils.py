import numpy as np

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168],[75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])


# scannet
SCANNET_SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SCANNET_SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator',
                                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])
SCANNET_DA_SEMANTIC_NAMES = np.array(["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
                                      "bookshelf", "desk", "ceiling", "unannotated"])
SCANNET_CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160],
    'ceiling': [0, 255, 0]
}
SCANNET_SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table',
                             8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture', 12: 'counter', 14: 'desk',
                             16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',  34: 'sink',
                             36: 'bathtub', 39: 'otherfurniture'}


# s3dis
S3DIS_SEMANTIC_NAMES = np.array(["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair",
                                 "sofa", "bookshelf", "board", "clutter"])
S3DIS_DA_SEMANTIC_NAMES = np.array(["wall", "floor", "chair", "sofa", "table", "door", "window", "bookshelf",
                                    "ceiling", "beam", "column", 'ignore'])

S3DIS_SEMANTIC_IDX2NAME = {1: 'ceiling', 2: 'floor', 3: 'wall', 4: 'beam', 5: 'column', 6: 'window', 7: 'door',
                           8: 'table', 9: 'chair', 10: 'sofa', 11:'bookshelf', 12: 'board', 13: 'clutter'}

S3DIS_CLASS_COLOR = {
    'ceiling':[0, 255, 0],
    'floor':[0, 0, 255],
    'wall':[0, 255, 255],
    'beam':[255, 255, 0],
    'column':[255, 0, 255],
    'window':[100, 100, 255],
    'door':[200, 200, 100],
    'table':[170, 120, 200],
    'chair':[255, 0, 0],
    'sofa':[200, 100, 100],
    'bookshelf':[10, 200, 100],
    'board':[200, 200, 200],
    'clutter':[50, 50, 50],
    'ignore': [0, 0, 0]
}
