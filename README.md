# DODA

**Data-oriented Sim-to-Real Domain Adaptation for 3D Indoor Semantic Segmentation**, ECCV 2022.

![framwork](./docs/framework.png)

**Authors**: Runyu Ding\*, Jihan Yang\*, Li Jiang, Xiaojuan Qi  (\* equal contribution)
[arXiv](https://arxiv.org/abs/2204.01599)


## Introduction
 In this work, we propose a Data-Oriented Domain Adaptation (DODA) framework on sim-to-real domain adaptation for 3D indoor semantic segmentation. Our empirical studies demonstrate two unique challengeds in this setting: the  point pattern gap and the context gap caused by different sensing mechanisms and layout placements across domains. Thus, we propose virtual scan simulation to imitate real-world point cloud patterns and tail-aware cuboid mixing to alleviate the interior context gap with a cuboid-based intermediate domain. The first unsupervised sim-to-real adaptation benchmark on 3D indoor semantic segmentation is also built on 3D-FRONT, ScanNet and S3DIS along with 8 popular UDA methods. 

## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.


## Getting Started
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage.

#### Supported features and ToDo List
- [ ] Release code
- [ ] Support pre-trained model
- [ ] Support other baseline methods

## ModelZoo

#### 3D-FRONT -> ScanNet

| method | mIoU | download |
|:------:|:--:|--|
| DODA (only VSS) | | |
| DODA | | |


#### 3D-FRONT -> S3DIS

| method | mIoU | download |
|:------:|:--:|--|
| DODA (only VSS) | | |
| DODA | | |


## Acknowledgments
Our code base is partially borrowed from [PointGroup](https://github.com/dvlab-research/PointGroup), [PointWeb](https://github.com/hszhao/PointWeb) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Citation

If you find this project useful in your research, please consider cite:
```
@article{ding2022doda,
  title={DODA: Data-oriented Sim-to-Real Domain Adaptation for 3D Indoor Semantic Segmentation},
  author={Ding, Runyu and Yang, Jihan and Jiang, Li and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2204.01599},
  year={2022}
}
```