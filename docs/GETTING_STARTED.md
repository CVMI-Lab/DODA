# Getting Started
The dataset configs are located within [cfgs/dataset_configs](cfgs/dataset_configs), and the model configs are located within [cfgs](cfgs) for different settings.

### Dataset Preparation
#### 3D-FRONT Dataset
- Please download the subsampled and pre-processed [3D-FRONT dataset](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/ETXTrSJmy8lLikn0I_zsOisB5utQnffuqp3dGYwv-IIzDw?e=tpHJqc) and organize it as follows.
    ```
    DODA
    ├── data
    │   ├── 3dfront
    │   │   │── density1250
    │   │   │── train_list.txt
    │   │   │── val_list.txt
    ├── cfgs
    ├── dataset
    ```

#### ScanNet Dataset
- Please download the [ScanNet Dataset](http://www.scan-net.org/) and follow [PointGroup](https://github.com/dvlab-research/PointGroup/blob/master/dataset/scannetv2/prepare_data_inst.py) to pre-process the dataset as follows.

    ```
    DODA
    ├── data
    │   ├── scannetv2
    │   │   │── train_group
    │   │   │   │── scene0000_00.pth
    │   │   │   │── ...
    │   │   │── val_group
    ├── cfgs
    ├── dataset
    ```

#### S3DIS Dataset
- Please download the [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html#Download) and follow [PointNet](https://github.com/charlesq34/pointnet/blob/master/sem_seg/collect_indoor3d_data.py) to pre-process the dataset as follows or directly download the pre-processed data [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/Ed4LYh7wwp1CkVp4OfpfAocBvVy52EPO4CtK0vSmKR3E6w?e=fgNfXl).
    ```
    DODA
    ├── data
    │   ├── s3dis
    │   │   │── trainval_fullarea
    │   │   │   │── Area_1_Conference_1.npy
    │   │   │   │── ...
    ├── cfgs
    ├── dataset
    ```


### Training and Inference

#### pretrain
```
sh scripts/train.sh ${NUM_GPUS} train --cfg_file ${CONFIG_FILE} ${PY_ARGS}
```
For instance, if you want to the train the pretrained model for 3D-FRONT $\rightarrow$ ScanNet with 8 GPUs:
```
sh scripts/train.sh 8 train --cfg_file cfgs/da_front3d_scannet/spconv.yaml
```

#### self-train
```
sh scripts/train.sh ${NUM_GPUS} st --cfg_file ${CONFIG_FILE} ${PY_ARGS}
```
For instance, if you want to the train the model for 3D-FRONT $\rightarrow$ ScanNet with 8 GPUs:
```
sh scripts/train.sh 8 st --cfg_file cfgs/da_front3d_scannet/spconv_st.yaml --weight output/da_front3d_scannet/spconv/default/ckpt/best_train.pth
```
Notice that you need to select the best model as your pretrain model, because the performance of adapted model is quite unstable.

#### Test a model
```
sh scripts/test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} ${PY_ARGS}
```
For instance, if you want to the test the model for 3D-FRONT $\rightarrow$ ScanNet with 8 GPUs:
```
sh scripts/test.sh 8 --cfg_file cfgs/da_front3d_scannet/spconv_st.yaml --ckpt output/da_front3d_scannet/spconv/default/default/ckpt/best_train.pth
```
Notice that you also need to focus on the performance of the best model.