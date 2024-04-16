## Installation

Preparation: A Ubuntu system with GPU with GPU memory larger than 13GB.

Install Nvidia driver and CUDA Toolkit.

```
$ nvidia-smi  # check driver
$ nvcc --version # check toolkit
```

Install `Python` and `NumPy`. Please make sure your NumPy version is at least 1.18.

Install `PyTorch` with `CUDA` -- A version than (PyTorch 1.5.1, CUDA 10.1) may be problematic.

Install `TensorFlow` (for `TensorBoard`) -- This repo is tested with TensorFlow 2.2.0.

Compile the CUDA code for [PointNet++](https://arxiv.org/abs/1706.02413), which is used in the backbone network:

```
cd pointnet2
python setup.py install
```

If there is a problem, please refer to [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch#building-only-the-cuda-kernels)

Compile the CUDA code for general 3D IoU calculation in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

Different from 3DIoUMatch, please install the full version of OpenPCDet since the sampling part needs the tools in OpenPCDet.

Install dependencies:

```
pip install -r requirements.txt
```

## Datasets

### ScanNet

Please follow the instructions in `scannet/README.md`. using the download script with
`-o $(pwd) --types _vh_clean_2.ply .aggregation.json _vh_clean_2.0.010000.segs.json .txt` options to download data.

### SUNRGB-D

Please follow the instructions in `sunrgbd/README.md`.

## Pre-training

Please run:

```shell
sh scripts/run_pretrain.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST>
```

For example:

```shell
sh scripts/run_pretrain.sh 0 pretrain_scannet scannet scannetv2_train_0.1.txt
```

```shell
sh scripts/run_pretrain.sh 0 pretrain_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt
```

## Training

Please run:

```shell
sh scripts/run_train.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <PRETRAIN_CKPT>
```

For example, use the downloaded models:

```shell
sh scripts/run_train.sh 0 train_scannet scannet scannetv2_train_0.1.txt ckpts/scan_0.1_pretrain.tar
```

```shell
sh scripts/run_train.sh 0 train_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt ckpts/sun_0.05_pretrain.tar
```

You may modify the script by adding `--view_stats`  to load labels on unlabeled data and view the statistics on the unlabeled data (e.g. average IoU, class prediction accuracy).

## Evaluation

Please run:

```shell
sh scripts/run_eval.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <CKPT>
```

For example, use the downloaded models:

```shell
sh scriptsrun_eval.sh 0 eval_scannet scannet scannetv2_train_0.1.txt ckpts/scan_0.1.tar
```

```shell
sh scripts/run_eval.sh 0 eval_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt ckpts/sun_0.05.tar
```

For evaluation with IoU optimization, please run:

Please run:

```shell
sh scripts/run_eval_opt.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <CKPT> <OPT_RATE>
```

The number of steps (of optimization) is by default 10.

## Acknowledgements

Our implementation uses code from the following repositories:

- [3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection](https://github.com/THU17cyz/3DIoUMatch)
- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://github.com/facebookresearch/votenet)
- [SESS: Self-Ensembling Semi-Supervised 3D Object Detection](https://github.com/Na-Z/sess)
- [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
