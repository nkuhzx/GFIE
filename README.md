# GFIE: A Dataset and Baseline for Gaze-Following from 2D to 3D in Indoor Environments

This repository is the pytorch implementation of our CVPR 2023 work: GFIE: A Dataset and Baseline for Gaze-Following from 2D to 3D in Indoor Environments

[![](https://shields.io/badge/homepage-website-pink?logo=appveyor&style=for-the-badge)](https://sites.google.com/view/gfie)

## Prerequisite
1. Clone our repo
```bash
git clone https://github.com/nkuhzx/GFIE
```

2. (optional) Use the `.yaml` file to re-create the environment we used
```bash
conda env create -f gfie.yml
conda activate gfie.yml
```

3. Set the available GPU in [`inference.py`](inference.py) and [`main.py`](inference.py)
```bash
os.environ['CUDA_VISIBLE_DEVICES'] = ID
```
`ID` is available gpu id.

## Dataset Preparation

1. Please download the [GFIE dataset](https://drive.google.com/drive/folders/1AKA1jCVdrMmLIXqTeNNCFo5VnUrAcplq?usp=sharing) and [CAD120 dataset](https://drive.google.com/drive/folders/1PNe5AYHd2pdMJin4YzO1ntsdpoZH7pQb?usp=sharing) from the Google Drive. 
2. Unzip rgb.zip and depth.zip into corresponding folders.
3. The file structure should be as follows:

```bash
├── GFIE_dataset/
|   ├── rgb/
|   |   ├── train/
|   |   |   ├── scene1/ # scene id
|   |   |   |   └── '*.jpg' # n frame jpg`
|   |   |   └── ...
|   |   ├── valid/
|   |   |   └── ...
|   |   ├── test/
|   |   |   └── ...
|   ├── depth/
|   |   ├── train/
|   |   |   ├── scene1/ # scene id
|   |   |   |   └── '*.npy' # n frame npy`
|   |   ├── valid/
|   |   |   └── ...
|   |   ├── test/
|   |   |   └── ...
|   ├── CameraKinect.npy
|   ├── train_annotation.txt
|   ├── valid_annotation.txt
|   └── test_annotation.txt
├── CAD120_dataset/
|   ├── rgb/
|   |   ├── D1S1A001/
|   |   |   └── 'RGB_*.png' # n frame png`
|   |   └── ...
|   ├── depth/
|   |   ├── D1S1A001/
|   |   |   └── 'Depth_*.png' # n frame png`
|   |   └── ...
|   ├── CameraKinect.npy
└── └── annotation.txt
```
`Note: The decompressed file is about 350 GB, please check the capacity of your hard disk to ensrure that the dataset can be stored.`


4. Then you need to modify the address of the configuration ([cad120evaluation.yaml](config/cad120evaluation.yaml) | [gfiebenchmark.yaml](config/gfiebenchmark.yaml))

[gfiebenchmark.yaml](config/gfiebenchmark.yaml)
```bash
DATASET:
  root_dir: "YOUR_PATH/GFIE_dataset"
```

[cad120evaluation.yaml](config/cad120evaluation.yaml) 
```bash
DATASET:
  root_dir: "YOUR_PATH/CAD120_dataset"
```

`YOUR_PATH` is the root path of `GFIE_dataset` and `CAD120_dataset`.

## Getting start

### Training

After all the prerequisites are met, you can train the GFIE baseline method we proposed in the paper.

1. Set the path `STORE_PATH` to save the model file in the [gfiebenchmark.yaml](config/gfiebenchmark.yaml)

```bash
TRAIN:
  store: "STORE_PATH"
```

2. Download the [pre-trained model weights](https://drive.google.com/file/d/1eXWy4-bg5BQeCHbyH6R1dbWceGCNKPe4/view?usp=sharing) to `PATH`, and then set the path of pre-trained weights in [gfiebenchmark.yaml](config/gfiebenchmark.yaml)

```bash
MODEL:
  backboneptpath: "PATH/ptbackbone.pt"
```

3. Then run the training procedure
```bash
python main.py
```

### Evaluation

1. Set the absolute path of the model weight `cpkt_PATH` in the [cad120evaluation.yaml](config/cad120evaluation.yaml) | [gfiebenchmark.yaml](config/gfiebenchmark.yaml)

```bash
OTHER:
  cpkt: "cpkt_PATH"
```

2. Run the inference program and the evaluation results will be displayed in the termainal.

```bash
# evaluation on GFIE dataset
python inference.py --mode gfie

# evaluation on CAD120 dataset
python inference.py --mode cad120
```

### Model weights

We also provide the model weights for evaluation.

[`gfiemodel.pt.tar`](https://drive.google.com/file/d/1VVpAC1z5sQA0niuA92nmQIRRnin7TvIH/view?usp=sharing)

## Citation
If you fine our dataset/code useful for your research, please cite our paper
```
@inproceedings{
hu2023gfie,
title={GFIE: a dataset and baseline for gaze-followiung from 2d to 3d in indoor environments},
author={Hu, Zhengxi and Yang, Yuxue and Zhai, Xiaolin and Yang, Dingye, and Zhou, Bohan and Liu, Jingtai},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2023}
}
```

## Acknowledgements

We would like to thank Eunji Chong for [her work publised on CVPR 2020](https://github.com/ejcgt/attention-target-detection) and others that have contributed to gaze-following.




