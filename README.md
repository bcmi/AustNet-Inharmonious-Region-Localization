# AustNet-Inharmonious-Region-Localization

![teaser](assets/teaser_.png)

This is the official code of the paper:
> Inharmonious Region Localization with Auxiliary Style Feature           
Penghao Wu, Li Niu, Liqing Zhang                                                             
[arXiv Paper](https://arxiv.org/abs/2210.02029), BMVC 2022


## Install
Clone this repo and build the environment

```
git clone https://github.com/bcmi/AustNet-Inharmonious-Region-Localization.git
cd AustNet-Inharmonious-Region-Localization
conda env create -f environment.yml --name Austnet
conda activate Austnet
```

Download the semantic segmentation network model weight through link [Google Drive](https://drive.google.com/file/d/1l1TZ6Nngwxc8g4qJT0rV3iDImstm3NuF/view?usp=share_link) or [Baidu Yun](https://pan.baidu.com/s/1SSRMI8QYCtRsG9E2zmOiOg) with code pfpy. Put the model weight in the HRNet-Semantic-Segmentation-HRNet-OCR folder.

## Datset
Please refer to [DIRL](https://github.com/bcmi/DIRL-Inharmonious-Region-Localization) to download the iHarmoney4 dataset.

## Training

To train AustNet, run

```
python train_austnet.py --dataset_root PATH_OF_THE_DATASET --logdir austnet_training_log --gpus NUMBER_OF_GPUS
```
To train AustNet_S, run
```
python train_austnet_s.py --dataset_root PATH_OF_THE_DATASET --logdir austnet_s_training_log --gpus NUMBER_OF_GPUS
```


## Pretrained Model

|Model| Google Drive Link| Baidu Yun Link|
|-----------|--------------|--------------|
| Austnet   | [Google Drive](https://drive.google.com/file/d/1q983Nr9ZW4UGTUzv8RNdpzpCOqBs-tID/view?usp=share_link) | [Baidu Yun](https://pan.baidu.com/s/1Z7r6p4LgJKqekZaJ3ctPpQ) code: m8ku   |
| Austnet_s | [Google Drive](https://drive.google.com/file/d/1A7q4mMiGe-s0jHXpccgkdHmMf65Fbad3/view?usp=share_link) | [Baidu Yun](https://pan.baidu.com/s/1LwAWRiFCceoX_wcLOtS5vQ) code: jrdi|


## Evaluation

To evaluate AustNet, run

```
python test_austnet.py --dataset_root PATH_OF_THE_DATASET --ckpt MODEL_WEIGHT_PATH
```
To evaluate AustNet_S, run
```
python test_austnet_s.py --dataset_root PATH_OF_THE_DATASET --ckpt MODEL_WEIGHT_PATH
```

## Citation

If you find our work or code helpful, please cite:
````
@inproceedings{Wu2022Inharmonious,
  title={Inharmonious Region Localization with Auxiliary Style Feature},
  author={Penghao Wu and Li Niu and Liqing Zhang},
  booktitle={BMVC},
  year={2022}
}
````

## Acknowledgement
Our code is based on repositories:
- [DIRL](https://github.com/bcmi/DIRL-Inharmonious-Region-Localization)
- [HRNet-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation)
