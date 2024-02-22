## CV-WSL-MIS
Exploring CNN and ViT for Weakly-Supervised Medical Image Segmentation

## Requirements
* Pytorch
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

## DataSets
We use the ACDC dataset which you can find here [Official](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). The pre-processed dataset i.e. scribble can be download via [Google Drive](https://drive.google.com/file/d/1XR_Id0wdvXY9QeKtdOdgJHKVJ-nVr2j1/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1UX5NqeIc8RL6io-FKFTcxQ) with code 'u8bh' and put in 'data' folder. You can also simulate the scribble annotations with other dataset with the 'code/scribbles_generator.py' file.


## Usage

1. Clone the repo:
```
git clone https://github.com/ziyangwang007/CV-WSL-MIS.git 
cd CV-WSL-MIS
```


2. Train the model
```
cd code
```
You can choose model, dataset, experiment name, iteration number, batch size and etc in your command line, or leave it with default option.

Uncertainty Aware Mean Teacher, Rotation -> [[Paper Link](https://www.sciencedirect.com/science/article/pii/S0031320321005215)]
```
python train_weakly_supervised_ustm_2D.py 
```
Uncertainty Aware Mean Teacher, Rotation, Vision Transformer -> [[Paper Link](https://ieeexplore.ieee.org/abstract/document/10195028)]

```
python train_weakly_supervised_UAMT_ViT_2D.py 
```

Mean Teacher, Cross Pseudo Supervision, CNN, Vision Transformer -> [[Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-44992-5_1)]
```
python train_weakly_supervised_DCDPL_2D.py 
```
Gated CRF Loss -> [[Link](https://arxiv.org/abs/1906.04651)]
```
python train_weakly_supervised_pCE_GatedCRFLoss_2D.py 
```

3. Test the model

Test CNN-based model
```
python test_2D_fully.py -root_path ../data/XXX --exp ACDC/XXX
```

Test ViT-based model
```
python test_2D_fully_ViT.py -root_path ../data/XXX --exp ACDC/XXX
```

## References

Please consider citing the following works, if you use in your research/projects:
```
@inproceedings{wang2023weakly,
  title={Weakly Supervised Medical Image Segmentation Through Dense Combinations of Dense Pseudo-Labels},
  author={Wang, Ziyang and Voiculescu, Irina},
  booktitle={MICCAI Workshop on Data Engineering in Medical Imaging},
  pages={1--10},
  year={2023},
  organization={Springer}
}
```


## Acknowledgement

This code is mainly borrowed WSL4MIS, UNet, SwinUNet, USTM, and etc.
