## CV-WSL-MIS
Exploring CNN and ViT for Weakly-Supervised Medical Image Segmentation

## Requirements
* [Pytorch]
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......


## Usage

1. Clone the repo:
```
git clone https://github.com/ziyangwang007/CV-SSL-MIS.git 
cd CV-SSL-MIS
```


2. Train the model
```
cd code
python train_XXX.py --root_path ../data/XXX --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX
```

3. Test the model
```
python test_XXX.py -root_path ../data/XXX --exp ACDC/XXX -model XXX --num_classes 4 --labeled_num XXX
```

You can choose model, dataset, experiment name, iteration number, batch size and etc in your command line, or leave it with default option.


## References

Please consider citing the following works, if you use in your research/projects:

	TBC



## Acknowledgement

This code is mainly borrowed xxx.
Some of the other code is based on xxx.
