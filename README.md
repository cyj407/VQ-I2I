# VQ-I2I
PyTorch implementaton of our ECCV 2022 paper "Vector Quantized Image-to-Image Translation".
You can visit our project website [here](https://cyj407.github.io/VQ-I2I/).

In this paper, we propose a novel unified framework which is able to tackle image-to-image translation, unconditional generation of input domains and diverse extension based on an existing image.
<img src='fig/teaser.png' width="800px">

## Paper
[Vector Quantized Image-to-Image Translation](http://arxiv.org/abs/2207.13286) \
[Yu-Jie Chen*](cyj407.cs09g@nctu.edu.tw), [Shin-I Cheng*](shinicheng.cs09g@nctu.edu.tw), [Wei-Chen Chiu](walon@cs.nctu.edu.tw), [Hung-Yu Tseng](hungyutseng@fb.com), [Hsin-Ying Lee](hlee5@snap.com) \
European Conference on Computer Vision (ECCV), 2022 (* equal contribution)

Please cite our paper if you find it useful for your research.  
```
@inproceedings{chen2022eccv,
 title = {Vector Quantized Image-to-Image Translation},
 author = {Yu-Jie Chen and Shin-I Cheng and Wei-Chen Chiu and Hung-Yu Tseng and Hsin-Ying Lee},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2022}
}
```

## Installation and Environment
- Prerequisities: Python 3.6 & Pytorch (at least 1.4.0) 
- Clone this repo
```
git clone https://github.com/cyj407/VQ-I2I.git
cd VQ-I2I
```

- We provide a conda environment script, please run the following command after cloning our repo.
```
conda env create -f vqi2i_env.yml
```
## Datasets
- Yosemite (winter, summer) dataset: You can follow the instructions in CycleGAN [website](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to download the Yosemite (winter, summer) dataset.
- AFHQ (cat, dog, wildlife) You can follow the instructions in StarGAN v2 [website](https://github.com/clovaai/stargan-v2) to download the AFHQ (cat, dog, wildlife) dataset.
- Portrait (portrait, photography): 6452 photography images from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 1811 painting images downloaded and cropped from [Wikiart](https://www.wikiart.org/).
- Cityscapes (street scene, semantic labeling): 3475 street scenes and the corresponding semantic labelings from [the cityscapes dataset](https://www.cityscapes-dataset.com/).
> Please save the dataset images separately, e.g. Yosemite dataset:
> - `trainA` directory for training summer images.
> - `trainB` directory for training winter images.
> - `testA` directory for testing summer images.
> - `testB` directory for testing winter images.
## First Stage
### Train
#### Unpaired I2I task
```
python unpair_train.py --device <gpu_num> --root_dir <dataset_path> \
--dataset <dataset_name>\
--epoch_start <epoch_start> --epoch_end <epoch_end>
```
- You can also append arguments for hyperparameters, e.g.: `--ne <ne> --ed <ed> --z_channel <z_channel>`. 
#### Paired I2I task
```
python pair_train.py --device <gpu_num> --root_dir <dataset_path> \
--dataset <dataset_name>\
--epoch_start <epoch_start> --epoch_end <epoch_end>
```
- Used on Cityscapes dataset only.
- You can also append arguments for hyperparameters, e.g.: `--ne <ne> --ed <ed> --z_channel <z_channel>`. 
### Test (unpaired I2I translation.)
- Save the translation results.
```
python save_transfer.py --device <gpu_num> --root_dir <dataset_path> --dataset <dataset_name> \
--checkpoint_dir <checkpoint_dir> --checkpoint_epoch <checkpoint_epoch> \
--save_name <save_dir_name>
```
- `--atob True`: transfer domain A to domain b; otherwise, B to A.
- `--intra_transfer True`: enable intra-domain translation.
- You can also modify arguments for hyperparameters, e.g.: `--ne <ne> --ed <ed> --z_channel <z_channel>`. 
### Using the pre-trained models
- Download the [pre-trained models](https://mega.nz/file/JTMRBYBQ#BEOd5INGad-j4pv50ma_oFzAzqBbkYZiQnPYXFtp4C0), here we provide the pre-trained models for the four datasets.
    - Yosemite(summer, winter)256X256: --ed 512, --ne 512, --z_channel 256 
    - AFHQ(cat, dog)256X256: --ed 256, --ne 256, --z_channel 256
    - Portrait(portrait, photography)256X256: --ed 256, --ne 256, --z_channel 256  
    - Cityscapes(street scene, semantic labeling)256X256: --ed 256, --ne 64, --z_channel 128 
    
## Second stage
### Train
```
python autoregressive_train.py --device <gpu_num> --root_dir <dataset_path> \
--dataset <dataset_name> --first_stage_model <first_stage_model_path> \
--epoch_start <epoch_start> --epoch_end <epoch_end>
```
- You can also append arguments for hyperparameters, e.g.: `--ne <ne> --ed <ed> --z_channel <z_channel>`. 
### Test
#### Using the pre-trained models
- Download the [pre-trained transformer models](https://mega.nz/file/VX0CUbiC#1QbbngU5BKJJz5SAEx6geZe8kidXt2XjFAtSfNeDRmo), here we provide the pre-trained transformer model for the Yosemite dataset.
    - Yosemite(summer, winter)256X256: --ed 512, --ne 512, --z_channel 256 
#### Unconditional Generation
```
python save_uncondtional.py --device <gpu_num> \
--root_dir <dataset_path> --dataset <dataset_name> \
--first_stage_model <first_stage_model_path> \
--transformer_model <second_stage_model_path> \
--save_name <save_dir_name>
```
- `--sty_domain 'B'`: specify to generate domain B style images
#### Image Extension/Completion
##### Image extension
```
python save_extension.py --device <gpu_num> \
--root_dir <dataset_path> --dataset <dataset_name> \
--first_stage_model <first_stage_model_path> \
--transformer_model <second_stage_model_path> \
--save_name <save_dir_name>
```
- `--input_domain B`: select domain B images from the testing set as input.
- `--sty_domain A`: select domain A as the referenced styles to achieve translation.
- `--double_extension True`: enable the double-sided extension; default `False`.
- `--pure_extension True`: only extend the input images without translation; default `False`.
- `--extend_w <extend_pixels>`: extends for 128/192 pixels on the width; default `128`.
##### Image completion
```
python save_completion.py -device <gpu_num> \
--root_dir <dataset_path> --dataset <dataset_name> \
--first_stage_model <first_stage_model_path> \
--transformer_model <second_stage_model_path> \
--save_name <save_dir_name>
```
- `--input_domain B`: select domain B images from the testing set as input.
- `--sty_domain A`: select domain A as the referenced styles to achieve translation.
- `--pure_completion True`: only extend the input images without translation; default `True`.
- `--partial_input top-left`: given top-left corner image as the input. There are two more options, `left-half` (given the left-half image as input), and `top-half` (given the top-half image as input).
#### Transitional Stylization
- The demonstration of all applications (includes transitional stylization) are in `VQ-I2I-Applications.ipynb`
## Acknowledgments
Our code is based on [VQGAN](https://github.com/CompVis/taming-transformers).
The implementation of the disentanglement architecture is borrowed from [MUNIT](https://github.com/NVlabs/MUNIT).