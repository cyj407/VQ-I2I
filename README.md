# VQ-I2I
## Environment
- Prerequisities: Python 3.6 & Pytorch (at least 1.4.0) 
- We provide a conda environment script, please run the following command after cloning our repo.
```
conda env create -f vqi2i_env.yml
```
## First stage -- translation
### Train
#### Unpaired I2I task
```
python unpair_train.py
```
#### Paired I2I task
```
python pair_train.py
```
### Test (unpaired I2I only.)
- Save the translation results.
```
python save_transfer.py --root_dir <dataset_path> --dataset <dataset_name> --checkpoint_model <checkpoint_path>
```

## Second stage -- image generation and extension
### Train
```
python autoregressive_train.py 0 <dataset_name> <ne> <ed> <z_channel> <epoch_start> <epoch_end>
```
### Test
- The detail of  unconditional image generation and image extensions results are in `autoregressive_translation.ipynb`