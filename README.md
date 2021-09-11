# VQ-I2I
## Environment
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
- Save thetranslation results.
```
python save_transfer.py
```
## Second stage -- image generation and extension
### Train
```
python autoregressive_train.py 0 <dataset_name> <ne> <ed> <z_channel> <epoch_start> <epoch_end>
```
### Test
- The detail of  unconditional image generation and image extensions results are in `autoregressive_translation.ipynb`