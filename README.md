# VQI2I First Stage
## Setting (b)
### Execute
```
python main_setting_b.py
```
- ONLY NEED TO MODIFY THE HYPERPARAMETER SETTING IN `main_setting_b.py`
### Log
- Use `dataset_combine` to load the dataset
    - **Class A label as 1, Class B label as 0**
    - Resize to 286 first, and crop to 256
- Configuration file `config_comb.yaml`
    - Import `taming_comb` module
    - **Only need to change the num of embeddings `ne` and embedding dims `ed` in the `main_setting_b.py`**
- `taming_comb` module
    - Modify `./models/vqgan.py`
        - Two decoders with function `decode_a` and `decode_b`
        - Function `get_last_layer` return a tuple of (dec_a.weight, dec_b.weight)
    - Modify `./losses/vqperceptual.py`
        - Close the `calculate_adaptive_weight` function