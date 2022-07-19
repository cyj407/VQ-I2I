import os, importlib
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 
from taming_comb.models.cond_transformer import * 
import torch.utils.data as data
from PIL import Image
import numpy as np
import time



def show_image(s):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    s = Image.fromarray(s)
    # display(s)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    # h, w, _ = image_numpy.shape

    # if aspect_ratio is None:
    #     pass
    # elif aspect_ratio > 1.0:
    #     image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    # elif aspect_ratio < 1.0:
    #     image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_tensor(im_data, image_dir, image_name):
    im = tensor2im(im_data)
    save_path = os.path.join(image_dir, str(image_name)) #+ '.png'
    save_image(im, save_path)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_rand_input(device, condition=None, h=16, w=28, original_w=16, original_h=16, codebook_size=512, div_w=1, div_h=1):
    z_random = torch.randint(codebook_size, (w*h,)).to(device) # [400]
    z_random = z_random.reshape( 1, h, w) # [1, 20, 20]    
    
    if(condition == None):
        return z_random
    else:
        # set left-top part as the input image
        z_random[:, :original_h//div_h, :original_w//div_w] = condition.reshape( 1, original_h, original_w)[:, :original_h//div_h, :original_w//div_w]

        ## idx as the input (original + random)
        return z_random.detach().clone() # [1, 20, 20]


def get_coord_idx(model, device, target_code_size=16):
    # coordinate encode as condition
    c_size = 256 # original image size
    coordinate = np.arange(c_size*c_size).reshape(c_size,c_size,1)/(c_size*c_size)
    coordinate = torch.from_numpy(coordinate) # [256, 256, 1]
    c = model.get_c(coordinate) # [1, 1, 256, 256]
    c = c.to(device)

    # encode with condition
    _, cidx = model.encode_to_c(c) # [1, 256]
    cidx = cidx.reshape( 1, target_code_size, target_code_size) # [1, 16, 16]

    return cidx


def sample_gen(idx, coord_idx, model, z_code_shape, original_w=0, original_h=0, temperature=2.0, top_k=5):

    start_t = time.time()
    window_size = 16
    print(z_code_shape)
    for i in range(0, z_code_shape[2]-0):
        if i <= window_size//2:
            local_i = i
        elif (z_code_shape[2]-i) < window_size//2:
            local_i = window_size -(z_code_shape[2]-i)
        else:
            local_i = window_size//2
      
        for j in range(0,z_code_shape[3]-0):

            if j <= window_size//2:
                local_j = j
            elif (z_code_shape[3]-j) < window_size//2:
                local_j = window_size - (z_code_shape[3]-j)
            else:
                local_j = window_size//2

            i_start = i-local_i
            i_end = i_start+int(window_size)
            j_start = j-local_j
            j_end = j_start+int(window_size)
            
            if(i >= original_h or j >= original_w):

                patch = idx[:,i_start:i_end,j_start:j_end]
                patch = patch.reshape(patch.shape[0],-1)
                cpatch = coord_idx[:, i_start:i_end, j_start:j_end]
                cpatch = cpatch.reshape(cpatch.shape[0], -1)

                patch = torch.cat((cpatch, patch), dim=1)
                logits,_ = model.transformer(patch[:,:-1]) # [1, x, 512]
                logits = logits[:, -window_size*window_size:, :] # [1, 256, 512]
                logits = logits.reshape(z_code_shape[0],window_size,window_size,-1)  # [1, 16, 16, 512]
                logits = logits[:,local_i,local_j,:] # [1, 512]   

                logits = logits/temperature # small not equal

                if top_k is not None:
                    logits = model.top_k_logits(logits, top_k)

                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx[:,i,j] = torch.multinomial(probs, num_samples=1)
    print(f"Time: {time.time() - start_t} seconds")
    return idx

def gen_uncond_indices(model, device, target_code_size=16, codebook_size=512):
    rand_idx = get_rand_input(device, condition=None, codebook_size=codebook_size, h=target_code_size, w=target_code_size)
    coord_idx = get_coord_idx(model, device)
    gen_content_idx = sample_gen(rand_idx, coord_idx, model,
                        z_code_shape=(1, codebook_size, target_code_size, target_code_size), 
                        temperature=5.0, top_k=2)
    return gen_content_idx

