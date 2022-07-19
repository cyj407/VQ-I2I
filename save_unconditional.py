import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 
from taming_comb.models.cond_transformer import * 
import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from IPython.display import clear_output
import time
import random


codebook_size = 512
window_size = 16
z_code_shape = (1, codebook_size,  16, 16)

def show_image(s):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    s = Image.fromarray(s)
    display(s)

class dataset_single(data.Dataset):
    def __init__(self, root, mode, _class, model, resize=256, cropsize=256):
        self.root = root
        
        # style information
        self.vqi2i = model
        self.label = 1 if _class == 'A' else 0
        
        images = os.listdir(os.path.join(self.root, mode + _class))
        self.img_path = [os.path.join(self.root, mode + _class, x) for x in images]
        self.size = len(self.img_path)
        self.input_dim = 3

        ## resize size
        transforms = [Resize((resize, resize), Image.BICUBIC)]
        if(mode == 'train'):
            transforms.append(RandomCrop(cropsize))
        else:
            transforms.append(CenterCrop(cropsize))

        transforms_flip = transforms.copy()
        transforms_flip.append(RandomHorizontalFlip(p=1))

        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        transforms_flip.append(ToTensor())
        transforms_flip.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transforms = Compose(transforms)
        self.transforms_flip = Compose(transforms_flip)
        return

    def __getitem__(self, index):
        print('Index: {}'.format(index))
        return self.load_img(self.img_path[index], self.input_dim)
        
    def __len__(self):
        return self.dataset_size

    def load_img(self, img_name, input_dim):
        _img = Image.open(img_name).convert('RGB')
        img = self.transforms(_img)
        # flip_img = self.transforms_flip(_img)
        
        img = img.unsqueeze(0) # make tensor2im workable
        # flip_img = flip_img.unsqueeze(0) # make tensor2im workable
        
        style = self.vqi2i.encode_style( img.to(device), self.label)
        # flip_style = self.vqi2i.encode_style( flip_img.to(device), self.label)
        print('Image Path: {}'.format(img_name))
        return {'img_name': img_name.split('/')[-1], 
                'image': img.to(device), 'style': style, 'label': self.label}

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


def get_rand_input(condition=None, h=16, w=28, original_w=16, original_h=16):
    z_random = torch.randint(codebook_size, (w*h,)).to(device) # [400]
    z_random = z_random.reshape(z_code_shape[0], h, w) # [1, 20, 20]    
    
    if(condition == None):
        return z_random
    else:
        # set left-top part as the input image
        z_random[:, :original_h, :original_w] = condition.reshape(z_code_shape[0], original_h, original_w)

        ## idx as the input (original + random)
        return z_random.detach().clone() # [1, 20, 20]


def get_cidx(target_code_size=16):
    # coordinate encode as condition
    c_size = 256//16 * target_code_size # original image size
    coordinate = np.arange(c_size*c_size).reshape(c_size,c_size,1)/(c_size*c_size)
    coordinate = torch.from_numpy(coordinate) # [256, 256, 1]
    c = model.get_c(coordinate) # [1, 1, 256, 256]
    c = c.to(device)

    # encode with condition
    _, cidx = model.encode_to_c(c) # [1, 256]
    cidx = cidx.reshape(z_code_shape[0], target_code_size, target_code_size) # [1, 16, 16]

    return cidx


def sample_gen(idx, style, cidx, z_code_shape, temperature=2.0, top_k=150, return_idx=False):

    start_t = time.time()
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
        
        patch = idx[:,i_start:i_end,j_start:j_end]
        patch = patch.reshape(patch.shape[0],-1)
        cpatch = cidx[:, i_start:i_end, j_start:j_end]
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
    if(return_idx):
        return idx


def gen_uncond_indices(target_code_size=16):
    idx = get_rand_input(condition=None, h=target_code_size, w=target_code_size)
    cidx = get_cidx()
    gen_idx = sample_gen(idx, _, cidx,
                        z_code_shape=(1, codebook_size, target_code_size, target_code_size), 
                        return_idx=True, temperature=5.0, top_k=2)
    return gen_idx



torch.cuda.empty_cache()


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='0',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='/eva_data0/dataset/summer2winter_yosemite',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='summer2winter_yosemite',
                    help="dataset directory name",
                    type=str)

    parser.add_argument("--first_stage_model", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_settingc_256_final_test/settingc_latest.pt',
                    help="first stage model",
                    type=str)

    parser.add_argument("--transformer_model", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_transformer_final_test/n_700.pt',
                    help="transformer model (second stage model)",
                    type=str)

    parser.add_argument("--save_name", default='./summer2winter_yosemite_uncond_gen',
                    help="save directory name",
                    type=str)

    parser.add_argument("--sample_num", default=10,
                    help="the total generation number",
                    type=int)

    parser.add_argument("--sty_domain", default='A',
                    help="the domain of unconditional generation (A or B)",
                    type=str)

    parser.add_argument("--ne", default=512,
                    help="the number of embedding",
                    type=int)

    parser.add_argument("--ed", default=512,
                    help="embedding dimension",
                    type=int)

    parser.add_argument("--z_channel",default=256,
                    help="z channel",
                    type=int)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    # load first stage + second stage model
    transformer_config = OmegaConf.load('transformer.yaml')
    transformer_config.model.params.f_path = args.first_stage_model
    transformer_config.model.params.first_stage_model_config.params.embed_dim = args.ed
    transformer_config.model.params.first_stage_model_config.params.n_embed = args.ne
    transformer_config.model.params.first_stage_model_config.params.ddconfig.z_channels = args.z_channel
    transformer_config.model.params.device = str(device)
    model = instantiate_from_config(transformer_config.model)
    if(os.path.isfile(args.transformer_model)):
        print('load ' + args.transformer_model)
        ck = torch.load( args.transformer_model, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print('Finish Loading!')
    
    os.makedirs(args.save_name, exist_ok=True)

    # encode coordinate as condition
    c_size = 256 # original image size
    coordinate = np.arange(c_size*c_size).reshape(c_size,c_size,1)/(c_size*c_size)
    coordinate = torch.from_numpy(coordinate) # [256, 256, 1]
    c = model.get_c(coordinate) # [1, 1, 256, 256]
    c = c.to(device)

    # encode with condition
    _, cidx = model.encode_to_c(c) # [1, 256]
    cidx = cidx.reshape(z_code_shape[0], 16, 16) # [1, 16, 16]


    if args.sty_domain == 'A':
        testA_set = dataset_single(args.root_dir, 'test', 'A', model.first_stage_model)
    else:
        testB_set = dataset_single(args.root_dir, 'test', 'B', model.first_stage_model)

    for i in range(args.sample_num):

        content_idx = gen_uncond_indices(target_code_size=16)

        if args.sty_domain == 'A':
            style_ref_img = testA_set[random.randint(0, testA_set.size-1)]
        else:
            style_ref_img = testB_set[random.randint(0, testB_set.size-1)]
        test_samples = model.decode_to_img(content_idx, 
                              (1, codebook_size, content_idx.shape[1], content_idx.shape[2]),
                              style_ref_img['style'], style_ref_img['label'])

        save_tensor(test_samples, args.save_name, '{}_{}'.format(i, style_ref_img['img_name']))