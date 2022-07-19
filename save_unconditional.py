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
import argparse


original_size = 16
# size = 20
h, w = 16, 24
div = 2
codebook_size = 512
window_size = 16
z_code_shape = (1, codebook_size,  h, w)

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
        return self.size

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
                'image': img.to(device), 'style': style,
                # 'flip_image': flip_img.to(device), 'flip_style': flip_style, 
                'label': self.label}

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


def get_content_z(img, label):
    # get content z_code
    z_code, z_indices, _ = model.encode_to_z(img, label) # [1, 256]
    z_indices_shape = z_indices.shape
    return z_code, z_indices, z_indices_shape


def show_recon(z_indices, style, label, img):
    print('Reconstruction')
    x_sample = model.decode_to_img(
        z_indices.reshape(z_code_shape[0], original_size, original_size), 
        (z_code_shape[0], codebook_size, original_size, original_size), style, label)
    show_image(x_sample)
    print(nn.L1Loss()(img, x_sample))
    print('---------------------------------')
    return x_sample

def get_rand_input(z_indices):
    
    z_random = torch.randint(codebook_size, (h*w,)).to(device) # [400]
    z_random = z_random.reshape(z_code_shape[0], h, w) # [1, 20, 20]

    # set left-top part as the input image (256x256)
    z_random[:, :h, :8] = z_indices.reshape(z_code_shape[0], h, 8)

    ## idx as the input (original + random)
    idx = z_random.detach().clone() # [1, 20, 20]
    return idx

def get_timestep():
    import time
    ts = time.time()
    import datetime
    st = datetime.datetime.fromtimestamp(ts).strftime('%m%d_%H_%M_%S')
    return st

def sythesize(idx, style, label, z_code_shape=z_code_shape, temperature=2.0, top_k=5, return_idx=True):
    start_t = time.time()
    # print(z_code_shape)
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
            
            if(i >= original_size or j >= original_size):

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
        
    #         print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
    #         x_sample = model.decode_to_img(idx, (1, codebook_size, h, w), style, label)
    #         show_image(x_sample)
    print(f"Time: {time.time() - start_t} seconds")
    if(return_idx):
        return idx
    x_sample = model.decode_to_img(idx, (1, codebook_size, h, w), style, label)
    print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
    
    return x_sample

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

    parser.add_argument("--ne", default=512,
                    help="the number of embedding",
                    type=int)

    parser.add_argument("--ed", default=512,
                    help="embedding dimension",
                    type=int)

    parser.add_argument("--z_channel",default=128,
                    help="z channel",
                    type=int)
    

    parser.add_argument("--epoch_start", default=1,
                    help="start from",
                    type=int)

    parser.add_argument("--epoch_end", default=1000,
                    help="end at",
                    type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # ONLY MODIFY SETTING HERE
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('device: ', device)


    f = '/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_transformer_final_test/n_700.pt' 
    #os.path.join(os.getcwd(), 'n_700.pt')

    transformer_config = OmegaConf.load('transformer.yaml')
    transformer_config.model.params.f_path = #os.path.join(
        args.first_stage_model, 
        # os.getcwd(), 'summer2winter_disentangle_model', 'new_latest.pt')
    transformer_config.model.params.device = str(device)
    transformer_config.model.params.first_stage_model_config.params.embed_dim = args.ed
    transformer_config.model.params.first_stage_model_config.params.n_embed = args.ne
    model = instantiate_from_config(transformer_config.model)


    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print('Finish Loading!')


    # coordinate encode as condition
    c_size = 256 # original image size
    coordinate = np.arange(c_size*c_size).reshape(c_size,c_size,1)/(c_size*c_size)
    coordinate = torch.from_numpy(coordinate) # [256, 256, 1]
    c = model.get_c(coordinate) # [1, 1, 256, 256]
    c = c.to(device)

    # encode with condition
    _, cidx = model.encode_to_c(c) # [1, 256]
    cidx = cidx.reshape(z_code_shape[0], 16, 16) # [1, 16, 16]


    save_dir = 'l2r_eccv_ext256_res'
    full_path = os.path.join(os.getcwd(), save_dir, 'full')
    right256 = os.path.join(os.getcwd(), save_dir, 'right256')

    os.makedirs(full_path, exist_ok=True)
    os.makedirs(right256, exist_ok=True)

    
    # A class
    testA_set = dataset_single(args.root_dir, 'test', 'A', model.first_stage_model)
    ## load test images
    for i in range(len(testA_set)):
        print(i)
        img = testA_set[i] # 2
        _, z_indices, z_indices_shape = get_content_z(img['image'], img['label'])
        z_indices = z_indices.reshape(1, 16, 16)

        # only extract partial indices
        z_indices = z_indices[:,:z_indices.shape[1], :z_indices.shape[2]//div]

        idx = get_rand_input(z_indices)

        gen_img = sythesize(idx, img['style'], label=img['label'], return_idx=False)
        right_256_tensor = gen_img[:, :, :, -256:]
        
        save_tensor(gen_img, full_path, img['img_name'])
        save_tensor(right_256_tensor, right256, img['img_name'])

    # B class
    testA_set = dataset_single(args.root_dir, 'test', 'B', model.first_stage_model)
    ## load test images
    for i in range(len(testA_set)):
        print(i)
        img = testA_set[i] # 2
        _, z_indices, z_indices_shape = get_content_z(img['image'], img['label'])
        z_indices = z_indices.reshape(1, 16, 16)

        # only extract partial indices
        z_indices = z_indices[:,:z_indices.shape[1], :z_indices.shape[2]//div]

        idx = get_rand_input(z_indices)

        gen_img = sythesize(idx, img['style'], label=img['label'], return_idx=False)
        right_256_tensor = gen_img[:, :, :, -256:]
        
        save_tensor(gen_img, full_path, img['img_name'])
        save_tensor(right_256_tensor, right256, img['img_name'])
