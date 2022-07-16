from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from dataset import dataset_single
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
from torchvision.utils import make_grid
import matplotlib as mpl
import random
import importlib
mpl.rcParams.update({'figure.max_open_warning': 0})

from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 

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


def load_eval_model(_path, config_file, ed, ne, target, z):
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_file)
    # print(config.model.params.n_embed)
    config.model.target = target
    config.model.z_channels = z
    config.model.resolution = 256
    config.model.params.n_embed = ne
    config.model.params.embed_dim = ed
    model_a = instantiate_from_config(config.model)
    ck = torch.load(_path, map_location=device)
    print(_path)
    model_a.load_state_dict(ck['model_state_dict'], strict=False)
    model_a = model_a.to(device)
    return model_a.eval()


def save_tensor(im_data, image_dir, image_name):
    im = tensor2im(im_data)
    save_path = os.path.join(image_dir, str(image_name)) #+ '.png'
    save_image(im, save_path)


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":

    # dataloader
    dataset = 'afhq'
    root = '/eva_data0/dataset/{}'.format(dataset)
    save_root = 'sample_afhq_withoutregression'

    mode = 'test'   # or 'train'
    config = 'config_comb.yaml' 
    ed = 256
    ne = 256
    img_size = 256    
    z = 128
    atob = False

    if(atob):
        validation_data_a = dataset_single(root, mode, 'A', img_size, img_size, flip=False)
        validation_data_b = dataset_single(root, mode, 'B', img_size, img_size, flip=False)
    else:
        validation_data_a = dataset_single(root, mode, 'B', img_size, img_size, flip=False)
        validation_data_b = dataset_single(root, mode, 'A', img_size, img_size, flip=False)
  
    # model_name = 'portrait_{}_{}_settingc_{}'.format(ed, ne, img_size)
    # model_name = 'afhq_cat2dog_256_256_settingc_256'
    model_name = 'afhq_256_256_settingc_256_withoutregression' #'summer2winter_disentangle_model'
    epoch_list = ['latest']

    ############################
    
    if(not os.path.isdir(save_root)):
        os.mkdir(save_root)

    for epoch in epoch_list:   
        m_path = os.path.join(os.getcwd(), model_name, 'settingc_{}.pt'.format(epoch))
        #m_path = "/eva_data3/VQI2I-setting-c/afhq_cat2dog_256_256_settingc_256/settingc_n_400.pt"
        config = 'config_comb.yaml'
        m = load_eval_model(m_path, config, ed, ne, 'taming_comb.models.vqgan.VQModelCrossGAN_ADAIN', z)
 
        if(atob):
            save_dir = '{}_a2b'.format(dataset)
        else:
            save_dir = '{}_b2a'.format(dataset)

        if(not os.path.isdir(os.path.join(os.getcwd(), save_root, save_dir))):
            os.mkdir(os.path.join(os.getcwd(), save_root, save_dir))

        # data loader
        test_loader_a = DataLoader(validation_data_a, batch_size=1, shuffle=False, pin_memory=True)
        test_loader_b = DataLoader(validation_data_b, batch_size=1, shuffle=False, pin_memory=True)
        
        test_img_name_a = validation_data_a.get_img_name()
        test_img_name_b = validation_data_b.get_img_name()
        
        
        # for idx1, data_A in enumerate(test_loader_a):
        for idx1 in range(len(validation_data_a)):
            if idx1 == 30: # num of collections
                break
            print('{}/{}'.format(idx1, len(validation_data_a)))
            img_name_a = test_img_name_a[idx1].rsplit('/', 1)[-1]

            data_A = validation_data_a[idx1].unsqueeze(0)

            data_A = data_A.to(device)
            
            if(not os.path.isdir(os.path.join(os.getcwd(), save_root, save_dir, img_name_a))):
                os.mkdir(os.path.join(os.getcwd(), save_root, save_dir, img_name_a))

            cur_dir = os.path.join(os.getcwd(), save_root, save_dir, img_name_a)
            save_tensor(data_A, cur_dir, 'input.jpg')
            '''
            ## intra-domain style transfer
            for i in range(20):
                data_A2 = validation_data_a[random.randint(0, len(validation_data_a)-1)].unsqueeze(0)
                data_A2 = data_A2.to(device)
                if(atob):
                    # s_a = m.encode_style(data_A, label=1)
                    s_a2 = m.encode_style(data_A2, label=1)
                else:
                    # s_a = m.encode_style(data_A, label=0)
                    s_a2 = m.encode_style(data_A2, label=0)
                with torch.no_grad():
                    if(atob):
                        # res, _, _ = m(data_A, label=1, cross=False, s_given=s_a2)
                        quant, diff, _, _ = m.encode(data_A, 1)
                        output = m.decode_a(quant, s_a2)
                    else:
                        quant, diff, _, _ = m.encode(data_A, 0)
                        output = m.decode_b(quant, s_a2)
                        # res, _, _ = m(data_A, label=0, cross=False, s_given=s_a2)
                save_tensor(output, cur_dir, 'trans_intra_{}.jpg'.format(i))                
            
            '''
            # for idx2, data_B in enumerate(test_loader_b):
            for idx2, _ in enumerate(test_loader_b):
                if idx2 == 1: # num of collections
                    break
                data_B = validation_data_b[random.randint(0, len(validation_data_b)-1)].unsqueeze(0)
                data_B = data_B.to(device)

                # img_name_b = test_img_name_b[idx2].rsplit('/', 1)[-1]
                if(atob):
                    # s_a = m.encode_style(data_A, label=1)
                    s_b = m.encode_style(data_B, label=0)
                else:
                    # s_a = m.encode_style(data_A, label=0)
                    s_b = m.encode_style(data_B, label=1)
                with torch.no_grad():
                    if(atob):
                        AcBs, _, _ = m(data_A, label=1, cross=True, s_given=s_b)
                    else:
                        AcBs, _, _ = m(data_A, label=0, cross=True, s_given=s_b)
                    res = AcBs
                save_tensor(res, cur_dir, 'trans_{}.jpg'.format(idx2))
                save_tensor(data_B, cur_dir, 'style.jpg')