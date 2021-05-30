"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from dataset import dataset_single, dataset_unpair
from torch.utils.data import DataLoader


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

def code_histogram(validation_originals, model):
    ze_a = model.quant_conv( model.encoder(validation_originals))   # (1, 256, 16, 16)
    zq_a, _, (_, _, enc_a_indice) = model.quantize(ze_a)
    # xrec = model.decode(zq_a)
    xrec = model.decode_b(zq_a)
    
    return enc_a_indice.cpu().numpy(), xrec#, histo.cpu().numpy()


def load_eval_model(_path, config_file, ed, ne):
    # a_path = os.path.join(os.getcwd(), 'a_model_w_3e-1', 'vqgan_880.pt')
    # model_a = torch.load(a_path, map_location=device)
    from omegaconf import OmegaConf
    from main_setting_a import get_obj_from_str, instantiate_from_config
    config = OmegaConf.load(config_file)
    # print(config.model.params.n_embed)

    config.model.params.n_embed = ne
    config.model.params.embed_dim = ed
    model_a = instantiate_from_config(config.model)
    ck = torch.load(_path, map_location=device)
    model_a.load_state_dict(ck['model_state_dict'], strict=False)
    model_a = model_a.to(device)
    # print(model_a.loss.discriminator)
    return model_a.eval()


def save_tensor(im_data, image_dir, image_name):
    im = tensor2im(im_data)
    save_path = os.path.join(os.getcwd(), 'res', image_dir, str(image_name) + '.png')
    save_image(im, save_path)


device = torch.device('cuda:0')


if __name__ == "__main__":

    # dataloader
    # root = '/eva_data/yujie/datasets/cat2dog'
    root = '/eva_data/yujie/datasets/afhq'
    
    _class = 'A'
    # model_name = 'both_afhq{}_'.format(_class)
    model_name = 'both_afhq_'#.format(_class)
    # epoch = 25
    mode = 'test'   # or 'train'
    config = 'config_comb.yaml'
    
    # if(_class == 'cat'):        
        # train_data = dataset_single(root, 'train', 'A')
    # if(_class == 'both'):
    validation_data = dataset_single(root, mode, _class, 256, 256)
    # else:
        # validation_data = dataset_unpair(root, mode, 286, 256)
    

    # m_inorm_path = os.path.join(os.getcwd(), 'cat_d_1_1e-1_512_512', 'vqgan_1100.pt')
    # m_inorm_path2 = os.path.join(os.getcwd(), 'cat_d_1_1e-1_512_512', 'vqgan_300.pt')
    # m_inorm_path2 = os.path.join(os.getcwd(), 'cat_d_1_1e-1_512_512', 'vqgan_300.pt')
    # m_lnorm_path = os.path.join(os.getcwd(), 'ln_d_a_model_1_1e-1', 'vqgan_950.pt')
    # m_inorm = load_eval_model(m_inorm_path, 'config_cat2dog.yaml')
    # m_lnorm = load_eval_model(m_lnorm_path, 'config_ln.yaml')
    # doc = ['512_512', '512_256', '512_128', '256_512', '256_256', '256_128']
    # doc = ['256_64', '256_128', '256_256', '256_512']
    doc = ['256_256']

    epochs = [i for i in range(20, 55, 5)]
    _d = '256_256'

    model_list = []
    # for _d in doc:
    for epoch in epochs:
        ed, ne = _d.split('_')
        ed, ne = int(ed), int(ne)
        # _name = _class + _d
        _name = model_name + _d + '_switch_upd'

        m_inorm_path = os.path.join(os.getcwd(), _name, 'vqgan_{}.pt'.format(epoch))
        m_inorm = load_eval_model(m_inorm_path, config, ed, ne)
        model_list.append(m_inorm)

    # print(model_list)

    ############################
    
    if(not os.path.isdir('res')):
        os.mkdir('res')

    
    # if(not os.path.isdir(os.path.join(os.getcwd(), 'res', 'originals2'))):
        # os.mkdir(os.path.join(os.getcwd(), 'res', 'originals2'))
    
    
    for epoch, _m in zip(epochs, model_list):
    # for _d, _m in zip(doc, model_list):
        save_dir = '{}{}_{}_{}_{}_a2b_all'.format(model_name, mode, _class, _d, epoch)
        print(save_dir)
        if(not os.path.isdir(os.path.join(os.getcwd(), 'res', save_dir))):
            os.mkdir(os.path.join(os.getcwd(), 'res', save_dir))
    
    
    # data loader
    test_loader = DataLoader(validation_data, batch_size=1, shuffle=False, pin_memory=True)
    
    for i, data in enumerate(test_loader):
        # print(data.shape)   # (1, 3, 256, 256)

        data = data.to(device)
        
        for epoch, _m in zip(epochs, model_list):
        # for _m in model_list:        
        # for _d, _m in zip(doc, model_list):
            _, xrec_in = code_histogram(data, _m)
            # _, xrec_in = code_histogram(data, m_inorm)
            # _, xrec_in2 = code_histogram(data, m_inorm2)
            # _, xrec_org = code_histogram(data, model_a)
            save_dir = '{}{}_{}_{}_{}_a2b_all'.format(model_name, mode, _class, _d, epoch)
                
                # save_dir = 'setB_trans_{}_{}_{}_{}'.format(mode, _class, _d, epoch)
                # print(save_dir)
            save_tensor(xrec_in, save_dir, i)
               
        # save_tensor(data, 'originals2', i)
        # save_tensor(xrec_in, 'cat_1100', i)
        # save_tensor(xrec_in2, 'cat_300', i)
        # save_tensor(xrec_org, 'cat_original', i)
        # save_tensor(xrec_ln, 'layer_norm', i)
        print(i)
    