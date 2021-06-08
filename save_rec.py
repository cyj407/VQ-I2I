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


def load_eval_model(_path, config_file, ed, ne, target, img_size):
    from omegaconf import OmegaConf
    from main_setting_a import get_obj_from_str, instantiate_from_config
    config = OmegaConf.load(config_file)
    # print(config.model.params.n_embed)
    config.model.target = target
    config.model.z_channels = img_size
    config.model.resolution = img_size
    config.model.params.n_embed = ne
    config.model.params.embed_dim = ed
    model_a = instantiate_from_config(config.model)
    ck = torch.load(_path, map_location=device)
    model_a.load_state_dict(ck['model_state_dict'], strict=False)
    model_a = model_a.to(device)
    return model_a.eval()


def save_tensor(im_data, image_dir, image_name):
    im = tensor2im(im_data)
    save_path = os.path.join(os.getcwd(), 'res', image_dir, str(image_name) + '.png')
    save_image(im, save_path)


device = torch.device('cuda:2')
print(torch.cuda.device_count())


if __name__ == "__main__":

    # dataloader
    root = '/eva_data/yujie/datasets/afhq'
    _class = 'B'
    epochs = [80]
    mode = 'test'   # or 'train'
    config = 'config_comb.yaml'    
    ed = 256
    ne = 512
    img_size = 128
    validation_data = dataset_single(root, mode, _class, img_size, img_size)
    # model_name = 'both_afhq_{}_{}_rec10_switch1'.format(ed, ne)
    model_name = 'both_afhq_{}_{}_2gloss_1dloss_img{}'.format(ed, ne, img_size)
    save_name = '2g1d_img{}_{}{}_{}_{}_b2a'.format( img_size, mode, _class, ed, ne)
    # save_name = 'tmp_{}_{}{}_{}_{}_b2a'.format( img_size, mode, _class, ed, ne)


    model_list = []
    for epoch in epochs:
        m_inorm_path = os.path.join(os.getcwd(), model_name, 'vqgan_{}.pt'.format(epoch))
        m_inorm = load_eval_model(m_inorm_path, config, ed, ne, 'taming_comb.models.vqgan.VQModelCrossGAN', img_size)
        model_list.append(m_inorm)

    ############################
    
    if(not os.path.isdir('res')):
        os.mkdir('res')

    # if(not os.path.isdir(os.path.join(os.getcwd(), 'res', 'originalsa'))):
    #     os.mkdir(os.path.join(os.getcwd(), 'res', 'originalsa'))
        
    for epoch, _m in zip(epochs, model_list):
        save_dir = '{}_{}'.format(save_name, epoch)
        print(save_dir)
        if(not os.path.isdir(os.path.join(os.getcwd(), 'res', save_dir))):
            os.mkdir(os.path.join(os.getcwd(), 'res', save_dir))
    
    
    # data loader
    test_loader = DataLoader(validation_data, batch_size=1, shuffle=False, pin_memory=True)
    
    for i, data in enumerate(test_loader):
        # print(data.shape)   # (1, 3, 256, 256)

        data = data.to(device)
        
        for epoch, _m in zip(epochs, model_list):

            # forward
            quant, _, _ = _m.encode(data)
            xrec_in = _m.decode_a(quant)

            save_dir = '{}_{}'.format(save_name, epoch)                
            save_tensor(xrec_in, save_dir, i)
               
        # save_tensor(data, 'originalsa', i)
        print(i)
    