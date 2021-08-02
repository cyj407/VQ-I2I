import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
from dataset import dataset_combine, dataset_unpair
from torch.utils.data import DataLoader
import os

from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 

from taming_comb.models.cond_transformer import * 

import argparse

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





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device",
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("dataset",
                    help="dataset",
                    type=str)
    
    parser.add_argument("ne",
                    help="the number of embedding",
                    type=int)

    parser.add_argument("ed",
                    help="embedding dimension",
                    type=int)

    parser.add_argument("z_channel",
                    help="z channel",
                    type=int)
    

    parser.add_argument("epoch_start",
                    help="start from",
                    type=int)

    parser.add_argument("epoch_end",
                    help="end at",
                    type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    # ONLY MODIFY SETTING HERE
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    batch_size = 1 # 128
    learning_rate = 1e-5       # 256/512 lr=4.5e-6 from 71 epochs
    ne = args.ne  # Enlarge
    ed = args.ed
    img_size = 256
    epoch_start = args.epoch_start
    epoch_end = args.epoch_end
    switch_weight = 0.1 # self-reconstruction : a2b/b2a = 10 : 1
    
    
    first_model_save_path = '{}_{}_{}_settingc_{}'.format(args.dataset, ed, ne, img_size)    # first stage model dir
    save_path = args.dataset + '{}_{}_{}_transformer'.format(args.dataset, ed, ne)    # second stage model dir
    print(save_path)
    root = '/home/jenny870207/data/' + args.dataset + '/'

    # load data
    train_data = dataset_unpair(root, 'train', img_size, img_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # load first stage model
    '''f = os.path.join(os.getcwd(), first_model_save_path, 'settingc_latest.pt')
    config = OmegaConf.load('config_comb.yaml')
    config.model.params.embed_dim = args.ed
    config.model.params.n_embed = args.ne
    config.model.z_channels = args.z_channel
    config.model.resolution = 256
    first_model = instantiate_from_config(config.model)
    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        first_model.load_state_dict(ck['model_state_dict'], strict=False)
    first_model = first_model.to(device)
    first_model.eval()'''

    # load second stage model
    f = os.path.join(os.getcwd(), save_path, 'latest.pt')
    
    transformer_config = OmegaConf.load('transformer.yaml')
    transformer_config.model.params.first_stage_model_config.params.embed_dim = args.ed
    transformer_config.model.params.first_stage_model_config.params.n_embed = args.ne
    transformer_config.model.params.first_stage_model_config.z_channels = args.z_channel
    transformer_config.model.params.first_stage_model_config.resolution = 256
    transformer_config.model.params.f_path = os.path.join(os.getcwd(), first_model_save_path, 'settingc_latest.pt')
    transformer_config.model.params.device = str(device)
    model = instantiate_from_config(transformer_config.model)

    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)
    model.train()

    # print(model.loss.discriminator)
    
    opt_transformer = torch.optim.Adam(model.transformer.parameters(),
                                lr=learning_rate, betas=(0.5, 0.999))
    

    if(os.path.isfile(f)):
        print('load ' + f)
        opt_transformer.load_state_dict(ck['opt_transformer_state_dict'])
        


    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)



    iterations = len(train_data) // batch_size
    iterations = iterations + 1 if len(train_data) % batch_size != 0 else iterations
    
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    loss_a = []
    loss_b = []
    
    for epoch in range(epoch_start, epoch_end+1):
        for i in range(iterations):

            dataA, dataB = next(iter(train_loader))
            dataA, dataB = dataA.to(device), dataB.to(device)

            # dataA
            opt_transformer.zero_grad()
            #x, c = model.get_xc(dataA)
            logits, target = model(dataA, 1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss_a.append(loss.item())
            loss.backward()
            opt_transformer.step()

            # dataB
            opt_transformer.zero_grad()
            #x, c = model.get_xc(dataB)
            logits, target = model(dataB, 0)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss_b.append(loss.item())
            loss.backward()
            opt_transformer.step()
            
            

            if (i+1) % 1000 == 0:
                _rec  = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                _rec += 'a loss: {:8f}, b loss: {:8f}\n'.format(
                            np.mean(loss_a[-1000:]), np.mean(loss_b[-1000:]))
    
                print(_rec)
                with open(os.path.join(os.getcwd(), save_path, 'loss.txt'), 'a') as f:
                    f.write(_rec)
                    f.close()
            
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'opt_transformer_state_dict': opt_transformer.state_dict(),
    
            }, os.path.join(os.getcwd(), save_path, 'latest.pt'))


        if(epoch % 20 == 0 and epoch >= 20):
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'opt_transformer_state_dict': opt_transformer.state_dict(),
                 
                }, os.path.join(os.getcwd(), save_path, 'n_{}.pt'.format(epoch)))
        

