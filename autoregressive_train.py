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


    parser.add_argument("--device", default='4',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='/eva_data0/dataset/summer2winter_yosemite',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='summer2winter_yosemite',
                    help="dataset directory name",
                    type=str)

    parser.add_argument("--first_stage_model", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_settingc_256_final_test/settingc_latest.pt',
                    help="first stage model directory",
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

    parser.add_argument("--epoch_end", default=900,
                    help="end at",
                    type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    # ONLY MODIFY SETTING HERE
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    batch_size = 1 # 128
    learning_rate = 1e-5       # 256/512 lr=4.5e-6 from 71 epochs
    img_size = 256
    

    save_path = '{}_{}_{}_transformer_final_test'.format(args.dataset, args.ed, args.ne)    # second stage model dir
    print(save_path)


    # load data
    train_data = dataset_unpair(args.root_dir, 'train', 'A', 'B', img_size, img_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # load second stage model
    transformer_config = OmegaConf.load('transformer.yaml')
    transformer_config.model.params.first_stage_model_config.params.embed_dim = args.ed
    transformer_config.model.params.first_stage_model_config.params.n_embed = args.ne
    transformer_config.model.params.first_stage_model_config.z_channels = args.z_channel
    transformer_config.model.params.f_path = os.path.join(os.getcwd(), args.first_stage_model)
    transformer_config.model.params.device = str(device)
    model = instantiate_from_config(transformer_config.model)

    f = os.path.join(os.getcwd(), save_path, 'latest.pt')
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
    
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    train_loss_a = []
    train_loss_b = []
    


    for epoch in range(args.epoch_start, args.epoch_end+1):
        for i in range(iterations):

            dataA, dataB = next(iter(train_loader))
            dataA, dataB = dataA.to(device), dataB.to(device)

        
            opt_transformer.zero_grad()

            coordinate = np.arange(img_size*img_size).reshape(img_size,img_size,1)/(img_size*img_size)
            coordinate = torch.from_numpy(coordinate)
            c = model.get_c(coordinate)
            c = c.to(device)

            # dataA
            logits, target = model(dataA, c, 1)
            loss_a = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            train_loss_a.append(loss_a.item())
            loss_a.backward()
        

            # dataB
            logits, target = model(dataB, c, 0)
            loss_b = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            train_loss_b.append(loss_b.item())
            loss_b.backward()

            opt_transformer.step()


            if (i+1) % 1000 == 0:
                _rec  = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                _rec += 'a loss: {:8f}, b loss: {:8f}\n'.format(
                            np.mean(train_loss_a[-1000:]), np.mean(train_loss_b[-1000:]))
    
                print(_rec)
                with open(os.path.join(os.getcwd(), save_path, 'loss.txt'), 'a') as f:
                    f.write(_rec)
                    f.close()

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'opt_transformer_state_dict': opt_transformer.state_dict(),
    
            }, os.path.join(os.getcwd(), save_path, 'latest.pt'))


        if(epoch % 20 == 0 and epoch >= 60):
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'opt_transformer_state_dict': opt_transformer.state_dict(),
                 
                }, os.path.join(os.getcwd(), save_path, 'n_{}.pt'.format(epoch)))
        

