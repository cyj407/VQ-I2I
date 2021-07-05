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


os.environ["CUDA_VISIBLE_DEVICES"]='0'


if __name__ == "__main__":

    # ONLY MODIFY SETTING HERE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1 # 128
    learning_rate = 1e-4        # 256/512 lr=4.5e-6 from 71 epochs
    ne = 128  # Enlarge
    ed = 128
    epoch_start = 1
    epoch_end = 150
    switch_weight = 0.1 # self-reconstruction : a2b/b2a = 10 : 1
    dataset = 'cityscapes'
    if(dataset == 'summer2winter'):
        dataset_dir = dataset + '_yosemite'
    else:
        dataset_dir = dataset
    
    # save_path = 'both_afhq_{}_{}_rec_switch_img128'.format(ed, ne)    # model dir
    save_path = 'afhq_{}_{}_settingc'.format(ed, ne)    # model dir
    print(save_path)
    root = '/home/jenny/VQVAE-CUT/dataset/afhq_cat2dog/'

    # load data
    train_data = dataset_unpair(root, 'train', 128, 128)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    f = os.path.join(os.getcwd(), save_path, 'settingc_latest.pt')
    config = OmegaConf.load('config_comb.yaml')
    config.model.target = 'taming_comb.models.vqgan.VQModelCrossGAN_ADAIN'
    config.model.base_learning_rate = learning_rate
    config.model.params.embed_dim = ed
    config.model.params.n_embed = ne
    config.model.z_channels = 128
    config.model.resolution = 128
    model = instantiate_from_config(config.model)
    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)
    model.train()

    # print(model.loss.discriminator)
    
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                                list(model.decoder_a.parameters())+
                                list(model.decoder_b.parameters())+
                                list(model.quantize.parameters())+
                                list(model.quant_conv.parameters())+
                                list(model.post_quant_conv.parameters())+
                                list(model.style_enc_a.parameters())+
                                list(model.style_enc_b.parameters())+
                                list(model.mlp_a.parameters())+
                                list(model.mlp_b.parameters()),
                                lr=learning_rate, betas=(0.5, 0.999))
    
    opt_disc_a = torch.optim.Adam(model.loss_a.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.999))
    
    opt_disc_b = torch.optim.Adam(model.loss_b.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.999))

    if(os.path.isfile(f)):
        print('load ' + f)
        opt_ae.load_state_dict(ck['opt_ae_state_dict'])
        opt_disc_a.load_state_dict(ck['opt_disc_a_state_dict'])
        opt_disc_b.load_state_dict(ck['opt_disc_b_state_dict'])


    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)

    train_ae_a_error = []
    train_ae_b_error = []
    train_disc_a_error = []
    train_disc_b_error = []
    train_disc_a2b_error = []
    train_disc_b2a_error = []
    train_res_rec_error = []
    
    train_style_a_loss = []
    train_style_b_loss = []

    iterations = len(train_data) // batch_size
    iterations = iterations + 1 if len(train_data) % batch_size != 0 else iterations
    for epoch in range(epoch_start, epoch_end+1):
        for i in range(iterations):

            dataA, dataB = next(iter(train_loader))
            dataA, dataB = dataA.to(device), dataB.to(device)

            
            ## Discriminator A
            opt_disc_a.zero_grad()
            
            fakeA, _, s_a_sampled = model(dataB, label=0, cross=True)
            
            b2a_loss, log = model.loss_a(_, dataA, fakeA, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")
            
            disc_a_loss = b2a_loss
            disc_a_loss.backward()
            opt_disc_a.step()

            
            ## Generator A
            opt_ae.zero_grad()
            
            recA, qlossA, _ = model(dataA, label=1, cross=False)

            aeloss_a, _ = model.loss_a(qlossA, dataA, recA, fake=fakeA, switch_weight=switch_weight, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=1), split="train")
            
            recA_from_fake, _, s_a = model(fakeA, label=1, cross=False)
            
            # style loss
            style_a_loss = torch.mean(torch.abs(s_a_sampled.to(s_a.device) - s_a))
            
            aeloss_a = aeloss_a + style_a_loss
            
            aeloss_a.backward()
            opt_ae.step()



            ## Discriminator B
            opt_disc_b.zero_grad()
            
            
            fakeB, _, s_b_sampled = model(dataA, label=1, cross=True)
            
            a2b_loss, log = model.loss_b(_, dataB, fakeB, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")
          
            disc_b_loss = a2b_loss 
            disc_b_loss.backward()
            opt_disc_b.step()


            ## Generator B
            opt_ae.zero_grad()
            
            recB, qlossB, _ = model(dataB, label=0, cross=False)

            aeloss_b, _ = model.loss_b(qlossB, dataB, recB, fake=fakeB, switch_weight=switch_weight, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=0), split="train")
            
            recB_from_fake, _, s_b = model(fakeB, label=0, cross=False)
            
            # style loss
            style_b_loss = torch.mean(torch.abs(s_b_sampled - s_b))
            
            aeloss_b = aeloss_b + style_b_loss
            
            aeloss_b.backward()
            opt_ae.step()
            

            # compute mse loss b/w input and reconstruction
            data = torch.cat((dataA, dataB), 0).to(device)
            rec = torch.cat((recA, recB), 0).to(device)
            recon_error = F.mse_loss( data, rec)

            train_res_rec_error.append(recon_error.item())
            train_ae_a_error.append(aeloss_a.item())
            train_ae_b_error.append(aeloss_b.item())
            train_disc_a_error.append(disc_a_loss.item())
            train_disc_b_error.append(disc_b_loss.item())
            train_disc_a2b_error.append(a2b_loss.item())
            train_disc_b2a_error.append(b2a_loss.item())
            
            train_style_a_loss.append(style_a_loss.item())
            train_style_b_loss.append(style_b_loss.item())

            if (i+1) % 100 == 0:
                _rec  = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                _rec += '(A domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                            np.mean(train_ae_a_error[-100:]), np.mean(train_disc_a_error[-100:]))
                _rec += '(B domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                            np.mean(train_ae_b_error[-100:]), np.mean(train_disc_b_error[-100:]))
                _rec += 'A vs A2B loss: {:8f}, B vs B2A loss: {:8f}\n'.format(
                            np.mean(train_disc_a2b_error[-100:]), np.mean(train_disc_b2a_error[-100:]))
                _rec += 'recon_error: {:8f}\n\n'.format(
                    np.mean(train_res_rec_error[-100:]))
                
                _rec += 'style_a_loss: {:8f}\n\n'.format(
                    np.mean(train_style_a_loss[-100:]))
                _rec += 'style_b_loss: {:8f}\n\n'.format(
                    np.mean(train_style_b_loss[-100:]))
                
                print(_rec)
                with open(os.path.join(os.getcwd(), save_path, 'loss.txt'), 'a') as f:
                    f.write(_rec)
                    f.close()
            
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'opt_ae_state_dict': opt_ae.state_dict(),
                'opt_disc_a_state_dict': opt_disc_a.state_dict(),
                'opt_disc_b_state_dict': opt_disc_b.state_dict()
            }, os.path.join(os.getcwd(), save_path, 'settingc_latest.pt'))


        if(epoch % 20 == 0 and epoch >= 20):
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'opt_ae_state_dict': opt_ae.state_dict(),
                    'opt_disc_a_state_dict': opt_disc_a.state_dict(),
                    'opt_disc_b_state_dict': opt_disc_b.state_dict()
                }, os.path.join(os.getcwd(), save_path, 'settingc_n_{}.pt'.format(epoch)))
            # torch.save(
            #     {
            #         'model_state_dict': model.state_dict(),
            #         'opt_ae_state_dict': opt_ae.state_dict(),
            #         'opt_disc_a_state_dict': opt_disc_a.state_dict(),
            #         'opt_disc_b_state_dict': opt_disc_b.state_dict()
            #     }, os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt'))

