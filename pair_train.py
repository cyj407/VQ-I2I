import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
from dataset import dataset_pair
from torch.utils.data import DataLoader
import os

from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 

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

    parser.add_argument("--device", default='0',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='/eva_data0/dataset/cityscapes',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='cityscapes',
                    help="dataset directory name",
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

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    # ONLY MODIFY SETTING HERE
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    batch_size = 1 # 128
    learning_rate = 1e-5       # 256/512 lr=4.5e-6 from 71 epochs
    img_size = 256
    switch_weight = 0.1 # self-reconstruction : a2b/b2a = 10 : 1
    
    
    save_path = '{}_{}_{}_pair'.format(args.dataset, args.ed, args.ne)    # model dir
    print(save_path)

    # load data
    train_data = dataset_pair(args.root_dir, 'train', img_size, img_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    f = os.path.join(os.getcwd(), save_path, 'settingc_latest.pt')
    config = OmegaConf.load('config_comb.yaml')
    config.model.target = 'taming_comb.models.vqgan.VQModelCrossGAN_ADAIN'
    config.model.base_learning_rate = learning_rate
    config.model.params.embed_dim = args.ed
    config.model.params.n_embed = args.ne
    config.model.z_channels = args.z_channel
    config.model.resolution = 256
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
    train_content_loss = []
    train_cross_recons_loss = []

    iterations = len(train_data) // batch_size
    iterations = iterations + 1 if len(train_data) % batch_size != 0 else iterations
    
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    for epoch in range(args.epoch_start, args.epoch_end+1):
        for i in range(iterations):

            dataA, dataB = next(iter(train_loader))
            dataA, dataB = dataA.to(device), dataB.to(device)

            ## Discriminator A
            opt_disc_a.zero_grad()
            
            s_a = model.encode_style(dataA, label=1)
            fakeA, _, _ = model(dataB, label=0, cross=True, s_given=s_a)

            recA, qlossA, _ = model(dataA, label=1, cross=False)
            
            b2a_loss, log = model.loss_a(_, dataA, fakeA, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")

            a_rec_d_loss, _ = model.loss_a(_, dataA, recA, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")
            
            disc_a_loss = 0.8*b2a_loss + 0.2*a_rec_d_loss
            disc_a_loss.backward()
            opt_disc_a.step()
            
            
            ## Discriminator B
            opt_disc_b.zero_grad()
            
            s_b = model.encode_style(dataB, label=0)
            fakeB, _, s_b_sampled = model(dataA, label=1, cross=True, s_given=s_b)

            recB, qlossB, _ = model(dataB, label=0, cross=False)
            
            a2b_loss, log = model.loss_b(_, dataB, fakeB, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")

            b_rec_d_loss, _ = model.loss_b(_, dataB, recB, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")
            
          
            disc_b_loss = 0.8*a2b_loss + 0.2*b_rec_d_loss
            disc_b_loss.backward()
            opt_disc_b.step()
       

            ## Generator 
            opt_ae.zero_grad()

            aeloss_a, _ = model.loss_a(qlossA, dataA, recA, fake=fakeA, switch_weight=switch_weight, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=1), split="train")
            
            
            # cross path with style a
            AtoBtoA, _, s_a_from_cross = model(fakeA, label=1, cross=False)
            

            aeloss_b, _ = model.loss_b(qlossB, dataB, recB, fake=fakeB, switch_weight=switch_weight, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=0), split="train")
            
            # cross path with style b
            BtoAtoB, _, s_b_from_cross = model(fakeB, label=0, cross=False)

            # style loss
            style_a_loss = torch.mean(torch.abs(s_a.detach() - s_a_from_cross)).to(device)
            style_b_loss = torch.mean(torch.abs(s_b.detach() - s_b_from_cross)).to(device)
            style_loss = 0.5*style_a_loss + 0.5*style_b_loss
            
            # content loss
            c_a, c_a_quan = model.encode_content(dataA)
            c_b, c_b_quan = model.encode_content(dataB)
            content_loss = torch.mean(torch.abs(c_a.detach() - c_b)).to(device)
            content_quan_loss = torch.mean(torch.abs(c_a_quan - c_b_quan.detach())).to(device)
            content_loss = 0.5*content_loss + 0.5*content_quan_loss 

            # cross reconstruction loss
            cross_recons_loss_a = torch.mean(torch.abs(dataA.detach() - fakeA)).to(device)
            cross_recons_loss_b = torch.mean(torch.abs(dataB.detach() - fakeB)).to(device)
            cross_recons_loss = 0.5*cross_recons_loss_a + 0.5*cross_recons_loss_b

            
            
            gen_loss = aeloss_a + aeloss_b + 3.0*cross_recons_loss + 0.5*(style_loss + content_loss) 
            gen_loss.backward()
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
            
            train_content_loss.append(content_loss.item())
            train_cross_recons_loss.append(cross_recons_loss.item())


            if (i+1) % 1000 == 0:
                _rec  = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                _rec += '(A domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                            np.mean(train_ae_a_error[-1000:]), np.mean(train_disc_a_error[-1000:]))
                _rec += '(B domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                            np.mean(train_ae_b_error[-1000:]), np.mean(train_disc_b_error[-1000:]))
                _rec += 'A vs A2B loss: {:8f}, B vs B2A loss: {:8f}\n'.format(
                            np.mean(train_disc_a2b_error[-1000:]), np.mean(train_disc_b2a_error[-1000:]))
                _rec += 'recon_error: {:8f}\n\n'.format(
                    np.mean(train_res_rec_error[-1000:]))
                
                _rec += 'style_a_loss: {:8f}\n\n'.format(
                    np.mean(train_style_a_loss[-1000:]))
                _rec += 'style_b_loss: {:8f}\n\n'.format(
                    np.mean(train_style_b_loss[-1000:]))
                
                _rec += 'content_loss: {:8f}\n\n'.format(
                    np.mean(train_content_loss[-1000:]))

                _rec += 'cross_recons_loss: {:8f}\n\n'.format(
                    np.mean(train_cross_recons_loss[-1000:]))

                
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
