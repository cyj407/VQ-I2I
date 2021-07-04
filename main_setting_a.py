import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
from dataset import dataset_single
from torch.utils.data import DataLoader
import os

device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")


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

    # dataloader
    # root = './root/'
    root = '/eva_data/yujie/datasets/afhq'
    
    ##### MODIFY HERE
    batch_size = 3
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _class = 'A'
    ed = 256
    ne = 512
    learning_rate = 4.5e-6

    train_data = dataset_single(root, 'train', _class, 286, 256)
    # validation_data = dataset_single(root, 'test', 'A')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    # model
    # save_path = 'afhqA_256_64'
    save_path = 'afhq{}_{}_{}'.format(_class, ed, ne)
    print(save_path)
    # b_model_3e-1_5e-1
    # 340, lr = 4.5e-6

    # a_model_w_3e-1
    # 701~900 : lr = 4.5e-6

    # a_model_4e-6
    # disc_start: 501 #301
    # disc_factor: 0.8 #0.8 / 0.3
    # disc_weight: 0.8 # 0.5 / 0.8

    f = os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt')
    config = OmegaConf.load('config_cat2dog.yaml')
    config.model.base_learning_rate = learning_rate
    config.model.params.embed_dim = ed
    config.model.params.n_embed = ne

    # print(torch.cuda.device_count())
    model = instantiate_from_config(config.model)
    # model = torch.nn.DataParallel(#model).to(device)
    #     model, device_ids=device_ids, output_device=device_ids[1]).to(device)


    ### important
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model.to(device)
    model.train()

    # print(model.loss.discriminator)
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                                list(model.decoder.parameters())+
                                list(model.quantize.parameters())+
                                list(model.quant_conv.parameters())+
                                list(model.post_quant_conv.parameters()),
                                lr=learning_rate, betas=(0.5, 0.9))
    # opt_ae = torch.nn.DataParallel(opt_ae, device_ids=device_ids)

    ### important
    # if isinstance(opt_ae, torch.nn.DataParallel):
    #     opt_ae = opt_ae.module

    opt_disc = torch.optim.Adam(model.loss.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.9))
    
    # opt_disc = torch.nn.DataParallel(opt_disc, device_ids=device_ids)

    ### important
    # if isinstance(opt_disc, torch.nn.DataParallel):
    #     opt_disc = opt_disc.module

    if(os.path.isfile(f)):
        print('load ' + f)
        opt_ae.load_state_dict(ck['opt_ae_state_dict'])
        opt_disc.load_state_dict(ck['opt_dic_state_dict'])
  
    train_res_ae_error = []
    train_res_disc_error = []
    train_res_rec_error = []

    iterations = len(train_data) // batch_size
    iterations = iterations + 1 if len(train_data) % batch_size != 0 else iterations
    # print(iterations)
    # epochs = 200
    # iterations = 1
    # global_step = 0

    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)


    for epoch in range(251, 300+1):
        for i in range(iterations):
            data = next(iter(train_loader))
            data = data.to(device)

            ### ???
            # data = model.get_input(data)
            xrec, qloss = model(data)

            ## Generator
            opt_ae.zero_grad()
            aeloss, _ = model.loss(qloss, data, xrec, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(), split="train")
            aeloss.backward()
            opt_ae.step()

            ## Discriminator
            opt_disc.zero_grad()
            discloss, log = model.loss(qloss, data, xrec, optimizer_idx=1, global_step=epoch,
                                    last_layer=model.get_last_layer(), split="train")
            

            discloss.backward()
            opt_disc.step()

            recon_error = F.mse_loss(data, xrec)
            train_res_rec_error.append(recon_error.item())
            train_res_ae_error.append(aeloss.item())
            train_res_disc_error.append(discloss.item())

            if (i+1) % 100 == 0:
                _rec = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                _rec += 'ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                    np.mean(train_res_ae_error[-100:]), np.mean(train_res_disc_error[-100:]))
                _rec += 'recon_error: {:8f}\n\n'.format(
                    np.mean(train_res_rec_error[-100:]))
                # print(_rec)
                with open(os.path.join(os.getcwd(), save_path, 'loss.txt'), 'a') as f:
                    f.write(_rec)
                    f.close()

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'opt_ae_state_dict': opt_ae.state_dict(),
                'opt_dic_state_dict': opt_disc.state_dict()
            }, os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt'))


        if(epoch % 10 == 0 and epoch >= 50):
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'opt_ae_state_dict': opt_ae.state_dict(),
                    'opt_dic_state_dict': opt_disc.state_dict()
                }, os.path.join(os.getcwd(), save_path, 'vqgan_{}.pt'.format(epoch)))
            # torch.save(
            #     {
            #         'model_state_dict': model.state_dict(),
            #         'opt_ae_state_dict': opt_ae.state_dict(),
            #         'opt_dic_state_dict': opt_disc.state_dict()
            #     }, os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt'))

