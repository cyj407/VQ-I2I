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
import torch.nn as nn
import os
from taming.models.vqgan import VQModel

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device_ids = [0, 1]
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


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

    # hyperparameters
    batch_size = 6# * len(device_ids)
    learning_rate = 4.5e-6

    # dataloader
    # root = './root/'
    root = '/eva_data/yujie/datasets/summer2winter_yosemite'
    train_data = dataset_single(root, 'train', 'A')
    # validation_data = dataset_single(root, 'test', 'A')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    # model
    save_path = 'a_model_1_1e-1'

    f = os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt')
    config = OmegaConf.load('config.yaml')
    model = instantiate_from_config(config.model)
    model = torch.nn.DataParallel(
        model, device_ids=device_ids, output_device=device_ids[0])
    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.module.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.train()


    opt_ae = torch.optim.Adam(list(model.module.encoder.parameters())+
                                list(model.module.decoder.parameters())+
                                list(model.module.quantize.parameters())+
                                list(model.module.quant_conv.parameters())+
                                list(model.module.post_quant_conv.parameters()),
                                lr=learning_rate, betas=(0.5, 0.9))

    opt_ae = nn.DataParallel(opt_ae, device_ids=device_ids)
    opt_disc = torch.optim.Adam(model.module.loss.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.9))
    opt_disc = nn.DataParallel(opt_disc, device_ids=device_ids)

    if(os.path.isfile(f)):
        print('load ' + f)
        opt_ae.module.load_state_dict(ck['opt_ae_state_dict'])
        opt_disc.module.load_state_dict(ck['opt_dic_state_dict'])
    

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


    for epoch in range(501, 700+1):
        for i in range(iterations):
            data = next(iter(train_loader))

            data = data.to(device)
            xrec, qloss = model(data)

            ## Generator
            opt_ae.zero_grad()
            aeloss, _ = model.module.loss(qloss, data, xrec, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.module.get_last_layer(), split="train")
            aeloss.backward()
            opt_ae.module.step()

            ## Discriminator
            opt_disc.zero_grad()
            discloss, log = model.module.loss(qloss, data, xrec, optimizer_idx=1, global_step=epoch,
                                    last_layer=model.module.get_last_layer(), split="train")
            

            discloss.backward()
            opt_disc.module.step()

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

        # if(epoch == 1):
        if(epoch % 20 == 0 and epoch >= 50):
            torch.save(
                {
                    'model_state_dict': model.module.state_dict(),
                    'opt_ae_state_dict': opt_ae.module.state_dict(),
                    'opt_dic_state_dict': opt_disc.module.state_dict()
                }, os.path.join(os.getcwd(), save_path, 'vqgan_{}.pt'.format(epoch)))
            torch.save(
                {
                    'model_state_dict': model.module.state_dict(),
                    'opt_ae_state_dict': opt_ae.module.state_dict(),
                    'opt_dic_state_dict': opt_disc.module.state_dict()
                }, os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt'))
        