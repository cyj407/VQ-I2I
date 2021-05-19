import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
from dataset import dataset_combine
from torch.utils.data import DataLoader
import os



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

    # ONLY MODIFY SETTING HERE
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    learning_rate = 4.5e-6
    ne = 512
    ed = 512
    epoch_start = 1
    epoch_end = 1
    root = '/eva_data/yujie/datasets/cat2dog'


    train_data = dataset_combine(root, 'train', 286, 256)
    # validation_data = dataset_single(root, 'test', 'A')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # model
    save_path = 'both_{}_{}'.format(ed, ne)

    f = os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt')
    config = OmegaConf.load('config_comb.yaml')
    config.model.base_learning_rate = learning_rate
    config.model.params.embed_dim = ed
    config.model.params.n_embed = ne
    model = instantiate_from_config(config.model)
    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model.to(device)
    model.train()

    # print(model.loss.discriminator)
    
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                                list(model.decoder_a.parameters())+
                                list(model.decoder_b.parameters())+
                                list(model.quantize.parameters())+
                                list(model.quant_conv.parameters())+
                                list(model.post_quant_conv.parameters()),
                                lr=learning_rate, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(model.loss.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.9))
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


    for epoch in range(epoch_start, epoch_end+1):
        for i in range(iterations):
            data, label = next(iter(train_loader))
            data = data.to(device)
            label = label.to(device)

            # print(label)
            ### ???
            # data = model.get_input(data)
            xrec, qloss = model(data, label)

            ## Generator
            opt_ae.zero_grad()
            # print(model.get_last_layer().shape)

            num_of_b = torch.count_nonzero(label).item()
            # print(num_of_b)

            aeloss, _ = model.loss(qloss, data, xrec, optimizer_idx=0, global_step=epoch,
                                    last_layer=(model.get_last_layer(), num_of_b), split="train")
            aeloss.backward()
            opt_ae.step()

            ## Discriminator
            opt_disc.zero_grad()
            discloss, log = model.loss(qloss, data, xrec, optimizer_idx=1, global_step=epoch,
                                    last_layer=(model.get_last_layer(), num_of_b), split="train")
            

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


        if(epoch % 50 == 0 and epoch >= 50):
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

