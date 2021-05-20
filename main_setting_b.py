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
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    batch_size = 3
    learning_rate = 4.5e-6
    ne = 64
    ed = 512
    epoch_start = 1
    epoch_end = 300
    save_path = 'both_{}_{}_sep_upd'.format(ed, ne)    # model dir
    root = '/eva_data/yujie/datasets/cat2dog'

    # load data
    train_data = dataset_unpair(root, 'train', 286, 256)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)


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
    opt_disc_a = torch.optim.Adam(model.loss_a.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.9))
    opt_disc_b = torch.optim.Adam(model.loss_b.discriminator.parameters(),
                                lr=learning_rate, betas=(0.5, 0.9))

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
    train_res_rec_error = []

    iterations = len(train_data) // batch_size
    iterations = iterations + 1 if len(train_data) % batch_size != 0 else iterations
    for epoch in range(epoch_start, epoch_end+1):
        for i in range(iterations):

            dataA, dataB = next(iter(train_loader))
            dataA, dataB = dataA.to(device), dataB.to(device)


            ## Generator A
            recA, qlossA = model(dataA, label=1)
            opt_ae.zero_grad()

            aeloss_a, _ = model.loss_a(qlossA, dataA, recA, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=1), split="train")
            aeloss_a.backward()
            opt_ae.step()

            ## Discriminator A
            opt_disc_a.zero_grad()
            discloss_a, log = model.loss_a(qlossA, dataA, recA, optimizer_idx=1, global_step=epoch,
                                    last_layer=model.get_last_layer(label=1), split="train")
            
            discloss_a.backward()
            opt_disc_a.step()

            ## Generator B
            recB, qlossB = model(dataB, label=0)
            opt_ae.zero_grad()

            aeloss_b, _ = model.loss_b(qlossB, dataB, recB, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=0), split="train")
            aeloss_b.backward()
            opt_ae.step()

            ## Discriminator B
            opt_disc_b.zero_grad()
            discloss_b, log = model.loss_b(qlossB, dataB, recB, optimizer_idx=1, global_step=epoch,
                                    last_layer=model.get_last_layer(label=0), split="train")
            
            discloss_b.backward()
            opt_disc_b.step()

            # compute mse loss b/w input and reconstruction
            data = torch.cat((dataA, dataB), 0).to(device)
            rec = torch.cat((recA, recB), 0).to(device)
            recon_error = F.mse_loss( data, rec)

            train_res_rec_error.append(recon_error.item())
            train_ae_a_error.append(aeloss_a.item())
            train_ae_b_error.append(aeloss_b.item())
            train_disc_a_error.append(discloss_a.item())
            train_disc_b_error.append(discloss_b.item())

            if (i+1) % 100 == 0:
                _rec  = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                _rec += '(A domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                            np.mean(train_ae_a_error[-100:]), np.mean(train_disc_a_error[-100:]))
                _rec += '(B domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                            np.mean(train_ae_b_error[-100:]), np.mean(train_disc_b_error[-100:]))
                _rec += 'recon_error: {:8f}\n\n'.format(
                    np.mean(train_res_rec_error[-100:]))
                # print(_rec)
                with open(os.path.join(os.getcwd(), save_path, 'loss.txt'), 'a') as f:
                    f.write(_rec)
                    f.close()
                break

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'opt_ae_state_dict': opt_ae.state_dict(),
                'opt_disc_a_state_dict': opt_disc.state_dict(),
                'opt_disc_b_state_dict': opt_disc.state_dict()
            }, os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt'))


        if(epoch % 50 == 0 and epoch >= 50):
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'opt_ae_state_dict': opt_ae.state_dict(),
                    'opt_disc_a_state_dict': opt_disc.state_dict(),
                    'opt_disc_b_state_dict': opt_disc.state_dict()
                }, os.path.join(os.getcwd(), save_path, 'vqgan_{}.pt'.format(epoch)))
            # torch.save(
            #     {
            #         'model_state_dict': model.state_dict(),
            #         'opt_ae_state_dict': opt_ae.state_dict(),
            #         'opt_disc_a_state_dict': opt_disc.state_dict(),
            #         'opt_disc_b_state_dict': opt_disc.state_dict()
            #     }, os.path.join(os.getcwd(), save_path, 'vqgan_latest.pt'))

