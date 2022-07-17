import torch
import torch.nn.functional as F
#import pytorch_lightning as pl
import argparse, os, sys, datetime, glob, importlib
import torch.nn as nn
from torch.autograd import Variable

from taming_comb.modules.diffusionmodules.model import * #Encoder, Decoder, VUNet
from taming_comb.modules.vqvae.quantize import VectorQuantizer
from taming_comb.modules.discriminator.model import NLayerDiscriminator
from taming_comb.modules.losses.vqperceptual import hinge_d_loss


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


class VQModel_ADAIN(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None
                 ):
        super(VQModel_ADAIN, self).__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder_a = Decoder(**ddconfig)
        self.decoder_b = Decoder(**ddconfig)
        self.loss_a = instantiate_from_config(lossconfig)
        self.loss_b = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        
        #style encoders
        self.style_enc_a = StyleEncoder(4, 3, 64, 8, norm='none', activ='relu', pad_type='reflect')
        
        self.style_enc_b = StyleEncoder(4, 3, 64, 8, norm='none', activ='relu', pad_type='reflect')
        
        # MLP to generate AdaIN parameters
        n_a = self.get_num_adain_params(self.decoder_a)
        n_b = self.get_num_adain_params(self.decoder_b)
        
        self.mlp_a = MLP(8, n_a, 256, 3, norm='none', activ='relu')  
        self.mlp_b = MLP(8, n_b, 256, 3, norm='none', activ='relu')
        


    def encode(self, x, label):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        #encode style
        if label == 1:
            style_encoded = self.style_enc_a(x)
        else:
            style_encoded = self.style_enc_b(x)
            
        style_encoded = style_encoded
         
        return quant, emb_loss, info, style_encoded
    
    
    def encode_style(self, x, label):
        if label == 1:
            style_encoded = self.style_enc_a(x)
        else:
            style_encoded = self.style_enc_b(x)
            
        return style_encoded
    
    
    def encode_content(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        return h, quant
            
    

    def decode_a(self, quant, style_a):
        # decode content and style codes to an image
        self.mlp_a = self.mlp_a.to(style_a.device)
        adain_params = self.mlp_a(style_a)
        self.assign_adain_params(adain_params, self.decoder_a)
 
        
        quant = self.post_quant_conv(quant)
        
        dec = self.decoder_a(quant)
        return dec

    def decode_b(self, quant, style_b):
        # decode content and style codes to an image
        self.mlp_b = self.mlp_b.to(style_b.device)
        adain_params = self.mlp_b(style_b)
        self.assign_adain_params(adain_params, self.decoder_b)
        
        quant = self.post_quant_conv(quant)
        dec = self.decoder_b(quant)
        return dec

 
    def forward(self, input, label):
        if(label == 1):
            quant, diff, _, style_encoded = self.encode(input, label)
            dec = self.decode_a(quant, style_encoded)
        else:
            quant, diff, _, style_encoded = self.encode(input, label)
            dec = self.decode_b(quant, style_encoded)

        return dec, diff


    def get_last_layer(self, label):
        if(label ==  1):
            return self.decoder_a.conv_out.weight
        else:
            return self.decoder_b.conv_out.weight

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1).to(adain_params.device)
                m.weight = std.contiguous().view(-1).to(adain_params.device)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

        
    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    

class VQModelCrossGAN_ADAIN(VQModel_ADAIN):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super(VQModelCrossGAN_ADAIN, self).__init__(
            ddconfig, lossconfig, n_embed, embed_dim
        )

    def forward(self, input, label, cross=False, s_given=False):
        quant, diff, _, s = self.encode(input, label)
        
        # sample
        '''s_a = Variable(torch.randn(input.size(0), 8, 1, 1).cuda())
        s_b = Variable(torch.randn(input.size(0), 8, 1, 1).cuda())'''
        
        if(label == 1):
            if cross == False:
                output = self.decode_a(quant, s)
            else:
                s = s_given
                output = self.decode_b(quant, s)
        else:
            if cross == False:
                output = self.decode_b(quant, s)
            else:
                s = s_given
                output = self.decode_a(quant, s)
        
        return output, diff, s
    
    


