import torch
import torch.nn.functional as F
#import pytorch_lightning as pl
import argparse, os, sys, datetime, glob, importlib
import torch.nn as nn
from torch.autograd import Variable

from taming_comb.modules.diffusionmodules.model import * #Encoder, Decoder, VUNet
from taming_comb.modules.vqvae.quantize import VectorQuantizer
from taming_comb.modules.StyleGAN2PatchDiscriminator.patch_discriminator import * #Encoder, Decoder, VUNet




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



def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1)


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

        #style patch GAN
        self.Dpatch_a = StyleGAN2PatchDiscriminator()
        self.Dpatch_b = StyleGAN2PatchDiscriminator()
        


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

def apply_random_crop(x, target_size, scale_range, num_crops=1, return_rect=False):
    # build grid
    B = x.size(0) * num_crops
    flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
    unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :, np.newaxis].repeat(B, target_size, 1, 1)
    unit_grid_y = unit_grid_x.transpose(1, 2)
    unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)


    #crops = []
    x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
    #for i in range(num_crops):
    scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
    sampling_grid = unit_grid * scale + offset
    crop = F.grid_sample(x, sampling_grid, align_corners=False)
    #crops.append(crop)
    #crop = torch.stack(crops, dim=1)
    crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))

    return crop    
    
from taming_comb.modules.discriminator.model import NLayerDiscriminator
from taming_comb.modules.losses.vqperceptual import hinge_d_loss
class VQModelCrossGAN_StyleGAN(VQModel_ADAIN):
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
        super(VQModelCrossGAN_StyleGAN, self).__init__(
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

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = apply_random_crop(
            x, 64,#self.opt.patch_size
            (1, 1),#self.opt.patch_min_scale, self.opt.patch_max_scale
            num_crops=1#self.opt.patch_num_crops
        )
        return crops

    def compute_patch_discriminator_losses(self, real, mix, domain):

        if domain == 'A':
            real_feat = self.Dpatch_a.extract_features(
                self.get_random_crops(real),
                aggregate=False#self.opt.patch_use_aggregation
            )
            target_feat = self.Dpatch_a.extract_features(self.get_random_crops(real))
            mix_feat = self.Dpatch_a.extract_features(self.get_random_crops(mix))

            real_loss = gan_loss(
                self.Dpatch_a.discriminate_features(real_feat, target_feat),
                should_be_classified_as_real=True,
            ) #* self.opt.lambda_PatchGAN

            fake_loss = gan_loss(
                self.Dpatch_a.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=False,
            ) #* self.opt.lambda_PatchGAN

        else:
            real_feat = self.Dpatch_b.extract_features(
                self.get_random_crops(real),
                aggregate=False#self.opt.patch_use_aggregation
            )
            target_feat = self.Dpatch_b.extract_features(self.get_random_crops(real))
            mix_feat = self.Dpatch_b.extract_features(self.get_random_crops(mix))

            real_loss = gan_loss(
                self.Dpatch_b.discriminate_features(real_feat, target_feat),
                should_be_classified_as_real=True,
            ) #* self.opt.lambda_PatchGAN

            fake_loss = gan_loss(
                self.Dpatch_b.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=False,
            ) #* self.opt.lambda_PatchGAN


        return (real_loss + fake_loss) * 0.5
    
    def compute_Dpatch_G_loss(self, real, mix, domain):
        if domain == 'A':
            real_feat = self.Dpatch_a.extract_features(
                        self.get_random_crops(real),
                        aggregate=False).detach()#self.opt.patch_use_aggregation
            mix_feat = self.Dpatch_a.extract_features(self.get_random_crops(mix))

            Dpatch_g_loss = gan_loss(
                self.Dpatch_a.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) #* self.opt.lambda_PatchGAN
        else:
            real_feat = self.Dpatch_b.extract_features(
                        self.get_random_crops(real),
                        aggregate=False).detach()#self.opt.patch_use_aggregation
            mix_feat = self.Dpatch_b.extract_features(self.get_random_crops(mix))

            Dpatch_g_loss = gan_loss(
                self.Dpatch_b.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) #* self.opt.lambda_PatchGAN

        return Dpatch_g_loss
    


