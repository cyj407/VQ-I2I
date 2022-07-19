import argparse, os, random, torch
from omegaconf import OmegaConf
from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 
from taming_comb.models.cond_transformer import * 
from dataset import dataset_single_enc_sty
from utils import get_rand_input, get_coord_idx, sample_gen, save_tensor


torch.cuda.empty_cache()

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='0',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='/eva_data0/dataset/summer2winter_yosemite',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='summer2winter_yosemite',
                    help="dataset directory name",
                    type=str)

    parser.add_argument("--first_stage_model", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_settingc_256_final_test/settingc_latest.pt',
                    help="first stage model",
                    type=str)

    parser.add_argument("--transformer_model", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_transformer_final_test/n_700.pt',
                    help="transformer model (second stage model)",
                    type=str)

    parser.add_argument("--save_name", default='./summer2winter_yosemite_completion',
                    help="save directory name",
                    type=str)

    parser.add_argument("--sample_num", default=5,
                    help="the total generation number",
                    type=int)

    parser.add_argument("--input_domain", default='B',
                    choices=['A', 'B'],
                    help="the input image domain",
                    type=str)

    parser.add_argument("--sty_domain", default='A',
                    choices=['A', 'B'],
                    help="the generated image domain",
                    type=str)

    parser.add_argument("--pure_completion", default=True,
                    help="set True to only complete the input domain image without translation",
                    type=str)

    parser.add_argument("--partial_input", default='top-left',
                    choices=['top-left', 'left-half', 'top-half'],
                    help="top-left: given 1/4 of top-left corner image",
                    type=str)

    parser.add_argument("--ne", default=512,
                    help="the number of embedding",
                    type=int)

    parser.add_argument("--ed", default=512,
                    help="embedding dimension",
                    type=int)

    parser.add_argument("--z_channel",default=256,
                    help="z channel",
                    type=int)


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    # load first stage + second stage model
    transformer_config = OmegaConf.load('transformer.yaml')
    transformer_config.model.params.f_path = args.first_stage_model
    transformer_config.model.params.first_stage_model_config.params.embed_dim = args.ed
    transformer_config.model.params.transformer_config.params.vocab_size = args.ne
    transformer_config.model.params.transformer_config.params.n_embd = args.ne
    transformer_config.model.params.cond_stage_config.params.n_embed = args.ne
    transformer_config.model.params.first_stage_model_config.params.n_embed = args.ne
    transformer_config.model.params.first_stage_model_config.params.ddconfig.z_channels = args.z_channel
    transformer_config.model.params.device = str(device)
    model = instantiate_from_config(transformer_config.model)
    if(os.path.isfile(args.transformer_model)):
        print('load ' + args.transformer_model)
        ck = torch.load( args.transformer_model, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print('Finish Loading!')
    
    os.makedirs(args.save_name, exist_ok=True)
    coord_idx = get_coord_idx(model, device)
    content_set = dataset_single_enc_sty(args.root_dir, 'test', args.input_domain, model.first_stage_model, device)
    style_set = dataset_single_enc_sty(args.root_dir, 'test', args.sty_domain, model.first_stage_model, device)
    if args.partial_input == 'top-left':
        div_w, div_h = 2, 2
    elif args.partial_input == 'left-half':
        div_w, div_h = 2, 1
    else: # top-half
        div_w, div_h = 1, 2


    ## load test images
    for i in range(len(content_set)):
        if i == args.sample_num:
            break

        print(i)
        img = content_set[i]

        _, z_indices, _ = model.encode_to_z(img['image'], img['label']) # [1, 256]
        
        # new_idx contains z_indices + random indices
        new_idx = get_rand_input(device, z_indices, h=16, w=16, div_w=div_w, div_h=div_h, codebook_size=args.ne)

        content_idx = sample_gen(new_idx, coord_idx, model, original_h=16//div_h, original_w=16//div_w,
                    z_code_shape=(1, args.ne, 16, 16))
        
        if args.pure_completion:
            test_samples = model.decode_to_img(content_idx, 
                              (1, args.ne, content_idx.shape[1], content_idx.shape[2]),
                              img['style'], img['label'])
        else:
            style_img = style_set[random.randint(0, len(style_set)-1)]
            test_samples = model.decode_to_img(content_idx, 
                                (1, args.ne, content_idx.shape[1], content_idx.shape[2]),
                                style_img['style'], style_img['label'])

        save_tensor(test_samples, args.save_name, 'inpaint_{}_{}'.format(args.partial_input, img['img_name']))
        save_tensor(img['image'], args.save_name, 'input_{}'.format( img['img_name']))