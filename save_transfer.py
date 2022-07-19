import torch, os, random, argparse
from dataset import dataset_single
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from utils import instantiate_from_config, save_tensor
from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 



def load_eval_model(_path, config_file, ed, ne, target, z):
    config = OmegaConf.load(config_file)
    # print(config.model.params.n_embed)
    config.model.target = target
    config.model.z_channels = z
    config.model.params.n_embed = ne
    config.model.params.embed_dim = ed
    model_a = instantiate_from_config(config.model)
    ck = torch.load(_path, map_location=device)
    print(_path)
    model_a.load_state_dict(ck['model_state_dict'], strict=False)
    model_a = model_a.to(device)
    return model_a.eval()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--device", default='0',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='/eva_data0/dataset/summer2winter_yosemite/',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='summer2winter_yosemite',
                    help="dataset directory name",
                    type=str)

    parser.add_argument("--checkpoint_dir", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_settingc_256_final_test/',
                    help="first stage model directory",
                    type=str)

    parser.add_argument("--checkpoint_epoch", default='latest', # or 'n_600' / 'n_400' ...
                    help="the number of the epoch used for the checkpoint",
                    type=str)

    parser.add_argument("--save_name", default='translation_summer2winter_yosemite',
                    help="dataset directory name",
                    type=str)
    
    parser.add_argument("--atob", default=True,
                    help="True: domain A--> domain B; False: domain B--> domain A",
                    type=bool)

    parser.add_argument("--intra_transfer", default=False,
                    help="intra-domain translation",
                    type=bool)
                    
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


    # dataloader
    mode = 'test'   # or 'train'
    img_size = 256
    if(args.atob):
        validation_data_a = dataset_single(args.root_dir, mode, 'A', img_size, img_size, flip=False)
        validation_data_b = dataset_single(args.root_dir, mode, 'B', img_size, img_size, flip=False)
    else:
        validation_data_a = dataset_single(args.root_dir, mode, 'B', img_size, img_size, flip=False)
        validation_data_b = dataset_single(args.root_dir, mode, 'A', img_size, img_size, flip=False)

    ############################
    
    # load first stage model
    m_path = os.path.join(os.getcwd(), args.checkpoint_dir, 'settingc_{}.pt'.format(args.checkpoint_epoch))
    config = 'config_comb.yaml'
    m = load_eval_model(m_path, config, args.ed, args.ne, 'taming_comb.models.vqgan.VQModelCrossGAN_ADAIN', args.z_channel)

    if(args.atob):
        save_dir = '{}_a2b'.format(args.dataset)
    else:
        save_dir = '{}_b2a'.format(args.dataset)


    # data loader
    test_loader_a = DataLoader(validation_data_a, batch_size=1, shuffle=False, pin_memory=True)
    test_loader_b = DataLoader(validation_data_b, batch_size=1, shuffle=False, pin_memory=True)
    
    test_img_name_a = validation_data_a.get_img_name()
    test_img_name_b = validation_data_b.get_img_name()
    
    
    for idx1 in range(len(validation_data_a)):
        if idx1 == 5: # sample first 5 images from the testing set
            break
        print('{}/{}'.format(idx1, len(validation_data_a)))
        img_name_a = test_img_name_a[idx1].rsplit('/', 1)[-1]

        data_A = validation_data_a[idx1].unsqueeze(0)
        data_A = data_A.to(device)

        os.makedirs(os.path.join(os.getcwd(), args.save_name, save_dir, img_name_a), exist_ok=True)
        
        cur_dir = os.path.join(os.getcwd(), args.save_name, save_dir, img_name_a)
        save_tensor(data_A, cur_dir, 'input.jpg')
        
        if args.intra_transfer:  ## intra-domain style transfer
            for i in range(3): # random choose for 20 style images from the same domain
                data_A2 = validation_data_a[random.randint(0, len(validation_data_a)-1)].unsqueeze(0)
                data_A2 = data_A2.to(device)
                with torch.no_grad():
                    if(args.atob):
                        s_a2 = m.encode_style(data_A2, label=1)
                        quant, diff, _, _ = m.encode(data_A, 1)
                        output = m.decode_a(quant, s_a2)
                    else:
                        s_a2 = m.encode_style(data_A2, label=0)
                        quant, diff, _, _ = m.encode(data_A, 0)
                        output = m.decode_b(quant, s_a2)
                save_tensor(output, cur_dir, 'trans_intra_{}.jpg'.format(i))
        
        for idx2, _ in enumerate(test_loader_b):
            if idx2 == 3: # generate 3 translation for each
                break
            data_B = validation_data_b[random.randint(0, len(validation_data_b)-1)].unsqueeze(0)
            data_B = data_B.to(device)

            img_name_b = test_img_name_b[idx2].rsplit('/', 1)[-1]
            with torch.no_grad():
                if(args.atob):
                    s_b = m.encode_style(data_B, label=0)
                    AcBs, _, _ = m(data_A, label=1, cross=True, s_given=s_b)
                else:
                    s_b = m.encode_style(data_B, label=1)
                    AcBs, _, _ = m(data_A, label=0, cross=True, s_given=s_b)
            save_tensor(AcBs, cur_dir, 'trans_{}.jpg'.format(idx2))
            # save_tensor(data_B, cur_dir, 'style_{}.jpg'.format(img_name_b))
        