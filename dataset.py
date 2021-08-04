import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import torch
import numpy as np

class dataset_single(data.Dataset):
    def __init__(self, root, mode, _class, resize=256, cropsize=256, flip=True):
        self.root = root

        # a
        images_a = os.listdir(os.path.join(self.root, mode + _class))
        self.A = [os.path.join(self.root, mode + _class, x) for x in images_a]
        self.A_size = len(self.A)

        # b
        # images_b = os.listdir(os.path.join(self.root, mode + 'B'))
        # self.B = [os.path.join(self.root, mode + 'B', x) for x in images_b]
        self.B_size = 0 #len(self.B)

        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = 3
        self.input_dim_B = 3

        ## resize size
        transforms = [Resize((resize, resize), Image.BICUBIC)]
        if(mode == 'train'):
            transforms.append(RandomCrop(cropsize))
        else:
            transforms.append(CenterCrop(cropsize))

        ## flip
        if(flip):
            transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        return

    def __getitem__(self, index):
        if(self.dataset_size == self.A_size):
            data_A = self.load_img(self.A[index], self.input_dim_A)
            # data_B = self.load_img(self.B[random.randint(0, self.B_size -1)], self.input_dim_B)
        # else:
            # data_A = self.load_img(self.A[random.randint(0, self.A_size -1)], self.input_dim_A)
            # data_B = self.load_img(self.B[index], self.input_dim_B)
        return data_A#, data_B

    def __len__(self):
        return self.dataset_size

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if(input_dim == 1):
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def get_img_name(self):
        return self.A

class dataset_unpair(data.Dataset):
    def __init__(self, root, mode, resize=256, cropsize=256):
        self.root = root

        # a
        images_a = os.listdir(os.path.join(self.root, mode + 'A'))
        self.A = [os.path.join(self.root, mode + 'A', x) for x in images_a]
        self.A_size = len(self.A)

        # b
        images_b = os.listdir(os.path.join(self.root, mode + 'B'))
        self.B = [os.path.join(self.root, mode + 'B', x) for x in images_b]
        self.B_size = len(self.B)

        self.labels_a = torch.Tensor([1] * len(images_a))
        self.labels_b = torch.Tensor([0] * len(images_b))

        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = 3
        self.input_dim_B = 3

        ## resize size
        transforms = [Resize((resize, resize), Image.BICUBIC)]
        if(mode == 'train'):
            transforms.append(RandomCrop(cropsize))
        else:
            transforms.append(CenterCrop(cropsize))

        ## flip
        transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        return

    def __getitem__(self, index):
        if(self.dataset_size == self.A_size):
            data_A = self.load_img(self.A[index], self.input_dim_A)
            data_B = self.load_img(self.B[random.randint(0, self.B_size -1)], self.input_dim_B)

        else:
            data_A = self.load_img(self.A[random.randint(0, self.A_size -1)], self.input_dim_A)
            data_B = self.load_img(self.B[index], self.input_dim_B)
        return data_A, data_B#, torch.Tensor([1, 0])

    def __len__(self):
        return self.dataset_size

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if(input_dim == 1):
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img


class dataset_combine(data.Dataset):
    def __init__(self, root, mode, resize=286, cropsize=256):
        self.root = root

        images_a = os.listdir(os.path.join(self.root, mode + 'A'))  # 1231
        images_b = os.listdir(os.path.join(self.root, mode + 'B'))  # 962
        A = [os.path.join(self.root, mode + 'A', x) for x in images_a]
        B = [os.path.join(self.root, mode + 'B', x) for x in images_b]
        self.imgs = A + B

        # 0 represents class A, 1 represents class B
        # labels = [0] * len(images_a) + [1] * len(images_b)
        
        # 1 represents class A, 0 represents class B
        labels = [1] * len(images_a) + [0] * len(images_b)
        labels = torch.Tensor(labels)
        self.labels = labels

        self.A_size = len(A)
        self.B_size = len(B)
        self.dataset_size = len(self.imgs)
        self.input_dim = 3

        # resize size
        transforms = [Resize((resize, resize), Image.BICUBIC)]
        if(mode == 'train'):
            transforms.append(RandomCrop(cropsize))
        else:
            transforms.append(CenterCrop(cropsize))

        # flip
        transforms.append(RandomHorizontalFlip(p=1.0))

        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        return

    def __getitem__(self, index):
        return self.load_img(self.imgs[index], self.input_dim), self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if(input_dim == 1):
            img = img[0, ...] * 0.299 + img[1, ...] * \
                0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def var(self):
        # np_a = []
        # np_b = []
        np_imgs = []    # (dataset_size, 3, H, W)
        for i in range(len(self.imgs)):
            img = self.load_img(self.imgs[i], self.input_dim)
            np_imgs.append(np.array(img))
            # if(i < self.A_size):
            #     np_a.append(np.array(img))
            # else:
            #     np_b.append(np.array(img))
        print(np.array(np_imgs).mean())
        data_variance = np.var( np.array(np_imgs) / 255.0)
        # var_a = np.var( np.array(np_a) / 255.0)
        # var_b = np.var( np.array(np_b) / 255.0)
        # print(var_a)
        # print(var_b)
        return data_variance


class dataset_pair(data.Dataset):
    def __init__(self, root, mode, resize=256, cropsize=256):
        self.root = root
        self.mode = mode
        # a
        images_a = os.listdir(os.path.join(self.root, mode + 'A'))
        self.A = [x for x in images_a]
        self.A_size = len(self.A)

        self.dataset_size = len(self.A)

        self.input_dim_A = 3
        self.input_dim_B = 3

        ## resize size
        transforms = [Resize((resize, resize), Image.BICUBIC)]
        if(mode == 'train'):
            transforms.append(RandomCrop(cropsize))
        else:
            transforms.append(CenterCrop(cropsize))

        self.hflip = 0.0
        if self.hflip > 0.5:
            transforms.append(RandomHorizontalFlip(p=1.0))



        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        return

    def __getitem__(self, index):

        flip_or_not = random.random()

        A = os.path.join(self.root, self.mode + 'A', self.A[index])
        data_A = self.load_img(A, self.input_dim_A, flip_or_not)

        B = os.path.join(self.root, self.mode + 'B', self.A[index][:-5] + 'B' + '.jpg')
        data_B = self.load_img(B, self.input_dim_B, flip_or_not)

      
        return data_A, data_B  

    def __len__(self):
        return self.dataset_size

    def load_img(self, img_name, input_dim, flip_or_not):
        #flip
        self.hflip = flip_or_not

        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if(input_dim == 1):
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img


