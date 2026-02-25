# Abstract class for the trainer
import os
import json

import numpy as np
from PIL import Image
# import webdataset as wds

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torchvision.datasets import ImageFolder
from PIL import Image

transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGBA')  # Ensure 4 channels (RGBA)
        return img

img_folder = ImageFolder(
    root= "Data/img/composed",  # Path to the dataset
    transform=transform_train,          # Apply transformations
    loader=custom_loader,         # Use the custom loader
)

class Trainer(object):
    """ Abstract class for each trainer """

    vit = None
    optim = None

    def __init__(self, args):
        """ Initialization of the Trainer """
        self.args = args
        self.writer = None if args.writer_log == "" else SummaryWriter(log_dir=args.writer_log)  # Tensorboard writer

    def get_data(self):
        """ class to load data """

        if self.args.model_mode == "img":

            data = np.load('Data/img_mask_complete_3.npz')
            mask = np.load('Data/img_mask_partial_3.npz')
            s = np.load('Data/Site_codebook.npy')
            site = np.concatenate((s,s,s),axis=0)
            img_loader = DataLoader(data_train, batch_size=self.args.bsize,
                                shuffle=False,
                                num_workers=8, pin_memory=True,
                                drop_last=True, sampler=None)
            # else: img_loader = np.zeros(site.shape[0])#DataLoader(TensorDataset(torch.tensor(np.zeros(site.shape[0]), dtype=torch.long)),batch_size=self.args.tsize,num_workers=self.args.num_workers, pin_memory=True)
            idx = data.files
            # ['T','S','L','A','R','W']
            data_cat = np.stack((data[idx[0]],data[idx[1]],data[idx[2]],data[idx[3]],data[idx[4]],data[idx[5]]),axis=1)
            mask_cat = np.stack((mask[idx[0]],mask[idx[1]],mask[idx[2]],mask[idx[3]],mask[idx[4]],mask[idx[5]]),axis=1)

            data_train = TensorDataset(torch.tensor(data_cat[:self.args.split], dtype=torch.long),torch.tensor(site[:self.args.split], dtype=torch.long),
                                       torch.tensor(mask_cat[:self.args.split], dtype=torch.long))
            test_loader = [site[self.args.split:],mask_cat[self.args.split:]]

        elif self.args.model_mode == "vec":

            data = np.load('Data/vec_mask_complete_3.npz')
            mask = np.load('Data/vec_mask_partial_3.npz')
            s = np.load('Data/Site_codebook.npy')
            site = np.concatenate((s,s,s),axis=0)
            if self.args.img_loss:
                img_loader = np.load('Data/img_batch_3.npy')
            else: img_loader = np.zeros(site.shape[0])#DataLoader(TensorDataset(torch.tensor(np.zeros(site.shape[0]), dtype=torch.long)),batch_size=self.args.tsize,num_workers=self.args.num_workers, pin_memory=True)
            idx = data.files
            # ['T','S','L','A','R','W']
            T = torch.tensor(data[idx[0]][:self.args.split], dtype=torch.float)
            mT = torch.tensor(mask[idx[0]][:self.args.split], dtype=torch.float)
            S = torch.tensor(data[idx[1]][:self.args.split], dtype=torch.float)
            mS = torch.tensor(mask[idx[1]][:self.args.split], dtype=torch.float)
            L = torch.tensor(data[idx[2]][:self.args.split], dtype=torch.float)
            mL = torch.tensor(mask[idx[2]][:self.args.split], dtype=torch.float)
            A = torch.tensor(data[idx[3]][:self.args.split], dtype=torch.float)
            mA = torch.tensor(mask[idx[3]][:self.args.split], dtype=torch.float)
            R = torch.tensor(data[idx[4]][:self.args.split], dtype=torch.float)
            mR = torch.tensor(mask[idx[4]][:self.args.split], dtype=torch.float)
            W = torch.tensor(data[idx[5]][:self.args.split], dtype=torch.float)
            mW = torch.tensor(mask[idx[5]][:self.args.split], dtype=torch.float)

            vT = torch.tensor(data[idx[0]][self.args.split:], dtype=torch.float)
            vmT = torch.tensor(mask[idx[0]][self.args.split:], dtype=torch.float)
            vS = torch.tensor(data[idx[1]][self.args.split:], dtype=torch.float)
            vmS = torch.tensor(mask[idx[1]][self.args.split:], dtype=torch.float)
            vL = torch.tensor(data[idx[2]][self.args.split:], dtype=torch.float)
            vmL = torch.tensor(mask[idx[2]][self.args.split:], dtype=torch.float)
            vA = torch.tensor(data[idx[3]][self.args.split:], dtype=torch.float)
            vmA = torch.tensor(mask[idx[3]][self.args.split:], dtype=torch.float)
            vR = torch.tensor(data[idx[4]][self.args.split:], dtype=torch.float)
            vmR = torch.tensor(mask[idx[4]][self.args.split:], dtype=torch.float)
            vW = torch.tensor(data[idx[5]][self.args.split:], dtype=torch.float)
            vmW = torch.tensor(mask[idx[5]][self.args.split:], dtype=torch.float)

            data_train = TensorDataset(T, S, L, A, R, W, torch.tensor(site[:self.args.split], dtype=torch.long), mT, mS, mL, mA, mR, mW)
            test_loader = [vT, vS, vL, vA, vR, vW, site[self.args.split:], vmT, vmS, vmL, vmA, vmR, vmW]

        
        elif self.args.model_mode == "hybrid":

            valid = np.load('Data/valid_num.npy')
            data = np.load('Data/vec_mask_complete_3.npz')
            s = np.load('Data/All_codebook_torch.npz')['site']
            site = np.concatenate((s,s,s),axis=0)
            # if self.args.img_loss:
            img_loader = DataLoader(img_folder, batch_size=self.args.bsize,
                            shuffle=False,
                            num_workers=8, pin_memory=True,
                            drop_last=True, sampler=None)
            # compose_img = ImageFolder(os.path.join(self.args.composed_img), transform=transforms.ToTensor())
            # else: img_loader = np.zeros(site.shape[0])#DataLoader(TensorDataset(torch.tensor(np.zeros(site.shape[0]), dtype=torch.long)),batch_size=self.args.tsize,num_workers=self.args.num_workers, pin_memory=True)
            idx = data.files
            # ['T','S','L','A','R','W']
            T = torch.tensor(data[idx[0]][:self.args.split], dtype=torch.float)
            S = torch.tensor(data[idx[1]][:self.args.split], dtype=torch.float)
            L = torch.tensor(data[idx[2]][:self.args.split], dtype=torch.float)
            A = torch.tensor(data[idx[3]][:self.args.split], dtype=torch.float)
            R = torch.tensor(data[idx[4]][:self.args.split], dtype=torch.float)
            W = torch.tensor(data[idx[5]][:self.args.split], dtype=torch.float)

            vT = torch.tensor(data[idx[0]][self.args.split:], dtype=torch.float)
            vS = torch.tensor(data[idx[1]][self.args.split:], dtype=torch.float)
            vL = torch.tensor(data[idx[2]][self.args.split:], dtype=torch.float)
            vA = torch.tensor(data[idx[3]][self.args.split:], dtype=torch.float)
            vR = torch.tensor(data[idx[4]][self.args.split:], dtype=torch.float)
            vW = torch.tensor(data[idx[5]][self.args.split:], dtype=torch.float)

            data_img = np.load('Data/img_mask_complete_3.npz')

            idx = data.files
            # ['T','S','L','A','R','W']
            data_cat = np.stack((data_img[idx[0]],data_img[idx[1]],data_img[idx[2]],data_img[idx[3]],data_img[idx[4]],data_img[idx[5]]),axis=1)

            data_train = TensorDataset(torch.tensor(data_cat[:self.args.split], dtype=torch.long),torch.tensor(site[:self.args.split], dtype=torch.long),
                                       T, S, L, A, R, W)
            test_loader = [data_cat[self.args.split:],site[self.args.split:], vT, vS, vL, vA, vR, vW] #vT, vS, vL, vA, vR, vW,

        train_loader = DataLoader(data_train, batch_size=self.args.bsize,
                                  shuffle=False if self.args.img_loss else True,
                                  num_workers=self.args.num_workers, pin_memory=True,drop_last=True, sampler=train_sampler)
        # test_loader = DataLoader(data_test, batch_size=self.args.tsize,
                                #  shuffle=False if self.args.is_multi_gpus else True,
                                #  num_workers=self.args.num_workers, pin_memory=True,sampler=test_sampler)

        return train_loader, test_loader, img_loader, torch.tensor(valid, dtype=torch.long)

    def get_network(self, archi, pretrained_file=None):
        pass

    def log_add_img(self, names, img, iter):
        """ Add an image in tensorboard"""
        if self.writer is None:
            return
        self.writer.add_image(tag=names, img_tensor=img, global_step=iter)

    def log_add_scalar(self, names, scalar, iter):
        """ Add scalar value in tensorboard """
        if self.writer is None:
            return
        if isinstance(scalar, dict):
            self.writer.add_scalars(main_tag=names, tag_scalar_dict=scalar, global_step=iter)
        else:
            self.writer.add_scalar(tag=names, scalar_value=scalar, global_step=iter)

    @staticmethod
    def get_optim(net, lr, mode="AdamW", **kwargs):
        """ Get the optimizer Adam or Adam with weight decay """
        if isinstance(net, list):
            params = []
            for n in net:
                params += list(n.parameters())
        else:
            params = net.parameters()

        if mode == "AdamW":
            return optim.AdamW(params, lr, weight_decay=1e-5, **kwargs)
        elif mode == "Adam":
            return optim.Adam(params, lr, **kwargs)
        return None

    @staticmethod
    def get_loss(mode="l1", **kwargs):
        """ return the loss """
        if mode == "l1":
            return nn.L1Loss()
        elif mode == "l2":
            return nn.MSELoss()
        elif mode == "cross_entropy":
            return nn.CrossEntropyLoss(**kwargs)
        elif mode == "mse":
            return nn.MSELoss(reduction='mean')
        return None

    def train_one_epoch(self, epoch):
        return

    def fit(self):
        pass

    @torch.no_grad()
    def eval(self):
        pass

    def sample(self, nb_sample):
        pass

    @staticmethod
    def all_gather(obj, gpus, reduce="mean"):
        """ Gather the value obj from all GPUs and return the mean or the sum """
        tensor_list = [torch.zeros(1) for _ in range(gpus)]
        dist.all_gather_object(tensor_list, obj)
        obj = torch.FloatTensor(tensor_list)
        if reduce == "mean":
            obj = obj.mean()
        elif reduce == "sum":
            obj = obj.sum()
        elif reduce == "none":
            pass
        else:
            raise NameError("reduction not known")

        return obj

    def save_network(self, model, path, iter=None, optimizer=None, global_epoch=None):
        """ Save the state of the model, including the iteration,
            the optimizer state and the current epoch"""
        if self.args.is_multi_gpus:
            torch.save({'iter': iter,
                        'global_epoch': global_epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       path)
        else:
            torch.save({'iter': iter,
                        'global_epoch': global_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       path)

