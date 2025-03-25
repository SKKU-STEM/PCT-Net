import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from imageio import imread

def random_resize_2x(img0, imgt, img1, p=0.2):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    return img0, imgt, img1

def random_crop_2x(img0, imgt, img1, crop_size = (224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w]
    imgt = imgt[x:x+h, y:y+w]
    img1 = img1[x:x+h, y:y+w]
    return img0, imgt, img1

def random_vertical_flip_2x(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
    return img0, imgt, img1

def random_horizontal_flip_2x(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
    return img0, imgt, img1

def random_rotate_2x(img0, imgt, img1, p = 0.5):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0))
        imgt = imgt.transpose((1, 0))
        img1 = img1.transpose((1, 0))
    return img0, imgt, img1

def random_reverse_time_2x(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
    return img0, imgt, img1

class Train_Dataset_2x(Dataset):
    def __init__(self, dataset_dir = '', augment = True):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        dataset_list = os.listdir(dataset_dir)
        for data_folder in dataset_list:
            data_list = os.listdir(f"{dataset_dir}/{data_folder}")
            step = 2
            for data_idx in range(len(data_list)):
                if data_idx + step < len(data_list):
                    self.img0_list.append(f"{dataset_dir}/{data_folder}/{data_list[data_idx]}")
                    self.imgt_list.append(f"{dataset_dir}/{data_folder}/{data_list[data_idx + 1]}")
                    self.img1_list.append(f"{dataset_dir}/{data_folder}/{data_list[data_idx + step]}")
                else:
                    break

    def __len__(self):
        return len(self.img0_list)

    def __getitem__(self, idx):
        img0 = imread(self.img0_list[idx])
        imgt = imread(self.imgt_list[idx])
        img1 = imread(self.img1_list[idx])

        if self.augment:
            img0, imgt, img1 = random_resize_2x(img0, imgt, img1)
            img0, imgt, img1 = random_crop_2x(img0, imgt, img1)
            img0, imgt, img1 = random_vertical_flip_2x(img0, imgt, img1)
            img0, imgt, img1 = random_horizontal_flip_2x(img0, imgt, img1)
            img0, imgt, img1 = random_rotate_2x(img0, imgt, img1)
            img0, imgt, img1 = random_reverse_time_2x(img0, imgt, img1)

        img0 = torch.from_numpy(img0.reshape(1, img0.shape[0], img0.shape[1]).astype(np.float32))
        imgt = torch.from_numpy(imgt.reshape(1, imgt.shape[0], imgt.shape[1]).astype(np.float32))
        img1 = torch.from_numpy(img1.reshape(1, img1.shape[0], img1.shape[1]).astype(np.float32))
        img0 = img0 / 255
        imgt = imgt / 255
        img1 = img1 / 255
    
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, imgt, img1, embt

class Test_Dataset_2x(Dataset):
    def __init__(self, dataset_dir = ''):
        self.dataset_dir = dataset_dir
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        dataset_list = os.listdir(dataset_dir)
        for data_folder in dataset_list:
            data_list = os.listdir(f"{dataset_dir}/{data_folder}")
            step = 2
            for data_idx in range(len(data_list)):
                if data_idx + step < len(data_list):
                    self.img0_list.append(f"{dataset_dir}/{data_folder}/{data_list[data_idx]}")
                    self.imgt_list.append(f"{dataset_dir}/{data_folder}/{data_list[data_idx + 1]}")
                    self.img1_list.append(f"{dataset_dir}/{data_folder}/{data_list[data_idx + step]}")
                else:
                    break

    def __len__(self):
        return len(self.img0_list)

    def __getitem__(self, idx):
        img0 = imread(self.img0_list[idx])
        imgt = imread(self.imgt_list[idx])
        img1 = imread(self.img1_list[idx])

        img0 = torch.from_numpy(img0.reshape(1, img0.shape[0], img0.shape[1]).astype(np.float32))
        imgt = torch.from_numpy(imgt.reshape(1, imgt.shape[0], imgt.shape[1]).astype(np.float32))
        img1 = torch.from_numpy(img1.reshape(1, img1.shape[0], img1.shape[1]).astype(np.float32))
        img0 = img0 / 255
        imgt = imgt / 255
        img1 = img1 / 255
    
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, imgt, img1, embt
    

class Test_Data_2x(Dataset):
    def __init__(self, dataset_dir = '', step = 2):
        self.dataset_dir = dataset_dir
        self.img0_list = []
        self.img1_list = []
        data_list = os.listdir(f"{dataset_dir}")
        for data_idx in range(0, len(data_list), step):
            if data_idx + step < len(data_list):
                self.img0_list.append(f"{dataset_dir}/{data_list[data_idx]}")
                self.img1_list.append(f"{dataset_dir}/{data_list[data_idx + step]}")
            else:
                break

    def __len__(self):
        return len(self.img0_list)

    def __getitem__(self, idx):
        img0 = imread(self.img0_list[idx])
        img1 = imread(self.img1_list[idx])

        img0 = torch.from_numpy(img0.reshape(1, img0.shape[0], img0.shape[1]).astype(np.float32))
        img1 = torch.from_numpy(img1.reshape(1, img1.shape[0], img1.shape[1]).astype(np.float32))
        img0 = img0 / 255
        img1 = img1 / 255
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, img1, embt
