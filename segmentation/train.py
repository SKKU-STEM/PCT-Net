import warnings
warnings.filterwarnings(action = "ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
import json
import hyperspy.api as hs
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from unet3d import UNet3D
    

class GEN_DATASET(Dataset):
    def __init__(self, dataset_dir, dataset_list):
        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        data = self.dataset_list[idx]
        image_dir = f"{self.dataset_dir}/input/{data}"
        target_dir = f"{self.dataset_dir}/target/{data}"

        image = np.load(image_dir).astype(np.float32)
        target = np.load(target_dir).astype(np.uint8)
        z, h, w = image.shape

        if np.random.random() < 0.5:
            image = image[:, ::-1]
            target = target[:, ::-1]
        
        if np.random.random() < 0.5:
            image = image[:, :, ::-1]
            target = target[:, :, ::-1]
        
        if np.random.random() < 0.5:
            image = image.transpose((0, 2, 1))
            target = target.transpose((0, 2, 1))

        image = torch.from_numpy(image.reshape(z, h, w).copy())
        #image = (image - image.mean()) / image.std()
        image /= 255
        target = torch.from_numpy(target.reshape(z, h, w).copy() > 0)
        
        return image, target
    
def load_dataset(dataset_dir, batch_size, num_workers):
    
    dataset_list = os.listdir(f"{dataset_dir}/input")
    np.random.shuffle(dataset_list)
    training_dataset_list = dataset_list[:int(len(dataset_list) * 4 / 5)]
    validation_dataset_list = dataset_list[int(len(dataset_list) * 4 / 5):]

    train_dataset = GEN_DATASET(dataset_dir = dataset_dir, dataset_list = training_dataset_list)
    valid_dataset = GEN_DATASET(dataset_dir = dataset_dir, dataset_list = validation_dataset_list)

    train_dataset = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = num_workers)
    valid_dataset = DataLoader(valid_dataset, batch_size, shuffle = False, num_workers = num_workers)

    return train_dataset, valid_dataset


def load_optimizer(lr, lr_decay_step, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma = 0.5)

    return criterion, optimizer, scheduler

def train(model, train_dataloader, valid_dataloader, device, optimizer, criterion):
    model.to(device)
    criterion = criterion.to(device)
    train_loss = 0
    train_acc = 0
    train_len = 0
    model.train()
    total_batch = len(train_dataloader)
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        x, y = x.view(-1, 1, x.shape[1], x.shape[2], x.shape[3]).to(device).float(), y.view(-1, y.shape[1], y.shape[2], y.shape[3]).to(device).long()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        output = torch.argmax(output, dim = 1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        batch_acc = (output == y).sum().item()
        train_acc += batch_acc
        batch_acc = batch_acc * 100 / (y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3])
        train_len += (y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3])
        print(f"BATCH : {i + 1} / {total_batch} | Batch loss : {loss.item():.3f} | Batch acc : {batch_acc:.3f} %")

    train_loss /= i + 1
    train_acc = train_acc * 100 / train_len

    with torch.no_grad():
        valid_loss = 0
        valid_acc = 0
        valid_len = 0
        model.eval()
        total_batch = len(valid_dataloader)
        for i, (x, y) in enumerate(valid_dataloader):
            x, y = x.view(-1, 1, x.shape[1], x.shape[2], x.shape[3]).to(device).float(), y.view(-1, y.shape[1], y.shape[2], y.shape[3]).to(device).long()
            output = model(x)
            loss = criterion(output, y)
            valid_loss += loss.item()
            output = torch.argmax(output, dim = 1).cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            batch_acc = (output == y).sum().item()
            valid_acc += batch_acc
            batch_acc = batch_acc * 100 / (y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3])
            valid_len += (y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3])
            print(f"BATCH : {i + 1} / {total_batch} | Batch loss : {loss.item():.3f} | Batch acc : {batch_acc:.3f} %")

        valid_loss /= i + 1
        valid_acc = valid_acc * 100 / valid_len

        return train_loss, train_acc, valid_loss, valid_acc



def set_training_log():
    df = pd.DataFrame({"Epoch" : [],
                       "train loss" : [],
                       "train accuracy (%)" : [],
                       "valid loss" : [],
                       "valid accuracy (%)" : [],
                       })
    return df

def training_log(model_save_path, df, epoch, train_loss, train_corr, valid_loss, valid_corr):
    df_idx = len(df)
    df.loc[df_idx, "Epoch"] = epoch
    df.loc[df_idx, "train loss"] = train_loss
    df.loc[df_idx, "train accuracy (%)"] = train_corr
    df.loc[df_idx, "valid loss"] = valid_loss
    df.loc[df_idx, "valid accuracy (%)"] = valid_corr

    df.to_csv(f'{model_save_path}/train_log.csv')

def main():
    
    gpu_id = "cuda:1"
    dataset_dir = "training_dataset"
    batch_size = 16
    num_workers = 8
    EPOCH = 100
    model_save_path = "train_model0"
    model_save_step = 1
    lr = 0.0000001
    lr_decay_step = 50
    
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    
    model = UNet3D(in_channels = 1, num_classes = 2, level_channels = [64, 128, 256], bottleneck_channel = 512)
    train_dataloader, valid_dataloader = load_dataset(dataset_dir = dataset_dir, 
                                                      batch_size = batch_size,
                                                      num_workers = num_workers
                                                      )
    criterion, optimizer, scheduler = load_optimizer(lr = lr, lr_decay_step = lr_decay_step, model = model)
    train_log = set_training_log()
    
    try:
        os.mkdir(model_save_path)
    except:
        pass

    for epoch in range(1, EPOCH + 1):
        train_loss, train_corr, valid_loss, valid_corr = train(model, train_dataloader, valid_dataloader, device, optimizer, criterion)
        scheduler.step()
        if epoch % model_save_step == 0:
            torch.save(model.state_dict(), f"{model_save_path}/model_epoch{epoch}.pt")
            training_log(model_save_path, train_log, epoch, train_loss, train_corr, valid_loss, valid_corr)
        print(f'EPOCH : {epoch}/{EPOCH}')
        print(f'Training Loss : {train_loss:.3f} | Training acc : {train_corr:.3f} %')
        print(f'Validation Loss : {valid_loss:.3f} | Validation acc : {valid_corr:.3f} %')


if __name__ == "__main__":
    main()

