import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import PIL.Image as pilimg
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.IFRNet import Model
from datasets import Train_Dataset_2x, Test_Dataset_2x
from metric import calculate_ssim, calculate_psnr, calculate_ie
import pandas as pd

def get_data_from_dataloader(data):
    if len(data) == 4:
        return data[0], data[1], data[2], data[3]
    elif len(data) == 8:
        img0 = torch.cat([data[0], data[0], data[0]], 0)
        imgt = torch.cat([data[1], data[2], data[3]], 0)
        img1 = torch.cat([data[4], data[4], data[4]], 0)
        embt = torch.cat([data[5], data[6], data[7]])
        return img0, imgt, img1, embt
    elif len(data) == 16:
        img0 = torch.cat([data[0], data[0], data[0], data[0], data[0], data[0], data[0]], 0)
        imgt = torch.cat([data[1], data[2], data[3], data[4], data[5], data[6], data[7]], 0)
        img1 = torch.cat([data[8], data[8], data[8], data[8], data[8], data[8], data[8]], 0)
        embt = torch.cat([data[9], data[10], data[11], data[12], data[13], data[14], data[15]])
        return img0, imgt, img1, embt
    elif len(data) == 32:
        img0 = torch.cat([data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0], data[0]], 0)
        imgt = torch.cat([data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]], 0)
        img1 = torch.cat([data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16], data[16]], 0)
        embt = torch.cat([data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31]])
        return img0, imgt, img1, embt
    

def convert_to_uint8(tensor):
    tensor = tensor.cpu().detach().numpy()[0]
    int_arr = tensor.copy()
    int_arr = int_arr.reshape(-1)
    int_arr.sort()
    int_min = int_arr[int(len(int_arr) * 0.001)]
    int_max = int_arr[-int(len(int_arr) * 0.001)]
    tensor = (tensor - int_min) * 255 / (int_max - int_min)
    tensor[tensor < 0] = 0
    tensor[tensor > 255] = 255
    tensor = tensor.astype(np.uint8)
    img = torch.from_numpy(tensor)
    img = img.view(1, tensor.shape[0], tensor.shape[1])
    return img

def main():
    epochs = 1000
    batch_size = 32
    lr = 1e-6
    num_workers = 8
    train_dataset_dir = "../train_dataset"
    valid_dataset_dir = "../valid_dataset"
    device = "cuda:3"
    model_save_dir = "model_2x"

    try:
        os.mkdir(model_save_dir)
    except:
        pass

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = Model(device = device)

    dataset_train = Train_Dataset_2x(dataset_dir = train_dataset_dir, augment = True)
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, num_workers = num_workers,
                                shuffle = True)

    dataset_test = Train_Dataset_2x(dataset_dir = valid_dataset_dir, augment = True)
    dataloader_test = DataLoader(dataset_test, batch_size = batch_size, num_workers = num_workers,
                                shuffle = False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    train_df = pd.DataFrame({"EPOCH" : [],
                            "Training Loss Rec" : [],
                            "Training Loss Geo" : [],
                            "Training Loss" : [],
                            "Validation Loss Rec" : [],
                            "Validation Loss Geo" : [],
                            "Validation Loss" : [],
                            "Test PSNR" : [],
                            "Test IE" : [],
                            "Test SSIM" : []
                            })
    model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss_rec = 0
        train_loss_geo = 0
        model.train()
        for i, data in enumerate(dataloader_train):
            img0, imgt, img1, embt = get_data_from_dataloader(data)

            img0 = img0.to(device)
            imgt = imgt.to(device)
            img1 = img1.to(device)
            embt = embt.to(device)

            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo = model(img0, img1, embt, imgt)
            loss = loss_rec + loss_geo
            loss.backward()
            optimizer.step()

            train_loss_rec += loss_rec.item()
            train_loss_geo += loss_geo.item()
            
            print(f"EPOCH : {epoch}/{epochs} iter : {i + 1}/{dataloader_train.__len__()} loss_rec : {loss_rec:4f} loss_geo : {loss_geo:4f}")
        
        train_loss_rec = train_loss_rec / (i + 1)
        train_loss_geo = train_loss_geo / (i + 1)
        train_loss = train_loss_rec + train_loss_geo

        with torch.no_grad():
            valid_loss_rec = 0
            valid_loss_geo = 0
            valid_psnr = 0
            valid_ie = 0
            valid_ssim = 0
            model.eval() 
            for i, data in enumerate(dataloader_test):
                img0, imgt, img1, embt = get_data_from_dataloader(data)
                img0 = img0.to(device)
                imgt = imgt.to(device)
                img1 = img1.to(device)
                embt = embt.to(device)
                imgt_pred, loss_rec, loss_geo = model(img0, img1, embt, imgt)
                valid_loss_rec += loss_rec.item()
                valid_loss_geo += loss_geo.item()

                imgt_pred = imgt_pred.cpu().detach()
                imgt = imgt.cpu()
                batch_psnr = 0
                batch_ie = 0
                batch_ssim = 0
                for j in range(len(imgt_pred)):
                    imgt_pred_i = convert_to_uint8(imgt_pred[j]).float()
                    imgt_i = convert_to_uint8(imgt[j]).float()
                    batch_psnr += calculate_psnr(imgt_pred_i, imgt_i)
                    batch_ie += calculate_ie(imgt_pred_i, imgt_i)
                    batch_ssim += calculate_ssim(imgt_pred_i, imgt_i)
                
                batch_psnr = batch_psnr / len(imgt_pred)
                batch_ie = batch_ie / len(imgt_pred)
                batch_ssim = batch_ssim / len(imgt_pred)

                valid_psnr += batch_psnr
                valid_ie += batch_ie
                valid_ssim += batch_ssim

                print(f"EPOCH : {epoch}/{epochs} iter : {i + 1}/{dataloader_test.__len__()} loss_rec : {loss_rec:4f} loss_geo : {loss_geo:4f}")
                print(f"batch PSNR : {batch_psnr} batch IE : {batch_ie} batch SSIM : {batch_ssim}")

            valid_loss_rec = valid_loss_rec / (i + 1)
            valid_loss_geo = valid_loss_geo / (i + 1)
            valid_loss = valid_loss_rec + valid_loss_geo
            valid_psnr = valid_psnr / (i + 1)
            valid_ie = valid_ie / (i + 1)
            valid_ssim = valid_ssim / (i + 1)

            train_df.loc[epoch - 1, "EPOCH"] = epoch
            train_df.loc[epoch - 1, "Training Loss Rec"] = train_loss_rec
            train_df.loc[epoch - 1, "Training Loss Geo"] = train_loss_geo
            train_df.loc[epoch - 1, "Training Loss"] = train_loss
            train_df.loc[epoch - 1, "Validation Loss Rec"] = valid_loss_rec
            train_df.loc[epoch - 1, "Validation Loss Geo"] = valid_loss_geo
            train_df.loc[epoch - 1, "Validation Loss"] = valid_loss
            train_df.loc[epoch - 1, "Test PSNR"] = valid_psnr.item()
            train_df.loc[epoch - 1, "Test IE"] = valid_ie.item()
            train_df.loc[epoch - 1, "Test SSIM"] = valid_ssim.item()

            train_df.to_csv(f"{model_save_dir}/train_log.csv")
            torch.save(model.state_dict(), f"{model_save_dir}/model_epoch{epoch}.pt")
        

if __name__ == "__main__":
    main()