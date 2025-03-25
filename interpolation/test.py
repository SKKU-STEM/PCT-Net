import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import numpy as np
import torch
import PIL.Image as pilimg
import hyperspy.api as hs
from torch.utils.data import DataLoader
from models.IFRNet import Model
from datasets import Test_Data_2x

def convert_to_uint8(tensor):
    tensor = tensor.cpu().detach().numpy()[0][0]
    int_arr = tensor.copy()
    int_arr = int_arr.reshape(-1)
    int_arr.sort()
    int_min = int_arr[int(len(int_arr) * 0.001)]
    int_max = int_arr[-int(len(int_arr) * 0.001)]
    tensor = (tensor - int_min) * 255 / (int_max - int_min)
    tensor[tensor < 0] = 0
    tensor[tensor > 255] = 255
    tensor = tensor.astype(np.uint8)
    img = pilimg.fromarray(tensor)
    return img

def main():
    device = "cuda:3"
    model_dir = "model_2x/model_epoch100.pt"

    sample_list = ["A", "A'"]
    step_list = range(1, 7)

    for sample in sample_list:
        for step_i in step_list:

            data_dir = f"test_dataset1/{sample}"
            step = 2**step_i
            
            device = torch.device(device if torch.cuda.is_available() else "cpu")
            model = Model(device = device)
            model.load_state_dict(torch.load(model_dir))
            step_range = range(0, int(np.log2(step)))
            
            model.to(device)

            for step_idx in step_range:
                if step_idx == 0:
                    data_src_dir = data_dir
                    result_dir = f"{data_dir}_result_step{step}_{step_idx}"            
                    dataset_test = Test_Data_2x(dataset_dir = data_src_dir, step = step)
                else:
                    data_src_dir = f"{data_dir}_result_step{step}_{step_idx-1}"
                    result_dir = f"{data_dir}_result_step{step}_{step_idx}"
                    dataset_test = Test_Data_2x(dataset_dir = data_src_dir, step = 1)

                dataloader_test = DataLoader(dataset_test, batch_size = 1, num_workers = 1,
                                            shuffle = False)

                try:
                    os.mkdir(result_dir)
                except:
                    pass

                with torch.no_grad():
                    model.eval() 
                    data_idx = 0
                    for i, data in enumerate(dataloader_test):
                        img0, img1, embt = data
                        img0 = img0.to(device)
                        img1 = img1.to(device)
                        embt = embt.to(device)
                        img_pred = model.inference(img0, img1, embt)
                        
                        img0 = convert_to_uint8(img0)
                        img0.save(f"{result_dir}/{data_idx:03}.tif")
                        data_idx += 1
                        img_pred = convert_to_uint8(img_pred)
                        img_pred.save(f"{result_dir}/{data_idx:03}.tif")
                        data_idx += 1
                    img1 = convert_to_uint8(img1)
                    img1.save(f"{result_dir}/{data_idx:03}.tif")
                if step_idx != 0:
                    shutil.rmtree(data_src_dir)
            
            img_list = os.listdir(result_dir)
            img_stack = []
            for img in img_list:
                s = hs.load(f"{result_dir}/{img}")
                img_stack.append(s.data)
            img_stack = np.array(img_stack)
            img_stack = hs.signals.Signal2D(img_stack)
            img_stack.save(f"{data_dir}_step{step}_stack.tif")
            shutil.rmtree(result_dir)

if __name__ == "__main__":
    main()