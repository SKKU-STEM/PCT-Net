import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import hyperspy.api as hs
import pandas as pd
import time
from tqdm import tqdm
from unet3d import UNet3D

def load_model(model_dir):
    model_state_dict = torch.load(model_dir, map_location = "cpu")
    in_channels = model_state_dict['a_block1.conv1.weight'].shape[1]
    num_classes = model_state_dict['s_block1.conv3.weight'].shape[0]
    level_channels = [model_state_dict['a_block1.conv2.weight'].shape[0], 
                    model_state_dict['a_block2.conv2.weight'].shape[0], 
                    model_state_dict['a_block3.conv2.weight'].shape[0]]
    bottleneck_channel = model_state_dict['bottleNeck.conv2.weight'].shape[0]

    model = UNet3D(in_channels = in_channels, num_classes = num_classes,
                level_channels = level_channels, bottleneck_channel = bottleneck_channel)

    model.load_state_dict(model_state_dict)
    
    return model

def load_input(data_dir, z_step, z_overlap, x_step, x_overlap):
    img_stack = hs.load(data_dir)
    img_stack = img_stack.data.astype(np.float32)

    input_tensor = torch.from_numpy(img_stack).float()
    input_tensor = input_tensor.view(-1, 1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2])
    #input_tensor = (input_tensor - input_tensor.mean()) / input_tensor.std()
    input_tensor /= 255
        
    z_idx = 0

    z_start = 0
    z_end = z_start + z_step
    z_idx_list = []

    while z_start < img_stack.shape[0] - z_overlap:
        z_idx_list.append([z_start, z_end])
        z_start = z_end - z_overlap
        z_end = z_start + z_step
        if z_end > img_stack.shape[0]:
            z_end = img_stack.shape[0]

    x_idx = 0
    x_start = 0
    x_end = x_start + x_step
    x_idx_list = []
    while x_start < img_stack.shape[2] - x_overlap:
        x_idx_list.append([x_start, x_end])
        x_start = x_end - x_overlap
        x_end = x_start + x_step
        if x_end > img_stack.shape[2]:
            x_end = img_stack.shape[2]
    x_idx_list

    input_list = []

    for z_idx in z_idx_list:
        for x_idx in x_idx_list:
            z_start = z_idx[0]
            z_end = z_idx[1]
            x_start = x_idx[0]
            x_end = x_idx[1]
            input_list.append(input_tensor[:, :, z_start:z_end, :, x_start:x_end])

    return input_list, z_idx_list, x_idx_list, img_stack.shape

def get_3Dporemap(model, input_list, z_idx_list, x_idx_list, data_shape, device):
    output_list = []
    with torch.no_grad():
        model.eval()
        model.to(device)
        for input_tensor in tqdm(input_list):
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            output = torch.argmax(output, dim = 1).cpu().detach().numpy()[0]
            output_list.append(output)

    total_output = np.zeros(data_shape)
    output_idx = 0
    for z_idx in z_idx_list:
        for x_idx in x_idx_list:
            z_start = z_idx[0]
            z_end = z_idx[1]
            x_start = x_idx[0]
            x_end = x_idx[1]
            total_output[z_start : z_end, :, x_start : x_end] += output_list[output_idx]
            output_idx += 1
    total_output_sig = hs.signals.Signal2D(total_output > 0)
    total_output_sig.change_dtype("uint8")
    
    return total_output_sig

def quantitative_analysis(total_output_sig, x_scale, y_scale, z_scale):
    
    xy = 0
    yz = 0
    zx = 0
    for i in range(total_output_sig.shape[0] - 1):
        xy += (total_output_sig[i] != total_output_sig[i + 1]).sum()

    for i in range(total_output_sig.shape[1] - 1):
        zx += (total_output_sig[:, i] != total_output_sig[:, i + 1]).sum()

    for i in range(total_output_sig.shape[2] - 1):
        yz += (total_output_sig[:, :, i] != total_output_sig[:, :, i + 1]).sum()

    surface_area = xy * x_scale * y_scale + yz * z_scale * y_scale + zx * z_scale * x_scale
    porosity = ((total_output_sig == 1).sum() / (total_output_sig.shape[0] * total_output_sig.shape[1] * total_output_sig.shape[2]))


    return porosity, surface_area

def segmented_volume_analysis(total_output_sig, x_scale, y_scale, z_scale, unit_scale, sampling_num):
    
    total_volume = total_output_sig.copy()
    x_pixel = int(unit_scale / x_scale)
    y_pixel = int(unit_scale / y_scale)
    z_pixel = int(unit_scale / z_scale)

    total_z_pixel, total_y_pixel, total_x_pixel = total_volume.shape

    analysis_arr = []
    for _ in range(sampling_num):
        x_pos = np.random.randint(total_x_pixel - x_pixel)
        y_pos = np.random.randint(total_y_pixel - y_pixel)
        z_pos = np.random.randint(total_z_pixel - z_pixel)

        segmented_volume = total_volume[z_pos : z_pos + z_pixel, y_pos : y_pos + y_pixel, x_pos : x_pos + x_pixel]
        porosity, surface_area = quantitative_analysis(segmented_volume, x_scale, y_scale, z_scale)
        analysis_arr.append([porosity, surface_area])
    
    analysis_arr = np.array(analysis_arr)

    return analysis_arr



def main():
    model_dir = "train_model0/model_epoch100.pt"
    src_dir = "test_data"
    device = "cuda:0"
    sampling_num = 100
    unit_scale_list = [0.5, 1, 2]

    data_list = os.listdir(src_dir)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir)

    result_df = pd.DataFrame({"data_dir" : [],
                              "porosity" : [],
                              "surface_area" : []})
    
    df_idx = 0
    for data in data_list:

        data_dir = f"{src_dir}/{data}"
        x_scale = 0.0064
        y_scale = 0.0064
        z_scale = 0.015
        z_step = 40
        z_overlap = 10
        x_step = 300
        x_overlap = 50
        print(data_dir)
        input_list, z_idx_list, x_idx_list, data_shape = load_input(data_dir, z_step, z_overlap, x_step, x_overlap)
        start_time = time.time()
        total_output_sig = get_3Dporemap(model, input_list, z_idx_list, x_idx_list, data_shape, device)
        porosity, surface_area = quantitative_analysis(total_output_sig.data, x_scale, y_scale, z_scale)
        end_time = time.time()
        print(end_time - start_time)
        #total_output_sig.save(f"{model_dir.split('/')[0]}/{data}_pore_map.tif")
        result_df.loc[df_idx, "data_dir"] = data
        result_df.loc[df_idx, "porosity"] = porosity
        result_df.loc[df_idx, "surface_area"] = surface_area
        #for unit_scale in unit_scale_list:
        #    analysis_arr = segmented_volume_analysis(total_output_sig.data, x_scale, y_scale, z_scale, unit_scale, sampling_num)
        #    np.savetxt(f"{model_dir.split('/')[0]}/{data}_segmented_volume{unit_scale}.csv", analysis_arr, delimiter = ",")

        df_idx += 1
    
    #result_df.to_csv(f"{model_dir.split('/')[0]}/quantitative_result.csv")


if __name__ == "__main__":
    main()

