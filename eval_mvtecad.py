import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

parser = argparse.ArgumentParser(description='Test Restormer on a dataset with multiple objects and sub-types')
parser.add_argument('--input_dir', required=True, type=str, help='Root directory of input images')
parser.add_argument('--result_dir', required=True, type=str, help='Root directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Motion_Deblurring',
                                                                                    'Single_Image_Defocus_Deblurring',
                                                                                    'Deraining',
                                                                                    'Real_Denoising',
                                                                                    'Gaussian_Gray_Denoising',
                                                                                    'Gaussian_Color_Denoising'])
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=64, help='Overlapping of different tiles')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'mvtecad_96kiters_net_g_96000.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] =  1
        parameters['out_channels'] =  1
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters

def calculate_metrics(restored_img, gt_img):
    """Calculate PSNR and SSIM metrics for two images."""
    psnr_value = psnr(gt_img, restored_img, data_range=gt_img.max() - gt_img.min())
    ssim_value = ssim(gt_img, restored_img, multichannel=True, data_range=gt_img.max() - gt_img.min())
    return psnr_value, ssim_value

task    = args.task
inp_dir = args.input_dir
res_dir = args.result_dir

# Set up result metrics storage
metrics_data = []

# Get model weights and parameters
parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
weights, parameters = get_weights_and_parameters(task, parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")

# Iterate through the object types and sub-types
for object_type in natsorted(os.listdir(inp_dir)):
    degraded_base_dir = os.path.join(inp_dir, object_type, 'Val', 'Degraded_image')
    gt_base_dir = os.path.join(inp_dir, object_type, 'Val', 'GT_clean_image')

    for object_subtype in natsorted(os.listdir(degraded_base_dir)):
        degraded_dir = os.path.join(degraded_base_dir, object_subtype)
        gt_dir = os.path.join(gt_base_dir, object_subtype)
        
        # Set up the result directory for this object type and sub-type
        output_dir = os.path.join(res_dir, 'MVTECAD', object_type, object_subtype)
        os.makedirs(output_dir, exist_ok=True)

        # Get the list of images
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(degraded_dir, '*.'+ext)))
        files = natsorted(files)

        if len(files) == 0:
            print(f'No files found in {degraded_dir}. Skipping...')
            continue

        with torch.no_grad():
            for file_ in tqdm(files, desc=f'Processing {object_type}/{object_subtype}'):
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()

                # Load images
                if task == 'Gaussian_Gray_Denoising':
                    degraded_img = load_gray_img(file_)
                    gt_img = load_gray_img(os.path.join(gt_dir, os.path.basename(file_)))
                else:
                    degraded_img = load_img(file_)
                    gt_img = load_img(os.path.join(gt_dir, os.path.basename(file_)))

                input_ = torch.from_numpy(degraded_img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

                # Pad the input if not multiple of 8
                height,width = input_.shape[2], input_.shape[3]
                H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
                padh = H-height if height%img_multiple_of!=0 else 0
                padw = W-width if width%img_multiple_of!=0 else 0
                input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                # Perform restoration
                if args.tile is None:
                    restored = model(input_)
                else:
                    # Tile-based processing
                    b, c, h, w = input_.shape
                    tile = min(args.tile, h, w)
                    assert tile % 8 == 0, "tile size should be multiple of 8"
                    tile_overlap = args.tile_overlap

                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                    E = torch.zeros(b, c, h, w).type_as(input_)
                    W = torch.zeros_like(E)

                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                            out_patch = model(in_patch)
                            out_patch_mask = torch.ones_like(out_patch)

                            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                    restored = E.div_(W)

                restored = torch.clamp(restored, 0, 1)

                # Unpad the output
                restored = restored[:,:,:height,:width]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = img_as_ubyte(restored[0])

                # Save the restored image
                output_file = os.path.join(output_dir, os.path.basename(file_))
                if task == 'Gaussian_Gray_Denoising':
                    save_gray_img(output_file, restored)
                else:
                    save_img(output_file, restored)

                # Calculate metrics
                psnr_value, ssim_value = calculate_metrics(restored, gt_img)
                metrics_data.append([object_type, object_subtype, os.path.basename(file_), psnr_value, ssim_value])

# Save metrics to a CSV file
metrics_df = pd.DataFrame(metrics_data, columns=['Object Type', 'Object Subtype', 'Image Name', 'PSNR (dB)', 'SSIM'])
metrics_df.to_csv(os.path.join(res_dir, 'restoration_metrics.csv'), index=False)
print("Processing complete. Metrics saved to 'restoration_metrics.csv'")

