import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util

import matplotlib.pyplot as plt

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='scunet_color_real_psnr', help='scunet_color_real_psnr, scunet_color_real_gan')
    parser.add_argument('--testset_name', type=str, default='real3', help='test set, bsd68 | set12')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_zoo', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--results', type=str, default='results', help='path of results')
    parser.add_argument('--ground_truth_name', type=str, default='ground_truth_name', help='path of ground truth images')
    parser.add_argument('--parent_path', type=str, default='/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/', help='provide the parent path of the Data')

    args = parser.parse_args()

    n_channels = 3

    result_name = args.testset_name + '_' + args.model_name     # fixed
    model_path = os.path.join(args.model_zoo, args.model_name+'.pth')
    
    gt_path = args.ground_truth_name

    parent_path = args.parent_path

    print("Parent path : ", parent_path)

    class_list = sorted(os.listdir(parent_path))
    # class_list = class_list

    print("Classes present : ", class_list)
    
    
    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------

    # get the list of directories at the testsets

    test_set_names = [name for name in os.listdir(args.testsets)]


    sub_class_paths = [parent_path+str(cls)+'/'+'Train/Degraded_image/' for cls in class_list]

    test_set_names.clear()

    for path in sub_class_paths:
        for sub_class in os.listdir(path):
            test_set_names.append(os.path.join(path, sub_class))

    ## Gt starts
    sub_class_paths_GT = [parent_path+str(cls)+'/'+'Train/GT_clean_image/' for cls in class_list]

    test_set_names_GT = []

    for path in sub_class_paths_GT:
        for sub_class in os.listdir(path):
            test_set_names_GT.append(os.path.join(path, sub_class))

    L_path = []
    
    for test_set_name in test_set_names:
        L_path.append(os.path.join(args.testsets, test_set_name)) # L_path, for Low-quality images

    G_path = []
    for test_set_name in test_set_names:
        G_path.append(os.path.join(gt_path, test_set_name)) # G_path, for Low-quality images

    L_path = test_set_names
    G_path = test_set_names_GT

    
    E_path = os.path.join(args.results, result_name)        # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_scunet import SCUNet as net
    model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)

    print("Loaded model : ", model_path)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.train()

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    # for k, v in model.named_parameters():
    #     v.requires_grad = False

    model = model.to(device)

    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    logger.info('model_name:{}'.format(args.model_name))
    # logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    G_paths = util.get_image_paths(G_path)


    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    running_loss = 0.
    total_img = 0.
    loss_list = []

    for epoch in range(500):
        print(f"=============== ***** Epoch : , {epoch} ***** ===============")
        for idx, (img, gt_img) in enumerate(zip(L_paths, G_paths)):
            # print("image : ", img)
            # print("GT imae path : ", gt_img)

            total_img += 1
            # print("*************************************************************")
            # print(gt_img, "\n", img)
            # print("*************************************************************")
            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            gt_img_name, ext = os.path.splitext(os.path.basename(gt_img))

            logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))

            img_L = util.imread_uint(img, n_channels=n_channels)
            img_G = util.imread_uint(gt_img, n_channels=n_channels)

            util.imshow(img_L) if args.show_img else None

            img_L = util.uint2tensor4(img_L)
            img_L = img_L.to(device)

            img_G = util.uint2tensor4(img_G)
            img_G = img_G.to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------

            #img_E = utils_model.test_mode(model, img_L, refield=64, min_size=512, mode=2)
            optimiser.zero_grad()

            img_E = model(img_L)
            # img_E = util.tensor2uint(img_E)
            # img_E = torch.tensor(img_E, device=device, dtype=torch.float)
            img_E.requires_grad_(True)

            # print(img_L.shape, img_G.shape, img_E.shape)

            loss = criterion(img_E, img_G)
            loss.backward()
            
            optimiser.step()

            running_loss += loss.item()
            
            loss_list.append(running_loss/total_img)

            if total_img%len(G_paths) == 0:
                print("loss : ", running_loss/total_img)

                # save the mode for every 50 epochs
    
        if (epoch+1) % 50 == 0:
            print("************************ INterim Model Saved********************************************")
            model_name = 'new_model' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_name)
            print("********************************************************************************")

        if epoch == 499:
            print("************************ Final Model Saved********************************************")
            model_name = 'final_new_model' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_name)
            print("********************************************************************************")

    plt.plot(loss_list)
    plt.show()

   
    torch.save(model.state_dict(), './muni.pth')

    
    
        # ------------------------------------
        # save results
        # ------------------------------------
        # util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

if __name__ == '__main__':

    main()