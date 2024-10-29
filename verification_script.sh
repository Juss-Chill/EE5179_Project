#!/bin/bash

#############################################
# Add your folders here
#############################################

# Define the list of folders and testset names
folders=(
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/bottle/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/capsule/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/grid/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/leather/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/pill/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/tile/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/transistor/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/zipper/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/cable/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/carpet/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/hazelnut/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/metal_nut/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/screw/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/toothbrush/Val/Degraded_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/wood/Val/Degraded_image/"
)

ground_truth_folders=(

    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/bottle/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/capsule/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/grid/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/leather/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/pill/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/tile/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/transistor/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/zipper/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/cable/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/carpet/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/hazelnut/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/metal_nut/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/screw/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/toothbrush/Val/GT_clean_image/"
    "/home/asl/Muni/EE5179/project/Denoising_Dataset_train_val/wood/Val/GT_clean_image/"
)

testset_names=("bottle" "capsule" "grid" "leather" "pill" "tile" "transistor" "zipper" "cable" "carpet" "hazelnut" "metal_nut" "screw" "toothbrush" "wood")

# Loop through each folder and run the command
for i in "${!folders[@]}"; do
    python3 main_test_scunet_real_application.py --model_name new_model499 \
        --testsets "${folders[i]}" \
        --testset_name "${testset_names[i]}" \
        --results ./scunet_499ep/ \
        --ground_truth_name "${ground_truth_folders[i]}"
done
