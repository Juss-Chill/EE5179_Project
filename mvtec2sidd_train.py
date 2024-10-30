import os
import shutil

def process_images(main_folder, output_folder):
    """
    Process images from the specified main folder and save them in the output folder
    following the renaming and folder structure as specified.
    """
    # Iterate through each object (e.g., 'bottle') in the main folder
    for obj_name in os.listdir(main_folder):
        obj_path = os.path.join(main_folder, obj_name)
        if not os.path.isdir(obj_path):
            continue

        # Iterate through each condition folder (e.g., 'broken_large') under 'Degraded_image'
        degraded_image_folder = os.path.join(obj_path, 'Train', 'Degraded_image')
        gt_clean_image_folder = os.path.join(obj_path, 'Train', 'GT_clean_image')

        if not os.path.exists(degraded_image_folder) or not os.path.exists(gt_clean_image_folder):
            print(f"Skipping {obj_name}, as the required folders are not present.")
            continue

        for condition in os.listdir(degraded_image_folder):
            degraded_condition_folder = os.path.join(degraded_image_folder, condition)
            gt_condition_folder = os.path.join(gt_clean_image_folder, condition)

            if not os.path.isdir(degraded_condition_folder) or not os.path.isdir(gt_condition_folder):
                print(f"Skipping {condition} for {obj_name}, as the required condition folders are not present.")
                continue

            # Process each image in the degraded condition folder
            for image_file in os.listdir(degraded_condition_folder):
                original_filename, ext = os.path.splitext(image_file)
                if ext.lower() not in ['.png']:
                    continue  # Skip non-image files

                # Define new names and directories for the images
                new_folder_name = f"{original_filename}_{obj_name}-{condition}"
                new_folder_path = os.path.join(output_folder, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)

                # Rename for noisy and clean images
                noisy_image_new_name = f"{original_filename}_NOISY_{obj_name}-{condition}{ext}"
                gt_image_new_name = f"{original_filename}_GT_{obj_name}-{condition}{ext}"

                # Full paths for the source and destination images
                noisy_image_src = os.path.join(degraded_condition_folder, image_file)
                noisy_image_dst = os.path.join(new_folder_path, noisy_image_new_name)

                gt_image_src = os.path.join(gt_condition_folder, image_file)
                gt_image_dst = os.path.join(new_folder_path, gt_image_new_name)

                # Copy the images to the new locations
                shutil.copy(noisy_image_src, noisy_image_dst)
                shutil.copy(gt_image_src, gt_image_dst)

                print(f"Copied and renamed {image_file} to {noisy_image_dst} and {gt_image_dst}")

if __name__ == "__main__":
    # Define the main input folder containing the original dataset
    main_folder = 'Denoising_Dataset_train_val'
    # Define the output folder where the new structure will be created
    output_folder = 'MVTEC/train'

    # Run the processing function
    process_images(main_folder, output_folder)

