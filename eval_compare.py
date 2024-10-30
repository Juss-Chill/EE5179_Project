import itertools
import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define possible values for tile and tile_overlap
tile_values = [64, 128, 256, 512]
tile_overlap_values = [16, 32, 64, 128, 256]

# Filter combinations to ensure tile_overlap < tile
combinations = [(tile, tile_overlap) for tile in tile_values for tile_overlap in tile_overlap_values if tile_overlap < tile]

# Directories and task definition
input_dir = "Denoising_Dataset_train_val"
task = "Real_Denoising"

# Create a dictionary to store PSNR and SSIM results for each combination
results = {}

# Loop over all valid combinations and run the evaluation
for tile, tile_overlap in combinations:
    result_dir = f"results/t_{tile}_ol_{tile_overlap}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Run the evaluation using subprocess
    cmd = [
        "python", "eval_mvtecad.py",  # Replace with the actual script name if different
        "--input_dir", input_dir,
        "--result_dir", result_dir,
        "--task", task,
        "--tile", str(tile),
        "--tile_overlap", str(tile_overlap)
    ]
    subprocess.run(cmd)

    # Read the resulting metrics CSV file
    metrics_file = os.path.join(result_dir, "restoration_metrics.csv")
    metrics_df = pd.read_csv(metrics_file)

    # Calculate object-wise PSNR and SSIM
    object_metrics = metrics_df.groupby(['Object Type']).agg({'PSNR (dB)': 'mean', 'SSIM': 'mean'}).reset_index()
    object_metrics['Tile'] = tile
    object_metrics['Tile Overlap'] = tile_overlap

    # Calculate overall PSNR and SSIM
    overall_psnr = metrics_df['PSNR (dB)'].mean()
    overall_ssim = metrics_df['SSIM'].mean()
    overall_row = pd.DataFrame([['Overall', overall_psnr, overall_ssim, tile, tile_overlap]],
                               columns=['Object Type', 'PSNR (dB)', 'SSIM', 'Tile', 'Tile Overlap'])

    # Append the overall metrics to the object metrics
    combined_metrics = pd.concat([object_metrics, overall_row], ignore_index=True)

    # Store results in the dictionary
    if (tile, tile_overlap) not in results:
        results[(tile, tile_overlap)] = combined_metrics

    # Save the combined metrics to a CSV file
    combined_metrics.to_csv(os.path.join(result_dir, "object_wise_metrics.csv"), index=False)

# Create 3D plots for each object and for the overall validation set
objects = combined_metrics['Object Type'].unique()
objects = [obj for obj in objects if obj != 'Overall']  # Exclude 'Overall'

# Plot PSNR and SSIM for each object
for obj in objects + ['Overall']:
    fig_psnr = plt.figure()
    ax_psnr = fig_psnr.add_subplot(111, projection='3d')
    fig_ssim = plt.figure()
    ax_ssim = fig_ssim.add_subplot(111, projection='3d')

    # Prepare data for plotting
    psnr_values = []
    ssim_values = []
    tile_values_plot = []
    tile_overlap_values_plot = []

    for (tile, tile_overlap), metrics_df in results.items():
        if obj in metrics_df['Object Type'].values:
            obj_data = metrics_df[metrics_df['Object Type'] == obj]
            psnr_value = obj_data['PSNR (dB)'].values[0]
            ssim_value = obj_data['SSIM'].values[0]

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            tile_values_plot.append(tile)
            tile_overlap_values_plot.append(tile_overlap)

    # Convert to numpy arrays for plotting
    tile_values_plot = np.array(tile_values_plot)
    tile_overlap_values_plot = np.array(tile_overlap_values_plot)
    psnr_values = np.array(psnr_values)
    ssim_values = np.array(ssim_values)

    # Plot PSNR
    ax_psnr.scatter(tile_values_plot, tile_overlap_values_plot, psnr_values, c='r', marker='o')
    ax_psnr.set_xlabel('Tile Size')
    ax_psnr.set_ylabel('Tile Overlap')
    ax_psnr.set_zlabel('PSNR (dB)')
    ax_psnr.set_title(f'PSNR for {obj}')
    ax_psnr.legend(['PSNR'])

    # Plot SSIM
    ax_ssim.scatter(tile_values_plot, tile_overlap_values_plot, ssim_values, c='b', marker='^')
    ax_ssim.set_xlabel('Tile Size')
    ax_ssim.set_ylabel('Tile Overlap')
    ax_ssim.set_zlabel('SSIM')
    ax_ssim.set_title(f'SSIM for {obj}')
    ax_ssim.legend(['SSIM'])

    # Save the plots
    fig_psnr.savefig(f"results/psnr_plot_{obj}.png")
    fig_ssim.savefig(f"results/ssim_plot_{obj}.png")

    plt.close(fig_psnr)
    plt.close(fig_ssim)

print("Evaluation completed. Plots and metrics are saved in the results directory.")

