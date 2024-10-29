import matplotlib.pyplot as plt
import numpy as np

# Data
classes = [
    "Bottle", "Cable", "Capsule", "Carpet", "Grid", "Hazelnut",
    "Leather", "Metal nut", "Pill", "Screw", "Tile", 
    "Truth brush", "Transistor", "Wood", "Zipper"
]

epochs = [50, 100, 150, 200]

# PSNR and SSIM values for each class and epoch
psnr_values = [
    [29.2687, 30.5046, 30.5237, 30.2843],  # Bottle
    [17.40879, 17.4441, 17.1526, 17.3423],  # Cable
    [16.51, 28.8095, 29.0696, 28.1470],     # Capsule
    [17.5406, 17.7514, 17.6359, 17.42],      # Carpet
    [16.5135, 16.5910, 16.637, 16.663],      # Grid
    [22.62002, 22.9756, 22.6856, 22.559],    # Hazelnut
    [23.551, 23.7281, 23.9969, 23.3753],     # Leather
    [17.2246, 17.1273, 16.8841, 17.0177],    # Metal nut
    [26.4830, 27.2398, 26.7589, 27.1981],    # Pill
    [17.8265, 17.822, 17.1993, 18.1474],      # Screw
    [17.6185, 17.3961, 16.7840, 17.2588],     # Tile
    [27.2639, 30.1762, 29.1604, 28.7491],     # Truth brush
    [20.5161, 21.8324, 21.4560, 21.751],      # Transistor
    [20.1684, 21.3461, 22.1571, 20.7846],     # Wood
    [20.842, 20.7025, 20.7978, 20.5258]        # Zipper
]

ssim_values = [
    [0.8835, 0.8866, 0.8894, 0.88636],  # Bottle
    [0.44996, 0.4458, 0.44425, 0.43312],  # Cable
    [0.2465, 0.8704, 0.87056, 0.8691],    # Capsule
    [0.29934, 0.2936, 0.2931, 0.299],     # Carpet
    [0.24656, 0.24819, 0.24547, 0.24967], # Grid
    [0.78044, 0.7754, 0.77903, 0.7645],   # Hazelnut
    [0.4025, 0.39674, 0.39769, 0.39140],   # Leather
    [0.5542, 0.5540, 0.55048, 0.54680],    # Metal nut
    [0.83851, 0.83708, 0.83934, 0.8301],   # Pill
    [0.7630, 0.76280, 0.7614, 0.7618],      # Screw
    [0.20306, 0.1934, 0.1895, 0.1846],      # Tile
    [0.8599, 0.87765, 0.8772, 0.8718],      # Truth brush
    [0.6821, 0.68409, 0.6820, 0.6795],       # Transistor
    [0.361780, 0.3622, 0.3813, 0.3202],      # Wood
    [0.48961, 0.4774, 0.18955, 0.4736]       # Zipper
]

# Set the width of the bars
bar_width = 0.2
x = np.arange(len(classes))

# Create subplots
fig, ax = plt.subplots(figsize=(12, 6))

# Plot PSNR
for i in range(len(epochs)):
    ax.bar(x + i * bar_width, [psnr[i] for psnr in psnr_values], bar_width, label=f'PSNR (Epochs {epochs[i]})')

# Adjust the x-ticks
ax.set_xticks(x + bar_width * (len(epochs) - 1) / 2)
ax.set_xticklabels(classes)

# Add labels and title
ax.set_ylabel('Values')
ax.set_title('PSNR Values by Class and Epochs')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Create subplots
fig, ax = plt.subplots(figsize=(12, 6))
# Plot SSIM
for i in range(len(epochs)):
    ax.bar(x + i * bar_width, [ssim[i] for ssim in ssim_values], bar_width, label=f'SSIM (Epochs {epochs[i]})')

# Adjust the x-ticks
ax.set_xticks(x + bar_width * (len(epochs) - 1) / 2)
ax.set_xticklabels(classes)

# Add labels and title
ax.set_ylabel('Values')
ax.set_title('SSIM Values by Class and Epochs')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()


average_psnr = np.average(psnr_values, axis=0)
average_ssim = np.average(ssim_values, axis=0)

print(f"Average PSNR : ", average_psnr)
print(f"Average SSIM : ", average_ssim)