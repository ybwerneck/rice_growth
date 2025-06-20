# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:13:42 2025

@author: yanbw
"""

import numpy as np
import matplotlib.pyplot as plt
from models_deploy import BiomassGrowthGPU  # Import GPU model
from basemodel import generate_wave  # Import wave generation utility

# Parameters
b0 = 0.1  # Initial biomass
dt = 0.01  # Time step
steps = 8000  # Number of simulation steps

# Define parameter grids
depth_values = np.linspace(0, 10, 30)  # Depths (0-10 meters)
maxa_values = np.linspace(10,40, 30)  # Max flood amplitudes (20-40 meters)

# Initialize arrays for water levels and biomass
final_biomass_matrix = np.zeros((len(depth_values), len(maxa_values)), dtype=np.float32)
water_levels_matrix = np.zeros((len(depth_values), len(maxa_values), steps), dtype=np.float32)

# Generate water level time series
for i, depth in enumerate(depth_values):
    for j, maxa in enumerate(maxa_values):
        water_levels_matrix[i, j] = generate_wave(maxa, 23, 60, steps=12000)[:steps]

# Prepare flattened inputs for GPU model
depths_flat = np.repeat(depth_values, len(maxa_values))  # Repeat depths for each maxa valuex
water_levels_flat = water_levels_matrix.reshape((-1, steps), order="C")  # Flatten water levels matrix

# Ensure GPU-compatible data types
depths_flat = np.array(depths_flat, dtype=np.float32)
water_levels_flat = np.array(water_levels_flat, dtype=np.float32)

# Run the GPU model
gpu_model = BiomassGrowthGPU(dt=dt)
biomass_results, _ = gpu_model.simulate_growth(b0, depths_flat, water_levels_flat, steps)



# Plot water levels: Min, max, and mean
min_water_levels = np.min(water_levels_matrix[:, :, :steps], axis=(0, 1))
max_water_levels = np.max(water_levels_matrix[:, :, :steps], axis=(0, 1))
mean_water_levels = np.mean(water_levels_matrix[:, :,:steps], axis=(0, 1))

plt.figure(figsize=(10, 6))
plt.fill_between(np.linspace(0,11,len(mean_water_levels)),range(len(min_water_levels)), min_water_levels, max_water_levels, alpha=0.2, label='Water Level Range')
plt.plot(np.linspace(0,11,len(mean_water_levels)),mean_water_levels, label='Mean Water Level', linestyle='--', color='blue')
plt.xlabel('Time')
plt.ylabel('Water Levels')
plt.legend()
plt.title('Water Levels with Min-Max Range')
plt.savefig('water_levels_shaded.png', dpi=300)

# Plot heatmap of final biomass
plt.figure(figsize=(10, 8))
plt.imshow(biomass_results[:,-1].reshape((len(depth_values),len(maxa_values)),order="A"), origin='lower', aspect='auto',
           extent=[maxa_values[0], maxa_values[-1], depth_values[0], depth_values[-1]],
           cmap='viridis')
plt.colorbar(label='Final Biomass')
plt.xlabel('Flood Amplitude (m)')
plt.ylabel('Plant Depth (m)')
plt.title('Heatmap of Final Biomass')
plt.savefig('biomasss_heatmap.png', dpi=300)

print("Simulation complete. Plots saved.")
