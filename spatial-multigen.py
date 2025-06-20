import os
import random
import numpy as np
import matplotlib.pyplot as plt
from models_deploy import BiomassGrowthGPU  # Import GPU model
from basemodel import generate_wave  # Import wave generation utility

# Simulation setup
grid_size = 25
x_coords = np.linspace(0, 1, grid_size)
y_coords = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x_coords, y_coords)
all_positions = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(grid_size * grid_size)]
total_positions = len(all_positions)

# Parameters (aligned with the first example)
max_depth = 10
dt = 0.01
steps = 8000  # Number of simulation steps
b0 = 0.1  # Initial biomass

# Generate water level time series (updated function)

# Initialize the GPU biomass model
gpu_model = BiomassGrowthGPU(dt=dt)

# Initial plants
initial_num_plants = 100
init_positions = random.sample(all_positions, initial_num_plants)
current_plants = [(pos, b0) for pos in init_positions]  # b0 is the initial biomass

# Simulation Loop
frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)
population_counts = []

generation = 0
while generation < 10 and current_plants:

    population_counts.append(len(current_plants))
    plt.figure(figsize=(10, 6))
    plt.clf()
    k=(generation+1)/10
    water_levels = generate_wave(24+k, 23, 60, steps=12000)[:steps]

    water_level = water_levels[generation]

    D = max_depth * np.exp(-(X**2) / (2 * 1))
    plt.imshow(D, cmap='Blues', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Waterbed Height (m)')
    plt.title(f'Generation {generation}: {len(current_plants)} Plants')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    all_x = [pos[0] for pos, _ in current_plants]
    all_y = [pos[1] for pos, _ in current_plants]
    plt.scatter(all_x, all_y, c='red', edgecolor='black', s=50, label='Non-reproducers')

    reproducing_positions = []
    new_plants = []

    # Prepare inputs for GPU model
    depths = np.array([max_depth * np.exp(-pos[0]**2 / (2 * 1)) for pos, _ in current_plants], dtype=np.float32)
    water_levels_flat = np.tile(water_levels, (len(current_plants), 1)).astype(np.float32)

    # Simulate growth on GPU
    biomasses, _ = gpu_model.simulate_growth(b0, depths, water_levels_flat, steps)

    # Process results
    for (pos, _), final_biomass in zip(current_plants, biomasses[:, -1]):
        if final_biomass >= 1.2:  # Biomass threshold for reproduction
            reproducing_positions.append(pos)
            for _ in range(5):  # Offspring count
                if(len(new_plants) <10000):
                    new_x = pos[0] + np.random.normal(0, 0.1)
                    new_y = pos[1] + np.random.normal(0, 0.1)
                    new_x = np.clip(new_x, 0, 1)
                    new_y = np.clip(new_y, 0, 1)
                    new_plants.append(((new_x, new_y), b0))  # Initial biomass
    #random.shuffle(new_plants)

    if reproducing_positions:
        repro_x = [pos[0] for pos in reproducing_positions]
        repro_y = [pos[1] for pos in reproducing_positions]
        plt.scatter(repro_x, repro_y, facecolors='green', edgecolors='black', s=50, linewidths=2, label='Reproducers')

    plt.legend()
    plt.savefig(os.path.join(frame_dir, f"frame_{generation}.png"), dpi=300)

    print(f"Generation {generation} - {len(current_plants)} plants:")
    print([(pos, round(bm, 2)) for pos, bm in current_plants])

    if len(new_plants) == 0:
        print("All plants died out.")
        break

    current_plants = new_plants
    generation += 1

plt.savefig("final_population.png", dpi=300)
import matplotlib.animation as animation
from PIL import Image

# Directory containing the frames
frame_dir = "frames"
frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])

# Create an animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')  # Hide axes for a cleaner animation
frame_img = plt.imshow(plt.imread(frame_files[0]))

def update_frame(frame_idx):
    frame_path = frame_files[frame_idx]
    frame_img.set_array(plt.imread(frame_path))
    ax.set_title(f"Generation {frame_idx}")
    return [frame_img]

ani = animation.FuncAnimation(
    fig, update_frame, frames=len(frame_files), interval=500, blit=True
)

# Save the animation as a video or GIF
output_file = "simulation_animation.gif"
ani.save(output_file, writer='pillow', fps=2)