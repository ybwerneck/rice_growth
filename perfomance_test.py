import numpy as np
import time
from basemodel import BiomassGrowthModel,generate_wave
from models_deploy import BiomassGrowthCPU, BiomassGrowthGPU
import matplotlib.pyplot as plt
# Parameters
b0 = 0.1  # Initial biomass
steps = 100  # Number of simulation steps
n_samples = 20  # Number of samples to run
depths = np.random.uniform(0, 10, n_samples)  # Random plant depths
water_levels =np.array([generate_wave(i, 23, 60,steps=steps) for i in np.random.uniform(30, 40, n_samples)  ] ) # Random water levels

# Function to check if two results are similar within a tolerance
def check_accuracy(base_results, optimized_results, tolerance=1e-1):
    return np.allclose(base_results, optimized_results, atol=tolerance)

# Base Model
print("Running Base Model...")
base_model = BiomassGrowthModel(dt=0.1)
base_biomass_results = []

start_time = time.perf_counter()
for i in range(n_samples):
    biomass, depth = base_model.simulate_growth(b0, depths[i], water_levels[i], steps)
    base_biomass_results.append(biomass)  # Collect the final biomass at each sample
base_biomass_results = np.array(base_biomass_results)
base_time = time.perf_counter() - start_time
print(f"Base Model Time: {base_time:.2f} seconds")



# GPU-Optimized Model
print("\nRunning GPU-Optimized Model...")
gpu_model = BiomassGrowthGPU(dt=0.1)
start_time = time.perf_counter()
gpu_biomass_results, _ = gpu_model.simulate_growth(b0, depths, water_levels, steps)
gpu_time = time.perf_counter() - start_time
print(f"GPU-Optimized Model Time: {gpu_time:.2f} seconds")

# Check Accuracy of GPU Model
gpu_accuracy = check_accuracy(base_biomass_results, gpu_biomass_results)
print(f"GPU-Optimized Model Accuracy: {'Pass' if gpu_accuracy else 'Fail'}")
plt.figure(figsize=(10, 6))
plt.plot(gpu_biomass_results.T,"-o",color="blue")
plt.plot(base_biomass_results.T,color="red")
plt.savefig('gpu_biomass_results.png')



# CPU-Optimized Model
print("\nRunning CPU-Optimized Model...")
cpu_model = BiomassGrowthCPU(dt=0.1)
start_time = time.perf_counter()
cpu_biomass_results, _ = cpu_model.simulate_growth(b0, depths, water_levels, steps)
cpu_time = time.perf_counter() - start_time
print(f"CPU-Optimized Model Time: {cpu_time:.2f} seconds")

# Check Accuracy of CPU Model
cpu_accuracy = check_accuracy(base_biomass_results, cpu_biomass_results)
print(f"CPU-Optimized Model Accuracy: {'Pass' if cpu_accuracy else 'Fail'}")


# Summary
print("\n--- Timing Summary ---")
print(f"Base Model: {base_time:.2f} seconds")
print(f"CPU-Optimized Model: {cpu_time:.2f} seconds")
print(f"GPU-Optimized Model: {gpu_time:.2f} seconds")

