# -*- coding: utf-8 -*-
"""
Extended Biomass Growth Model with CPU and GPU JIT Support
"""

import numpy as np
from numba import njit, cuda

from basemodel import BiomassGrowthModel  # Import the base class
from numba import get_num_threads, set_num_threads
from multiprocessing import Pool
import numpy as np
from numba import njit
import math
@njit
def exp_drowning_response_jit(x,a,k):
    if -a < x :
        return  math.exp(-k * (x)) / math.exp(-k * (-1*a)) 
    else:
        return 0



@njit
def simulate_growth_single(b0, depth, water_levels, adv, dt):
    """
    Simulate the growth for a single depth level.
    """
    B = b0
    biomass_over_time = []
    depth_time = []

    for i in range(adv):
        plant_height = B
        height_sub = plant_height - water_levels[i] + depth
        rate = exp_drowning_response_jit(height_sub, 5, 3) * 0.1 * 30
        B += B * dt * rate 
        biomass_over_time.append(B)
        depth_time.append(height_sub)

    return biomass_over_time, depth_time


class BiomassGrowthCPU:
    def __init__(self, dt):
        self.dt = dt

    def simulate_growth(self, b0, depths, water_levels, adv):
        """
        Parallelize growth simulation across multiple depths.
        """
        # Prepare input arguments
        args = [(b0, depth, water_levels[t], adv, self.dt) for t, depth in enumerate(depths)]

        # Use multiprocessing to parallelize
        with Pool() as pool:
            results = pool.starmap(simulate_growth_single, args)

        # Separate results
        biomass_results = [r[0] for r in results]
        depth_results = [r[1] for r in results]

        return biomass_results, depth_results
@cuda.jit
def gpu_growth_kernel(biomass_results, depth_results, b0, depths, water_levels, adv, dt, a, k):
    idx = cuda.grid(1)
    n_depths = depths.shape[0]  # Number of depths
    if idx < n_depths:
        B = b0
        depth = depths[idx]
        for t in range(adv):
            plant_height = B
            height_sub = plant_height - water_levels[idx, t] + depth
            rate = exp_drowning_response_jit(height_sub, a, k) * 0.1 * 30
            B += B * dt * rate 
            biomass_results[idx, t] = B   # Store time-series result
            depth_results[idx, t] = height_sub  # Store corresponding height_sub
    
class BiomassGrowthGPU(BiomassGrowthModel):
    def simulate_growth(self, b0, depths, water_levels, adv):
        # Ensure proper data types
        n_depths = len(depths)
        n_steps = water_levels.shape[1]
        depths = np.array(depths, dtype=np.float32)
        water_levels = np.array(water_levels, dtype=np.float32)

        # Allocate memory for results
        biomass_results = np.zeros((n_depths, n_steps), dtype=np.float32)
        depth_results = np.zeros((n_depths, n_steps), dtype=np.float32)

        # Configure GPU kernel execution
        threads_per_block = 256
        blocks_per_grid = (n_depths + threads_per_block - 1) // threads_per_block


        # Launch GPU kernel
        gpu_growth_kernel[blocks_per_grid, threads_per_block](
            biomass_results,
            depth_results,
            b0,
            depths,
            water_levels,
            adv,
            self.dt,
            2,  # a parameter
            2,  # k parameter
        )
        return biomass_results, depth_results



