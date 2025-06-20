# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 00:28:52 2025

@author: yanbw
"""

import numpy as np
import matplotlib.pyplot as plt



def exp_drowning_response(x,a,k):
    if -a < x :
        return  np.exp(-k * (x)) / np.exp(-k * (-1*a)) 
    else:
        return 0


class BiomassGrowthModel:
    def __init__(self,  dt=0.1,f=lambda x : exp_drowning_response(x,a=5,k=1)):
       
       ##MULTIGEN PARAMETERS
        self.biomass_threshold = .3
        self.sigma_dispersion = 0.1
        self.offspring_count = 3
   
        ##NUMERICAL METHOD PARAMETERS
        self.dt = dt


       ##RESPONSE FUNCTION
        self.f=f
    
    def simulate_growth(self, b0, depth, water_levels,adv):
        B = b0
        S = 0
        biomass_over_time = []
        seeding_effort_time = []
        depth_time = []
        for i in range(adv):
            plant_height = B
            height_sub = plant_height - water_levels[i] + depth
            depth_time.append(height_sub)
            rate = self.f(height_sub) * 0.1*30
            B += B * self.dt *rate 
            biomass_over_time.append(B)
        return biomass_over_time, depth_time



##floodplains model
def generate_wave(A_base=7*5, nivel_medio=22.70, D=60,FPh=20,steps=120):
    """
    Gera uma única onda senoidal.
    
    Parâmetros:
    A_base (float): Amplitude da onda.
    nivel_medio (float): Nível médio da onda.
    duracao (int): Duração da onda em unidades de tempo.
    
    Retorna:
    numpy.ndarray: Onda gerada.
    """
    T =  np.linspace(0, D, steps)
    onda = nivel_medio + A_base * np.sin(2 * np.pi * (1 / D) * (T))
    onda = nivel_medio + A_base * np.sin(2 * np.pi * (1 / D) * (T-3))-FPh
    ##nivel de transborde
    onda = np.where(onda < 0, 0, onda)
    onda = np.where(onda > 10, 10, onda)
    
    return onda

if(__name__ == "__main__"):
    a = 5  # Drowning threshold
    x_values = np.linspace(-a, 2, 500)  # Submersion levels
    k_values = [.1, .3, .5]  # Decay rates to test
    D_T_values = [1,2 , 4]  # Different values of D_T to test

    # Define the function parameters
    M = 1  # Maximum value
    k = 3 # Decay rate (adjustable for smooth transition)

    D_T=5
    T = 12
    dt = 0.01
    steps = int(T / dt)
    water_levels = generate_wave(10,23,60,steps=1200)

    steps=1000


    DT=3
    K=1
    colors = ["red", "blue", "green"]
    model = BiomassGrowthModel(f=lambda x:exp_drowning_response(x,DT,K))
    plant_depths = [1, 3, 5]  # Different depths
    biomass_results = [model.simulate_growth(0.001, depth, water_levels,steps) for depth in plant_depths]

    y_values = [exp_drowning_response(x,DT, K) for x in x_values]

    # Plot results
    plt.figure(figsize=(12*1.2, 8*1.2))
    time = np.linspace(0, steps*dt , steps)
    print(np.shape(biomass_results))
    for i, depth in enumerate(plant_depths):
        # Plant height (biomass) curve
        plt.plot(time, biomass_results[i][0], c=colors[i], label=f'Plant canopy height (m) at altitude {depth} m ')
        # Distance between canopy and water surface curve (dashed)
        plt.plot(time, biomass_results[i][1], "--", c=colors[i], label=f'Distance between canopy and water surface (m)')

    # Plot water level with a distinct color and linestyle
    plt.hlines(-DT,xmin=0,xmax=steps*dt,linestyle="--",label="Drowning threshold")

    plt.plot(time, water_levels[:steps], linestyle='--', label='Water level', color='black')
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.title("Biomass Growth of Plants at Different Depths with Seasonal Water Level")
    plt.legend()
    plt.savefig("biomass_growth.png", dpi=300)

