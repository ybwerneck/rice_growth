# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 00:18:44 2025

@author: yanbw
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the function parameters
M = 1  # Maximum value
k = 5  # Decay rate (adjustable for smooth transition)

# Define the function
def gen_exp_dec(a):
    def custom_function(x):
        if x > 1:
            return 0
        elif -5 < x <= 1:
            return M * np.exp(-k * x/5)/1000
        else:
            return 0
    return custom_function
# Generate x values
x_values = np.linspace(-20, 1, 500)
y_values = [gen_exp_dec(1)(x) for x in x_values]

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=r'Growth rate', color='blue')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(-0.4, color='red', linestyle='--', label=r'Threshold of submersion survival (0.4m)')
plt.axvline(0, color='green', linestyle='--', label=r'Water Level')

plt.xlabel("Distance between plant canopy and the water level (m)")
plt.ylabel(r"Growth rate (m.day^-1)")
plt.title("Custom Response Function")
plt.legend()
plt.show()
###old code
if(True):
    # Plot varying k
    plt.figure(figsize=(12, 6))
    for k in k_values:
        y_values = [exp_drowning_response(x, D_T, k) for x in x_values]
        plt.plot(x_values, y_values, "--",label=f"k = {k}",color="red")
    plt.title("Exponential Drowning Response with Varying k (D_T=5)")
    plt.xlabel("Submersion Level (x)")
    plt.ylabel("Response")
    plt.legend()
    plt.grid(True)

    # Plot varying D_T
    for D_T in D_T_values:
        y_values = [exp_drowning_response(x, D_T, 3) for x in x_values]
        plt.plot(x_values, y_values,"--",label=f"D_T = {D_T}",color="gray")
    plt.title("Exponential Drowning Response with Varying D_T (k=3)")
    plt.xlabel("Submersion Level (x)")
    plt.ylabel("Response")
    plt.legend()
    plt.show()
    # Plot the function
    plt.plot(x_values, y_values, label=r'Growth rate', color='blue')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    #plt.axvline(-0.4, color='red', linestyle='--', label=r'Threshold of submersion survival (0.4m)')
    plt.axvline(0, color='green', linestyle='--', label=r'Water Level')

    plt.xlabel("Distance between plant canopy and the water level (m)")
    plt.ylabel(r"Growth rate coeficient")
    plt.title("Custom Response Function")
    plt.legend()
    plt.savefig("custom_response_function.png", dpi=300)
