# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 18:43:44 2025

@author: yanbw
"""

import matplotlib.pyplot as plt

import numpy as np

    

def generate_wave(A_base=7*5, nivel_medio=22.70, duracao=60):
    """
    Gera uma única onda senoidal.
    
    Parâmetros:
    A_base (float): Amplitude da onda.
    nivel_medio (float): Nível médio da onda.
    duracao (int): Duração da onda em unidades de tempo.
    
    Retorna:
    numpy.ndarray: Onda gerada.
    """
    t = np.arange(0, duracao + 1,step=0.01)
    onda = nivel_medio + A_base * np.sin(2 * np.pi * (1 / duracao) * t)
    onda = nivel_medio + A_base * np.sin(2 * np.pi * (1 / duracao) * t)-20
    ##nivel de transborde
    onda = np.where(onda < 0, 0, onda)
    onda = np.where(onda > 20, 20, onda)
    
    
    return onda


for A in [30,40,50]:
# Exemplo de uso:
    onda = generate_wave(A_base=A)
    
    plt.plot(onda)
plt.show()