# pipeline deducere best p for a given m

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle

# d

N = 1000

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)


spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10

spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
spatiu_serie_timp = np.add(spatiu_serie_timp, spatiu_semnal)

MSE_best_p = np.zeros(N)

for m in range(2, N):
    MSE_best = 1e9    
    for p in range(2,N):
        if (m + p >= N):
            continue
        if (p > m):
            continue
        
        y = spatiu_serie_timp[N - m:][::-1]
        N_y = y.shape[0]
        Y = np.zeros((m,p))
 
        for i in range(m):
            Y[i] = spatiu_serie_timp[N - p - 1 - i:N - 1 - i][::-1]
 
        x_star = np.linalg.inv(Y.T @ Y) @ Y.T @ y

        # generez predictiile pentru spatiul timp

        predicted_value_space = np.linspace(N - m, N - 1, m)
        # primele p valori sunt ground_truth, restul vor fi prezise
        predicted_values = spatiu_serie_timp[N - m - p:N - m][::-1]

        for i in range(m):
            predicted_N = predicted_values.shape[0]
            predicted_value = x_star.T @ predicted_values[:p]
            predicted_values = np.concatenate((np.array([predicted_value]), predicted_values))
            
        MSE = (np.sum((spatiu_serie_timp[N - m:] - predicted_values[:predicted_N - p + 1][::-1])**2)) / m
        
        if (MSE <= MSE_best):
            MSE_best = MSE
            MSE_best_p[m] = p
        
        print(f'Processed m = {m} p = {p}')
        print(f'best MSE = {MSE_best}')