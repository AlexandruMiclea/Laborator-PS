import numpy as np
import matplotlib.pyplot as plt
import scipy

# d

N = 1000

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

# 1 indexed
spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10

spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
spatiu_serie_timp = np.add(spatiu_serie_timp, spatiu_semnal)

# reference: https://matplotlib.org/stable/gallery/widgets/slider_demo.html
# https://www.geeksforgeeks.org/matplotlib-slider-widget/

# slidere pentru setarea valorilor lui p si m

MSE_best = 1e9
MSE_best_m_p = (0,0)

for m in range(N - 1, 1, -1):
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

        predicted_value_space = np.linspace(0, N - 1, N)
        predicted_values = np.zeros(N)

        for i in range(m):
            idx = N - i - 1
            time_series_true = spatiu_serie_timp[idx - p:idx]
            
            predicted_values[idx] = x_star.T @ time_series_true[::-1]
            
        MSE = (np.sum((spatiu_serie_timp[N - m:] - predicted_values[N - m:])**2)) / m
        
        if (MSE <= MSE_best):
            MSE_best = MSE
            MSE_best_m_p = (m, p)
        
        print(f'Processed m = {m} p = {p}')
        print(f'best MSE = {MSE_best}')