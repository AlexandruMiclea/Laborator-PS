import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pickle
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('plots/Exercitiul_4.pdf')

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

# pentru ca dureaza mult sa calculez, am salvat pickles pentru L din {400, 500. 600}
L = 500

if os.path.isfile(f'pickles/SSA_{L}.pickle'):
    with open(f'pickles/SSA_{L}.pickle', 'rb') as pickle_file:
        Xi_mean = pickle.load(pickle_file)
else:
    hankel_matrix = np.zeros((L, N - L + 1))
    
    for j in range(N - L + 1):
        hankel_matrix[:,j] = spatiu_serie_timp[j:j + L]
    
    
    U, sigma, V_transpus = np.linalg.svd(hankel_matrix, full_matrices=False)
    
    Xi = np.zeros((L, L, N - L + 1))
    Xi_mean = np.zeros((L, L, N - L + 1))
    
    for i in range(min(L, N - L + 1)):
        Xi[i] = np.outer((sigma[i] * U[:,i]), V_transpus[i,:])
    
    min_dim = min(L, N - L + 1)
    max_dim = max(L, N - L + 1)
    for mat_idx in range(L):
        Xi_flip = np.fliplr(Xi[mat_idx,:,:])
        Xi_flip_mean = np.zeros(Xi_flip.shape)
        
        min_diag_dim = -Xi_flip.shape[0] + 1
        max_diag_dim = Xi_flip.shape[1] - 1
        for dim in range(min_diag_dim, max_diag_dim + 1):
            # fiecare dim reprezinta antidiagonala cu indicele dim
            antidiag = np.diag(Xi_flip, dim)
            
            # antidiag_count trebuie sa fie numarul de elemente in diagonala de pe
            # ordinul dim, raportat la o matrice de dimensiune max_dim*max_dim
            antidiag_count = max_dim - np.abs(dim)
            antidiag_mean = np.mean(antidiag)
            new_antidiag = np.ones(antidiag_count) * antidiag_mean
            new_antidiag_matrix = np.diag(new_antidiag, dim)
            assert(new_antidiag_matrix.shape == (max_dim, max_dim))
            Xi_flip_mean += new_antidiag_matrix[:Xi_flip.shape[0], :Xi_flip.shape[1]]
        Xi_mean[mat_idx] = np.fliplr(Xi_flip_mean).copy()
    
    with open(f'pickles/SSA_{L}.pickle', 'wb') as pickle_file:
        pickle.dump(Xi_mean[:10], pickle_file)
    
# un parametru prin care spun cate componente afisez in plot
# (pentru pickles P <= 10)
P = 3

plt.figure()
plt.title(f"Serie timp descompusa cu SSA, L = {L}")
plt.plot(spatiu_serie_timp, label = "Serie timp originala")

for i in range(P):
    plt.plot(Xi_mean[i,:,0], label = f"Componenta de ordin {i + 1}")

plt.legend()
plt.show()

pp.savefig()
plt.savefig(f"plots/Exercitiul_4_{L}.svg", format='svg')
plt.savefig(f"plots/Exercitiul_4_{L}.png", format='png')

pp.close()