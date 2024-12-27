import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

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

L = 500

hankel_matrix = np.zeros((L, N - L + 1))

for j in range(N - L + 1):
    hankel_matrix[:,j] = spatiu_serie_timp[j:j + L]

# matricile S1 si S2 sunt simetrice
    
S1 = hankel_matrix @ hankel_matrix.T
S2 = hankel_matrix.T @ hankel_matrix

U, sigma, V_transpose = np.linalg.svd(hankel_matrix)

eigval1, eigvec1 = np.linalg.eig(S1)
eigval2, eigvec2 = np.linalg.eig(S2)

eigval1 = np.abs(eigval1)
eigval2 = np.abs(eigval2)

# din sigma scad vectorul de valori proprii de lungime mai mica

if (eigval1.shape > eigval2.shape):
    vector = eigval2
else:
    vector = eigval1

vector_sorted = np.sort(vector)[::-1]
matrice_aux = sigma - np.sqrt(vector_sorted)
rezultat_eigval = np.linalg.norm(matrice_aux)

print("Lambda = Sigma**2: " + str(rezultat_eigval <= 1e-6))

sort_indices = np.argsort(eigval1)[::-1]
eigvec1 = eigvec1[:,sort_indices]
eigvec1 = np.abs(eigvec1)
U = np.abs(U)
matrice_aux2 = U - eigvec1
rezultat_eigvec1 = np.linalg.norm(matrice_aux2)

print("U = eigvec(H @ H.t): " + str(rezultat_eigvec1 <= 1e-6))


sort_indices = np.argsort(eigval2)[::-1]
eigvec2 = eigvec2[:,sort_indices]
eigvec2 = np.abs(eigvec2)
V_transpose = np.abs(V_transpose)
V = V_transpose.T
matrice_aux3 = V - eigvec2
rezultat_eigvec2 = np.linalg.norm(matrice_aux3)

print("V = eigvec(H.t @ H): " + str(rezultat_eigvec2 <= 1e-6))