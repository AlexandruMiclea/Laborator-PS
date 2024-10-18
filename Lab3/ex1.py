import numpy as np
import matplotlib.pyplot as plt

matrice_fourier_8 = np.ndarray((8,8), dtype=np.complex64)

for x in range(8):
    for y in range(8):
        matrice_fourier_8[x][y] = np.exp(-2*np.pi*1j*x*y/8)
        
# print (matrice_fourier_8)

fig,axs = plt.subplots(4,2)
fig.suptitle("Coordonatele reale si imaginare ale matricei Fourier N = 8\nAlbastru: real, Rosu: imaginar")

for line in range(8):
    axs[line // 2][line % 2].set_title(f"Linia {line + 1}")
    axs[line // 2][line % 2].plot(np.real(matrice_fourier_8[line]), label = 'real')
    axs[line // 2][line % 2].plot(np.imag(matrice_fourier_8[line]), label = 'imaginar', color='red')
    #axs[line // 2][line % 2].legend()
    
plt.savefig('plots/Exercitiul_1.svg', format = "svg")

# impart matricea fourier la sqrt(8) pentru a aduce elementele de pe diagonala principala la 1
matrice_fourier_8 = matrice_fourier_8 / np.sqrt(8)

matrice_H = np.transpose(np.conjugate(matrice_fourier_8))
matrice_produs = np.matmul(matrice_fourier_8,matrice_H)
matrice_identitate = np.identity(8)

matrice_aux = matrice_produs - matrice_identitate

rezultat = np.linalg.norm(matrice_aux)

# eroarea in cazul meu este pana in a 7-a zecimala, a fost necesar sa modific atol
print(np.allclose(rezultat, 0, atol = 1e-7))