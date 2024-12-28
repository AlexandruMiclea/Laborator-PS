import numpy as np
import matplotlib.pyplot as plt
import scipy
from cvxopt import matrix
from matplotlib.backends.backend_pdf import PdfPages

# coeficientii sunt luati in ordinea c0,..,cn-1
coeficienti_polinom = [1, 2, -3]


def radacini_polinom(coeficienti: list):
    coeficienti_np = np.array(coeficienti)
    N = coeficienti_np.shape[0]

    matrice_companion = np.zeros((N,N))

    matrice_companion[:,N - 1] = -coeficienti_np[::-1]
    ones = np.ones(N - 1)
    matrice_companion += np.diag(ones, k = -1)

    eigvals, _ = np.linalg.eig(matrice_companion)
    
    return eigvals

for idx, i in enumerate(radacini_polinom(coeficienti_polinom)):
    print(f"Radacina {idx}: {i}")