import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('plots/Exercitiul_5.pdf')

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

p = 80
m = 100

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

# coeficientii sunt luati in ordinea c0,..,cn-1

def radacini_polinom(coeficienti):
    N = coeficienti.shape[0]

    matrice_companion = np.zeros((N,N))

    matrice_companion[:,N - 1] = -coeficienti[::-1]
    ones = np.ones(N - 1)
    matrice_companion += np.diag(ones, k = -1)

    eigvals, _ = np.linalg.eig(matrice_companion)
    
    return eigvals

# c0 este 1, tre sa il adaug
# coeficientii se potrivesc pentru L*t[y]+L**2*t[y]...
x_star = np.insert(x_star, x_star.shape[0], 1)
coefs = radacini_polinom(x_star)
coefs_norm = np.abs(coefs)

points = np.linspace(0, 2 * np.pi, 10000)

plt.title("Radacinile polinomului caracteristic modelului AR")
plt.plot(np.sin(points), np.cos(points), label = "Cercul unitate")
plt.plot(coefs.real, coefs.imag, '.', label = "Radacina a polinomului")
plt.xlabel("Partea Reala")
plt.ylabel("Partea Imaginara")
plt.legend()
plt.show()

if ([x < 1 for x in coefs_norm] is not None):
    print("Seria de timp nu este stationara!")
else:
    print("Seria de timp este stationara!")

pp.savefig()
plt.savefig("plots/Exercitiul_5.svg", format='svg')
plt.savefig("plots/Exercitiul_5.png", format='png')    
pp.close()