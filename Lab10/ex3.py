import numpy as np
import matplotlib.pyplot as plt
import scipy
from cvxopt import matrix
from l1regls import l1regls
from matplotlib.backends.backend_pdf import PdfPages

N = 1000
pp = PdfPages('plots/Exercitiul_3.pdf')

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10
spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
spatiu_serie_timp = np.add(spatiu_serie_timp, spatiu_semnal)

# d

# am generat best values folosind aux_brute_best_params
# rezultatele pot varia in functie de zgomot!

p = 150
m = 100
s = 80

y = spatiu_serie_timp[N - m:][::-1]
N_y = y.shape[0]
Y = np.zeros((m,p))

for i in range(m):
    Y[i] = spatiu_serie_timp[N - p - 1 - i:N - 1 - i][::-1]


# L1
Y_cvx = matrix(Y)
y_cvx = matrix(y)
x_cvx = l1regls(Y_cvx, y_cvx)

x_star_l1 = np.array(x_cvx)[:,0]

# greedy

# dintr-un motiv sau altul, adaugarea unui regresor nu imi garanteaza un loss
# mai mic decat cel gasit pana acum (este probabil o eroare din modul in care
# calculez x_star_dense)
x_star_dense = np.linalg.inv(Y.T @ Y) @ Y.T @ y
x_star_last = np.zeros((p))
min_loss = 1e9

# extragem din x_star s solutii, pe restul le setam la zero
for param in range(s):
    #if (param > 15):
     #   break
    x_star_current = x_star_last.copy()
    #print(np.linalg.norm(x_star_current, ord = 0))
        
    min_idx = -1
    for i in range(p):
        if (x_star_current[i] != 0):
            continue
        x_star_current[i] = x_star_dense[i]
        
        # verific pierderea pentru acest index
        loss = np.linalg.norm((Y @ x_star_current) - y, ord = 2)**2
        if (loss <= min_loss):
            min_loss = loss
            idx = i
        x_star_current[i] = 0
            
    print(f"loss pentru param {param + 1}: {min_loss}")
    #print(idx)
    x_star_last[idx] = x_star_dense[idx]
    
x_star_greedy = x_star_last

# generez predictiile pentru spatiul timp

predicted_value_space = np.linspace(N - m, N - 1, m)
# primele p valori sunt ground_truth, restul vor fi prezise
predicted_values_greedy = spatiu_serie_timp[N - m - p:N - m][::-1]
predicted_values_l1 = spatiu_serie_timp[N - m - p:N - m][::-1]

for i in range(m):
    predicted_N = predicted_values_greedy.shape[0]
    predicted_value_greedy = x_star_greedy.T @ predicted_values_greedy[:p]
    predicted_values_greedy = np.concatenate((np.array([predicted_value_greedy]), predicted_values_greedy))
    predicted_value_l1 = x_star_l1.T @ predicted_values_l1[:p]
    predicted_values_l1 = np.concatenate((np.array([predicted_value_l1]), predicted_values_l1))

fig, ax = plt.subplots()
ax.plot(spatiu_serie_timp, label = "Serie timp originala")
ax.plot(predicted_value_space, predicted_values_greedy[:predicted_N - p + 1][::-1], label='Valori prezise cu greedy')
ax.plot(predicted_value_space, predicted_values_l1[:predicted_N - p + 1][::-1], label='Valori prezise cu regularizare L1')

ax.set_title("Predictii cu metoda greedy si regularizare L1")
fig.legend()
fig.show()

pp.savefig()
plt.savefig("plots/Exercitiul_3.svg", format='svg')
plt.savefig("plots/Exercitiul_3.png", format='png')

pp.close()