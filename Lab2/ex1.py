import numpy as np
import matplotlib.pyplot as plt

nr_reale = np.linspace(0, 1, 50000)
A = 2
functie_cos = lambda t: A*np.cos(2*np.pi*2*t)
phi = np.pi/2
functie_sin = lambda t: A*np.sin(2*np.pi*2*t + phi)

plt.plot(nr_reale, functie_cos(nr_reale))
plt.plot(nr_reale, functie_sin(nr_reale), linestyle='dashed')

plt.savefig("plots/Exercitiul_1.svg", format="svg")
plt.show()