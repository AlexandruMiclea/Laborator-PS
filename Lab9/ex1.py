import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

N = 1000
pp = PdfPages('plots/Exercitiul_1.pdf')

fig, axs = plt.subplots(4)
plt.title('Serie timp')

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

# 1 indexed
spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10

axs[0].plot(spatiu_serie_timp)
axs[0].set_title('Ecuatie grad 2')
spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
axs[1].plot(noise)
axs[1].set_title('Zgomot')
spatiu_serie_timp += np.add(spatiu_serie_timp, spatiu_semnal)
axs[2].plot(spatiu_semnal)
axs[2].set_title('Semnal')

plt.xlabel("Ordin esantion")
plt.ylabel("Valoare")
plt.plot(spatiu_serie_timp)
plt.show()

pp.savefig()
plt.savefig("plots/Exercitiul_1.svg", format='svg')
plt.savefig("plots/Exercitiul_1.png", format='png')

pp.close()
