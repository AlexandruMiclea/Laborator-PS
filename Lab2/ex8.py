import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile

nr_pct = 500000

spatiu = np.linspace(-np.pi/2, np.pi/2, nr_pct)
#spatiu = np.linspace(0, np.pi/2, nr_pct)
semnal_sin = lambda t: np.sin(t)
semnal_t = lambda t: t
semnal_pade = lambda t: (t - ((7 * t**3) / 60)) / (1 + (t**2 / 20))

fig,(ax1, ax2) = plt.subplots(2,2)
ax1[0].set_title('Grafic standard')
ax1[0].plot(spatiu, semnal_sin(spatiu), label = 'sin(t)')
ax1[0].plot(spatiu, semnal_t(spatiu), linestyle='dashed', label = 't')
ax1[0].plot(spatiu, semnal_pade(spatiu), linestyle = 'dashdot', label = 'pade(t)')
ax1[0].legend()

ax1[1].set_title('Grafic de eroare')
ax1[1].plot(spatiu, np.abs(semnal_sin(spatiu) - semnal_t(spatiu)), linestyle = 'dashed', label = 'sin(t) / t')
ax1[1].plot(spatiu, np.abs(semnal_sin(spatiu) - semnal_pade(spatiu)), linestyle = 'dashdot', label = 'sin(t) / pade(t)')
ax1[1].plot(spatiu, np.abs(semnal_pade(spatiu) - semnal_t(spatiu)), label = 't / pade(t)')
ax1[1].legend()

ax2[0].set_title('Grafic logaritmic')
ax2[0].plot(spatiu, semnal_sin(spatiu))
ax2[0].plot(spatiu, semnal_t(spatiu), linestyle='dashed')
ax2[0].plot(spatiu, semnal_pade(spatiu), linestyle = 'dashdot')
ax2[0].set_yscale("log")

# primesc erori pentru ca log primeste valori < 0,
# on my machine it works

ax2[1].set_title('Grafic de eroare')
ax2[1].plot(spatiu, np.abs(np.log(semnal_sin(spatiu)) - np.log(semnal_t(spatiu))), linestyle = 'dashed', label = 'sin(t) / t')
ax2[1].plot(spatiu, np.abs(np.log(semnal_sin(spatiu)) - np.log(semnal_pade(spatiu))), linestyle = 'dashdot', label = 'sin(t) / pade(t)')
ax2[1].plot(spatiu, np.abs(np.log(semnal_pade(spatiu)) - np.log(semnal_t(spatiu))), label = 't / pade(t)')
ax2[1].legend()

plt.savefig("plots/Exercitiul_8_aproximare_Pade.svg", format="svg")

plt.show()

