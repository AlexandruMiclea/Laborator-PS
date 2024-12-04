import numpy as np
import matplotlib.pyplot as plt

frecventa = 10
nr_pct = 500000

spatiu = np.linspace(0, 1, nr_pct)
semnal_sin = lambda t: np.sin(2*np.pi*frecventa*t)

# TODO for self does sawtooth rely on sin?
semnal_st = lambda t: np.mod(2*np.pi*frecventa/4*t, 2) - 1
semnal_mixt = lambda t: semnal_sin(t) + semnal_st(t)

fig,axs = plt.subplots(3)
axs[0].set_title('Semnal Sinusoidal')
axs[0].plot(spatiu, semnal_sin(spatiu))
axs[1].set_title('Semnal Sawtooth')
axs[1].plot(spatiu, semnal_st(spatiu))
axs[2].set_title('Semnal Mixt')
axs[2].plot(spatiu, semnal_mixt(spatiu))

plt.savefig("plots/Exercitiul_4_semnale_diferite.svg", format="svg")
plt.show()