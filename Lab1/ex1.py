import numpy as np
import matplotlib.pyplot as plt

# a
nr_reale = np.linspace(0, 0.03, 2000)

# b
x_t = lambda t: np.cos(520*np.pi*t + np.pi/3)
y_t = lambda t: np.cos(280*np.pi*t - np.pi/3)
z_t = lambda t: np.cos(120*np.pi*t + np.pi/3)

fig,axs = plt.subplots(3)
fig.suptitle('Semnale')
axs[0].plot(nr_reale, x_t(nr_reale))
axs[1].plot(nr_reale, y_t(nr_reale))
axs[2].plot(nr_reale, z_t(nr_reale))

for ax in axs:
    ax.grid()

plt.savefig("plots/Exercitiul_1b.svg", format="svg")

# c

# fs este 1 / T, fs este 200 => T = 1 / 200
T = 1 / 200

nr_esantionare = np.linspace(0, 0.03, 2000 // 200)

fig,axs = plt.subplots(3)
fig.suptitle('Semnale esantionate')
axs[0].plot(nr_reale, x_t(nr_reale))
axs[0].stem(nr_esantionare, x_t(nr_esantionare))
axs[1].plot(nr_reale, y_t(nr_reale))
axs[1].stem(nr_esantionare, y_t(nr_esantionare))
axs[2].plot(nr_reale, z_t(nr_reale))
axs[2].stem(nr_esantionare, z_t(nr_esantionare))

for ax in axs:
    ax.grid()

plt.savefig("plots/Exercitiul_1c.svg", format="svg")
plt.show()
