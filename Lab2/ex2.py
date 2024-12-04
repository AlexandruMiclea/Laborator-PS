import numpy as np
import matplotlib.pyplot as plt

nr_reale = np.linspace(0, 1, 5000)

A = 1
f0 = 50
semnal = lambda t, phi: A*np.sin(2*np.pi*f0*t + phi)

z = np.random.normal(size=5000)

calc_gamma = lambda SNR: np.sqrt((np.linalg.norm(semnal(nr_reale, 0), ord=2)**2)/(SNR*np.linalg.norm(z, ord=2)**2))

gamma1 = calc_gamma(0.1)
gamma2 = calc_gamma(1)
gamma3 = calc_gamma(10)
gamma4 = calc_gamma(100)

semnal_zgomot_1 = lambda t: semnal(t, 0) + gamma1 * z
semnal_zgomot_2 = lambda t: semnal(t, 0) + gamma2 * z
semnal_zgomot_3 = lambda t: semnal(t, 0) + gamma3 * z
semnal_zgomot_4 = lambda t: semnal(t, 0) + gamma4 * z

plt.plot(nr_reale, semnal(nr_reale, 0))
plt.plot(nr_reale, semnal(nr_reale, 1))
plt.plot(nr_reale, semnal(nr_reale, 5))
plt.plot(nr_reale, semnal(nr_reale, 8))

plt.savefig("plots/Exercitiul_2_faze.svg", format="svg")

plt.show()

fig,axs = plt.subplots(4)
fig.suptitle('Semnale cu SNR diferit')
axs[0].set_title('SNR 100')
axs[0].plot(nr_reale, semnal_zgomot_4(nr_reale))
axs[1].set_title('SNR 10')
axs[1].plot(nr_reale, semnal_zgomot_3(nr_reale))
axs[2].set_title('SNR 1')
axs[2].plot(nr_reale, semnal_zgomot_2(nr_reale))
axs[3].set_title('SNR 0.1')
axs[3].plot(nr_reale, semnal_zgomot_1(nr_reale))

plt.savefig("plots/Exercitiul_2_SNR.svg", format="svg")

plt.show()