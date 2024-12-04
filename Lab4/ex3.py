import numpy as np
import matplotlib.pyplot as plt

esantioane_plot = 10000
esantioane_teoretic = 101 # 1 kHz

spatiu_semnal = np.linspace(0,0.1,esantioane_plot)
spatiu_teoretic = np.linspace(0,0.1,esantioane_teoretic)
functie_semnal = lambda A, f, t, fz: A * np.sin(2 * np.pi * f * t + fz)

semnal_1 = lambda t: functie_semnal(1, 30, t, 0)
semnal_2 = lambda t: functie_semnal(1, 130, t, 0)
semnal_3 = lambda t: functie_semnal(1, 230, t, 0)

fig, axs = plt.subplots(4)
fig.suptitle('Esantionare cu frecventa supra-Nyquist')
axs[0].plot(spatiu_semnal, semnal_1(spatiu_semnal))
axs[1].plot(spatiu_semnal, semnal_1(spatiu_semnal))
axs[1].scatter(spatiu_teoretic, semnal_1(spatiu_teoretic), color = 'yellow')
axs[1].plot(spatiu_teoretic, semnal_1(spatiu_teoretic), linestyle = 'dotted', color = 'black')
axs[2].plot(spatiu_semnal, semnal_2(spatiu_semnal), color = 'purple')
axs[2].scatter(spatiu_teoretic, semnal_2(spatiu_teoretic), color = 'yellow')
axs[2].plot(spatiu_teoretic, semnal_2(spatiu_teoretic), linestyle = 'dotted', color = 'black')
axs[3].plot(spatiu_semnal, semnal_3(spatiu_semnal), color = 'green')
axs[3].scatter(spatiu_teoretic, semnal_3(spatiu_teoretic), color = 'yellow')
axs[3].plot(spatiu_teoretic, semnal_3(spatiu_teoretic), linestyle = 'dotted', color = 'black')

plt.savefig("plots/Exercitiul_3.svg", format='svg')
plt.savefig("plots/Exercitiul_3.png", format='png')
plt.savefig("plots/Exercitiul_3.pdf", format='pdf')