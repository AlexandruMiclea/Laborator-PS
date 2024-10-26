import numpy as np
import matplotlib.pyplot as plt

esantioane_plot = 10000
esantioane_teoretic = 11 # 100 Hz -> 10 esantioane in 100 ms -> 100 esantioane in 1000 ms

# am luat 11 esantioane. in reprezentarea din numpy, al 11-lea esantion coincide
# cu primul esantion. asadar, frecventa "teoretica" este de 100 Hz
# astfel, pentru orice semnal cu frecventa = f + k*fs, unde fs este 100 Hz,
# esantioanele vor coincide

# mai jos am analizat frecventele pe un spatiu de 100 ms

spatiu_semnal = np.linspace(0,0.1,esantioane_plot)
spatiu_teoretic = np.linspace(0,0.1,esantioane_teoretic)
functie_semnal = lambda A, f, t, fz: A * np.sin(2 * np.pi * f * t + fz)

semnal_1 = lambda t: functie_semnal(1, 30, t, 0)
semnal_2 = lambda t: functie_semnal(1, 130, t, 0)
semnal_3 = lambda t: functie_semnal(1, 230, t, 0)

fig, axs = plt.subplots(4)
fig.suptitle('Esantionare cu frecventa sub-Nyquist')
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

plt.savefig("plots/Exercitiul_2.svg", format='svg')
plt.savefig("plots/Exercitiul_2.png", format='png')
plt.savefig("plots/Exercitiul_2.pdf", format='pdf')