import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

# sa cantam la contrabas

esantioane_plot = 10000

spatiu_semnal = np.linspace(0,0.1,esantioane_plot)
functie_semnal = lambda A, f, t, fz: A * np.sin(2 * np.pi * f * t + fz)

fsemnal_1 = lambda t: functie_semnal(1, 40, t, 0)
fsemnal_2 = lambda t: functie_semnal(1, 80, t, 0)
fsemnal_3 = lambda t: functie_semnal(1, 160, t, 0)
fsemnal_4 = lambda t: functie_semnal(1, 200, t, 0)

semnal_1 = fsemnal_2(spatiu_semnal) + fsemnal_3(spatiu_semnal)
semnal_2 = fsemnal_1(spatiu_semnal) + fsemnal_4(spatiu_semnal)

semnal_final = np.concatenate((
        np.concatenate(([semnal_1 for i in range(3)])),
        np.concatenate(([semnal_1 + semnal_2 for i in range(3)])),
        np.concatenate(([semnal_2 for i in range(3)]))
    ))

scipy.io.wavfile.write('wavs/contrabas_4.wav', int(10e4), semnal_final)

spatiu_semnal_final = spatiu_semnal = np.linspace(0,0.3,3 * esantioane_plot)
semnal_final = np.concat((semnal_1, semnal_1 + semnal_2, semnal_2))

esantioane_simulare = 42 * 3 # 410 Hz
spatiu_esantioane = np.linspace(0,0.3,esantioane_simulare)

# cum frecventele emise de contrabas se afla in spectrul [40Hz, 200Hz], conform
# teoremei Nyquist-Shannon, am avea nevoie sa esantionam semnalul cu o frecventa
# > 2 * 200Hz, sau altfel spus, > 400 Hz

plt.plot(spatiu_semnal_final, semnal_final)

def f_esantionare(t):
    if (t < 0.1):
        return fsemnal_2(t) + fsemnal_3(t)
    elif (t < 0.2):
        return fsemnal_1(t) + fsemnal_2(t) + fsemnal_3(t) + fsemnal_4(t)
    else:
        return fsemnal_1(t) + fsemnal_4(t)
    
esantioane_y = np.zeros(esantioane_simulare) 

for i in range(esantioane_simulare):
    esantioane_y[i] = f_esantionare(spatiu_esantioane[i])

plt.scatter(spatiu_esantioane, esantioane_y, color = 'orange')
plt.plot(spatiu_esantioane, esantioane_y, linestyle= 'dotted', color = 'black')

plt.savefig("plots/Exercitiul_4.svg", format='svg')
plt.savefig("plots/Exercitiul_4.png", format='png')
plt.savefig("plots/Exercitiul_4.pdf", format='pdf')