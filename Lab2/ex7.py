import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile

frecventa = 100
nr_pct = 1000

spatiu_1 = np.linspace(0, 0.3, nr_pct)
spatiu_2 = np.linspace(0, 0.3, int(nr_pct / 4))
spatiu_3 = np.linspace(0, 0.3, int(nr_pct / 16))
semnal_1 = lambda t: np.sin(2*np.pi*frecventa*t)

fig, axs = plt.subplots(3)
fig.suptitle('Esantionare / frecventa')
axs[0].plot(spatiu_1, semnal_1(spatiu_1))
axs[1].plot(spatiu_2, semnal_1(spatiu_2))
axs[2].plot(spatiu_3[1:], semnal_1(spatiu_3[1:]))

plt.savefig("plots/Exercitiul_7_esantionare_mica.svg", format="svg")
plt.show()

# frecventa de esantionare este importanta in a reprezenta in mod fidel
# un semnal. se poate observa cum pe masura ce micsoram frecventa de 
# esantionare, semnalul devine mai "jagged". Asta se observa cel mai clar
# inspre varful unei oscilatii, unde daca frecventa de esantionare este mica,
# semnalul esantionat "taie" curba
# de asemenea, pentru o frecventa de esantionare mica, sarind si un singur 
# esantion este suficient sa faca semnalul nostru sa piarda informatie 

# observatiile precedente au fost facute pentru un semnal sinusoidal de frecventa
# 20 Hz

# daca crestem frecventa semnalului sinusoidal la 1000 Hz,
# reducerea ratei de esantionare este cu atat mai vizibila,
# practic pare ca insusi frecventa sinusoidei a fost redusa

# daca frecventa de esantionare este mai mare ca frecventa
# semnalului, atunci putem aproximativ reconstitui forma
# semnalului initial (cu un anume factor de jaggedness)

# cu cat mai mica frecventa de esantionare, cu atat mai
# "jagged" va arata semnalul

# daca rata de esantionare este un multiplu al frecventei, 
# atunci semnalul discret obtinut va avea aceeasi forma

# pentru o rata de esantionare care nu este multiplu, se va obtine
# un semnal discret diferit de cel continuu