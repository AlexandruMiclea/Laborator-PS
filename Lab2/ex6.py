import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile

frecventa_esantionare = 800

spatiu = np.linspace(0, 1, frecventa_esantionare)
semnal_0 = lambda t: np.sin(2*np.pi*frecventa_esantionare*2*t)
semnal_1 = lambda t: np.sin(2*np.pi*frecventa_esantionare*t)
semnal_12 = lambda t: np.sin(2*np.pi*frecventa_esantionare/1.5*t)
semnal_2 = lambda t: np.sin(2*np.pi*frecventa_esantionare/2*t)
semnal_3 = lambda t: np.sin(2*np.pi*frecventa_esantionare/4*t)
semnal_4 = lambda t: np.sin(2*np.pi*0*t)

fig, axs = plt.subplots(6)
fig.suptitle("Esantionare / frecventa")
axs[0].plot(spatiu, semnal_1(spatiu))
axs[5].plot(spatiu, semnal_12(spatiu))
axs[1].plot(spatiu, semnal_2(spatiu))
axs[2].plot(spatiu, semnal_3(spatiu))
axs[3].plot(spatiu, semnal_4(spatiu))
axs[4].plot(spatiu, semnal_0(spatiu))

plt.savefig("plots/Exercitiul_6_esantionare_mare.svg", format="svg")

plt.show()

# pentru rata de esantionare dubla fata de frecventa semnalului,
# putem deduce care este forma semnalului original prin
# preluarea valorilor pozitive
# (pentru fiecare miscare in semnalul discret dublu esantionat,
# luam doar a doua valoare esantionata)

# pentru orice coeficient > 2 insa, pierdem forma originala a semnalului
