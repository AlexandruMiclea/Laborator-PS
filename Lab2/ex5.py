import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile

frecventa_1 = 432
frecventa_2 = 440
nr_pct = 500000

spatiu = np.linspace(0, 1, nr_pct)
semnal_1 = lambda t: np.sin(2*np.pi*frecventa_1*t)
semnal_2 = lambda t: np.sin(2*np.pi*frecventa_2*t)

# TODO for self does sawtooth rely on sin?

semnal_final = np.concat((semnal_1(spatiu), semnal_2(spatiu)))

rate = int(10e4)
scipy.io.wavfile.write('audio/semnal5.wav', rate, semnal_final)

# in mijlocul fisierului, sunetul devine mai "inalt"
# 432 Hz suna mai calm, mai dulce, 440 Hz suna mai alert