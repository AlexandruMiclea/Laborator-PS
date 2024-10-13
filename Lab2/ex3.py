import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plt_img
import sounddevice
import scipy.signal
import scipy.io.wavfile
import time

rate = int(10e4)
fs = 44100

# a

frecventa = 400
nr_pct = 500000

spatiu_a = np.linspace(0, 4, nr_pct)   
semnal_a = lambda t: np.sin(2*np.pi*frecventa*t)

scipy.io.wavfile.write('audio/semnal3_1.wav', rate, semnal_a(spatiu_a))

# TODO for self de ce nu merge sounddevice
#fs = 48000
#sounddevice.play(semnal_a(spatiu_a), fs)
#sounddevice.wait()
#time.sleep(1)

x,y = scipy.io.wavfile.read('audio/semnal3_1.wav')
plt.plot(spatiu_a, y)
plt.savefig("plots/Exercitiul_3_semnal_din_wav.svg", format="svg")

plt.show()

# # b

frecventa = 800
durata = 3
nr_esant = 400

spatiu_b = np.linspace(0, durata, 500000)
semnal_b = lambda t: np.sin(2*np.pi*frecventa*t)

scipy.io.wavfile.write('audio/semnal3_2.wav', rate, semnal_b(spatiu_b))

# # c

frecventa = 240

spatiu_c = np.linspace(0, 20/frecventa, 500000)
semnal_c = lambda t: np.mod(frecventa*t, 1)

scipy.io.wavfile.write('audio/semnal3_3.wav', rate, semnal_c(spatiu_c))

# # d

frecventa = 300
# # ca sa nu imi dea np.sign() = 0 (in cazul in care am 0)
eps = 10**-6

spatiu_d = np.linspace(0 + eps, 30/frecventa - eps, 500000)
semnal_d = lambda t: np.sign(np.sin(2*np.pi*frecventa*t))

scipy.io.wavfile.write('audio/semnal3_4.wav', rate, semnal_d(spatiu_d))