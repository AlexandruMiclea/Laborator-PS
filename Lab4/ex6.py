import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

freq, semnal = scipy.io.wavfile.read('wavs/vocale.wav')

# semnal pe 2 canale
# voi lua doar semnalul de pe canalul 1

semnal = semnal[:,1]


lungime_canal = semnal.shape[0] // 100
deplasare = lungime_canal // 2


# 199 reprezinta cele 100 de canale (1%) plus cele 99 de canale care sunt 
# formate din 50% din fiecare canal

canale = np.ndarray((199, lungime_canal))

for i in range(199):
    canale[i] = semnal[deplasare * i : deplasare * i + lungime_canal]
    

fft_canale = np.ndarray((199, 4082), dtype = np.complex128) # 199 de canale, pentru fiecare fac fft
                                      # cu frecventa de 20 kHz 
    
for i in range(canale.shape[0]):
    fft_canale[i] = np.fft.fft(canale[i])
    
modul_complex = lambda x: np.sqrt(x.real**2 + x.imag**2)
fft_canale = modul_complex(fft_canale)
    
plt.figure()
plt.specgram(fft_canale)