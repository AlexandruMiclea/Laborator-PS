import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

rata_esantionare, semnal = scipy.io.wavfile.read('wavs/vocale.wav')

# semnal pe 2 canale
# voi lua doar semnalul de pe canalul 1
semnal = semnal[:,1]

# lungimea unei grupe denota valorile semnalului pe o lungime determinata de
# formula: N / (44100 * 100) (unde N este de fapt lungime_inregistrare (s) * 44100)

lungime_grupa = semnal.shape[0] // 100
deplasare = lungime_grupa // 2

# 199 reprezinta cele 100 de canale (1%) plus cele 99 de canale care sunt
# formate din 50% din fiecare canal
grupe = np.ndarray((199, lungime_grupa))

for i in range(199):
    grupe[i] = semnal[deplasare * i : deplasare * i + lungime_grupa]

 # 199 de canale, pentru fiecare fac fft pentru a afla ce frecvente compun acel semnal
fft_grupe = np.ndarray((grupe.shape[0], grupe.shape[1]), dtype = np.complex128)

for i in range(grupe.shape[0]):
    fft_grupe[i] = np.fft.fft(grupe[i])
    
# fft imi intoarce un vector de dimensiune identica, care arata prezenta unei frecvente
# de la 0Hz la maxim posibil, apoi de la minim posibil la -1
# pentru a afla ce frecventa reprezinta un index al vectorului,
# ma pot folosi de formula (idx * sampling_rate / N)

# de retinut faptul ca semnalul se presupune a fi centrat in 0 Hz,
# deci pentru un sampling rate de 44.1 kHz, frecventa maxima pe care o putem 
# detecta corect va undeva fi in preajma lui 22 kHz.

modul_complex = lambda x: np.sqrt(x.real**2 + x.imag**2)
fft_grupe = modul_complex(fft_grupe).T

# configurez o imagine care sa imi arate frecventele pentru o jumatate de grupa
# (pe principiul ca in a doua jumatate se vor afla frecventele negative)

# pe axa X o sa vreau timpul in milisecunde, pe axa Y valoarea frecventei in Hz

plt.figure()
plt.imshow(fft_grupe[:fft_grupe.shape[0] // 2:,:],
           cmap = 'magma',
           aspect = 'auto',
           norm = 'log',
           origin = 'lower',
           extent = [0, fft_grupe.shape[0] * fft_grupe.shape[1] * 100 / (rata_esantionare / 1000 * 199),
                     0, (rata_esantionare / 2)])

plt.colorbar(label='Prevalenta unei frecvente')
plt.xlabel('Momentul timp (ms)')
plt.ylabel('Frecventa prezenta (Hz)')

plt.savefig("plots/Exercitiul_6.svg", format='svg')
plt.savefig("plots/Exercitiul_6.png", format='png')
plt.savefig("plots/Exercitiul_6.pdf", format='pdf')