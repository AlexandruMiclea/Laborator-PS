import numpy as np
import matplotlib.pyplot as plt

fs = 10 # Hz
N = fs * 1000
f1 = 12 # Hz
f2 = 43 # Hz
f3 = 31 # Hz

spatiu_semnal = np.linspace(0,1,N)
fct_semnal = lambda t: 0.5 * np.sin(2*np.pi*t*f1) + 3 * np.sin(2*np.pi*t*f2) + 2.4 * np.sin(2*np.pi*t*f3)

semnal = fct_semnal(spatiu_semnal)

plt.figure()
plt.plot(spatiu_semnal, semnal)
plt.axhline(0, color = 'black')
plt.title(f"Semnalul sinusoidal de frecvente {f1}, {f2}, {f3} Hz")
plt.savefig("plots/Exercitiul_3_semnal.svg", format='svg')

cerc_unitate = lambda n, omega: np.exp(-2 * np.pi * 1j * omega * n / N)

frecvente_analizate = np.arange(0,101,1)
transformata_fourier = np.ndarray(101, dtype = np.complex128)

modul_complex = lambda numar: np.sqrt(numar.real**2 + numar.imag**2)

for omega in frecvente_analizate:
    transformata_fourier[omega] = np.sum([semnal[i] * cerc_unitate(i, omega) for i in range(fs*1000)])
    
plt.figure()
plt.xlabel("Frecventa (Hz)")
plt.ylabel("|X(ω)|")
plt.title(f"Transformata Fourier")
plt.stem(frecvente_analizate, modul_complex(transformata_fourier))
plt.savefig("plots/Exercitiul_3_transformata.svg", format='svg')