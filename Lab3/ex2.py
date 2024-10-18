import numpy as np
import matplotlib.pyplot as plt

fs = 10 # Hz
f = 12 # Hz
faza = 0

spatiu_semnal = np.linspace(0,1,fs * 1000)
fct_semnal = lambda t: np.sin(2*np.pi*t*f + faza)

semnal = fct_semnal(spatiu_semnal)

cerc_unitate = lambda n, omega: np.exp(-2 * np.pi * 1j * omega * n)

semnal_complex = np.ndarray(fs * 1000, dtype=np.complex128)
culoare = np.ndarray((3, fs * 1000))

functie_culoare = lambda dist: (dist * 0.35 + 0.4, dist * 0.65 + 0.2, dist * 0.45 + 0.25)
distanta_euclidiana = lambda comp: np.sqrt(comp.real**2 + comp.imag**2)

# afisez semnalul 

plt.figure()
plt.axhline(0, color = 'black')
plt.title(f"Semnalul sinusoidal cu f = {f}")
plt.plot(spatiu_semnal, semnal)
plt.savefig("plots/Exercitiul_2_semnal.svg", format='svg')

# TODO pentru un numar in intervalul [0,1] vreau un color code
# dupa care sa vad cum il mapez functiei


omega_list = [3, 5, 9, 12]

for omega in omega_list:
    for i in range(fs * 1000):
        semnal_complex[i] = semnal[i] * cerc_unitate(spatiu_semnal[i], omega)
        
    
        
    plt.figure()
    plt.title(f"Reprezentarea pe planul complex al semnalului cu f = {f} Hz, ω = {omega}")
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.xlabel("Real")
    plt.ylabel("Imaginar")
    plt.axhline(0, color = 'black')
    plt.axvline(0, color = 'black')
    plt.plot(semnal_complex.real, semnal_complex.imag, color = 'purple')
    plt.savefig(f"plots/Exercitiul_2_omega_{omega}.svg", format='svg')