import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import pickle
import os

esantioane = 10000

f1 = 200
f2 = 680
f3 = 2305

spatiu_semnal = np.linspace(0,1,esantioane)

functie_semnal = lambda A, f, t, fz: A * np.sin(2 * np.pi * f * t + fz)

semnal_smecher = lambda t: (functie_semnal(2, f1, t, 0)
                                + functie_semnal(0.2, f2, t, np.pi/4)
                                + functie_semnal(1, f3, t, np.pi/8))
                                
plt.figure()
plt.title('Semnal analizat')
plt.plot(spatiu_semnal, semnal_smecher(spatiu_semnal))

plt.savefig("plots/Exercitiul_1_semnal.svg", format='svg')
plt.savefig("plots/Exercitiul_1_semnal.png", format='png')
plt.savefig("plots/Exercitiul_1_semnal.pdf", format='pdf')

# liste folosite pentru plotarea diferentelor de timp
timp_standard = np.zeros(7)
timp_numpy = np.zeros(7)

lista_dimensiuni_vector = [128, 256, 512, 1024, 2048, 4096, 8192]

for (idx,dimensiune) in enumerate(lista_dimensiuni_vector):
    cerc_unitate = lambda n, omega: np.exp(-2 * np.pi * 1j * omega * n / dimensiune)    
    # implementare standard

    frecvente_analizate = np.arange(0,dimensiune,1)
    transformata_fourier = np.ndarray(dimensiune, dtype = np.complex128)
    
    spatiu_semnal = np.linspace(0,1,dimensiune)
    semnal = semnal_smecher(spatiu_semnal)
    
    # start dft implementation
    if os.path.isfile(f'pickles/dft_{dimensiune}.pickle'):
        with open(f'pickles/dft_{dimensiune}.pickle', 'rb') as pickle_file:
            transformata_fourier = pickle.load(pickle_file)
    else:
        start_time = perf_counter()
        
        for omega in range(dimensiune):
            transformata_fourier[omega] = np.sum([semnal[i] * cerc_unitate(i, omega) for i in range(dimensiune)])
            
        end_time = perf_counter()
        
        with open(f'pickles/dft_{dimensiune}.pickle', 'wb') as pickle_file:
            pickle.dump(transformata_fourier, pickle_file)
    
        timp_standard[idx] = end_time - start_time
    
    # start fft implementation
    if os.path.isfile(f'pickles/np_fft_{dimensiune}.pickle'):
        with open(f'pickles/np_fft_{dimensiune}.pickle', 'rb') as pickle_file:
            dft_numpy = pickle.load(pickle_file)
    else:
        start_time = perf_counter()
        
        dft_numpy = np.fft.fft(semnal)
        
        end_time = perf_counter()
        
        timp_numpy[idx] = end_time - start_time
        
        with open(f'pickles/np_fft_{dimensiune}.pickle', 'wb') as pickle_file:
            pickle.dump(dft_numpy, pickle_file)

if os.path.isfile(f'pickles/dft_timp.pickle'):
    with open(f'pickles/dft_timp.pickle', 'rb') as pickle_file:
        timp_standard = pickle.load(pickle_file)
else:
    with open(f'pickles/dft_timp.pickle', 'wb') as pickle_file:
        pickle.dump(timp_standard, pickle_file)
        
if os.path.isfile(f'pickles/np_fft_timp.pickle'):
    with open(f'pickles/np_fft_timp.pickle', 'rb') as pickle_file:
        timp_numpy = pickle.load(pickle_file)
else:
    with open(f'pickles/np_fft_timp.pickle', 'wb') as pickle_file:
        pickle.dump(timp_numpy, pickle_file)
    
plt.figure()
plt.title('Durata executare implementare DFT vs np.fft.fft')
plt.yscale('log')
plt.xlabel('Dimensiune vector')
plt.ylabel('Durata (s)')
plt.plot(lista_dimensiuni_vector, timp_numpy, label = 'np.fft.fft')
plt.plot(lista_dimensiuni_vector, timp_standard, label = 'DFT propriu')
plt.legend()

plt.savefig("plots/Exercitiul_1_durata.svg", format='svg')
plt.savefig("plots/Exercitiul_1_durata.png", format='png')
plt.savefig("plots/Exercitiul_1_durata.pdf", format='pdf')