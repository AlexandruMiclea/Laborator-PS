import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('plots/Exercitiul_3.pdf')

def rectangle_window(N: int):
    #w(n) = 1
    return np.ones(N)

def hanning_window(N: int):
    #w(n) = 0.5[1 − cos( 2πnN )]
    
    array = np.ndarray(N, dtype = np.float64)
    
    for i in range(N):
        array[i] = 0.5 * (1 - np.cos((2*np.pi*i)/N))
        
    return array

spatiu_esantioane = np.linspace(0, 1, 4000)
spatiu_semnal = 1 * np.sin(2 * np.pi * 100 * spatiu_esantioane)

#plt.plot(spatiu_esantioane, spatiu_semnal)

spatiu_hanning = np.pad(hanning_window(200), (0,200)) * spatiu_semnal[:400]
spatiu_dreptunghi = np.pad(rectangle_window(200), (0,200)) * spatiu_semnal[:400]

plt.figure()
plt.title('Filtru Hanning')
plt.plot(spatiu_esantioane[:400], spatiu_hanning)
plt.savefig("plots/Exercitiul_3_Hanning.svg", format='svg')
plt.savefig("plots/Exercitiul_3_Hanning.png", format='png')
pp.savefig()

plt.figure()
plt.title('Filtru Dreptunghi')
plt.plot(spatiu_esantioane[:400], spatiu_dreptunghi)
plt.savefig("plots/Exercitiul_3_dreptunghi.svg", format='svg')
plt.savefig("plots/Exercitiul_3_dreptunghi.png", format='png')
pp.savefig()

pp.close()