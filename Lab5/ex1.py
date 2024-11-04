import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('plots/Exercitiul_1.pdf')

# a

# semnalul a fost esantionat din ora in ora
# intr-o ora sunt 3600 de secunde
# astfel, frecventa in Hertzi este 1 / 3600 = 0.0002(7)

Fs = 1 / 3600

# b

# avem 18288 de esantioane, fiecare reprezentand numarul de masini care au trecut
# intr-o ora

# 18288 / 24 = 762 zile

# esantioanele pornesc din ziua de 25-08-2012 si se opresc in ziua de 25-09-2014
# (2014 are 366 de zile fiind an bisect, 25 august -> 25 septembrie = 31 de zile)
# 365 + 366 + 31 = 731 + 31 = 762 zile math checks out

# c

# folosindu-ne de teorema de esantionare Nyquist-Shannon
# si de faptul ca frecventa de esantionare este corecta (1 / 3600) Hz,
# frecventa maxima care poate fi prezenta in semnal este cel mult 1 / 2 * Fs
# anume 1 / 7200 ~= 0.00013(8)

# d

# citesc datele in format np.str512
y = np.genfromtxt('aux/Train.csv', delimiter=',', dtype=None)

# ma folosesc doar de coloana 2, care reprezinta numarul de masini
x = np.array(y[1:,2], dtype = np.uint64)

N = x.shape[0]

# transformata fourier
X = np.fft.fft(x)

# iau valoarea absoluta a componentelor
X = np.abs(X / N)

# iau jumatate de spectru
X = X[:N//2]

# vector frecvente
f = Fs * np.linspace(0, N//2, N//2) / N

plt.figure()
#plt.xscale('log')
plt.plot(f, X)
plt.title('Transformata Fourier initiala')
plt.ylabel('|X(ω)|')
plt.xlabel('Frecventa (Hz)')

pp.savefig()
plt.savefig("plots/Exercitiul_1_FFT_necentrat.svg", format='svg')
plt.savefig("plots/Exercitiul_1_FFT_necentrat.png", format='png')

# e

print(f'Componenta continua a semnalului este de {X[0]}')

# am prezenta o componenta continua in semnal, 

x2 = x - X[0]
X2 = np.fft.fft(x2)
X2 = np.abs(X2 / N)
X2 = X2[:N//2]

plt.figure()
plt.title('Transformata centrata in 0')
plt.plot(f, X2)
plt.ylabel('|X(ω)|')
plt.xlabel('Frecventa (Hz)')
plt.xscale('log')

pp.savefig()
plt.savefig("plots/Exercitiul_1_FFT_centrat.svg", format='svg')
plt.savefig("plots/Exercitiul_1_FFT_centrat.png", format='png')
# f 

print()

# extrag frecventele punctelor cu modulul cel mai mare
top_4 = np.argsort(X2)[-4:]

for idx,i in enumerate(top_4[::-1]):
    # afisez timpul in ore la care au loc trecerile
    print(f'Locul {idx + 1} tine de frecventa {f[i]} Hz ({X2[i]} media), care reprezinta {(1 / f[i]) / 3600} ore')
    
# g

# 25 august 2012 a fost o zi de sambata
# 1000 de esantioane -> 1000 de ore -> 1000 / 24 ~= 42 zile
# 42 de zile dupa 25 august -> 6 octombrie (sambata)
# stim asadar ca 44 de zile inseamna luni, 8 octombrie

# mergand in sens invers, 44 * 24 = 1056, ceea ce inseamna ca la index 1056
# gasesc prima zi de luni dupa indexul 1000
# ca sa plotez o luna, avand in vedere ca sunt in octombrie, trebuie sa
# merg 31 de zile in fata, 31 * 24 = 744

plt.figure()
plt.title('Luna de trafic')
plt.plot(x[1056:1056+744])
plt.xlabel('Esantionul')
plt.ylabel('Nr. masini')

pp.savefig()
plt.savefig("plots/Exercitiul_1_luna.svg", format='svg')
plt.savefig("plots/Exercitiul_1_luna.png", format='png')

# i

# fac top 10 valori, iar cea cu frecventa cea mai inalta este 

top_10 = np.argsort(X2)[-10:]
top_10 = top_10[::-1]

componente = list()

for (idx, val) in enumerate(X2):
    if val > 2:
        componente.append([idx, val])

componente = np.array(componente)

pp.close()