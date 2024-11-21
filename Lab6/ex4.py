import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy

pp = PdfPages('plots/Exercitiul_4.pdf')

y = np.genfromtxt('aux/Train.csv', delimiter=',', dtype=None)
x = np.array(y[1:,2], dtype = np.uint64)
N = x.shape[0]

# iau doar primele 3 zile din anul 2
semnal = x[365 * 24:365 * 24 + 72] 
N = 72

# b

plt.title('Filtrarea semnalului cu Rolling Average Filter')
plt.plot(semnal, label = 'Semnal initial')

w = 5

for w in [5, 9, 13, 17]:
    x2 = np.convolve(semnal, np.ones(w), 'valid') / w
    es = np.arange(0, 71, 71 / x2.shape[0])
    plt.plot(es, x2, label = f'Semnal filtrat cu dimensiunea w = {w}')

plt.xlabel('Ore')
plt.ylabel('Masini')
plt.legend()

plt.savefig("plots/Exercitiul_4_b.svg", format='svg')
plt.savefig("plots/Exercitiul_4_b.png", format='png')
pp.savefig()

# c

# presupunerea este ca orice frecventa mai mare de 1 / 3600 este zgomot,
# intrucat datele pe care le folosesc sunt sigur ca au fost esantionate la
# acea frecventa

# cum frecventa Nyquist = 1 / 7200 iar cea de esantionare este 1 / 3600
# rezulta ca valoarea normalizata la care voi implementa filtrul low-pass este
# de 0.5

Wn = 0.5

# d

butter_b, butter_a = scipy.signal.butter(5, Wn, btype='low')
rp_list = [1, 3, 5, 13]

semnal_butter = scipy.signal.filtfilt(butter_b, butter_a, semnal)

plt.figure()
plt.title('Filtrarea semnalului cu Low-Pass filters (fsz = 5)')
plt.plot(semnal, label = 'Semnal initial')
plt.plot(semnal_butter, label = 'Semnal filtrat cu filtru Butterworth')

for rp in rp_list:
    cheby_b, cheby_a = scipy.signal.cheby1(5, rp, Wn, btype='low')
    semnal_cheby = scipy.signal.filtfilt(cheby_b, cheby_a, semnal)
    plt.plot(semnal_cheby, label = f'Semnal filtrat cu filtru Chebyshev, rp = {rp}dB')

plt.xlabel('Ore')
plt.ylabel('Masini')
plt.legend()

plt.savefig("plots/Exercitiul_4_d.svg", format='svg')
plt.savefig("plots/Exercitiul_4_d.png", format='png')
pp.savefig()

# e

# filtrele Chebyshev de 5 si 13 decibeli filtreaza prea mult diferentele de 
# volum, deci nu m-as mai atinge de ele in cazul de fata

# despre butterworth, acesta se apropie cel mai mult de numarul de masini
# insa nu preia in detaliu diferentele produse (cel mai vizibil loc in care se intampla asta)
# este intre orele 2-10

# dintre toate personal as alege Chebyshev cu 1dB

# f

for fsz in [3, 7]:
    butter_b, butter_a = scipy.signal.butter(fsz, Wn, btype='low')
    rp_list = [1, 3]

    semnal_butter = scipy.signal.filtfilt(butter_b, butter_a, semnal)

    plt.figure()
    plt.title(f'Filtrarea semnalului cu Low-Pass filters (fsz = {fsz})')
    plt.plot(semnal, label = 'Semnal initial')
    plt.plot(semnal_butter, label = 'Semnal filtrat cu filtru Butterworth')

    for rp in rp_list:
        cheby_b, cheby_a = scipy.signal.cheby1(fsz, rp, Wn, btype='low')
        semnal_cheby = scipy.signal.filtfilt(cheby_b, cheby_a, semnal)
        plt.plot(semnal_cheby, label = f'Semnal filtrat cu filtru Chebyshev, rp = {rp}dB')

    plt.xlabel('Ore')
    plt.ylabel('Masini')
    plt.legend()

    plt.savefig(f"plots/Exercitiul_4_f_fsz_{fsz}.svg", format='svg')
    plt.savefig(f"plots/Exercitiul_4_f_fsz_{fsz}.png", format='png')
    pp.savefig()

# chebyshev fsz 3 dB 1 best in town

pp.close()