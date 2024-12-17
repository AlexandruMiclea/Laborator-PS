import numpy as np
import matplotlib.pyplot as plt
import scipy
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.backends.backend_pdf import PdfPages

N = 1000
pp = PdfPages('plots/Exercitiul_4.pdf')

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10

spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
spatiu_serie_timp = np.add(spatiu_serie_timp, spatiu_semnal)

model = ARIMA(spatiu_serie_timp)
rezultat = model.fit()
print(rezultat.summary())

#pp.savefig()
#plt.savefig("plots/Exercitiul_4.svg", format='svg')
#plt.savefig("plots/Exercitiul_4.png", format='png')

pp.close()
