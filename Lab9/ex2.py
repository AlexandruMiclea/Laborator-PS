import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

N = 1000
pp = PdfPages('plots/Exercitiul_2.pdf')

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10

spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
spatiu_serie_timp = np.add(spatiu_serie_timp, spatiu_semnal)

def exponential_mean(timeseries, alpha):
    result_array = [timeseries[0]]
    
    for i in range(1, timeseries.shape[0]):
        result_array.append(alpha * timeseries[i] + (1 - alpha) * result_array[i - 1])
    
    return result_array


# valoarea care imi ofera o medie aproape de seria mea de timp, fara sa imi preia din zgomot
alpha = 0.15

spatiu_serie_timp_mediata = exponential_mean(spatiu_serie_timp, alpha)

# alternating minimization

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2)
ax.plot(spatiu_serie_timp, label = "Serie timp originala")
line, = ax.plot(spatiu_serie_timp_mediata, label = "Serie timp mediata")

def update_mean(val):
    
    alpha = alpha_slider.val
    
    spatiu_serie_timp_mediata = exponential_mean(spatiu_serie_timp, alpha)
    
    line.set_ydata(spatiu_serie_timp_mediata)
    fig.canvas.draw_idle()

ax_alpha = plt.axes([0.25, 0.1, 0.65, 0.03])
alpha_slider = plt.Slider(ax = ax_alpha, label = 'Alpha', valmin = 0, valmax = 1, valinit = alpha)
alpha_slider.on_changed(update_mean)

pp.savefig()
plt.savefig("plots/Exercitiul_2.svg", format='svg')
plt.savefig("plots/Exercitiul_2.png", format='png')

pp.close()