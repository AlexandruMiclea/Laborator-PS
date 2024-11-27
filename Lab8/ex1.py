import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

N = 1000
pp = PdfPages('plots/Exercitiul_1.pdf')

# a

fig, axs = plt.subplots(4)
plt.title('Serie timp')

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

# 1 indexed
spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10

axs[0].plot(spatiu_serie_timp)
axs[0].set_title('Ecuatie grad 2')
spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
axs[1].plot(noise)
axs[1].set_title('Zgomot')
spatiu_serie_timp += np.add(spatiu_serie_timp, spatiu_semnal)
axs[2].plot(spatiu_semnal)
axs[2].set_title('Semnal')

plt.xlabel("Ordin esantion")
plt.ylabel("Valoare")
plt.plot(spatiu_serie_timp)
plt.show()

pp.savefig()
plt.savefig("plots/Exercitiul_1a.svg", format='svg')
plt.savefig("plots/Exercitiul_1a.png", format='png')

# b

def self_autocorrelation(timeseries):
    
    N = timeseries.shape[0]
    
    ret_values = np.zeros(2 * N - 1)
    
    for idx in range(2 * N - 1):
        
        if (idx < N):
            idx1_left = N - 1 - idx
            idx1_right = N
            idx2_left = 0
            idx2_right = idx + 1
            
        else:
            idx1_left = 0
            idx1_right = 2 * N - idx
            idx2_left = idx - N
            idx2_right = N
            
        sub_1 = timeseries[idx1_left:idx1_right]
        sub_2 = timeseries[idx2_left:idx2_right]
        
        ret_values[idx] = np.dot(sub_1, sub_2)
    
    return ret_values


test = np.correlate(spatiu_serie_timp, spatiu_serie_timp, "full")
test2 = self_autocorrelation(spatiu_serie_timp)

#print(np.sum(spatiu_serie_timp))

fig, axs = plt.subplots(2)
#plt.title('Self-autocorrelation vs NumPy')
axs[0].plot(test / np.sum(spatiu_serie_timp**2))
axs[0].set_title("NumPy")
axs[1].plot(test2 / np.sum(spatiu_serie_timp**2))
axs[1].set_title("Autocorelare implementata")
plt.show()

pp.savefig()
plt.savefig("plots/Exercitiul_1b.svg", format='svg')
plt.savefig("plots/Exercitiul_1b.png", format='png')

norm_test = np.linalg.norm(test)
norm_test2 = np.linalg.norm(test2)

print (f"It is {np.abs(norm_test - norm_test2) <= 1e6} that my autocorrelation is the best!")

# c

p = 8

y = spatiu_serie_timp[-N + p::][::-1]

Y = np.zeros((N - p,p))

for i in range(N - p):
    Y[i] = spatiu_serie_timp[N - p - 1 - i:N - 1 - i][::-1]


x_star = np.linalg.inv(Y.T @ Y) @ Y.T @ y

predictii = spatiu_serie_timp[N - p:N][::-1]

for i in range(1000):
    lungime_predictii = predictii.shape[0]
    predicted_value = x_star.T @ predictii[lungime_predictii - p:lungime_predictii]
    predictii = np.concatenate((predictii, np.array([predicted_value])))
    
predictii = predictii[p:]

# M = n - p

plt.figure()
plt.title("Serie timp cu predictii")
plt.xlabel("Ordin esantion")
plt.ylabel("Valoare")
plt.plot(spatiu_serie_timp, label = "Serie timp originala")
spatiu_predictii = np.linspace(N, 2 * N - 1, N)
plt.plot(spatiu_predictii, predictii, label=f'Valori prezise pentru p = {p}')
plt.axvline(p, label = 'Numarul predictiilor luate in model', color = 'red')
plt.legend()
plt.show()

pp.savefig()
plt.savefig("plots/Exercitiul_1c.svg", format='svg')
plt.savefig("plots/Exercitiul_1c.png", format='png')

# d

# reference: https://matplotlib.org/stable/gallery/widgets/slider_demo.html
# https://www.geeksforgeeks.org/matplotlib-slider-widget/

# slidere pentru setarea valorilor lui p si m

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.4)
ax.plot(spatiu_serie_timp, label = "Serie timp originala")
spatiu_predictii = np.linspace(N, 2 * N - 1, N)
line, = ax.plot(spatiu_predictii, predictii, label=f'Valori prezise')
line_2 = ax.axvline(8, label = 'Numarul predictiilor luate in model', color = 'red')

def update_predictions(val):
    
    p = p_slider.val
    m = m_slider.val
    
    y = spatiu_serie_timp[N - m:][::-1]

    Y = np.zeros((m,p))

    for i in range(m):
        Y[i] = spatiu_serie_timp[N - p - 1 - i:N - 1 - i][::-1]

    x_star = np.linalg.inv(Y.T @ Y) @ Y.T @ y

    predictii = spatiu_serie_timp[N - p:N][::-1]

    for i in range(N):
        lungime_predictii = predictii.shape[0]
        predicted_value = x_star.T @ predictii[lungime_predictii - p:lungime_predictii]
        predictii = np.concatenate((predictii, np.array([predicted_value])))
        
    predictii = predictii[p:]
    
    spatiu_predictii = np.linspace(N, 2 * N - 1, N)
    line.set_ydata(predictii)
    line_2.set_xdata([N - m])
    fig.canvas.draw_idle()

ax_p = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_m = plt.axes([0.25, 0.1, 0.65, 0.03])
p_slider = plt.Slider(ax = ax_p, label = 'P', valmin = 0, valmax = 1000, valinit = 8, valstep = 1)
m_slider = plt.Slider(ax = ax_m, label = 'M', valmin = 0, valmax = 1000, valinit = 992, valstep = 1)
p_slider.on_changed(update_predictions)
m_slider.on_changed(update_predictions)

fig.legend()
fig.show()

pp.savefig()
plt.savefig("plots/Exercitiul_1d.svg", format='svg')
plt.savefig("plots/Exercitiul_1d.png", format='png')

pp.close()