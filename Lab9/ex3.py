import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

N = 1000
pp = PdfPages('plots/Exercitiul_3.pdf')

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.randn(N) / 30

spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
spatiu_serie_timp = np.add(spatiu_serie_timp, spatiu_semnal)

q = 500

epsilon_N = np.random.randn(N)
mu = np.mean(spatiu_serie_timp)

# calculez thetas
epsilon = epsilon_N[:q + 1][::-1] # shape 50
epsilon = np.concatenate((epsilon, [mu]))
eps_matrix = np.reshape(epsilon, (epsilon.shape[0], 1)) # shape (51,1)
eps_matrix = eps_matrix[1: eps_matrix.shape[0] - 1,:] # shape (49, 1) (nu am eps[i] si media)

thetas = np.linalg.lstsq(eps_matrix.T, [spatiu_serie_timp[q - 1]])[0] # shape 49
thetas = np.insert(thetas, 0, 1)
thetas = np.insert(thetas, thetas.shape[0], 1)


print(spatiu_serie_timp[q])
print(thetas @ epsilon)

spatiu_serie_timp_ma = np.zeros((N - q))

for i in range(q + 1, N):
     epsilon = epsilon_N[i-q:i + 1][::-1] # shape 50
     epsilon = np.concatenate((epsilon, [mu])) # shape 51
     spatiu_serie_timp_ma[i - q] = thetas @ epsilon

spatiu_ox = np.linspace(0, 999, 1000)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2)
ax.plot(spatiu_serie_timp, label = "Serie timp originala")
line, = ax.plot(spatiu_ox[q:], spatiu_serie_timp_ma, label = "Serie timp mediata")
fig.legend()
ax.set_ylim([0, 4.1])
fig.show()

def update_ma(val):
    
    q = q_slider.val
    
    epsilon = epsilon_N[:q]
    mu = np.mean(spatiu_serie_timp)
    epsilon = np.concatenate((epsilon, [mu]))

    spatiu_serie_timp_ma = np.zeros((N - q))

    for i in range(q, N):
        vector_1 = np.concatenate(([1], spatiu_serie_timp[i - q: i - 1], [1]))
        spatiu_serie_timp_ma[i - q] = np.dot(vector_1, epsilon)


    spatiu_ox = np.linspace(0, 999, 1000)
    
    line.set_ydata(spatiu_serie_timp_ma)
    line.set_xdata(spatiu_ox[q:])
    fig.canvas.draw_idle()

ax_alpha = plt.axes([0.25, 0.1, 0.65, 0.03])
q_slider = plt.Slider(ax = ax_alpha, label = 'q', valmin = 0, valmax = N, valstep = 1, valinit = q)
q_slider.on_changed(update_ma)

pp.savefig()
plt.savefig("plots/Exercitiul_3.svg", format='svg')
plt.savefig("plots/Exercitiul_3.png", format='png')

# pp.close()