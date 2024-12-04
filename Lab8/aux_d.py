import numpy as np
import matplotlib.pyplot as plt
import scipy

# d

N = 1000

ec_grad2  = lambda x: x**2
spatiu_semnal = np.linspace(0,1,N)
semnal = lambda f1, f2: 0.1 * np.sin(2 * np.pi * f1 * spatiu_semnal) + 0.2 * np.sin(2 * np.pi * f2 * spatiu_semnal)

# 1 indexed
spatiu_ecuatie = np.linspace(1,2,N)
spatiu_serie_timp = ec_grad2(spatiu_ecuatie)
spatiu_semnal = semnal(3,6)

noise = np.random.random(size=N) / 10

spatiu_serie_timp = np.add(spatiu_serie_timp, noise)
spatiu_serie_timp = np.add(spatiu_serie_timp, spatiu_semnal)

# reference: https://matplotlib.org/stable/gallery/widgets/slider_demo.html
# https://www.geeksforgeeks.org/matplotlib-slider-widget/

# slidere pentru setarea valorilor lui p si m

p = 16
m = 256

y = spatiu_serie_timp[N - m:][::-1]
N_y = y.shape[0]
Y = np.zeros((m,p))

for i in range(m):
    Y[i] = spatiu_serie_timp[N - p - 1 - i:N - 1 - i][::-1]

x_star = np.linalg.inv(Y.T @ Y) @ Y.T @ y

# generez predictiile pentru spatiul timp

predicted_value_space = np.linspace(0, N - 1, N)
predicted_values = np.zeros(N)

for i in range(m):
    idx = N - i - 1
    time_series_true = spatiu_serie_timp[idx - p:idx]
    
    predicted_values[idx] = x_star.T @ time_series_true[::-1]

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.4)
ax.plot(spatiu_serie_timp, label = "Serie timp originala")
line, = ax.plot(predicted_value_space[N - m:], predicted_values[N - m:], label=f'Valori prezise')
line_2 = ax.axvline(N - m, label = 'Numarul predictiilor luate in model', color = 'red')

def update_predictions(val):
    
    p = p_slider.val
    m = m_slider.val
    
    y = spatiu_serie_timp[N - m:][::-1]
    N_y = y.shape[0]
    Y = np.zeros((m,p))

    for i in range(m):
        Y[i] = spatiu_serie_timp[N - p - 1 - i:N - 1 - i][::-1]

    x_star = np.linalg.inv(Y.T @ Y) @ Y.T @ y

    # generez predictiile pentru spatiul timp

    predicted_value_space = np.linspace(0, N - 1, N)
    predicted_values = np.zeros(N)

    for i in range(m):
        idx = N - i - 1
        time_series_true = spatiu_serie_timp[idx - p:idx]
        
        predicted_values[idx] = x_star.T @ time_series_true[::-1]
        
    
    line.set_ydata(predicted_values[N-m:])
    line.set_xdata(predicted_value_space[N-m:])
    line_2.set_xdata([N - m])
    fig.canvas.draw_idle()

ax_p = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_m = plt.axes([0.25, 0.1, 0.65, 0.03])
p_slider = plt.Slider(ax = ax_p, label = 'P', valmin = 0, valmax = 1000, valinit = p, valstep = 1)
m_slider = plt.Slider(ax = ax_m, label = 'M', valmin = 0, valmax = 1000, valinit = m, valstep = 1)
p_slider.on_changed(update_predictions)
m_slider.on_changed(update_predictions)

fig.legend()
fig.show()