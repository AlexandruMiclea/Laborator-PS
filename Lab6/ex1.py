import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('plots/Exercitiul_1.pdf')

x = np.random.random(100)

plt.figure()
plt.title('Vector start')
plt.plot(x)

pp.savefig()
plt.savefig("plots/Exercitiul_1_start.svg", format='svg')
plt.savefig("plots/Exercitiul_1_start.png", format='png')

for i in range(3):
    x = np.convolve(x,x)
    plt.figure()
    plt.title(f'Convolutia {i + 1}')
    plt.plot(x)
    
    pp.savefig()
    plt.savefig(f"plots/Exercitiul_1_iter{i+1}.svg", format='svg')
    plt.savefig(f"plots/Exercitiul_1_iter{i+1}.png", format='png')

# convolutia presupune un sir de adunari si inmultiri
# inmultirea a 2 valori aleatoare rezulta intr-o noua valoare aleatoare
# adunarea valorilor aleatoare va duce in reprezentarea unui clopot gaussian

pp.close()