import numpy as np
import matplotlib.pyplot as plt

# a

frecventa = 400
nr_esant = 1600
nr_pct = frecventa * nr_esant

spatiu_a = np.linspace(0, 4, nr_pct)   
esant_a = np.linspace(0, 4, nr_esant)
semnal_a = lambda t: np.sin(2*np.pi*frecventa*t)

plt.figure()
plt.title('Subpct a')
plt.plot(spatiu_a, semnal_a(spatiu_a))
plt.stem(esant_a, semnal_a(esant_a))
plt.savefig("plots/Exercitiul_2a.svg", format="svg")
#plt.show()

# b

frecventa = 800
durata = 3
nr_esant = 400

spatiu_b = np.linspace(0, durata, 200000)
esant_b = np.linspace(0, durata, nr_esant)
semnal_b = lambda t: np.sin(2*np.pi*frecventa*t)

plt.figure()
plt.title('Subpct b')
plt.plot(spatiu_b, semnal_b(spatiu_b))
plt.stem(esant_b, semnal_b(esant_b))
plt.savefig("plots/Exercitiul_2b.svg", format="svg")
#plt.show()

# c

frecventa = 240

spatiu_c = np.linspace(0, 20/frecventa, 1000)
semnal_c = lambda t: np.mod(frecventa*t, 1)

plt.figure()
plt.title('Subpct c')
plt.plot(spatiu_c, semnal_c(spatiu_c))
plt.savefig("plots/Exercitiul_2c.svg", format="svg")
#plt.show()

# d

frecventa = 300
# ca sa nu imi dea np.sign() = 0 (in cazul in care am 0)
eps = 10**-6

spatiu_d = np.linspace(0 + eps, 30/frecventa - eps, 10000)
semnal_d = lambda t: np.sign(np.sin(2*np.pi*frecventa*t))

plt.figure()
plt.title('Subpct d')
plt.plot(spatiu_d, semnal_d(spatiu_d))
plt.savefig("plots/Exercitiul_2d.svg", format="svg")
#plt.show()

# e

semnal_e = np.random.rand(128,128)

plt.figure()
plt.title('Subpct e')
plt.savefig("plots/Exercitiul_2e.svg", format="svg")
#plt.imshow(semnal_e, cmap='gray')

# f

# easter eggerino

semnal_f = np.ndarray((128,128))

plt.figure()
plt.title('Subpct f')
plt.savefig("plots/Exercitiul_2f.svg", format="svg")
plt.imshow(semnal_f, cmap='gray')