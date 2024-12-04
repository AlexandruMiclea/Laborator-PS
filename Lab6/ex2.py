import numpy as np

N = 20

p = np.random.randint(0,10,size=N)
q = np.random.randint(0,10,size=N)

# varianta 0 -> inmultire standard

r0 = np.zeros(2 * N - 1)

for i in range(N):
    for j in range(N):
        r0[i + j] += p[i] * q[j]
        
# varianta 1 -> np.convolve

r1 = np.convolve(p,q)

# varianta 2 -> fft -> inmultire -> ifft

p_fft = np.fft.fft(p, 2 * N - 1)
q_fft = np.fft.fft(q, 2 * N - 1)

r2 = np.abs(np.fft.ifft(np.multiply(p_fft, q_fft)))

all_good = np.allclose(r0, r1) and np.allclose(r1, r2)
print(all_good)