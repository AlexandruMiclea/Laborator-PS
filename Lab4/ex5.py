import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

freq, signal = scipy.io.wavfile.read('wavs/vocale.wav')


# vocalele a si e se pot distinge cu usurinta (frecventa lui A este mai activa
# in jurul frecventei de 1000Hz, pe cand frecventa lui E este mai activa sub
# 1000 Hz si peste 1000 Hz)
# intre E si I diferenta sta in activarea frecventei in jurul a 1000 Hz
# O este asemanator cu I, doar ca activarea frecventelor sub 1000 Hz este mai mare
# U este destul de asemanator, de asemenea, cu O, dar cu mai putin "zgomot"
# in frecventele > 3000 Hz


# in urma inregistrarii audio obtin
# doua canale audio (left si right)

plt.figure()
plt.title('Canal audio 0')
plt.plot(signal[:,0])

plt.figure()
plt.title('Canal audio 1')
plt.plot(signal[:,1])

scipy.io.wavfile.write('wavs/canal_0_5.wav', 44100, signal[:,0])
scipy.io.wavfile.write('wavs/canal_1_5.wav', 44100, signal[:,1])