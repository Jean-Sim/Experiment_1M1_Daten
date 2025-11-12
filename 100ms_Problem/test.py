from praktikum import cassy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
#### daten
data = cassy.CassyDaten("nur_stange_1.labx")
t = data.messung(1).datenreihe('t').werte[:20]
U = data.messung(1).datenreihe('U_B1').werte[:20]
#### peak
idx_peak = np.argmax(U)

t_peak = t[idx_peak]
U_peak = U[idx_peak]
print(t[(idx_peak-2):(idx_peak+4)])
print(t_peak)
#### fit
def sinus(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

guess = [1.0, 1.0, 0.0, 0.0]
params, cov = curve_fit(sinus, t[idx_peak-5:idx_peak+6], U[idx_peak-5:idx_peak+6], p0=guess)
amplitude , angular_speed, phase_shift, start_val = params
#### Fehlerrechnung
x_peak = (np.pi/2 - phase_shift) / angular_speed
dxdB = -(np.pi/2 - phase_shift) / (angular_speed**2)
dxdC = -1.0 / angular_speed
sigma_x_peak_sq = (dxdB**2)*cov[1,1] + (dxdC**2)*cov[2,2] + 2*dxdB*dxdC*cov[1,2]
sigma_x_peak = np.sqrt(abs(sigma_x_peak_sq))
print (cov)
print(sigma_x_peak)
#### plots
x_fit = np.linspace(min(t[idx_peak-5:idx_peak+6]), max(t[idx_peak-5:idx_peak+6]), 500)
y_fit = sinus(x_fit, *params)
plt.plot(t,U, "r+",label='Messwerte')
plt.plot([t_peak],[U_peak],"b+", label='Messwerte Peak')
plt.plot([x_peak], sinus(x_peak,*params), "g+", label='Interpolierter Peak')
plt.plot(x_fit,y_fit, label='sinus fit')
plt.ylabel("U [V]")
plt.xlabel("t [s]")
plt.legend()
plt.show()