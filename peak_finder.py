from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import find_peaks

def data_peaks(U_data):
    peaks, _ = find_peaks(U_data)
    return peaks


def sin_fit_peak(idx_peak, data):
    try:
        t, U = data
        def cos(x, A, B, t_peak, D):
            return A * np.cos(B * (x - t_peak)) + D

        guess = [1.0, 2*np.pi/1.6, t[idx_peak], 0.0]
        para, cov = curve_fit(cos, t[idx_peak - 4:idx_peak + 5], U[idx_peak - 4:idx_peak + 5], p0=guess)
        amplitude, angular_speed, t_peak, start_val = para
        sigma_t_peak = np.sqrt(np.diag(cov))[2]

        return t_peak, sigma_t_peak
    except ValueError:
        print("not enough values in range")