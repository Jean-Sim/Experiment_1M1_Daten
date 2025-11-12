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
        params, cov = curve_fit(cos, t[idx_peak - 4:idx_peak + 5], U[idx_peak - 4:idx_peak + 5], p0=guess)
        amplitude, angular_speed, t_peak, start_val = params
    #### Fehlerrechnung
    #t_peak = (np.pi / 2 - phase_shift) / angular_speed
    #dtdB = -(np.pi / 2 - phase_shift) / (angular_speed ** 2)
    #dtdC = -1.0 / angular_speed
    #sigma_t_peak_sq = (dtdB ** 2) * cov[1, 1] + (dtdC ** 2) * cov[2, 2] + 2 * dtdB * dtdC * cov[1, 2]
        sigma_t_peak = np.sqrt(np.diag(cov))[2]

        return t_peak, sigma_t_peak
    except ValueError:
        print("not enough values in range")


def bootstrap_tpeak(idx_peak, data, n_boot=300):
    t_segment = data[0][idx_peak - 8:idx_peak + 9]
    U_segment = data[1][idx_peak - 8:idx_peak + 9]
    params_fit = [1.0, 2*np.pi/1.6, t_segment[8], 0.0]

    def cos(x, A, B, t_peak, D):
        return A * np.cos(B * (x - t_peak)) + D

    A, B, t_peak, D = params_fit
    residuals = U_segment - cos(t_segment, *params_fit)
    t_peaks_boot = []

    for _ in range(n_boot):
        # Resample residuals with replacement
        resampled_resid = np.random.choice(residuals, size=len(residuals), replace=True)
        U_boot = cos(t_segment, *params_fit) + resampled_resid

        # Fit bootstrap sample
        try:
            params_b, _ = curve_fit(cos, t_segment, U_boot, p0=params_fit)
            t_peaks_boot.append(params_b[2])
        except Exception:
            # If fit fails, skip this bootstrap iteration
            continue

    t_peaks_boot = np.array(t_peaks_boot)
    sigma_t_boot = np.std(t_peaks_boot)
    return sigma_t_boot, np.mean(t_peaks_boot)