import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sigfig import round as sig_round

def linear_regression(x_data, y_data, y_err):
    def linear_model(x, a, b):
        return a * x + b

    popt, cov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
    a_fit, b_fit = popt
    a_err, b_err = np.sqrt(np.diag(cov))

    y_fit = linear_model(x_data, a_fit, b_fit)
    residuals = y_data - y_fit

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.errorbar(x_data, y_data, yerr=y_err, fmt='o', capsize=3, label='Data',markersize=3)
    ax1.plot(x_data, y_fit, 'r-', label=f'Fit: y = ({round(a_fit,6)} ± {sig_round(a_err,sigfigs=2)})x + ({round(b_fit,4)} ± {sig_round(b_err,sigfigs=2)})')
    ax1.set_ylabel("t [s]")
    ax1.legend()
    ax1.grid(True)

    ax2.errorbar(x_data, residuals, yerr=y_err, fmt='o', color='black', capsize=3)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("peaks [na]")
    ax2.set_ylabel("Residuals")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Slope (a): {round(a_fit,6)} ± {sig_round(a_err,sigfigs=2)}")
    print(f"Intercept (b): {round(b_fit,4)} ± {sig_round(b_err,sigfigs=2)}")
    return popt


def count_within_sigmas(x, y, y_err, popt):
    def fit_func(x, a, b):
        return a * x + b

    y_fit = fit_func(x, *popt)
    residuals = y - y_fit
    normalized_residuals = np.abs(residuals) / y_err

    within_1sigma = np.sum(normalized_residuals <= 1)
    within_2sigma = np.sum(normalized_residuals <= 2)
    within_3sigma = np.sum(normalized_residuals <= 3)

    n = len(x)
    return {
        "within_1σ": (round(within_1sigma,3), round(within_1sigma / n,3)),
        "within_2σ": (round(within_2sigma,3), round(within_2sigma / n,3)),
        "within_3σ": (round(within_3sigma,3), round(within_3sigma / n,3)),
    }