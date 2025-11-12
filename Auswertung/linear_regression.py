import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_regression(x_data, y_data, y_err):
    def linear_model(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
    a_fit, b_fit = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

    y_fit = linear_model(x_data, a_fit, b_fit)
    residuals = y_data - y_fit

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.errorbar(x_data, y_data, yerr=y_err, fmt='o', capsize=3, label='Data')
    ax1.plot(x_data, y_fit, 'r-', label=f'Fit: y = ({round(a_fit,7)} ± {round(a_err,8)})x + ({round(b_fit,6)} ± {round(b_err,6)})')
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)


    ax2.errorbar(x_data, residuals, yerr=y_err, fmt='o', color='black', capsize=3)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residuals")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


    print(f"Slope (a): {round(a_fit,8)} ± {round(a_err,7)}")
    print(f"Intercept (b): {round(b_fit,6)} ± {round(b_err,6)}")
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


def chi_squared(x, y, y_err, popt):
    def fit_func(x, a, b):
        return a * x + b

    y_fit = fit_func(x, *popt)
    residuals = y - y_fit
    chi2 = np.sum((residuals / y_err) ** 2)
    dof = len(x) - len(popt)
    chi2_reduced = chi2 / dof
    return round(chi2,3), round(chi2_reduced,3), round(dof,3)


