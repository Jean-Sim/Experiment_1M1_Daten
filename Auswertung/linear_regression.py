import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Define the linear model ---
def linear_regression(x_data, y_data, y_err):
    def linear_model(x, a, b):
        return a * x + b

# --- Load your data ---
# Replace these with your actual data arrays:
# x_data = np.array([...])
# y_data = np.array([...])
# y_err = np.array([...]) # individual uncertainties (1σ)

# Example synthetic data (for demo)
#np.random.seed(0)
#x_data = np.linspace(0, 10, 109)
#y_true = 2.5 * x_data + 5
#y_err = np.random.uniform(0.2, 1.0, len(x_data))
#y_data = y_true + np.random.normal(0, y_err)

# --- Perform weighted linear regression ---
    popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
    a_fit, b_fit = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

# --- Compute fitted values and residuals ---
    y_fit = linear_model(x_data, a_fit, b_fit)
    residuals = y_data - y_fit

# --- Plot setup ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

# --- Upper plot: Data with fit ---
    ax1.errorbar(x_data, y_data, yerr=y_err, fmt='o', capsize=3, label='Data')
    ax1.plot(x_data, y_fit, 'r-', label=f'Fit: y = ({round(a_fit,7)} ± {round(a_err,8)})x + ({round(b_fit,6)} ± {round(b_err,6)})')
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)

# --- Lower plot: Residuals with error bars ---
    ax2.errorbar(x_data, residuals, yerr=y_err, fmt='o', color='black', capsize=3)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residuals")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# --- Print results ---
    print(f"Slope (a): {round(a_fit,8)} ± {round(a_err,7)}")
    print(f"Intercept (b): {round(b_fit,6)} ± {round(b_err,6)}")
    return popt


def count_within_sigmas(x, y, y_err, popt):
    def fit_func(x, a, b):
        return a * x + b
    """
    Counts how many data points lie within 1σ, 2σ, and 3σ
    of the fitted model (given the individual uncertainties).

    Parameters
    ----------
    x : array-like
        x-values
    y : array-like
        measured y-values
    y_err : array-like
        1σ uncertainties on y
    fit_func : callable
        model function, e.g. linear_model(x, a, b)
    popt : list or tuple
        best-fit parameters

    Returns
    -------
    dict with counts and fractions
    """
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
    """
    Calculates the chi-squared (χ²) and reduced chi-squared (χ²/dof)
    for a given fit.

    Parameters
    ----------
    x : array-like
        x data points
    y : array-like
        measured y data points
    y_err : array-like
        1σ uncertainties on y
    fit_func : callable
        model function, e.g. linear_model(x, a, b)
    popt : list or tuple
        best-fit parameters from curve_fit or similar

    Returns
    -------
    chi2 : float
        Total chi-squared value
    chi2_reduced : float
        Reduced chi-squared (χ² / dof)
    dof : int
        Degrees of freedom
    """
    y_fit = fit_func(x, *popt)
    residuals = y - y_fit
    chi2 = np.sum((residuals / y_err) ** 2)
    dof = len(x) - len(popt)
    chi2_reduced = chi2 / dof
    return round(chi2,3), round(chi2_reduced,3), round(dof,3)

