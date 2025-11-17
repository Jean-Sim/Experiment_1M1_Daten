import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_phase_shift_two_periods(T1, T2, t_max=200, title_prefix="Phasenverschiebung"):
    t = np.linspace(0, t_max, 5000)

    f1 = 1 / T1
    f2 = 1 / T2

    phi1 = 2 * np.pi * f1 * t
    phi2 = 2 * np.pi * f2 * t

    delta_phi = (phi2 - phi1 + np.pi) % (2 * np.pi) - np.pi

    delta_t = delta_phi / (2 * np.pi * f1)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, delta_phi, label=r'$\Delta \phi$ (rad)')
    plt.xlabel("Zeit [s]")
    plt.ylabel("Phasenverschiebung [rad]")
    plt.title(f"{title_prefix} (Winkel)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, delta_t, color='orange', label=r'$\Delta t$ (s)')
    plt.xlabel("Zeit [s]")
    plt.ylabel("Zeitverschiebung [s]")
    plt.title(f"{title_prefix} (Zeit)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return t, delta_phi, delta_t

def plot_measurement_table_matplotlib(
    M1, M2, M3, M4, M5, M6,
    names_without=["Messung 1", "Messung 2", "Messung 3"],
    names_with=["Messung 4", "Messung 5", "Messung 6"]
):

    labels = (
        [""] * 3 +
        [""] * 3
    )
    names = names_without + names_with

    cell_data = [
        [M1[0], M1[1], M1[2], M1[3]],
        [M2[0], M2[1], M2[2], M2[3]],
        [M3[0], M3[1], M3[2], M3[3]],
        [M4[0], M4[1], M4[2], M4[3]],
        [M5[0], M5[1], M5[2], M5[3]],
        [M6[0], M6[1], M6[2], M6[3]],
    ]

    row_labels = [
        f"{labels[i]}\n{names[i]}" for i in range(6)
    ]

    col_labels = [
        "Periodendauer [s]",
        "Unsicherheit [s]",
        "χ²",
        "Güte der Messung [χ²/f]"
    ]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis('off')

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.3)  # Breite, Höhe

    ax.set_title("Messreihenübersicht – Periodendauer und Fit-Güte", fontsize=12, pad=20)
    plt.tight_layout()
    plt.show()


M1 = [1.656686, 1.2e-05, 109.352, 1.022]
M2 = [1.65684, 1.2e-05, 105.649, 0.997]
M3 = [1.656648, 1.8e-05, 110.796, 1.035]
M4 = [1.65615, 1.5e-05, 106.522, 1.005]
M5 = [1.655292, 6.3e-06, 109.99, 1.028]
M6 = [1.655154, 7.2e-06, 103.975, 0.981]
#plot_measurement_table_matplotlib( M1, M2, M3, M4, M5, M6,)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


g_final = 9.826
sigma = 0.0072
g_EKG = 9.811
def g_histogram(mu,sigma,mark):
    N = 200000

    data = np.random.normal(mu, sigma, N)
    percent = np.mean(data <= mark) * 100

    print(f"Wert {mark} liegt bei {percent}% der Vrteilung.")

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=80, density=True, alpha=0.6, color='blue', label="Normalverteilung")
    plt.axvline(mark, color='red', linewidth=2, label=f"g_BKG = {mark}")

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    plt.plot(x, norm.pdf(x, mu, sigma), 'k-', label="Theor. Normalverteilung")

    plt.title("Histogramm von g_final mit markiertem g_BKG")
    plt.xlabel("g [m/s^2]")
    plt.ylabel("Dichte")
    plt.legend()
    plt.grid(True)

    plt.show()

g_histogram(g_final, sigma, g_EKG)
