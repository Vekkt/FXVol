import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FREQ_TO_DAYS = {
    "D": 1,
    "W": 7,
    "M": 30,
    "Y": 365,
}


def read_excel_from_bbg_file(file_path):
    df = pd.read_excel(file_path, header=None, skiprows=3)
    columns_of_interest = [0, 1, 3, 5, 7, 9]
    df = df.iloc[:, columns_of_interest]

    df.columns = ["TTM", "ATM", "25C", "25P", "10C", "10P"]
    df["TTM"] = df["TTM"].apply(lambda x: int(x[:-1]) * FREQ_TO_DAYS[x[-1]]) / 365  # Using annualized TTM
    df = df.set_index("TTM")
    df = df / 100  # implied vols quoted in %
    return df[["10C", "25C",  "ATM", "25P", "10P"]]


def plot_volatility_surface(surface):
    X, Y = np.meshgrid(surface.columns.values, surface.index.values)
    Z = surface.values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', rcount=100, ccount=100)

    ax.set_xlabel('Strikes')
    ax.set_ylabel('Maturities')
    ax.set_zlabel('Implied Volatility')

    return fig
