from utils import read_excel_from_bbg_file, plot_volatility_surface
from interpolation import interpolate_vol_surface
import matplotlib.pyplot as plt


SPOT = 144.45
USD_RATE = 0.053
JPY_RATE = -0.0009


vol_surface = read_excel_from_bbg_file('data/bbgnoadj.xlsx')
surface = interpolate_vol_surface(vol_surface, SPOT, USD_RATE, JPY_RATE, 1)

fig = plot_volatility_surface(surface)
plt.show()
