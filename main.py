from utils import read_excel, plot_market_volatility_surface
from interpolation import interpolate_vol_surface
import matplotlib.pyplot as plt


SPOT = 147.64
USD_RATE = 0.05119
JPY_RATE = 0.00111


vol_surface = read_excel('data/usdjpy_18082024.xlsx')
surface = interpolate_vol_surface(vol_surface, SPOT, USD_RATE, JPY_RATE, 1)

fig = plot_market_volatility_surface(surface)
plt.show()
