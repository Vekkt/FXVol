import pandas as pd
from scipy.interpolate import griddata

from pricing import *
from itertools import product


def get_strike_ua(forward, delta, base_rf, ttm, vol):
    if delta == "ATM":
        return atm_strike_ua(forward, vol, ttm)

    delta, opt = (int(delta[:-1]) / 100, delta[-1])
    delta = -delta if opt == 'P' else delta
    if ttm <= 1:
        return strike_from_spot_delta_ua(forward, delta, vol, ttm, base_rf, opt)

    return strike_from_forward_delta_ua(forward, delta, vol, ttm, opt)


def map_strikes(vol_surface, spot, base_rf, quote_rf):
    def func(value):
        vol = value.iloc[0]
        ttm, delta = value.name[0], value.name[1]
        forward = fx_forward(spot, base_rf, quote_rf, ttm)
        return get_strike_ua(forward, delta, base_rf, ttm, vol)

    return vol_surface.stack().to_frame().apply(func, axis=1).unstack()


def interpolate_surface(df, maturity_cutoff):
    x = df["Strike"]
    y = df["TTM"]
    z = df["IV"]

    mask_linear = y <= maturity_cutoff
    mask_cubic = y >= maturity_cutoff

    x_linear, y_linear, z_linear = x[mask_linear], y[mask_linear], z[mask_linear]
    x_cubic, y_cubic, z_cubic = x[mask_cubic], y[mask_cubic], z[mask_cubic]

    spot_strikes = df[df['TTM'] == min(df['TTM'])].sort_values(by='Strike')
    atm_spot_strike = spot_strikes.iloc[len(spot_strikes) // 2]['Strike']

    # Define the grid for interpolation - more samples around spot ATM strike
    xi = np.hstack((-np.geomspace(-atm_spot_strike, -x.min(), num=500)[::-1],
                    np.geomspace(atm_spot_strike, x.max(), num=501)[1:],))
    yi = np.linspace(y.min(), y.max(), num=1000)
    xi, yi = np.meshgrid(xi, yi)

    # Linear interpolation for expirations <= 1 year
    z_linear_interp = griddata((x_linear, y_linear), z_linear, (xi, yi), method='linear')

    # Cubic interpolation for expirations >= 1 year
    z_cubic_interp = griddata((x_cubic, y_cubic), z_cubic, (xi, yi), method='cubic')

    # Merge the interpolated surfaces
    z_interp = np.where(yi <= 1, z_linear_interp, z_cubic_interp)

    surface = pd.DataFrame(z_interp)
    surface.columns = xi[0]
    surface.index = yi[:, 0]
    return surface


def interpolate_vol_surface(vol_surface, spot, base_rf, quote_rf, maturity_cutoff):
    strike_surface = map_strikes(vol_surface, spot, base_rf, quote_rf)

    strikes, ttms, vols = [], [], []
    for ttm, delta in product(vol_surface.index, vol_surface.columns):
        strikes.append(strike_surface.loc[ttm, delta])
        ttms.append(ttm)
        vols.append(vol_surface.loc[ttm, delta])

    df = pd.DataFrame([ttms, strikes, vols]).T
    df.columns = ['TTM', 'Strike', 'IV']

    return interpolate_surface(df, maturity_cutoff)


