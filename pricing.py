import numpy as np

from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import root_scalar


DELTA_SEARCH_MAX_ITER = 10000
DELTA_SEARCH_TOL = 1e-3


def get_opt_factor(opt):
    assert opt in ["CALL", "PUT", "C", "P"], "Option type must be one of 'CALL' (or 'C') and 'PUT' (or 'P')"
    return 1 if opt in ["CALL", "C"] else -1


def fx_forward(spot, base_rf, quote_rf, ttm):
    return spot * np.exp((quote_rf - base_rf) * ttm)


def fx_forward_value(spot, strike, base_rf, quote_rf, ttm):
    return spot * exp(-base_rf * ttm) - strike * exp(-quote_rf * ttm)


def d1(forward, strike, ttm, vol):
    return (log(forward / strike) + 0.5 * vol ** 2 * ttm) / (vol * sqrt(ttm))


def d2(forward, strike, ttm, vol):
    return d1(forward, strike, ttm, vol) - vol * sqrt(ttm)


def bs_vanilla_price(forward, strike, vol, ttm, quote_rf, opt):
    d1_ = d1(forward, strike, ttm, vol)
    d2_ = d2(forward, strike, ttm, vol)
    phi = get_opt_factor(opt)
    return phi * exp(-quote_rf * ttm) * (forward * norm.cdf(phi * d1_) - strike * norm.cdf(phi * d2_))


def spot_delta_ua(forward, strike, vol, ttm, base_rf, opt):
    phi = get_opt_factor(opt)
    d1_ = d1(forward, strike, ttm, vol)
    return exp(-base_rf * ttm) * phi * norm.cdf(phi * d1_)


def spot_delta_pa(forward, strike, vol, ttm, base_rf, opt):
    phi = get_opt_factor(opt)
    d2_ = d2(forward, strike, ttm, vol)
    return strike / forward * exp(-base_rf * ttm) * phi * norm.cdf(phi * d2_)


def forward_delta_ua(forward, strike, vol, ttm, opt):
    phi = get_opt_factor(opt)
    d1_ = d1(forward, strike, ttm, vol)
    return phi * norm.cdf(phi * d1_)


def forward_delta_pa(forward, strike, vol, ttm, opt):
    phi = get_opt_factor(opt)
    d2_ = d2(forward, strike, ttm, vol)
    return strike / forward * phi * norm.cdf(phi * d2_)


def atm_strike_ua(forward, vol, ttm):
    return forward * exp(0.5 * vol ** 2 * ttm)


def atm_strike_pa(forward, vol, ttm):
    return forward * exp(-0.5 * vol ** 2 * ttm)


def strike_from_spot_delta_ua(forward, delta, vol, ttm, base_rf, opt):
    phi = get_opt_factor(opt)
    h = norm.ppf(phi * exp(base_rf * ttm) * delta)
    return forward * exp(0.5 * vol ** 2 * ttm - phi * vol * sqrt(ttm) * h)


def strike_from_forward_delta_ua(forward, delta, vol, ttm, opt):
    phi = get_opt_factor(opt)
    h = norm.ppf(phi * delta)
    return forward * exp(0.5 * vol ** 2 * ttm - phi * vol * sqrt(ttm) * h)


def strike_from_spot_delta_pa(forward, delta, vol, ttm, base_rf, opt):
    phi = get_opt_factor(opt)

    def func(strike):
        h = norm.ppf(phi * delta * forward / strike * exp(base_rf * ttm))
        return log(forward) - 0.5 * vol ** 2 * ttm - phi * vol * sqrt(ttm) * h - log(strike)

    res = root_scalar(func, x0=forward)

    assert res.converged, "Could not find a strike."
    return res.root


def strike_from_forward_delta_pa(forward, delta, vol, ttm, opt):
    phi = get_opt_factor(opt)

    def func(strike):
        h = norm.ppf(phi * delta * forward / strike)
        return log(forward) - 0.5 * vol ** 2 * ttm - phi * vol * sqrt(ttm) * h - log(strike)

    res = root_scalar(func, x0=forward)

    assert res.converged, "Could not find a strike."
    return res.root
