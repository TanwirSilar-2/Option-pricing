import math
from typing import Literal, Tuple
from scipy.stats import norm  # standard normal CDF (N) and PDF (n)

OptionType = Literal["call", "put"]

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Tuple[float, float]:
    '''
    Calculate d1 and d2 for the Black-Scholes model. 
    Takes S, K, T, r, sigma and optional q as inputs
    '''
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, sigma must be positive; T and sigma cannot be zero.")

    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return d1, d2

def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: OptionType = "call", q: float = 0.0) -> float:
    '''
        Calculate the Black-Scholes price of a European call or put option.
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility of the underlying asset (annualized)
        kind: "call" for call option, "put" for put option
        q: Continuous dividend yield (default is 0.0)
    '''
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)  # dividend discount factor
    df_r = math.exp(-r * T)  # risk-free discount factor

    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    else:
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    
def greeks(S: float, K: float, T: float, r: float, sigma: float, kind: OptionType = "call", q: float = 0.0):
    '''    
        Calculate the Greeks for a European call or put option using the Black-Scholes model.
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility of the underlying asset (annualized)
        kind: "call" for call option, "put" for put option
        q: Continuous dividend yield (default is 0.0)
        Returns a dictionary with delta, gamma, vega, theta (per year), and rho.
    '''
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)

    if kind == "call":
        delta = df_q * cdf_d1
        theta = (-S * df_q * pdf_d1 * sigma / (2 * math.sqrt(T))) - r * K * df_r * norm.cdf(d2) + q * S * df_q * cdf_d1
        rho   = K * T * df_r * norm.cdf(d2)
    else:
        delta = df_q * (cdf_d1 - 1.0)
        theta = (-S * df_q * pdf_d1 * sigma / (2 * math.sqrt(T))) + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
        rho   = -K * T * df_r * norm.cdf(-d2)

    gamma = df_q * pdf_d1 / (S * sigma * math.sqrt(T))
    vega  = S * df_q * pdf_d1 * math.sqrt(T)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,                # per 1.0 vol change
        "theta_per_year": theta,     # divide by 365 for per-day
        "rho": rho,
    }

def implied_vol_newton(target_price: float,
                       S: float, K: float, T: float, r: float,
                       kind: OptionType = "call", q: float = 0.0,
                       sigma_init: float = 0.2, tol: float = 1e-6, max_iter: int = 100) -> float:
    sigma = max(1e-6, min(5.0, sigma_init))
    '''
    Calculate the implied volatility using Newton's method.
    '''
    for _ in range(max_iter):
        d1, _ = _d1_d2(S, K, T, r, sigma, q)
        price = bs_price(S, K, T, r, sigma, kind, q)
        diff = target_price - price
        if abs(diff) < tol:
            return sigma

        vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-12:
            break

        sigma = sigma + diff / vega
        sigma = max(1e-6, min(5.0, sigma))
    return sigma

