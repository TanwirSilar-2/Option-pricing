# app.py
# Black–Scholes interactive UI with standard plots
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import requests


from bsm import bs_price, greeks, implied_vol_newton
from datetime import datetime

# ---------------------------
# Helpers for sweeps
# ---------------------------
def sweep_over_spot(S, K, T, r, sigma, kind, q, n_points=160):
    S_grid = np.linspace(max(0.1, 0.5 * S), 1.5 * S, n_points)
    price_grid, delta_grid, gamma_grid, vega_grid = [], [], [], []
    for s in S_grid:
        price_grid.append(bs_price(s, K, T, r, sigma, kind, q))
        g = greeks(s, K, T, r, sigma, kind, q)
        delta_grid.append(g["delta"])
        gamma_grid.append(g["gamma"])
        vega_grid.append(g["vega"])
    df = pd.DataFrame(
        {"S": S_grid, "Price": price_grid, "Delta": delta_grid, "Gamma": gamma_grid, "Vega": vega_grid}
    )
    return df

def sweep_over_vol(S, K, T, r, kind, q, n_points=160, vol_min=0.05, vol_max=1.00):
    sig_grid = np.linspace(vol_min, vol_max, n_points)
    prices = [bs_price(S, K, T, r, s, kind, q) for s in sig_grid]
    df = pd.DataFrame({"Sigma": sig_grid, "Price": prices})
    return df

def sweep_over_time(S, K, r, sigma, kind, q, n_points=160, days_min=1, days_max=365):
    days_grid = np.linspace(days_min, days_max, n_points)
    T_grid = days_grid / 365.0
    price_grid, theta_day = [], []
    for t in T_grid:
        price_grid.append(bs_price(S, K, t, r, sigma, kind, q))
        th = greeks(S, K, t, r, sigma, kind, q)["theta_per_year"] / 365.0
        theta_day.append(th)
    df = pd.DataFrame({"Days": days_grid, "Price": price_grid, "Theta_per_day": theta_day})
    return df

def fetch_option_chain_ind(symbol: str):
    """
    Fetch the option chain for a given symbol.
    Gets the nearest expiry and extracts call options with their IVs. 
    Returns a DataFrame with calls and puts.
    """

    # ------------ Config ------------
    SYMBOL = symbol   # change to any NSE equity symbol
    BASE_URL = "https://www.nseindia.com"
    CHAIN_URL = f"{BASE_URL}/api/option-chain-equities?symbol={SYMBOL}"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"{BASE_URL}/option-chain"
    }

    # ------------ Fetch JSON from NSE (with session + cookies) ------------
    session = requests.Session()

    # Prime cookies by hitting the option-chain page
    session.get(f"{BASE_URL}/option-chain", headers=HEADERS, timeout=10)

    # Now fetch the option-chain JSON
    resp = session.get(CHAIN_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    # ------------ Pick an expiry (nearest by default) ------------
    expiry_list = payload.get("records", {}).get("expiryDates", [])
    if not expiry_list:
        raise RuntimeError("No expiry dates found in NSE response.")
    expiry = expiry_list[0]  # nearest expiry
    print("Using expiry:", expiry)

    # ------------ Extract CALL strikes and IVs for that expiry ------------
    records = payload.get("records", {}).get("data", [])
    rows = []
    for rec in records:
        if rec.get("expiryDate") != expiry:
            continue
        ce = rec.get("CE")
        iv = ce.get("impliedVolatility")  # Typically already in % (e.g., 18.42)
        if not ce or iv is None:
            continue

        strike = rec.get("strikePrice")
        
        lastPrice = ce.get("lastPrice")
        bid = ce.get("bidPrice")
        ask = ce.get("askPrice")
        S = ce.get("underlyingValue")  # Current underlying price
        expiry_date = datetime.strptime(ce.get("expiryDate"), "%d-%b-%Y").date()
        T = (expiry_date - datetime.today().date()).days / 365.0
        if iv is None:
            continue
        rows.append({"strike": strike, "impliedVolatility": iv/100, "lastPrice": lastPrice, "bid": bid, "ask": ask, "T": T, "expiry": expiry, "UnderlyingPrice": S})

    df_calls = pd.DataFrame(rows).dropna(subset=["impliedVolatility"]).sort_values("strike")
    if df_calls.empty:
        raise RuntimeError("No call IV data found for the selected expiry.")

    return df_calls


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Black–Scholes Option Calculator", layout="wide")
st.title("Black–Scholes Option Pricing")
st.caption("European options • pricing, Greeks, implied volatility, and standard sensitivity plots")

# ---------------------------
# Sidebar: Inputs
# ---------------------------
with st.sidebar:

    tkr = "HAL"
    st.header(f"Inputs ({tkr})")

    
    calls = fetch_option_chain_ind(tkr)
    # use mid price where possible
    valid = (calls["bid"] > 0) & (calls["ask"] > 0)
    calls.loc[valid, "mid"] = (calls.loc[valid, "bid"] + calls.loc[valid, "ask"]) / 2.0
    calls.loc[~valid, "mid"] = calls.loc[~valid, "lastPrice"]

    T = float(calls.iloc[0]["T"])  # use the first row's T as the default T(assumes all calls have same T)
    spotPrice = calls.iloc[0]["UnderlyingPrice"]  # use the first row's UnderlyingPrice for the default S

    S = st.number_input("Spot S", min_value=0.01, value=spotPrice, step=1.0)
    q = st.number_input("Dividend yield q (%)", min_value=0.0, value=0.0, step=0.1) / 100.0

    # Core contract parameters
    K = st.number_input("Strike K", min_value=0.01, value=float(round(S, 0)), step=1.0)
    days = st.number_input("Days to expiry", min_value=1, value=int(T*365.0), step=1)
    T = days / 365.0  # convert to years 
    r = st.number_input("Risk-free r (%)", min_value=0.0, value=10.0, step=0.25) / 100.0
    kind = st.selectbox("Option type", options=["call", "put"], index=0)

    # Volatility or IV from price
    mode = st.radio("Volatility mode", ["Use volatility", "Implied from market price"], index=0)
    if mode == "Use volatility":
        sigma = st.number_input("Volatility σ (%)", min_value=0.01, value=25.0, step=0.5) / 100.0
        target_px = None
    else:
        target_px = st.number_input("Observed option price", min_value=0.0, value=5.00, step=0.25)
        sigma = None

    # Plot resolution controls
    st.subheader("Plot resolution")
    n_pts = st.slider("Points per curve", min_value=50, max_value=400, value=160, step=10)

run = st.button("Compute")


    
    


# ---------------------------
# Compute & Display
# ---------------------------
if run:
    try:
        # If user chose IV from price, solve for sigma first
        if sigma is None and target_px is not None:
            sigma = implied_vol_newton(target_px, S, K, T, r, kind, q)
            st.success(f"Implied volatility (from price {target_px:.4f}): {sigma*100:.2f}%")

        # Price & Greeks at the chosen point
        price = bs_price(S, K, T, r, sigma, kind, q)
        g = greeks(S, K, T, r, sigma, kind, q)

        st.subheader("Calculated Values")
        c1, c2, c3 = st.columns(3)
        c1.metric("Option price", f"{price:.4f}")
        c2.metric("Delta", f"{g['delta']:.4f}")
        c3.metric("Gamma", f"{g['gamma']:.6f}")
        c1.metric("Vega (per 1.0 σ)", f"{g['vega']:.4f}")
        c2.metric("Theta (per day)", f"{g['theta_per_year']/365:.6f}")
        # c3.metric("r (annual %)", f"{r*100:.2f}%")
        # c1.metric("q (dividend %)", f"{q*100:.2f}%")
        # c1.metric("Rho", f"{g['rho']:.4f}")
        # c2.metric("Theta (per year)", f"{g['theta_per_year']:.4f}")

        st.divider()

        # Tabs for standard graphs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Vs Spot (S)", "Vs Volatility (σ)", "Vs Time (T)", "Call–Put Parity", "Inputs Echo", "Smile Plot"]
        )

        # --- Tab 1: Vs Spot (S) ---
        with tab1:
            st.markdown("### Sensitivity to Spot (S)")
            df_s = sweep_over_spot(S, K, T, r, sigma, kind, q, n_points=n_pts)

            cA, cB = st.columns(2)
            with cA:
                # st.plotly_chart(px.line(df_s, x="S", y="Price", title="Option Price vs Spot S"), use_container_width=True)
                st.plotly_chart(px.line(df_s, x="S", y="Delta", title="Delta vs Spot S"), use_container_width=True)
                st.plotly_chart(px.line(df_s, x="S", y="Vega", title="Vega vs Spot S"), use_container_width=True)
            with cB:
                st.plotly_chart(px.line(df_s, x="S", y="Gamma", title="Gamma vs Spot S"), use_container_width=True)
                
                

            st.caption(
                # "Typical shapes: Price↑ with S for calls (↓ for puts). "
                "Delta approaches 1 (calls) when deep ITM; 0 when deep OTM. "
                "Gamma peaks near ATM. Vega is largest near ATM and falls in the tails."
            )

        # --- Tab 2: Vs Volatility (σ) ---
        with tab2:
            st.markdown("### Sensitivity to Volatility (σ)")
            vol_min = st.slider("Min σ (%)", 1, 100, 5) / 100.0
            vol_max = st.slider("Max σ (%)", 5, 300, 100) / 100.0
            if vol_max <= vol_min:
                st.warning("Max σ must be greater than Min σ.")
            else:
                df_v = sweep_over_vol(S, K, T, r, kind, q, n_points=n_pts, vol_min=vol_min, vol_max=vol_max)
                st.plotly_chart(px.line(df_v, x="Sigma", y="Price", title="Option Price vs Volatility σ"), use_container_width=True)
                st.caption("Black–Scholes price is increasing and convex in σ (for European options).")

        # --- Tab 3: Vs Time (T) ---
        with tab3:
            st.markdown("### Sensitivity to Time to Expiry (T)")
            min_days = st.slider("Min days", 1, 180, 1)
            max_days = st.slider("Max days", 5, 720, min(365,days))
            if max_days <= min_days:
                st.warning("Max days must be greater than Min days.")
            else:
                df_t = sweep_over_time(S, K, r, sigma, kind, q, n_points=n_pts, days_min=min_days, days_max=max_days)
                c1_, c2_ = st.columns(2)
                with c1_:
                    st.plotly_chart(px.line(df_t, x="Days", y="Price", title="Option Price vs Time (days)"), use_container_width=True)
                with c2_:
                    st.plotly_chart(px.line(df_t, x="Days", y="Theta_per_day", title="Theta (per day) vs Time (days)"), use_container_width=True)
                st.caption(
                    "Theta is typically negative (time decay). Shapes can vary near expiry or deep ITM/OTM."
                )

        # --- Tab 4: Call–Put Parity ---
        with tab4:
            st.markdown("### Call–Put Parity")
            call_px = bs_price(S, K, T, r, sigma, "call", q)
            put_px  = bs_price(S, K, T, r, sigma, "put",  q)
            lhs = call_px - put_px
            rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

            colp1, colp2, colp3 = st.columns(3)
            colp1.metric("Call − Put", f"{lhs:.6f}")
            colp2.metric("S e^(−qT) − K e^(−rT)", f"{rhs:.6f}")
            colp3.metric("Difference", f"{abs(lhs - rhs):.6e}")

            st.caption("Parity should hold up to tiny floating-point noise.")

        # --- Tab 5: Inputs Echo (for reproducibility) ---
        with tab5:
            st.write(
                pd.Series(
                    {
                        "S": S,
                        "K": K,
                        "T (years)": T,
                        "Days to expiry": days,
                        "r (annual)": r,
                        "q (annual)": q,
                        "σ (annual)": sigma,
                        "Type": kind,
                    }
                )
            )

        # --- Tab 6: Smile Plot (if AAPL prefilled) ---
        with tab6: 
            

            try:
                st.markdown(f"### Smile Plot for {tkr} Calls")
                calls = fetch_option_chain_ind(tkr)
                # use mid price where possible
                valid = (calls["bid"] > 0) & (calls["ask"] > 0)
                calls.loc[valid, "mid"] = (calls.loc[valid, "bid"] + calls.loc[valid, "ask"]) / 2.0
                calls.loc[~valid, "mid"] = calls.loc[~valid, "lastPrice"]

                T_chain = calls.iloc[0]["T"]  # use the first row's T (assumes all calls have same T)
                S = calls.iloc[0]["UnderlyingPrice"]  # use the first row's UnderlyingPrice


                # compute your IV per strike
                iv_me = []
                for _, row in calls.iterrows():
                    try:
                        iv_me.append(implied_vol_newton(float(row["mid"]), S, float(row["strike"]), T_chain, r, "call", q))
                    except Exception:
                        iv_me.append(np.nan)
                calls["iv_me"] = iv_me

                st.write("Nearest \(T\) (years):", T_chain)
                st.plotly_chart(
                    px.scatter(calls, x="strike", y="impliedVolatility", title="Market IV vs Strike (NSE)"),
                    use_container_width=True
                )
                st.plotly_chart(
                    px.scatter(calls, x="strike", y="iv_me", title="My IV vs Strike"),
                    use_container_width=True
                )
                callsLong = calls.melt(
                    id_vars="strike",
                    value_vars = ["iv_me", "impliedVolatility"],
                    var_name = "IV_Type",
                    value_name="IV_Value"
                )
                st.plotly_chart(
                    px.scatter(callsLong, x="strike", y="IV_Value", color="IV_Type", title="Consolidated IV vs Strike",
                                labels={"strike": "Strike Price", "IV_Value": "Implied Volatility (%)"},
                                hover_data=["IV_Type", "strike", "IV_Value"]
                    ),
                    use_container_width=True
                )
                st.caption("Small diffs are normal: quotes, dividends, and microstructure.")
            except Exception as e:
                st.warning(f"Could not fetch/plot smile: {e}")


    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.info("Set your inputs in the sidebar, then click **Compute**.")