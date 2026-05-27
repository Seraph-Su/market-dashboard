import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from io import StringIO


# ── Fetch S&P 500 components ──────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500():
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0][["Symbol", "Security", "GICS Sector"]]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


# ── Run screener ──────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="篩選中，首次執行約需 1-2 分鐘…")
def run_screener():
    df_sp500 = fetch_sp500()
    tickers  = df_sp500["Symbol"].tolist()

    end   = datetime.today()
    start = end - timedelta(days=120)
    raw   = yf.download(
        tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )
    close = raw["Close"].copy()
    close.columns = [c[1] if isinstance(c, tuple) else c for c in close.columns]
    close = close.dropna(how="all")

    if len(close) < 65:
        raise ValueError("價格資料不足，無法計算 EMA60")

    ema60 = close.ewm(span=60, adjust=False).mean()
    cond_above = close.iloc[-1] > ema60.iloc[-1]
    ema_diff   = ema60.diff()
    slope_ok   = (ema_diff.iloc[-6:] > 0).all(axis=0)

    passed = cond_above & slope_ok
    passed_tickers = passed[passed].index.tolist()

    if not passed_tickers:
        return pd.DataFrame()

    rows = []
    info_map = df_sp500.set_index("Symbol")[["Security", "GICS Sector"]].to_dict("index")

    for sym in passed_tickers:
        try:
            fi = yf.Ticker(sym).fast_info
            mktcap = getattr(fi, "market_cap", None)
            if mktcap is None or mktcap < 2_000_000_000:
                continue
            price   = round(float(close[sym].iloc[-1]), 2)
            ema_val = round(float(ema60[sym].iloc[-1]), 2)
            dev     = round((price / ema_val - 1) * 100, 2)
            diffs = ema_diff[sym].dropna()
            streak = 0
            for v in reversed(diffs.tolist()):
                if v > 0:
                    streak += 1
                else:
                    break
            rows.append({
                "代號":        sym,
                "公司名稱":    info_map.get(sym, {}).get("Security", ""),
                "板塊":        info_map.get(sym, {}).get("GICS Sector", ""),
                "股價":        price,
                "EMA60":       ema_val,
                "乖離率":      f"{'+' if dev>=0 else ''}{dev}%",
                "連續向上天數": streak,
                "市值(億美金)": round(mktcap / 1e8, 1),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("連續向上天數", ascending=False).reset_index(drop=True)
    result.index += 1
    return result


# ── Page ──────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🔍 每日選股")
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem'>"
        "篩選條件：① EMA60 斜率連續向上 &gt;5 個工作天 &nbsp;｜&nbsp; "
        "② 股價在 EMA60 上方 &nbsp;｜&nbsp; "
        "③ 市值 &gt; 20 億美金 &nbsp;｜&nbsp; "
        "資料每日自動更新"
        "</span>",
        unsafe_allow_html=True,
    )
with col_refresh:
    if st.button("🔄 重新篩選", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

try:
    with st.spinner("篩選中，首次執行約需 1-2 分鐘…"):
        result = run_screener()

    as_of = datetime.today().strftime("%Y-%m-%d")

    if result.empty:
        st.info("今日無符合條件的股票。")
    else:
        st.markdown(
            f"<span style='color:#4ade80;font-size:0.9rem;font-weight:700'>"
            f"✅ 符合條件：{len(result)} 支股票</span>"
            f"<span style='color:#475569;font-size:0.75rem'> &nbsp;（截至 {as_of}）</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

        def highlight_dev(val):
            v = float(val.replace("%", "").replace("+", ""))
            if v > 10:
                return "color: #f87171"
            if v > 5:
                return "color: #fbbf24"
            return "color: #4ade80"

        styled = result.style.applymap(highlight_dev, subset=["乖離率"])
        st.dataframe(styled, use_container_width=True, height=min(600, 60 + len(result) * 35))

except Exception as e:
    st.error(f"篩選失敗：{e}")
