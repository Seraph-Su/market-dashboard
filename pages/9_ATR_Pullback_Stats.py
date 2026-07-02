import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# ── ATR 狀態分箱（固定門檻，跨個股可比較）────────────────────────
ATR_BINS = [
    ("深收縮",  -np.inf, 0.90, "#4ade80"),
    ("收縮",     0.90,   1.00, "#86efac"),
    ("正常",     1.00,   1.10, "#94a3b8"),
    ("偏擴張",   1.10,   1.25, "#fbbf24"),
    ("急擴張",   1.25,  np.inf, "#f87171"),
]
POS_BINS = [
    ("高點附近（≤5%）",   -0.05, 10.0),
    ("回檔中（5~15%）",   -0.15, -0.05),
    ("深回檔（>15%）",    -10.0, -0.15),
]
QUICK = ["GLW", "NVDA", "AVGO", "AMD", "TSM", "MU", "VRT", "ANET", "PLTR", "AAPL"]
DEFAULT_POOL = ("NVDA, AVGO, AMD, TSM, MU, MRVL, ASML, AMAT, LRCX, KLAC, "
                "ANET, VRT, SMCI, DELL, ORCL, PLTR, APP, CRWD, CEG")
MIN_OWN_SAMPLES = 750   # 自身樣本日低於此數 → 預設改用股票池橫斷面統計

# ── 資料與計算 ────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ohlc(ticker: str, period: str) -> pd.DataFrame | None:
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "High", "Low"]].dropna()
    return df if len(df) >= 280 else None

def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    prev = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _atr_label(x: float) -> str:
    for name, lo, hi, _ in ATR_BINS:
        if lo <= x < hi:
            return name
    return "正常"

def _pos_label(d: float) -> str:
    for name, lo, hi in POS_BINS:
        if lo <= d < hi:
            return name
    return POS_BINS[-1][0]

def build_samples(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """每個交易日一個樣本：當日 ATR 擴張度、距高點位置、未來 horizon 日的回檔結果"""
    cl = df["Close"]
    clv = cl.values
    exp_ratio = (wilder_atr(df, 14) / wilder_atr(df, 50)).values
    dist_high = (cl / cl.rolling(252).max() - 1).values
    rows = []
    for i in range(252, len(clv) - horizon):
        fwd = clv[i + 1: i + horizon + 1]
        mae = min(fwd.min() / clv[i] - 1, 0)
        rows.append({
            "atr_state": _atr_label(exp_ratio[i]),
            "pos_state": _pos_label(dist_high[i]),
            "mae":  mae,
            "pb5":  mae <= -0.05,
            "pb10": mae <= -0.10,
            "fwd":  clv[i + horizon] / clv[i] - 1,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=86400, show_spinner=False)
def build_pool_samples(tickers: tuple, horizon: int, period: str = "10y") -> pd.DataFrame:
    """橫斷面彙總：把整個股票池的每日樣本疊起來，供新上市股借用統計"""
    raw = yf.download(list(tickers), period=period, interval="1d",
                      auto_adjust=True, progress=False)
    frames = []
    for t in tickers:
        try:
            d = pd.DataFrame({"Close": raw["Close"][t], "High": raw["High"][t],
                              "Low": raw["Low"][t]}).dropna()
            if len(d) >= 400:
                frames.append(build_samples(d, horizon))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def state_table(samples: pd.DataFrame, pos_state: str,
                cur_atr_state: str | None) -> pd.DataFrame:
    sub = samples[samples["pos_state"] == pos_state]
    rows = []
    for name, *_ in ATR_BINS:
        g = sub[sub["atr_state"] == name]
        mark = " ◀ 現在" if name == cur_atr_state else ""
        if len(g) == 0:
            rows.append({"ATR 狀態": name + mark, "樣本日": 0, "回檔≥5%": "—",
                         "回檔≥10%": "—", "MAE 中位": "—", "期末報酬中位": "—"})
            continue
        rows.append({
            "ATR 狀態":  name + mark,
            "樣本日":    len(g),
            "回檔≥5%":   f"{g['pb5'].mean() * 100:.0f}%",
            "回檔≥10%":  f"{g['pb10'].mean() * 100:.0f}%",
            "MAE 中位":  f"{g['mae'].median() * 100:.1f}%",
            "期末報酬中位": f"{g['fwd'].median() * 100:+.1f}%",
        })
    return pd.DataFrame(rows)

# ── Page ──────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🌡️ ATR 狀態 vs 未來回檔機率")
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem'>"
        "統計單一個股在各種 ATR 擴張度（ATR14 ÷ ATR50）× 距高點位置下，"
        "未來 N 日發生回檔的歷史機率，並標示目前狀態 &nbsp;｜&nbsp; 資料每日自動更新"
        "</span>",
        unsafe_allow_html=True,
    )
with col_refresh:
    if st.button("🔄 重新計算", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    ticker_raw = st.text_input("股票代號", placeholder="GLW / 2330.TW", key="ap_ticker")
with c2:
    years = st.selectbox("統計期間", ["10y", "5y", "max"], index=0,
                         help="單一個股需要較長歷史才有足夠樣本；10 年約 2,500 個交易日")
with c3:
    horizon = st.selectbox("觀察期（交易日）", [20, 60], index=0)

chip_cols = st.columns(len(QUICK))
chip_clicked = None
for col, t in zip(chip_cols, QUICK):
    with col:
        if st.button(t, key=f"ap_chip_{t}", use_container_width=True):
            chip_clicked = t
ticker = (chip_clicked or ticker_raw.strip().upper()
          or st.session_state.get("ap_ticker_val", ""))
if chip_clicked:
    st.session_state["ap_ticker_val"] = chip_clicked

if not ticker:
    st.markdown(
        "<div style='text-align:center;padding:3rem;color:#334155'>"
        "<div style='font-size:2.5rem'>🌡️</div>"
        "<div style='font-size:1rem;margin-top:1rem'>輸入股票代號或點選常用清單</div>"
        "</div>", unsafe_allow_html=True)
    st.stop()

try:
    with st.spinner(f"抓取 {ticker} {years} 資料並統計中…"):
        df = fetch_ohlc(ticker, years)
    if df is None:
        st.error(f"找不到 {ticker} 的資料或歷史不足 400 日。台股請加 .TW。")
        st.stop()

    # 目前狀態
    atr14 = wilder_atr(df, 14)
    atr50 = wilder_atr(df, 50)
    cl = df["Close"]
    exp_now = float(atr14.iloc[-1] / atr50.iloc[-1])
    atr_pct_now = float(atr14.iloc[-1] / cl.iloc[-1])
    dist_now = float(cl.iloc[-1] / cl.rolling(252, min_periods=60).max().iloc[-1] - 1)
    cur_atr_state = _atr_label(exp_now)
    cur_pos_state = _pos_label(dist_now)
    exp_series = atr14 / atr50
    exp_pctile = float((exp_series < exp_now).mean() * 100)
    state_color = dict((n, c) for n, _, _, c in ATR_BINS)[cur_atr_state]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("收盤", f"{cl.iloc[-1]:,.2f}")
    m2.metric("ATR14（%）", f"{atr_pct_now * 100:.1f}%")
    m3.metric("ATR 擴張度", f"{exp_now:.2f}",
              help="ATR14 ÷ ATR50。<1 收縮、>1.25 急擴張")
    m4.metric("擴張度歷史分位", f"P{exp_pctile:.0f}")
    m5.metric("距 52 週高點", f"{dist_now * 100:.1f}%")

    st.markdown(
        f"<span style='background:{state_color}22;color:{state_color};"
        f"padding:4px 12px;border-radius:10px;font-size:0.85rem;font-weight:700'>"
        f"目前狀態：{cur_atr_state} × {cur_pos_state}</span>"
        f"<span style='color:#475569;font-size:0.75rem'>"
        f"&nbsp;&nbsp;截至 {df.index[-1].date()}，共 {len(df)} 個交易日</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 統計來源：自身歷史 or 股票池橫斷面（新上市股樣本不足時）
    own_n = max(0, len(df) - 252 - horizon)
    short_hist = own_n < MIN_OWN_SAMPLES
    if short_hist:
        st.warning(
            f"⚠️ {ticker} 自身可統計樣本僅 {own_n} 個交易日（門檻 {MIN_OWN_SAMPLES}），"
            f"單股統計不可靠，已預設改用同類股票池的橫斷面統計。")
    use_pool = st.checkbox(
        f"用同類股票池橫斷面統計（自身樣本 {own_n} 日）",
        value=short_hist,
        help="狀態（擴張度、距高點）仍用該股自己計算；各狀態格的回檔機率改為借用整個池的歷史。前提：池內個股與該股屬同類型。")
    if use_pool:
        pool_text = st.text_input("統計用股票池（逗號分隔）", value=DEFAULT_POOL)
        pool = tuple(sorted({t.strip().upper() for t in pool_text.split(",")
                             if t.strip() and t.strip().upper() != ticker}))
        with st.spinner(f"下載 {len(pool)} 檔股票池資料中（快取至明日）…"):
            samples = build_pool_samples(pool, horizon)
        src_note = f"統計來源：股票池 {len(pool)} 檔 × 10 年橫斷面，共 {len(samples)} 個樣本日"
    else:
        samples = build_samples(df, horizon)
        src_note = f"統計來源：{ticker} 自身歷史，共 {len(samples)} 個樣本日"

    if samples.empty:
        st.error("樣本不足，無法統計。請改用股票池統計或加長統計期間。")
        st.stop()
    st.markdown(f"<span style='color:#475569;font-size:0.75rem'>{src_note}</span>",
                unsafe_allow_html=True)

    # 統計表：三個位置狀態各一張，目前所在的排最前
    pos_order = sorted([p[0] for p in POS_BINS],
                       key=lambda p: 0 if p == cur_pos_state else 1)
    for pos in pos_order:
        is_cur = pos == cur_pos_state
        tag = ("<span style='color:#60a5fa;font-size:0.75rem;font-weight:700'>"
               "&nbsp;← 目前位置</span>") if is_cur else ""
        st.markdown(f"#### {pos}{tag}", unsafe_allow_html=True)
        tbl = state_table(samples, pos, cur_atr_state if is_cur else None)
        st.dataframe(tbl.set_index("ATR 狀態"), use_container_width=True)
        n_small = tbl["樣本日"].astype(int)
        if (n_small[n_small > 0] < 30).any():
            st.markdown(
                "<span style='color:#854d0e;font-size:0.68rem'>"
                "⚠ 部分格子樣本日 < 30，該格數字僅供參考</span>",
                unsafe_allow_html=True)

    st.markdown(
        f"<div style='color:#334155;font-size:0.68rem;margin-top:4px'>"
        f"回檔≥5%／≥10% ＝ 未來 {horizon} 日內收盤曾比當日低 5%／10% 的機率；"
        f"MAE ＝ 期間最深回落；期末報酬 ＝ 第 {horizon} 日收盤相對當日。"
        f"樣本為每日滑動視窗（有重疊，自相關會讓數字比表面上更不確定）。"
        f"位置狀態要分開看的原因：回檔本身就會撐大 ATR，混在一起會把"
        f"「已經在跌」誤讀成「ATR 大所以要跌」。"
        f"</div>", unsafe_allow_html=True,
    )
    st.markdown("---")

    # 走勢圖：價格 + ATR 擴張度
    st.markdown("#### 價格與 ATR 擴張度走勢（近 2 年）")
    tail = min(len(df), 500)
    d2 = df.tail(tail)
    e2 = exp_series.tail(tail)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d2.index, y=d2["Close"], name="收盤", yaxis="y1",
        line=dict(color="#e2e8f0", width=1.4)))
    fig.add_trace(go.Scatter(
        x=e2.index, y=e2, name="ATR14/ATR50", yaxis="y2",
        line=dict(color="#fbbf24", width=1.2)))
    fig.add_hline(y=1.25, line=dict(color="#f87171", width=1, dash="dot"),
                  yref="y2", annotation_text="急擴張 1.25",
                  annotation_font=dict(size=10, color="#f87171"))
    fig.add_hline(y=1.00, line=dict(color="#4ade80", width=1, dash="dot"),
                  yref="y2", annotation_text="收縮 1.00",
                  annotation_font=dict(size=10, color="#4ade80"))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=8),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(orientation="h", font=dict(size=10, color="#94a3b8")),
        xaxis=dict(showgrid=False, color="#475569", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1e293b", color="#475569", tickfont=dict(size=10)),
        yaxis2=dict(overlaying="y", side="right", color="#854d0e",
                    tickfont=dict(size=10), showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("📖 指標說明與用法"):
        st.markdown(f"""
**ATR（Average True Range，平均真實區間）**：真實區間取「當日高低差、高點與昨收差、低點與昨收差」三者最大值（涵蓋跳空），再取平均。ATR14 ÷ ATR50 衡量短期波動相對中期的擴張程度。

**分箱門檻（固定，跨個股可比）**

| ATR 狀態 | 擴張度 | 一般含義 |
|---|---|---|
| 深收縮 | < 0.90 | 波動極低，常見於盤整末端 |
| 收縮 | 0.90 ~ 1.00 | 平靜上行，歷史上回檔機率最低 |
| 正常 | 1.00 ~ 1.10 | 基準狀態 |
| 偏擴張 | 1.10 ~ 1.25 | 波動升溫 |
| 急擴張 | > 1.25 | 未來路徑最顛簸，深回檔機率明顯升高 |

**與加碼系統的搭配**：在「高點附近 × 收縮」狀態突破加碼最安全；「急擴張」狀態要嘛縮碼、要嘛把停損按 ATR 放寬（ATR 制部位會自動做到）。注意 ATR 擴張預測的是「顛簸程度」而非「趨勢方向」——急擴張時期末報酬中位數通常仍為正。

⚠️ 歷史統計不代表未來，且不同 regime（多頭/熊市）下的機率差異很大，本頁不構成投資建議。
""")

except Exception as e:
    st.error(f"計算失敗：{e}")
