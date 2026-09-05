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
    ("高點附近（≤5%）",    -0.05, 10.0),
    ("距高點 5~15%",      -0.15, -0.05),
    ("距高點 >15%",       -10.0, -0.15),
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
    """每個交易日一個樣本：當日 ATR 擴張度、距高點位置、趨勢方向、未來 horizon 日的回檔結果"""
    cl = df["Close"]
    clv = cl.values
    exp_ratio = (wilder_atr(df, 14) / wilder_atr(df, 50)).values
    dist_high = (cl / cl.rolling(252).max() - 1).values
    ema60 = cl.ewm(span=60, adjust=False).mean().values
    rows = []
    for i in range(252, len(clv) - horizon):
        fwd = clv[i + 1: i + horizon + 1]
        mae = min(fwd.min() / clv[i] - 1, 0)
        rows.append({
            "atr_state": _atr_label(exp_ratio[i]),
            "pos_state": _pos_label(dist_high[i]),
            "trend":     "上行" if clv[i] > ema60[i] else "下行",
            "mae":  mae,
            "pb5":  mae <= -0.05,
            "pb10": mae <= -0.10,
            "fwd":  clv[i + horizon] / clv[i] - 1,
        })
    return pd.DataFrame(rows)
def find_pullbacks(df: pd.DataFrame, atr: pd.Series, min_depth: float = 0.03) -> list:
    """已完成的回檔事件（創高→回落→再創高），深度以 % 與 ATR 倍數表示。劇本徽章用。"""
    cl = df["Close"].values
    atr_v = atr.values
    peak_i, trough_i, out = 0, 0, []
    for i in range(1, len(cl)):
        if cl[i] >= cl[peak_i]:
            depth = 1 - cl[trough_i] / cl[peak_i]
            if depth >= min_depth:
                a = atr_v[peak_i]
                out.append({"depth": depth,
                            "depth_atr": (cl[peak_i] - cl[trough_i]) / a if a > 0 else np.nan})
            peak_i, trough_i = i, i
        elif cl[i] < cl[trough_i]:
            trough_i = i
    return out
def pullback_episodes(df: pd.DataFrame, min_depth: float = 0.03) -> list:
    """所有回檔事件（含尚未收復的）。resolved=True 表示已重新創高，
    recover_days ＝ 從前高到重新創高的交易日數（水下時間）。"""
    cl = df["Close"].values
    peak_i, trough_i, eps = 0, 0, []
    for i in range(1, len(cl)):
        if cl[i] >= cl[peak_i]:
            depth = 1 - cl[trough_i] / cl[peak_i]
            if depth >= min_depth:
                eps.append({"depth": depth, "recover_days": i - peak_i,
                            "resolved": True})
            peak_i, trough_i = i, i
        elif cl[i] < cl[trough_i]:
            trough_i = i
    depth = 1 - cl[trough_i] / cl[peak_i]      # 尾端未了結的回檔
    if depth >= min_depth:
        eps.append({"depth": depth, "recover_days": None, "resolved": False})
    return eps
def recovery_by_depth(all_eps: list,
                      thresholds=(0.05, 0.08, 0.10, 0.15, 0.20, 0.30)) -> pd.DataFrame:
    """限時收復率：P(N 個交易日內收復前高 | 回檔曾觸及深度 ≥ X)。
    「最終收復」對現在位於高點附近的股票是套套邏輯（過去回檔按定義都收復了），
    限時版把拖太久的解套計為失敗，才有辨識度。"""
    rows = []
    for x in thresholds:
        hit = [e for e in all_eps if e["depth"] >= x]
        if not hit:
            continue
        n = len(hit)
        w60  = sum(1 for e in hit if e["resolved"] and e["recover_days"] <= 60)
        w120 = sum(1 for e in hit if e["resolved"] and e["recover_days"] <= 120)
        ever = sum(e["resolved"] for e in hit)
        rows.append({
            "回檔觸及深度": f"≥ {x*100:.0f}%",
            "事件數": n,
            "60日內收復": f"{w60 / n * 100:.0f}%",
            "120日內收復": f"{w120 / n * 100:.0f}%",
            "最終收復": f"{ever / n * 100:.0f}%",
        })
    return pd.DataFrame(rows)
@st.cache_data(ttl=86400, show_spinner=False)
def build_pool_samples(tickers: tuple, horizon: int, period: str = "10y") -> pd.DataFrame:
    """橫斷面彙總：把整個股票池的每日樣本疊起來，供新上市股借用統計。
    分批下載＋單檔補抓，避免整批被 Yahoo 限流時全軍覆沒。"""
    frames, done = [], set()
    BATCH = 6
    for i in range(0, len(tickers), BATCH):
        batch = list(tickers[i:i + BATCH])
        try:
            raw = yf.download(batch, period=period, interval="1d",
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                continue
            for t in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        d = pd.DataFrame({"Close": raw["Close"][t],
                                          "High": raw["High"][t],
                                          "Low": raw["Low"][t]}).dropna()
                    else:
                        d = raw[["Close", "High", "Low"]].dropna()
                    if len(d) >= 400:
                        frames.append(build_samples(d, horizon))
                        done.add(t)
                except Exception:
                    pass
        except Exception:
            pass
    # 缺的單檔補抓一次
    for t in set(tickers) - done:
        try:
            d = yf.download(t, period=period, interval="1d",
                            auto_adjust=True, progress=False)
            if d is None or d.empty:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d[["Close", "High", "Low"]].dropna()
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
    ema60_now = float(cl.ewm(span=60, adjust=False).mean().iloc[-1])
    cur_trend = "上行" if float(cl.iloc[-1]) > ema60_now else "下行"
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
        f"目前狀態：{cur_atr_state} × {cur_pos_state} × {cur_trend}"
        f"{'（低點回升）' if cur_trend == '上行' and cur_pos_state != POS_BINS[0][0] else ''}"
        f"{'（高點回落）' if cur_trend == '下行' and cur_pos_state != POS_BINS[0][0] else ''}</span>"
        f"<span style='color:#475569;font-size:0.75rem'>"
        f"&nbsp;&nbsp;截至 {df.index[-1].date()}，共 {len(df)} 個交易日</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### 🎯 ATR 停損與機械加碼計算器")
    st.markdown(
        "<span style='color:#64748b;font-size:0.75rem'>"
        "追蹤停損＝進場後最高收盤 − N×ATR14；機械加碼＝收盤每高於上次進場 2×ATR14 加一筆 0.5R"
        "（回測：追蹤 3ATR 有效、2ATR 轉負；僅適用半年 ≥150% 動能股與修復型產業龍頭）"
        "</span>", unsafe_allow_html=True)

    # 回測同款 ATR（14 日簡單平均；與本頁 Wilder 版差異約 5~10%，規則一律用此版）
    _tr = pd.concat([df["High"] - df["Low"],
                     (df["High"] - cl.shift(1)).abs(),
                     (df["Low"] - cl.shift(1)).abs()], axis=1).max(axis=1)
    _atr14 = _tr.rolling(14).mean()
    _atr5 = _tr.rolling(5).mean()
    _atr_now = float(_atr14.iloc[-1])
    _px = float(cl.iloc[-1])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _entry_px = st.number_input("進場價（0 = 用最新收盤）", min_value=0.0, value=0.0,
                                    step=0.01, format="%.2f", key="atr_entry_px")
    with k2:
        _entry_date = st.date_input("進場日（算進場後最高收盤）", value=None, key="atr_entry_date",
                                    help="留空 = 用近 63 個交易日最高收盤")
    with k3:
        _r_usd = st.number_input("R（美元，帳戶 1%）", min_value=1.0, value=930.0, step=10.0,
                                 key="atr_r_usd")
    with k4:
        _last_add = st.number_input("上次加碼價（0 = 從進場價起算）", min_value=0.0, value=0.0,
                                    step=0.01, format="%.2f", key="atr_last_add")
    k5, k6 = st.columns(2)
    with k5:
        _trail_n = st.select_slider("追蹤停損寬度（×ATR14）", options=[2.0, 2.5, 3.0, 3.5, 4.0],
                                    value=3.0, key="atr_trail_n")
    with k6:
        _step_n = st.select_slider("加碼間距（×ATR14）", options=[1.5, 2.0, 2.5, 3.0, 4.0],
                                   value=2.0, key="atr_step_n")

    _entry = _entry_px if _entry_px > 0 else _px
    if _entry_date is not None:
        _since = cl[cl.index >= pd.Timestamp(_entry_date)]
        _peak = float(_since.max()) if len(_since) else float(cl.iloc[-63:].max())
        _peak_date = _since.idxmax().date() if len(_since) else cl.iloc[-63:].idxmax().date()
    else:
        _peak = float(cl.iloc[-63:].max()); _peak_date = cl.iloc[-63:].idxmax().date()

    # ── 停損 ──
    _trail = _peak - _trail_n * _atr_now
    _trail2 = _peak - 2.0 * _atr_now
    # 最近前波支撐（前後 3 日最低的擺盪低點，且低於現價 5% 以上）—— 基本倉停損參考
    _lo = df["Low"]
    _swings = [(_lo.index[i].date(), float(_lo.iloc[i]))
               for i in range(len(_lo) - 4, max(len(_lo) - 120, 3), -1)
               if _lo.iloc[i] == _lo.iloc[i - 3:i + 4].min() and _lo.iloc[i] < _px * 0.95]
    _swing = _swings[0] if _swings else None

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("ATR14（回測同款）", f"{_atr_now:,.2f}", f"{_atr_now / _px * 100:.1f}% of price",
              delta_color="off")
    s2.metric(f"追蹤停損 {_trail_n:g}×ATR", f"{_trail:,.2f}",
              f"{(_trail / _px - 1) * 100:+.1f}% vs 現價", delta_color="off",
              help=f"進場後最高收盤 {_peak:,.2f}（{_peak_date}）− {_trail_n:g}×{_atr_now:.2f}。只往上移不往下。")
    s3.metric("對照：2×ATR", f"{_trail2:,.2f}", f"{(_trail2 / _px - 1) * 100:+.1f}%",
              delta_color="off", help="回測中 2ATR 追蹤是唯一讓機械加碼轉負的參數，僅供對照。")
    s4.metric("最近前波支撐（基本倉停損參考）",
              f"{_swing[1]:,.2f}" if _swing else "—",
              f"{(_swing[1] / _px - 1) * 100:+.1f}%（{_swing[0]}）" if _swing else "近 120 日無合格擺盪低點",
              delta_color="off")
    if _px <= _trail:
        st.error(f"⚠️ 現價 {_px:,.2f} 已在 {_trail_n:g}×ATR 追蹤停損 {_trail:,.2f} 之下——加碼單應已出場。")

    # ── 進場後的歷史 T 觸發（若填了進場日）──
    _hist = []
    if _entry_date is not None and len(_since) > 1:
        _last = _entry
        _i0 = cl.index.get_loc(_since.index[0])
        for _i in range(_i0, len(cl)):
            _a = _atr14.iloc[_i]
            if pd.notna(_a) and cl.iloc[_i] >= _last + _step_n * _a:
                _hist.append({"訊號日（收盤）": cl.index[_i].date(), "收盤": round(float(cl.iloc[_i]), 2),
                              "當日 ATR14": round(float(_a), 2),
                              "今值 vs 該價": f"{(_px / float(cl.iloc[_i]) - 1) * 100:+.1f}%"})
                _last = float(cl.iloc[_i])
    _today_is_trigger = bool(_hist) and _hist[-1]["訊號日（收盤）"] == cl.index[-1].date()

    # ── 加碼梯 ──
    # 起算價：手填上次加碼價 > 進場後最後一次歷史觸發價 > 進場價
    _ref = _last_add if _last_add > 0 else (float(_hist[-1]["收盤"]) if _hist else _entry)
    _add_sh = int(0.5 * _r_usd / (_trail_n * _atr_now)) if _atr_now > 0 else 0
    _ladder = []
    _p = _ref
    for _k in range(1, 4):
        _p = _p + _step_n * _atr_now
        _ladder.append({"第幾筆": f"#{_k}", "觸發價（收盤 ≥）": round(_p, 2),
                        "距現價": f"{(_p / _px - 1) * 100:+.1f}%",
                        "股數（0.5R ÷ 3ATR）": _add_sh, "名目（美元）": f"{_add_sh * _p:,.0f}"})
    _next = _ladder[0]["觸發價（收盤 ≥）"]
    st.markdown(
        f"<div style='margin:6px 0 4px'>"
        f"<span style='color:#94a3b8;font-size:0.8rem'>加碼起算價 {_ref:,.2f}　→　"
        f"<b style='color:#e2e8f0'>下一個加碼觸發：收盤 ≥ {_next:,.2f}</b>"
        f"（距現價 {(_next / _px - 1) * 100:+.1f}%）　每筆 0.5R = ${0.5 * _r_usd:,.0f} ÷ "
        f"({_trail_n:g}×{_atr_now:.2f}) = <b>{_add_sh} 股</b>，隔日開盤市價進</span></div>",
        unsafe_allow_html=True)
    if _today_is_trigger:
        st.success(f"🔔 今日收盤 {_px:,.2f} 觸發機械加碼（隔日開盤進 {_add_sh} 股，加碼單追蹤停損 {_trail:,.2f}）")
    elif _last_add > 0 and _px >= _next:
        st.success(f"🔔 今日收盤 {_px:,.2f} ≥ 觸發價 {_next:,.2f}：機械加碼條件成立（隔日開盤進 {_add_sh} 股，停損 {_trail:,.2f}）")
    st.dataframe(pd.DataFrame(_ladder).set_index("第幾筆"), use_container_width=True)
    if _entry_date is not None:
        if _hist:
            st.markdown(f"<span style='color:#94a3b8;font-size:0.8rem'>自 {_entry_date} 起、進場價 {_entry:,.2f}，"
                        f"照規則會出現的加碼觸發（共 {len(_hist)} 次；錯過的不補，加碼梯已從最後一次觸發價起算）：</span>",
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(_hist).set_index("訊號日（收盤）"), use_container_width=True)
        else:
            st.markdown("<span style='color:#64748b;font-size:0.78rem'>進場後尚無加碼觸發。</span>",
                        unsafe_allow_html=True)

    # ── 回季線加碼（A）狀態 ──
    _ema60 = float(cl.ewm(span=60, adjust=False).mean().iloc[-1])
    _dev60 = _px / _ema60 - 1
    _ratio = float(_atr5.iloc[-1] / _atr14.iloc[-1]) if _atr_now > 0 else float("nan")
    _a_ok_pos = -0.03 <= _dev60 <= 0.06
    _a_ok_atr = _ratio < 1.0
    st.markdown(
        f"<div style='color:#94a3b8;font-size:0.78rem;margin-top:4px'>"
        f"回季線加碼（A）：距季線 {_dev60 * 100:+.1f}% {'✓' if _a_ok_pos else '✗'}（需 −3%～+6%）　"
        f"ATR5/ATR14 {_ratio:.2f} {'✓' if _a_ok_atr else '✗'}（需 <1）　"
        + ("<b style='color:#4ade80'>→ A 成立</b>" if (_a_ok_pos and _a_ok_atr)
           else ("<b style='color:#f87171'>→ 近季線但 ATR 急擴張，不加</b>" if (_a_ok_pos and _ratio >= 1.3)
                 else "→ 未成立"))
        + "　｜　A 與 T 並行：A 出現就做（位置最好），T 為常規引擎。單股名目上限：一般 10%、怪物股 20%；財報前只砍加碼單。"
        f"</div>", unsafe_allow_html=True)
    # ── 劇本判定（風險端三指標：波動水位與叢聚具持續性，可作分軌依據）──
    crack_share = float((exp_series.iloc[-252:] >= 1.25).mean())
    _pbs = find_pullbacks(df, atr14, 0.03)
    med_pb_atr = float(np.median([p["depth_atr"] for p in _pbs])) if _pbs else float("nan")
    if crack_share >= 0.5:
        pb_label, pb_color = "慢性風暴型", "#f87171"
        pb_desc = ("近一年過半日子處於急擴張——回檔加碼軌的條件基本不會湊齊。"
                   "劇本：ATR 定小倉進場、突破創高加小碼、移動停損抱趨勢、高點附近急擴張事件減碼。")
    elif crack_share >= 0.2 or atr_pct_now >= 0.06:
        pb_label, pb_color = "高波動輪轉型", "#fbbf24"
        pb_desc = ("狀態會輪轉但波動偏高——雙軌加碼皆可用，"
                   "倉位由 ATR 公式自動縮小、停損寬度對照下方回檔中位數（ATR 倍）。")
    else:
        pb_label, pb_color = "輪轉型", "#4ade80"
        pb_desc = "狀態正常輪轉——雙軌加碼皆可用，耐心等「低波動回檔到季線」的高勝率設定出現。"
    st.markdown(
        f"<div style='margin-top:6px'>"
        f"<span style='background:{pb_color}22;color:{pb_color};padding:4px 12px;"
        f"border-radius:10px;font-size:0.85rem;font-weight:700'>劇本：{pb_label}</span>"
        f"<span style='color:#475569;font-size:0.75rem'>&nbsp;&nbsp;"
        f"急擴張占比（近一年）{crack_share*100:.0f}%｜ATR% {atr_pct_now*100:.1f}%"
        + (f"｜回檔中位 {med_pb_atr:.1f}×ATR" if med_pb_atr == med_pb_atr else "")
        + f"<br>{pb_desc}</span></div>",
        unsafe_allow_html=True)
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
        if use_pool:
            st.error("股票池資料下載失敗（可能被 Yahoo 暫時限流）。"
                     "請稍等一分鐘後按「🔄 重新計算」再試。")
        else:
            st.error("樣本不足，無法統計。請勾選股票池統計或加長統計期間。")
        st.stop()
    # 趨勢濾網：區分「從高點跌下來」和「從低點漲回來」（距高點距離相同、含義相反）
    trend_opts = ["全部", "上行（EMA60 之上）", "下行（EMA60 之下）"]
    trend_sel = st.radio(
        "趨勢濾網", trend_opts, horizontal=True,
        index=1 if cur_trend == "上行" else 2,
        help="同樣是「距高點 -10%」，趨勢下行是從高點跌下來、上行是從低點漲回來，"
             "歷史統計差異很大。預設選該股目前的趨勢方向。")
    if trend_sel.startswith("上行"):
        samples = samples[samples["trend"] == "上行"]
    elif trend_sel.startswith("下行"):
        samples = samples[samples["trend"] == "下行"]
    st.markdown(f"<span style='color:#475569;font-size:0.75rem'>{src_note}"
                f"（趨勢濾網後 {len(samples)} 個）</span>",
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
    # ══ 回檔收復率（含未收復事件，無倖存者偏差）══════════════════
    cur_dd_pct = -dist_now
    st.markdown("#### 回檔收復率 vs 深度")
    eps = pullback_episodes(df, min_depth=0.03)
    if eps:
        rec_tbl = recovery_by_depth(eps)
        st.dataframe(rec_tbl.set_index("回檔觸及深度"), use_container_width=True)
        if cur_dd_pct > 0.03:
            hit = [e for e in eps if e["depth"] >= cur_dd_pct]
            if hit:
                n = len(hit)
                w60 = sum(1 for e in hit if e["resolved"] and e["recover_days"] <= 60)
                w120 = sum(1 for e in hit if e["resolved"] and e["recover_days"] <= 120)
                st.markdown(
                    f"<span style='color:#60a5fa;font-size:0.8rem;font-weight:700'>"
                    f"目前回檔 {cur_dd_pct*100:.1f}%：歷史上觸及此深度 {n} 次，"
                    f"60 日內收復 {w60/n*100:.0f}%、120 日內收復 {w120/n*100:.0f}%</span>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#334155;font-size:0.68rem'>"
            "回答「回檔觸及 X% 之後，N 個交易日內收復前高的機率」。"
            "重點看限時欄：「最終收復」對目前在高點附近的股票是套套邏輯——"
            "過去的回檔按定義都收復了（不然它現在不會在高點），"
            "限時版把拖太久的解套計為失敗，才能區分「很快回來」和「回得來但等不起」。"
            "含尚未收復的進行中事件，收復率為保守下限。統計用該股自身歷史，"
            "資料範圍跟隨上方「統計期間」設定，新上市股請謹慎解讀。"
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<span style='color:#64748b;font-size:0.78rem'>期間內沒有 ≥3% 的回檔事件。</span>",
            unsafe_allow_html=True)
    st.markdown("---")
    # ══ 各 ATR 狀態的報酬貢獻（判斷適不適合狀態調倉）════════════
    st.markdown("#### 各 ATR 狀態的報酬貢獻")
    r_ser = cl.pct_change()
    state_prev = exp_series.shift(1)   # 前一日狀態決定今日歸屬，避免前視
    last252 = set(df.index[-252:])
    contrib_rows = []
    for name, lo, hi, _c in ATR_BINS:
        mask = state_prev.apply(lambda x: pd.notna(x) and lo <= x < hi)
        r_all = r_ser[mask].dropna()
        r_1y = r_ser[mask & df.index.isin(last252)].dropna()
        contrib_rows.append({
            "ATR 狀態": name,
            "全期天數": len(r_all),
            "全期累積報酬": "{:+.0f}%".format(((1 + r_all).prod() - 1) * 100) if len(r_all) else "—",
            "近一年天數": len(r_1y),
            "近一年累積": "{:+.1f}%".format(((1 + r_1y).prod() - 1) * 100) if len(r_1y) else "—",
        })
    st.dataframe(pd.DataFrame(contrib_rows).set_index("ATR 狀態"),
                 use_container_width=True)
    st.markdown(
        "<div style='color:#334155;font-size:0.68rem'>"
        "判讀：這是<b>歷史描述，不是加減碼時刻表</b>。持續性檢驗（池內 18 檔、前後半段對照）"
        "顯示「哪個狀態賺錢」不延續（延續率 39%，隨機 33%，排序相關中位 -0.50）——"
        "照歷史分布客製狀態調倉已驗證無效。此表的正確用途：了解該股過去的行為模式、"
        "解釋近期報酬來源。ATR 狀態可持續預測的是<b>風險</b>（急擴張＝MAE 更深、路徑更顛），"
        "請用於碼數與停損寬度，勿用於預測哪個狀態會賺錢。歸屬以前一日狀態計算（無前視）。"
        "</div>", unsafe_allow_html=True,
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
