import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  .metric-card {
    background: #1a1f2e; border-radius: 10px; padding: 14px 16px;
    border: 1px solid #2d3748; margin-bottom: 8px;
  }
  .section-hdr {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #475569; margin-bottom: 6px; margin-top: 4px;
  }
  .result-card {
    background: #1a1f2e; border-radius: 10px; padding: 16px 18px;
    border: 1px solid #2d3748; margin-bottom: 10px;
  }
  .safe-card    { border-color: #16a34a !important; background: linear-gradient(135deg, #052e16, #0a3622) !important; }
  .caution-card { border-color: #d97706 !important; background: linear-gradient(135deg, #1c1500, #3a2800) !important; }
  .danger-card  { border-color: #dc2626 !important; background: linear-gradient(135deg, #2d0000, #450a0a) !important; }
  .info-label { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }
  .depth-badge { display: inline-block; padding: 3px 10px; border-radius: 10px; font-size: 0.72rem; font-weight: 700; }
  .depth-shallow { background: #14532d; color: #4ade80; }
  .depth-medium  { background: #451a03; color: #fbbf24; }
  .depth-deep    { background: #7f1d1d; color: #f87171; }
  .warn-red    { background:#450a0a; border:1px solid #dc2626; border-radius:8px; padding:10px 14px; color:#fca5a5; font-size:0.82rem; margin-bottom:10px; }
  .warn-yellow { background:#3a2800; border:1px solid #d97706; border-radius:8px; padding:10px 14px; color:#fcd34d; font-size:0.82rem; margin-bottom:10px; }
  .warn-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px; padding:10px 14px; color:#86efac; font-size:0.82rem; margin-bottom:10px; }
  .stress-row { display:flex; justify-content:space-between; padding:4px 0;
                border-bottom:1px solid #1e293b; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Data fetch ───────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock(ticker: str, years: int = 3):
    period = f"{years}y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty or len(df) < 60:
        return None
    close = df['Close'].squeeze().dropna()
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in close.index],
        'close': close.tolist(),
        'n': len(close),
    }

# ── Core calculations ────────────────────────────────────────────────────
def calc_drawdown_stats(close_arr, dates_arr):
    close = np.array(close_arr, dtype=float)
    n     = len(close)

    # 52-week rolling high
    win       = min(252, n)
    roll_high = pd.Series(close).rolling(win, min_periods=1).max().values
    dd_series = (close / roll_high - 1) * 100  # always <= 0

    current_price = float(close[-1])
    current_high  = float(roll_high[-1])
    current_dd    = float(dd_series[-1])

    # Identify historical drawdown episodes (trough per episode)
    troughs = []
    in_ep   = False
    trough  = 0.0

    for i in range(n):
        dd = dd_series[i]
        if not in_ep:
            if dd < -1.5:
                in_ep  = True
                trough = dd
        else:
            if dd < trough:
                trough = dd
            if dd > -0.5 or i == n - 1:
                troughs.append(trough)
                in_ep = False

    if in_ep:
        troughs.append(trough)

    troughs_arr = np.array(troughs) if troughs else np.array([])

    if len(troughs_arr) >= 3:
        p25 = float(np.percentile(troughs_arr, 25))   # deeper (more negative)
        p50 = float(np.percentile(troughs_arr, 50))   # median
        p75 = float(np.percentile(troughs_arr, 75))   # shallower (less negative)
        max_dd = float(np.min(troughs_arr))            # worst ever
        n_ep   = len(troughs_arr)
    else:
        # Fallback: use percentiles of the raw drawdown series
        valid = dd_series[dd_series < -1]
        if len(valid) > 0:
            p25 = float(np.percentile(valid, 25))
            p50 = float(np.percentile(valid, 50))
            p75 = float(np.percentile(valid, 75))
            max_dd = float(np.min(valid))
        else:
            p25 = -20.0; p50 = -10.0; p75 = -5.0; max_dd = -30.0
        n_ep = len(troughs_arr)

    return {
        'current_price': current_price,
        'current_high':  current_high,
        'current_dd':    current_dd,
        'p25': p25, 'p50': p50, 'p75': p75,
        'max_dd':  max_dd,
        'n_ep':    n_ep,
        'troughs': troughs_arr.tolist(),
        'dd_series': dd_series.tolist(),
    }

def classify_depth(current_dd, p25, p75):
    """p75 is shallower (less neg), p25 is deeper (more neg)"""
    if current_dd >= p75:
        return 'shallow'
    elif current_dd > p25:
        return 'medium'
    else:
        return 'deep'

def profit_multiplier(profit_pct):
    if profit_pct > 20:  return 1.00, "獲利 > 20%，安全墊充裕"
    if profit_pct > 10:  return 0.90, "獲利 10~20%，安全墊充足"
    if profit_pct >  5:  return 0.75, "獲利 5~10%，安全墊尚可"
    if profit_pct >  0:  return 0.50, "獲利 0~5%，安全墊偏薄，保守加碼"
    if profit_pct > -10: return 0.25, "目前虧損，壓低比例，謹慎為上"
    return 0.00, "虧損 > 10%，不建議加碼，避免越攤越深"

def calc_recommendation(depth, profit_pct, current_price, cost, current_shares):
    BASE = {'shallow': 0.25, 'medium': 0.50, 'deep': 1.00}
    base_ratio = BASE[depth]
    mult, mult_note = profit_multiplier(profit_pct)
    final_ratio = base_ratio * mult
    add_shares  = current_shares * final_ratio

    if add_shares > 0:
        current_value  = current_shares * cost
        add_value      = add_shares * current_price
        new_total      = current_shares + add_shares
        new_avg_cost   = (current_value + add_value) / new_total
    else:
        new_avg_cost = cost
        add_shares   = 0.0

    # Risk label
    if profit_pct > 10:
        risk, risk_label = 'safe',    '安全'
    elif profit_pct > 0:
        risk, risk_label = 'caution', '謹慎'
    else:
        risk, risk_label = 'danger',  '高風險'

    return {
        'base_ratio':  base_ratio,
        'mult':        mult,
        'mult_note':   mult_note,
        'final_ratio': final_ratio,
        'add_shares':  add_shares,
        'new_avg_cost': new_avg_cost,
        'risk':        risk,
        'risk_label':  risk_label,
    }

# ── Page header ──────────────────────────────────────────────────────────
st.markdown("## 📉 加碼比例計算器")
st.markdown(
    "<span style='color:#64748b;font-size:0.78rem'>"
    "輸入股票代號與持股資訊，依回檔深度與成本安全墊計算建議加碼比例"
    "</span>", unsafe_allow_html=True)
st.markdown("---")

# ── Input controls ───────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">輸入參數</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([2, 1.6, 1.6, 1])
with c1:
    ticker_raw = st.text_input(
        "股票代號", placeholder="AAPL / 2330.TW", key="pa_ticker")
with c2:
    cost = st.number_input(
        "持股成本（每股均價）", min_value=0.01, value=100.0, step=0.01, key="pa_cost")
with c3:
    shares = st.number_input(
        "現有持股數量（股）", min_value=1, value=100, step=1, key="pa_shares")
with c4:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    calc_btn = st.button("計算 →", type='primary', use_container_width=True)

QUICK = ['AAPL','NVDA','TSLA','MSFT','GOOGL','META','AMZN','CRWD','MU','2330.TW']
st.markdown('<div class="section-hdr" style="margin-top:6px">常用</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(QUICK))
chip_clicked = None
for col, t in zip(chip_cols, QUICK):
    with col:
        if st.button(t, key=f'pa_chip_{t}', use_container_width=True):
            chip_clicked = t

if chip_clicked:
    st.session_state['pa_ticker_val'] = chip_clicked
elif calc_btn and ticker_raw.strip():
    st.session_state['pa_ticker_val'] = ticker_raw.strip().upper()

ticker = st.session_state.get('pa_ticker_val', '')

if not ticker:
    st.markdown(
        "<div style='text-align:center;padding:4rem;color:#334155'>"
        "<div style='font-size:2.5rem'>📉</div>"
        "<div style='font-size:1rem;margin-top:1rem'>輸入股票代號後按計算，或點選常用清單</div>"
        "<div style='font-size:0.8rem;margin-top:0.4rem;color:#475569'>台股請加 .TW，例如 2330.TW</div>"
        "</div>", unsafe_allow_html=True)
    st.stop()

# ── Fetch & calculate ────────────────────────────────────────────────────
with st.spinner(f'抓取 {ticker} 資料中…'):
    data = fetch_stock(ticker, years=3)

if data is None:
    st.error(f'找不到 **{ticker}** 的資料。請確認代碼正確（台股請加 .TW，例如 2330.TW）。')
    st.stop()

stats   = calc_drawdown_stats(data['close'], data['dates'])
cp      = stats['current_price']
h52w    = stats['current_high']
cur_dd  = stats['current_dd']
profit  = (cp / cost - 1) * 100
depth   = classify_depth(cur_dd, stats['p25'], stats['p75'])
rec     = calc_recommendation(depth, profit, cp, cost, shares)

# ── Title row ────────────────────────────────────────────────────────────
st.markdown(
    f"### {ticker}&nbsp;"
    f"<span style='font-size:0.85rem;color:#64748b'>{data['dates'][0]} ～ {data['dates'][-1]}"
    f"　{data['n']} 個交易日</span>",
    unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("當前股價",   f"${cp:,.2f}")
with m2: st.metric("52週高點",   f"${h52w:,.2f}")
with m3: st.metric("距高點回檔", f"{cur_dd:.1f}%")
with m4: st.metric("距成本損益", f"{profit:+.1f}%")
with m5: st.metric("持股成本",   f"${cost:,.2f}")

st.markdown("---")

# ── Depth classification + recommendation ────────────────────────────────
DEPTH_INFO = {
    'shallow': ('淺回檔', 'depth-shallow', '回檔幅度低於歷史 75% 的修正事件'),
    'medium':  ('中等回檔', 'depth-medium', '回檔幅度介於歷史 25th～75th 百分位'),
    'deep':    ('深度回檔', 'depth-deep',   '回檔幅度超過歷史 75% 的修正事件'),
}
d_label, d_cls, d_desc = DEPTH_INFO[depth]
risk_card_cls  = {'safe': 'safe-card', 'caution': 'caution-card', 'danger': 'danger-card'}[rec['risk']]
risk_color_map = {'safe': '#4ade80',   'caution': '#fbbf24',       'danger': '#f87171'}
risk_color     = risk_color_map[rec['risk']]
profit_color   = '#4ade80' if profit > 10 else '#fbbf24' if profit > 0 else '#f87171'

left_col, right_col = st.columns([1.8, 1])

with left_col:
    # ── Drawdown depth panel ──
    st.markdown('<div class="section-hdr">回檔深度分析</div>', unsafe_allow_html=True)

    if stats['n_ep'] >= 3:
        hist_html = (
            f"<div style='display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px'>"
            f"<div><div class='info-label'>歷史中位回檔</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:#fbbf24'>{stats['p50']:.1f}%</div></div>"
            f"<div><div class='info-label'>淺（75th）</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:#4ade80'>{stats['p75']:.1f}%</div></div>"
            f"<div><div class='info-label'>深（25th）</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:#f87171'>{stats['p25']:.1f}%</div></div>"
            f"<div><div class='info-label'>歷史最深</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:#f87171'>{stats['max_dd']:.1f}%</div></div>"
            f"<div><div class='info-label'>回檔次數</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:#94a3b8'>{stats['n_ep']} 次</div></div>"
            f"</div>"
        )
    else:
        hist_html = (
            "<div style='color:#64748b;font-size:0.78rem;margin-bottom:14px'>"
            "⚠ 歷史回檔次數不足，統計採用替代方法，請謹慎解讀</div>"
        )

    vs_html = (
        f"<div style='display:flex;align-items:center;gap:20px'>"
        f"<div>"
        f"<div class='info-label'>當前回檔</div>"
        f"<div style='font-size:2.2rem;font-weight:800;color:#f87171;line-height:1.1'>{cur_dd:.1f}%</div>"
        f"</div>"
        f"<div style='font-size:1.4rem;color:#334155'>vs</div>"
        f"<div>"
        f"<div class='info-label'>歷史中位</div>"
        f"<div style='font-size:2.2rem;font-weight:800;color:#fbbf24;line-height:1.1'>{stats['p50']:.1f}%</div>"
        f"</div>"
        f"<div style='padding-left:20px;border-left:1px solid #2d3748'>"
        f"<span class='depth-badge {d_cls}'>{d_label}</span>"
        f"<div style='font-size:0.7rem;color:#64748b;margin-top:6px;max-width:190px'>{d_desc}</div>"
        f"</div>"
        f"</div>"
    )

    st.markdown(
        f"<div class='result-card'>{hist_html}{vs_html}</div>",
        unsafe_allow_html=True)

    # ── Profit margin panel ──
    pm_note = (
        "安全墊充足，加碼風險可控" if profit > 10 else
        "安全墊偏薄，建議保守加碼" if profit > 0 else
        "目前持倉虧損，加碼前請審慎評估"
    )
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='section-hdr' style='margin-top:0'>成本安全墊</div>"
        f"<div style='display:flex;align-items:baseline;gap:14px'>"
        f"<div style='font-size:2rem;font-weight:800;color:{profit_color}'>{profit:+.1f}%</div>"
        f"<div style='color:{profit_color};font-size:0.8rem'>{pm_note}</div>"
        f"</div>"
        f"<div style='font-size:0.72rem;color:#64748b;margin-top:6px'>"
        f"持股成本 ${cost:,.2f} → 當前 ${cp:,.2f}　｜　安全墊係數 {rec['mult']:.2f}（{rec['mult_note']}）</div>"
        f"</div>",
        unsafe_allow_html=True)

with right_col:
    # ── Recommended ratio ──
    ratio_display = f"{rec['final_ratio']*100:.0f}%" if rec['final_ratio'] > 0 else "0%"
    ratio_color   = risk_color if rec['final_ratio'] > 0 else '#f87171'

    st.markdown(
        f"<div class='result-card {risk_card_cls}'>"
        f"<div class='info-label'>建議加碼比例（相對持股）</div>"
        f"<div style='font-size:3.2rem;font-weight:800;color:{ratio_color};line-height:1.1;margin:6px 0'>"
        f"{ratio_display}</div>"
        f"<div style='font-size:0.72rem;color:#94a3b8'>"
        f"基準 {rec['base_ratio']*100:.0f}%（{d_label}）× 安全墊係數 {rec['mult']:.2f}"
        f"</div>"
        f"<div style='margin-top:10px;padding-top:10px;border-top:1px solid #334155'>"
        f"<span style='background:{risk_color}22;color:{risk_color};"
        f"padding:3px 10px;border-radius:10px;font-size:0.72rem;font-weight:700'>"
        f"風險標籤：{rec['risk_label']}</span>"
        f"</div>"
        f"<div style='margin-top:10px;background:#1c1400;border:1px solid #854d0e;"
        f"border-radius:6px;padding:7px 10px;color:#fde68a;font-size:0.72rem;font-weight:600'>"
        f"⚠️ 本加碼比例僅供參考，過去不代表未來"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True)

    # ── Post-add calculation ──
    if rec['add_shares'] > 0:
        add_cost  = rec['add_shares'] * cp
        new_total = shares + rec['add_shares']
        new_pnl   = (cp / rec['new_avg_cost'] - 1) * 100
        new_pnl_c = '#4ade80' if new_pnl >= 0 else '#f87171'

        st.markdown(
            f"<div class='result-card'>"
            f"<div class='section-hdr' style='margin-top:0'>加碼後試算</div>"
            f"<div class='stress-row'><span style='color:#64748b'>加碼股數</span>"
            f"<span style='color:#e2e8f0;font-weight:600'>{rec['add_shares']:.0f} 股</span></div>"
            f"<div class='stress-row'><span style='color:#64748b'>加碼金額</span>"
            f"<span style='color:#e2e8f0;font-weight:600'>${add_cost:,.0f}</span></div>"
            f"<div class='stress-row'><span style='color:#64748b'>加碼後總股數</span>"
            f"<span style='color:#e2e8f0;font-weight:600'>{new_total:.0f} 股</span></div>"
            f"<div class='stress-row'><span style='color:#64748b'>加碼後新均價</span>"
            f"<span style='color:#4ade80;font-weight:700'>${rec['new_avg_cost']:,.2f}</span></div>"
            f"<div class='stress-row' style='border:none'><span style='color:#64748b'>新均價現損益</span>"
            f"<span style='color:{new_pnl_c};font-weight:700'>{new_pnl:+.1f}%</span></div>"
            f"</div>",
            unsafe_allow_html=True)

    # ── Stress test ──
    if stats['max_dd'] < -0.5:
        # Worst case: price drops from current 52w high to historical max drawdown level
        stress_price = h52w * (1 + stats['max_dd'] / 100)
        avg_for_stress = rec['new_avg_cost'] if rec['add_shares'] > 0 else cost
        stress_pnl = (stress_price / avg_for_stress - 1) * 100
        stress_c   = '#f87171' if stress_pnl < -20 else '#fbbf24' if stress_pnl < -10 else '#94a3b8'

        st.markdown(
            f"<div class='result-card'>"
            f"<div class='section-hdr' style='margin-top:0'>壓力測試（歷史最大回檔）</div>"
            f"<div style='font-size:0.72rem;color:#64748b;margin-bottom:8px'>"
            f"若從 52 週高點再跌至歷史最深 {stats['max_dd']:.1f}%</div>"
            f"<div class='stress-row'><span style='color:#64748b'>壓力測試價格</span>"
            f"<span style='color:#f87171;font-weight:600'>${stress_price:,.2f}</span></div>"
            f"<div class='stress-row' style='border:none'><span style='color:#64748b'>加碼後損益估算</span>"
            f"<span style='color:{stress_c};font-weight:700;font-size:1.1rem'>{stress_pnl:+.1f}%</span></div>"
            f"</div>",
            unsafe_allow_html=True)

# ── 自訂加碼模擬器 ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-hdr">自訂加碼模擬器</div>', unsafe_allow_html=True)
st.markdown(
    "<span style='color:#64748b;font-size:0.78rem'>"
    "不想照建議比例？拖動滑桿輸入自訂比例，即時看新均價落在歷史回檔哪個風險帶"
    "</span>", unsafe_allow_html=True)

custom_pct = st.slider(
    "自訂加碼比例（% of 現有持股）",
    min_value=0, max_value=300,
    value=max(0, int(rec['final_ratio'] * 100)),
    step=5, key="custom_ratio_slider",
    help="100% = 加碼等同現有持股數；200% = 加碼現有持股兩倍"
)

custom_ratio      = custom_pct / 100
custom_add_shares = shares * custom_ratio
custom_add_amount = custom_add_shares * cp
custom_new_total  = shares + custom_add_shares
custom_new_avg    = (shares * cost + custom_add_shares * cp) / custom_new_total if custom_ratio > 0 else cost
custom_pnl        = (cp / custom_new_avg - 1) * 100

# 新均價距 52 週高點的隱含回檔幅度
implied_dd_of_avg = (custom_new_avg / h52w - 1) * 100   # 負值代表新均價在高點以下

# 歷史上幾次修正曾跌穿此位置？
troughs_list = stats['troughs']
if len(troughs_list) > 0 and implied_dd_of_avg < 0:
    n_deeper   = sum(1 for t in troughs_list if t < implied_dd_of_avg)
    pct_deeper = n_deeper / len(troughs_list) * 100
else:
    n_deeper   = 0
    pct_deeper = 0.0

# 壓力測試（自訂比例下的最深回檔損益）
stress_price_c = h52w * (1 + stats['max_dd'] / 100) if stats['max_dd'] < -0.5 else cp
stress_pnl_c   = (stress_price_c / custom_new_avg - 1) * 100

sc1, sc2 = st.columns([1, 1])

with sc1:
    pnl_c = '#4ade80' if custom_pnl >= 0 else '#f87171'
    rec_pct = int(rec['final_ratio'] * 100)
    if custom_pct == rec_pct:
        badge = '<span style="background:#1e1b4b;color:#a5b4fc;padding:2px 8px;border-radius:8px;font-size:0.65rem">與建議相同</span>'
    elif custom_pct > rec_pct:
        badge = '<span style="background:#3b1515;color:#f87171;padding:2px 8px;border-radius:8px;font-size:0.65rem">↑ 高於建議</span>'
    else:
        badge = '<span style="background:#14532d;color:#4ade80;padding:2px 8px;border-radius:8px;font-size:0.65rem">↓ 低於建議</span>'

    st.markdown(
        f"<div class='result-card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>"
        f"<div class='section-hdr' style='margin:0'>試算結果（自訂 {custom_pct}%）</div>{badge}</div>"
        f"<div class='stress-row'><span style='color:#64748b'>加碼股數</span>"
        f"<span style='color:#e2e8f0;font-weight:600'>{custom_add_shares:.0f} 股</span></div>"
        f"<div class='stress-row'><span style='color:#64748b'>加碼金額</span>"
        f"<span style='color:#e2e8f0;font-weight:600'>${custom_add_amount:,.0f}</span></div>"
        f"<div class='stress-row'><span style='color:#64748b'>加碼後總股數</span>"
        f"<span style='color:#e2e8f0;font-weight:600'>{custom_new_total:.0f} 股</span></div>"
        f"<div class='stress-row'><span style='color:#64748b'>加碼後新均價</span>"
        f"<span style='color:#60a5fa;font-weight:800;font-size:1.15rem'>${custom_new_avg:,.2f}</span></div>"
        f"<div class='stress-row' style='border:none'><span style='color:#64748b'>新均價現損益</span>"
        f"<span style='color:{pnl_c};font-weight:700'>{custom_pnl:+.1f}%</span></div>"
        f"</div>",
        unsafe_allow_html=True)

with sc2:
    # 安全帶判斷
    if pct_deeper <= 20:
        s_color, s_bg, s_border = '#4ade80', '#052e16', '#16a34a'
        s_icon, s_label = '🟢', '相對安全區'
        s_desc = f'僅 {pct_deeper:.0f}% 的歷史修正曾跌穿此均價'
    elif pct_deeper <= 50:
        s_color, s_bg, s_border = '#fbbf24', '#3a2800', '#d97706'
        s_icon, s_label = '🟡', '中等風險區'
        s_desc = f'{pct_deeper:.0f}% 的歷史修正曾跌穿此均價'
    else:
        s_color, s_bg, s_border = '#f87171', '#450a0a', '#dc2626'
        s_icon, s_label = '🔴', '高風險區'
        s_desc = f'{pct_deeper:.0f}% 的歷史修正曾跌穿此均價'

    stress_c = '#f87171' if stress_pnl_c < -20 else '#fbbf24' if stress_pnl_c < -10 else '#94a3b8'

    # 進度條：新均價在歷史回檔分佈中的位置
    # 0% = 52w high, max_dd% = 最深，bar 顯示 implied_dd 落在哪裡
    bar_pct = 0
    if stats['max_dd'] < 0 and implied_dd_of_avg < 0:
        bar_pct = min(100, max(0, implied_dd_of_avg / stats['max_dd'] * 100))
    # bar_pct: 0% = 靠近高點（安全），100% = 靠近歷史最深（危險）
    # 顯示時反過來：已佔用的空間代表「距高點多遠」
    bar_used = 100 - bar_pct   # 剩餘空間（未觸及最深的緩衝）

    st.markdown(
        f"<div class='result-card'>"
        f"<div class='section-hdr' style='margin-top:0'>新均價 vs 歷史回檔分佈</div>"
        # 數字
        f"<div style='margin-bottom:10px'>"
        f"<div style='font-size:0.68rem;color:#64748b'>新均價距 52 週高點</div>"
        f"<div style='font-size:1.9rem;font-weight:800;color:#60a5fa;line-height:1.1'>{implied_dd_of_avg:.1f}%</div>"
        f"<div style='font-size:0.65rem;color:#475569;margin-top:2px'>"
        f"${custom_new_avg:,.2f} vs 52 週高 ${h52w:,.2f}</div>"
        f"</div>"
        # 進度條
        f"<div style='margin-bottom:10px'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.6rem;color:#475569;margin-bottom:3px'>"
        f"<span>52 週高點 0%</span><span>歷史最深 {stats['max_dd']:.1f}%</span></div>"
        f"<div style='background:#1e293b;border-radius:4px;height:10px;position:relative;overflow:hidden'>"
        f"<div style='height:100%;width:{100-bar_pct:.0f}%;background:linear-gradient(90deg,#4ade80,#fbbf24,#f87171);border-radius:4px'></div>"
        f"<div style='position:absolute;top:0;left:{100-bar_pct:.0f}%;height:100%;width:2px;background:#60a5fa'></div>"
        f"</div>"
        f"<div style='font-size:0.6rem;color:#60a5fa;margin-top:2px;text-align:left;padding-left:{max(0,100-bar_pct-5):.0f}%'>▲ 新均價</div>"
        f"</div>"
        # 安全帶
        f"<div style='background:{s_bg};border:1px solid {s_border};border-radius:8px;padding:9px 12px;margin-bottom:8px'>"
        f"<div style='color:{s_color};font-size:0.82rem;font-weight:700'>{s_icon} {s_label}</div>"
        f"<div style='color:{s_color};font-size:0.72rem;margin-top:2px'>{s_desc}</div>"
        f"<div style='font-size:0.62rem;color:#475569;margin-top:3px'>"
        f"共 {len(troughs_list)} 次歷史修正，{n_deeper} 次曾跌穿</div>"
        f"</div>"
        # 壓力測試
        f"<div class='stress-row'><span style='color:#64748b'>壓力測試（歷史最深 {stats['max_dd']:.1f}%）</span>"
        f"<span style='color:{stress_c};font-weight:700'>{stress_pnl_c:+.1f}%</span></div>"
        f"<div style='font-size:0.62rem;color:#475569;margin-top:3px'>"
        f"若跌至 ${stress_price_c:,.2f}，自訂加碼後的損益</div>"
        f"</div>",
        unsafe_allow_html=True)

# ── Drawdown history chart ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-hdr">歷史回檔走勢（距 52 週高點）</div>', unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data['dates'], y=stats['dd_series'],
    name='距高點回檔',
    line=dict(color='#f87171', width=1.5),
    fill='tozeroy', fillcolor='rgba(248,113,113,0.08)',
    mode='lines',
))
# Reference lines
ref_lines = [
    (stats['p75'], '#4ade80', f"淺 P75 {stats['p75']:.1f}%", 'right'),
    (stats['p50'], '#fbbf24', f"中位 {stats['p50']:.1f}%",   'right'),
    (stats['p25'], '#f87171', f"深 P25 {stats['p25']:.1f}%", 'right'),
]
for val, col, lbl, pos in ref_lines:
    if val != 0:
        fig.add_hline(
            y=val, line=dict(color=col, width=1, dash='dot'),
            annotation_text=lbl,
            annotation_font=dict(size=10, color=col),
            annotation_position=pos,
        )
fig.add_hline(
    y=cur_dd,
    line=dict(color='#a78bfa', width=2, dash='dash'),
    annotation_text=f'現在 {cur_dd:.1f}%',
    annotation_font=dict(size=11, color='#a78bfa'),
    annotation_position='left',
)
fig.update_layout(
    height=200, margin=dict(l=10, r=100, t=10, b=8),
    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    showlegend=False,
    xaxis=dict(showgrid=False, color='#475569', tickfont=dict(size=10)),
    yaxis=dict(gridcolor='#1e293b', color='#475569',
               tickfont=dict(size=10), ticksuffix='%'),
)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ── Episode trough histogram ─────────────────────────────────────────────
if len(stats['troughs']) >= 3:
    st.markdown('<div class="section-hdr">歷史回檔分佈（每次修正的最深點）</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=stats['troughs'], nbinsx=20,
        marker=dict(color='#60a5fa', opacity=0.7, line=dict(color='#1e40af', width=0.5)),
        name='回檔次數',
    ))
    fig2.add_vline(x=cur_dd, line=dict(color='#a78bfa', width=2, dash='dash'),
                   annotation_text=f'現在 {cur_dd:.1f}%',
                   annotation_font=dict(size=10, color='#a78bfa'),
                   annotation_position='top right')
    fig2.add_vline(x=stats['p50'], line=dict(color='#fbbf24', width=1.5, dash='dot'),
                   annotation_text=f'中位 {stats["p50"]:.1f}%',
                   annotation_font=dict(size=10, color='#fbbf24'))
    if implied_dd_of_avg < 0:
        fig2.add_vline(x=implied_dd_of_avg,
                       line=dict(color='#60a5fa', width=2, dash='dot'),
                       annotation_text=f'自訂新均價 {implied_dd_of_avg:.1f}%',
                       annotation_font=dict(size=10, color='#60a5fa'),
                       annotation_position='top left')
    fig2.update_layout(
        height=160, margin=dict(l=10, r=10, t=10, b=8),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        showlegend=False,
        xaxis=dict(showgrid=False, color='#475569', tickfont=dict(size=10), ticksuffix='%'),
        yaxis=dict(gridcolor='#1e293b', color='#475569', tickfont=dict(size=10)),
        bargap=0.1,
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

# ── Logic explanation ─────────────────────────────────────────────────────
with st.expander("📋 計算邏輯說明"):
    st.markdown(f"""
**回檔深度分類**（依 {ticker} 歷史 {stats['n_ep']} 次回檔事件）

| 分類 | 條件 | 基準加碼比例 |
|---|---|---|
| 淺回檔 | 回檔幅度 > {stats['p75']:.1f}%（P75） | 25% |
| 中等回檔 | {stats['p25']:.1f}% ～ {stats['p75']:.1f}% | 50% |
| 深度回檔 | 回檔幅度 < {stats['p25']:.1f}%（P25） | 100% |

**安全墊係數**（依持股成本調整）

| 獲利幅度 | 係數 |
|---|---|
| > +20% | 1.00 |
| +10% ～ +20% | 0.90 |
| +5% ～ +10% | 0.75 |
| 0% ～ +5% | 0.50 |
| -10% ～ 0% | 0.25 |
| < -10% | 0.00（不建議加碼）|

**最終加碼比例** = 基準比例 × 安全墊係數（相對於現有持股數量）

**壓力測試**：假設從當前 52 週高點（${h52w:,.2f}）再跌至歷史最大回檔幅度（{stats['max_dd']:.1f}%），
即股價觸及 ${h52w*(1+stats['max_dd']/100):,.2f}，計算加碼後的損益。

---
⚠️ 本工具僅供量化參考，不構成投資建議。歷史回檔統計不代表未來表現，請搭配基本面與大盤環境綜合判斷。
""")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.68rem'>"
    "數據來源：Yahoo Finance（yfinance）｜ 台股代碼請加 .TW（例如 2330.TW）｜ 本工具不構成投資建議"
    "</div>", unsafe_allow_html=True)
