import streamlit as st
import pandas as pd
import yfinance as yf
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
  .warn-red    { background:#450a0a; border:1px solid #dc2626; border-radius:8px;
                 padding:10px 14px; color:#fca5a5; font-size:0.82rem; margin-bottom:10px; }
  .warn-yellow { background:#3a2800; border:1px solid #d97706; border-radius:8px;
                 padding:10px 14px; color:#fcd34d; font-size:0.82rem; margin-bottom:10px; }
  .warn-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
                 padding:10px 14px; color:#86efac; font-size:0.82rem; margin-bottom:10px; }
  .warn-blue   { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
                 padding:10px 14px; color:#93c5fd; font-size:0.82rem; margin-bottom:10px; }
  .tbl-row { display:grid; gap:4px; font-size:0.73rem; padding:5px 0;
             border-bottom:1px solid #111827; }
  .tbl-hdr { display:grid; gap:4px; font-size:0.67rem; color:#475569;
             padding:4px 0; border-bottom:1px solid #1e293b; }
  .bar-wrap { display:flex; align-items:center; gap:6px; margin:4px 0; font-size:0.72rem; }
  .bar-track { flex:1; background:#1e293b; border-radius:2px; height:7px; overflow:hidden; }
  .bar-fill  { height:100%; border-radius:2px; }
  .chip-now  { background:#1e1b4b; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_calc(ticker: str, period: str, span: int, ma_type: str):
    df = yf.download(ticker, period=period, interval='1d',
                     auto_adjust=True, progress=False)
    if df.empty or len(df) < span + 5:
        return None
    close = df['Close'].squeeze()
    if ma_type == 'EMA':
        ma = close.ewm(span=span, adjust=False).mean()
    else:
        ma = close.rolling(span).mean()
    dev = (close - ma) / ma * 100
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in close.index],
        'close': close.tolist(),
        'ma':    ma.tolist(),
        'dev':   dev.tolist(),
        'n':     len(close),
    }

def threshold_stats(close_arr, dev_arr, thresholds):
    close = np.array(close_arr, dtype=float)
    dev   = np.array(dev_arr,   dtype=float)
    rows  = []
    for thr in thresholds:
        sigs, last = [], -999
        for i in range(len(dev) - 20):
            if not np.isnan(dev[i]) and dev[i] >= thr and (i - last) > 10:
                sigs.append(i); last = i
        if len(sigs) < 2:
            rows.append((thr, 0, None, None, None, None)); continue
        r5, r10, r20, dd = [], [], [], []
        for idx in sigs:
            e = close[idx]
            if idx + 5  < len(close): r5.append( (close[idx+5]  - e)/e*100)
            if idx + 10 < len(close): r10.append((close[idx+10] - e)/e*100)
            if idx + 20 < len(close): r20.append((close[idx+20] - e)/e*100)
            if idx + 20 < len(close):
                dd.append(close[idx:idx+21].min() / e * 100 - 100 < -5)
        rows.append((
            thr, len(sigs),
            float(np.mean(r5))  if r5  else None,
            float(np.mean(r10)) if r10 else None,
            float(np.mean(r20)) if r20 else None,
            float(np.mean(dd))  if dd  else None,
        ))
    return rows

def below_episodes(dev_arr, dates_arr):
    dev = np.array(dev_arr, dtype=float)
    eps, in_ep = [], False
    for i in range(1, len(dev)):
        if np.isnan(dev[i]): continue
        if not in_ep and dev[i-1] >= 0 and dev[i] < 0:
            in_ep, start_i, trough, trough_i = True, i, dev[i], i
        elif in_ep:
            if dev[i] < trough: trough, trough_i = dev[i], i
            if dev[i] >= 0 or i == len(dev)-1:
                eps.append(dict(start=dates_arr[start_i],
                                end=dates_arr[i] if dev[i] >= 0 else '至今',
                                days=i-start_i, trough=trough,
                                trough_date=dates_arr[trough_i]))
                in_ep = False
    return eps

def sign(v, d=1):
    if v is None or np.isnan(v): return '—'
    return f"+{v:.{d}f}%" if v >= 0 else f"{v:.{d}f}%"

def dev_color(v):
    if v is None or np.isnan(v): return '#94a3b8'
    if v > 25: return '#f87171'
    if v > 15: return '#fbbf24'
    if v >= 0: return '#4ade80'
    return '#f87171'

def ret_color(v):
    if v is None: return '#94a3b8'
    return '#4ade80' if v > 0 else '#f87171'

def dd_color(v):
    if v is None: return '#94a3b8'
    return '#f87171' if v > 0.6 else '#fbbf24' if v > 0.4 else '#4ade80'

def bar_html(val, max_abs, color, reverse=False):
    w = min(100, abs(val) / max_abs * 100) if max_abs else 0
    justify = 'justify-content:flex-end;' if reverse else ''
    return (f"<div class='bar-track' style='{justify}'>"
            f"<div class='bar-fill' style='width:{w:.0f}%;background:{color}'></div></div>")

def warn_box(dev, max_dev, mean, std, label):
    if dev < 0:
        return 'blue', f'↓ 股價低於{label}（{sign(dev)}），觀察是否重新站上均線'
    ratio = dev / max_dev if max_dev > 0 else 0
    if ratio >= 0.85:
        return 'red', f'⚠ 接近歷史最大乖離（{sign(max_dev)}），建議啟動移動停損'
    if dev > mean + 2*std:
        return 'yellow', f'◎ 超過均值 +2σ（{mean+2*std:.1f}%），漲勢可能趨緩'
    if dev > mean + std:
        return 'yellow', f'△ 超過均值 +1σ（{mean+std:.1f}%），留意乖離是否繼續擴大'
    return 'green', f'✓ 乖離率正常（{sign(dev)}），技術壓力不大'

# ── Win-rate helpers ──────────────────────────────────────────────────────────
WR_BUCKETS = [
    ("<-15%",   None, -15),
    ("-15~-10%", -15, -10),
    ("-10~-5%", -10,  -5),
    ("-5~0%",    -5,   0),
    ("0~5%",      0,   5),
    ("5~10%",     5,  10),
    ("10~15%",   10,  15),
    (">15%",     15, None),
]

def get_bucket_label(dev_val):
    for label, lo, hi in WR_BUCKETS:
        if lo is None and dev_val < hi:            return label
        if hi is None and dev_val >= lo:           return label
        if lo is not None and hi is not None and lo <= dev_val < hi: return label
    return None

def wr_color(wr):
    if wr >= 80: return '#4ade80'
    if wr >= 65: return '#86efac'
    if wr >= 50: return '#fbbf24'
    return '#f87171'

def calc_wr_table(close_arr, dates_arr, forward=20, min_n=5):
    """
    計算 EMA(60) 乖離率加碼勝率表。
    只使用 EMA(60) 斜率 > 0（季線向上）的日期。
    """
    close_s  = pd.Series(close_arr, index=pd.to_datetime(dates_arr), dtype=float)
    ema260   = close_s.ewm(span=60, adjust=False).mean()
    slope260 = ema260.diff()
    dev260   = (close_s / ema260 - 1) * 100

    df = pd.DataFrame({
        'close':   close_s,
        'ema260':  ema260,
        'slope':   slope260,
        'dev':     dev260,
    })
    # 前向報酬
    df['fwd']     = df['close'].shift(-forward) / df['close'] - 1
    df['fwd_pos'] = df['fwd'] > 0

    # 去除 EMA 收斂期前 260 筆及最後 forward 筆
    df_valid = df.iloc[260:-forward].copy()
    uptrend  = df_valid['slope'] > 0

    curr_dev260   = round(float(dev260.iloc[-1]), 2)
    curr_slope260 = float(slope260.iloc[-1])
    curr_bkt      = get_bucket_label(curr_dev260)

    rows = []
    for label, lo, hi in WR_BUCKETS:
        if lo is None:
            mask = uptrend & (df_valid['dev'] < hi)
        elif hi is None:
            mask = uptrend & (df_valid['dev'] >= lo)
        else:
            mask = uptrend & (df_valid['dev'] >= lo) & (df_valid['dev'] < hi)
        sub = df_valid[mask]
        if len(sub) < min_n:
            continue
        wr  = round(sub['fwd_pos'].mean() * 100, 1)
        avg = round(sub['fwd'].mean() * 100, 2)
        ev  = round(wr / 100 * avg, 2)
        rows.append({
            '乖離率區間': label,
            '勝率':      wr,
            '平均報酬':   avg,
            '期望值':     ev,
            'n':         len(sub),
        })

    return pd.DataFrame(rows) if rows else None, curr_dev260, curr_slope260, curr_bkt

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 📐 季線乖離率分析")
st.markdown("<span style='color:#64748b;font-size:0.78rem'>輸入代碼自動抓取 Yahoo Finance 資料，計算 EMA/SMA 乖離率及歷史分佈</span>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([3, 1.8, 1.2, 1.2, 1.2])
with c1:
    raw_input = st.text_input("代碼", placeholder="CRWD / 2330.TW",
                               label_visibility='collapsed', key='dev_raw')
with c2:
    period = st.selectbox("期間", ['1y','2y','5y'], index=1,
                           format_func=lambda x: {'1y':'1 年','2y':'2 年','5y':'5 年'}[x],
                           label_visibility='collapsed')
with c3:
    span = st.number_input("週期", min_value=2, max_value=300, value=60,
                            label_visibility='collapsed')
with c4:
    ma_type = st.selectbox("類型", ['EMA','SMA'], label_visibility='collapsed')
with c5:
    go_btn = st.button("查詢 →", use_container_width=True, type='primary')

# Quick chips
QUICK = ['CRWD','GLW','GEV','SNDK','TER','MU','CAT','DHI','TOL','NVDA','TSLA','AAPL']
st.markdown('<div class="section-hdr" style="margin-top:8px">常用</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(QUICK))
chip_clicked = None
for col, t in zip(chip_cols, QUICK):
    with col:
        if st.button(t, key=f'chip_{t}', use_container_width=True):
            chip_clicked = t

# Resolve ticker
if chip_clicked:
    st.session_state['dev_ticker'] = chip_clicked
elif go_btn and raw_input.strip():
    st.session_state['dev_ticker'] = raw_input.strip().upper()
ticker = st.session_state.get('dev_ticker', '')

# ── Main display ──────────────────────────────────────────────────────────────
if not ticker:
    st.markdown(
        "<div style='text-align:center;padding:4rem;color:#334155'>"
        "<div style='font-size:2.5rem'>📐</div>"
        "<div style='font-size:1rem;margin-top:1rem'>輸入股票代碼後按查詢，或點選常用清單</div>"
        "<div style='font-size:0.8rem;margin-top:0.4rem;color:#475569'>台股請加 .TW，例如 2330.TW</div>"
        "</div>", unsafe_allow_html=True)
    st.stop()

with st.spinner(f'抓取 {ticker} 資料中…'):
    data = fetch_and_calc(ticker, period, span, ma_type)

if data is None:
    st.error(f'找不到 **{ticker}** 的資料。請確認代碼正確（台股請加 .TW，例如 2330.TW）。')
    st.stop()

dates  = data['dates']
close  = np.array(data['close'])
ma_arr = np.array(data['ma'])
dev    = np.array(data['dev'])
ma_lbl = f"{ma_type}({span})"
valid  = dev[~np.isnan(dev)]
neg    = valid[valid < 0]
pos    = valid[valid > 0]
cur_price = close[-1]
cur_ma    = ma_arr[-1]
cur_dev   = dev[-1]
max_dev   = float(np.nanmax(dev))
min_dev   = float(np.nanmin(dev))
max_date  = dates[int(np.nanargmax(dev))]
min_date  = dates[int(np.nanargmin(dev))]
mean_all  = float(np.nanmean(valid))
std_all   = float(np.nanstd(valid))

# Percentiles
def pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else 0.0

p75 = pct(valid, 75); p85 = pct(valid, 85); p90 = pct(valid, 90)
p95 = pct(valid, 95); p99 = pct(valid, 99)
n10 = pct(neg, 10); n25 = pct(neg, 25); n50 = pct(neg, 50)

# ── Header & metrics ──────────────────────────────────────────────────────────
st.markdown(
    f"### {ticker} "
    f"<span style='color:#64748b;font-size:0.85rem'>"
    f"{dates[0]} ～ {dates[-1]}　{data['n']} 個交易日</span>",
    unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("最新收盤價", f"${cur_price:,.2f}")
with m2: st.metric(ma_lbl, f"${cur_ma:,.2f}")
with m3: st.metric("當前乖離率", sign(cur_dev))
with m4: st.metric("歷史最大", sign(max_dev), delta=max_date, delta_color='off')
with m5: st.metric("歷史最小", sign(min_dev), delta=min_date, delta_color='off')

# Warning
wtype, wmsg = warn_box(cur_dev, max_dev, mean_all, std_all, ma_lbl)
st.markdown(f'<div class="warn-{wtype}">{wmsg}</div>', unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col, dist_col = st.columns([3, 1])
with chart_col:
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=dates, y=close.tolist(), name='收盤價',
                               line=dict(color='#60a5fa', width=1.5), mode='lines'))
    fig_p.add_trace(go.Scatter(x=dates, y=ma_arr.tolist(), name=ma_lbl,
                               line=dict(color='#f87171', width=1.5, dash='dot'), mode='lines'))
    fig_p.update_layout(
        height=240, margin=dict(l=8, r=8, t=30, b=8),
        title=dict(text=f'{ticker} 股價與 {ma_lbl}', font=dict(size=12, color='#94a3b8')),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        legend=dict(orientation='h', y=1.1, font=dict(color='#94a3b8', size=11)),
        xaxis=dict(showgrid=False, color='#475569', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#1e293b', color='#475569', tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=dates, y=dev.tolist(), name='乖離率',
                               line=dict(color='#a78bfa', width=1.5), mode='lines',
                               fill='tozeroy', fillcolor='rgba(167,139,250,0.08)'))
    for val, col, dash, lbl, pos_ in [
        (0,                  '#475569', 'solid',  '0',  'right'),
        (mean_all + std_all, '#fbbf24', 'dot',    '+1σ','right'),
        (mean_all + 2*std_all,'#f87171','dash',   '+2σ','right'),
    ]:
        fig_d.add_hline(y=val, line=dict(color=col, width=1, dash=dash),
                        annotation_text=lbl,
                        annotation_font=dict(size=10, color=col),
                        annotation_position=pos_)
    fig_d.add_hline(y=cur_dev,
                    line=dict(color=dev_color(cur_dev), width=1.5, dash='dot'),
                    annotation_text=f'現在 {sign(cur_dev)}',
                    annotation_font=dict(size=10, color=dev_color(cur_dev)),
                    annotation_position='left')
    fig_d.update_layout(
        height=190, margin=dict(l=8, r=55, t=30, b=8),
        title=dict(text='乖離率走勢（%）', font=dict(size=12, color='#94a3b8')),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        showlegend=False,
        xaxis=dict(showgrid=False, color='#475569', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#1e293b', color='#475569',
                   tickfont=dict(size=10), ticksuffix='%'),
    )
    st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar': False})

with dist_col:
    max_abs = max(abs(p99) if p99 else 1, abs(n10) if n10 else 1) * 1.15
    st.markdown('<div class="section-hdr">正乖離百分位</div>', unsafe_allow_html=True)
    for lbl, val, col in [('99th', p99,'#f87171'), ('95th',p95,'#fb923c'),
                           ('90th', p90,'#fbbf24'), ('85th',p85,'#a3e635'),
                           ('75th', p75,'#4ade80')]:
        st.markdown(
            f"<div class='bar-wrap'>"
            f"<span style='color:#475569;width:32px'>{lbl}</span>"
            f"{bar_html(val, max_abs, col)}"
            f"<span style='color:#94a3b8;width:46px;text-align:right'>{val:+.1f}%</span>"
            f"</div>", unsafe_allow_html=True)
    if cur_dev > 0:
        c = dev_color(cur_dev)
        st.markdown(
            f"<div class='bar-wrap' style='border-top:1px solid #1e293b;padding-top:5px;margin-top:3px'>"
            f"<span style='color:{c};width:32px;font-weight:700'>現在</span>"
            f"{bar_html(cur_dev, max_abs, c)}"
            f"<span style='color:{c};width:46px;text-align:right;font-weight:700'>{cur_dev:+.1f}%</span>"
            f"</div>", unsafe_allow_html=True)
    if len(neg) > 0:
        st.markdown('<div class="section-hdr" style="margin-top:12px">負乖離百分位</div>',
                    unsafe_allow_html=True)
        for lbl, val, col in [('10th',n10,'#f87171'),('25th',n25,'#fbbf24'),('50th',n50,'#94a3b8')]:
            st.markdown(
                f"<div class='bar-wrap'>"
                f"<span style='color:#475569;width:32px'>{lbl}</span>"
                f"{bar_html(val, max_abs, col, reverse=True)}"
                f"<span style='color:#94a3b8;width:46px;text-align:right'>{val:+.1f}%</span>"
                f"</div>", unsafe_allow_html=True)
        if cur_dev < 0:
            c = '#f87171'
            st.markdown(
                f"<div class='bar-wrap' style='border-top:1px solid #1e293b;padding-top:5px;margin-top:3px'>"
                f"<span style='color:{c};width:32px;font-weight:700'>現在</span>"
                f"{bar_html(cur_dev, max_abs, c, reverse=True)}"
                f"<span style='color:{c};width:46px;text-align:right;font-weight:700'>{cur_dev:+.1f}%</span>"
                f"</div>", unsafe_allow_html=True)
        st.markdown('<div class="section-hdr" style="margin-top:10px">跌入各深度機率</div>',
                    unsafe_allow_html=True)
        eps_all = below_episodes(dev.tolist(), dates)
        troughs = [e['trough'] for e in eps_all]
        for thresh in [-5, -10, -15, -20]:
            prob = sum(1 for t in troughs if t <= thresh) / len(troughs) if troughs else 0
            w = prob * 100
            col = '#f87171' if prob > 0.3 else '#fbbf24' if prob > 0.1 else '#4ade80'
            st.markdown(
                f"<div class='bar-wrap'>"
                f"<span style='color:#475569;width:32px'>{thresh}%</span>"
                f"<div class='bar-track'><div class='bar-fill' style='width:{w:.0f}%;background:{col}'></div></div>"
                f"<span style='color:#94a3b8;width:46px;text-align:right'>{prob*100:.0f}%</span>"
                f"</div>", unsafe_allow_html=True)

# ── Threshold & episode tables ────────────────────────────────────────────────
st.markdown("---")
ta_col, ep_col = st.columns(2)
with ta_col:
    st.markdown('<div class="section-hdr">超過閾值後的後續表現</div>', unsafe_allow_html=True)
    raw_thrs = sorted(set([
        round(mean_all + std_all), round(p75), round(p85), round(p90), round(p95), round(p99)
    ]))
    thresholds = [t for t in raw_thrs if t > mean_all]
    rows = threshold_stats(close.tolist(), dev.tolist(), thresholds)
    st.markdown(
        "<div class='tbl-hdr' style='grid-template-columns:70px 40px 60px 60px 60px 68px'>"
        "<span>閾值</span><span>次數</span><span>5日</span><span>10日</span><span>20日</span>"
        "<span>回落>5%</span></div>", unsafe_allow_html=True)
    for thr, n, r5, r10, r20, dd in rows:
        is_here = (cur_dev >= thr and
                   (rows.index((thr,n,r5,r10,r20,dd)) == len(rows)-1 or
                    cur_dev < rows[rows.index((thr,n,r5,r10,r20,dd))+1][0]))
        bg = "background:#1e1b4b;border-radius:4px;padding:2px 4px;" if is_here else ""
        if n < 2:
            content = (f"<span style='color:#475569'>{thr:+.0f}%</span>"
                       f"<span style='color:#334155'>N/A</span>"
                       f"<span style='color:#334155'>—</span><span style='color:#334155'>—</span>"
                       f"<span style='color:#334155'>—</span><span style='color:#334155'>—</span>")
        else:
            tc = '#a5b4fc' if is_here else '#94a3b8'
            content = (
                f"<span style='color:{tc};font-weight:{'700' if is_here else '400'}'>{thr:+.0f}%</span>"
                f"<span style='color:#64748b'>{n}</span>"
                f"<span style='color:{ret_color(r5)}'>{sign(r5)}</span>"
                f"<span style='color:{ret_color(r10)}'>{sign(r10)}</span>"
                f"<span style='color:{ret_color(r20)}'>{sign(r20)}</span>"
                f"<span style='color:{dd_color(dd)}'>{f'{dd*100:.0f}%' if dd is not None else '—'}</span>"
            )
        st.markdown(
            f"<div class='tbl-row' style='grid-template-columns:70px 40px 60px 60px 60px 68px;{bg}'>"
            f"{content}</div>", unsafe_allow_html=True)

with ep_col:
    st.markdown('<div class="section-hdr">跌破季線事件記錄</div>', unsafe_allow_html=True)
    eps = below_episodes(dev.tolist(), dates)
    if not eps:
        st.markdown("<span style='color:#475569;font-size:0.8rem'>歷史上未曾跌破季線</span>",
                    unsafe_allow_html=True)
    else:
        troughs = [e['trough'] for e in eps]
        avg_days = int(np.mean([e['days'] for e in eps]))
        below_pct = len(valid[valid < 0]) / len(valid) * 100
        st.markdown(
            f"<div style='font-size:0.72rem;color:#64748b;margin-bottom:8px'>"
            f"共 {len(eps)} 次 ｜ 歷史 {below_pct:.0f}% 時間在均線下方 ｜"
            f" 平均持續 {avg_days} 日 ｜ 平均最深 {np.mean(troughs):+.1f}%</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='tbl-hdr' style='grid-template-columns:88px 88px 40px 70px'>"
            "<span>開始</span><span>結束</span><span>天數</span><span>最深跌幅</span></div>",
            unsafe_allow_html=True)
        for e in reversed(eps[-14:]):
            is_now = e['end'] == '至今'
            tc = '#f87171' if e['trough'] < -10 else '#fbbf24' if e['trough'] < -5 else '#94a3b8'
            bg = "background:#1e293b;border-radius:4px;padding:2px 4px;" if is_now else ""
            st.markdown(
                f"<div class='tbl-row' style='grid-template-columns:88px 88px 40px 70px;{bg}'>"
                f"<span style='color:#64748b'>{e['start']}</span>"
                f"<span style='color:{'#a5b4fc' if is_now else '#64748b'}'>{e['end']}</span>"
                f"<span style='color:#475569'>{e['days']}日</span>"
                f"<span style='color:{tc};font-weight:500'>{e['trough']:+.1f}%</span>"
                f"</div>", unsafe_allow_html=True)

# ── EMA(60) 加碼勝率 ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-hdr">EMA(60) 季線加碼勝率（持有20日 · 季線向上期間）</div>',
            unsafe_allow_html=True)

if len(close) < 200:
    st.markdown(
        "<div class='warn-yellow'>⚠ 資料不足 200 天，EMA(60) 統計樣本可能偏少，"
        "勝率統計樣本可能偏少，請謹慎解讀。</div>",
        unsafe_allow_html=True)

wr_tbl, curr_dev260, curr_slope260, curr_bkt260 = calc_wr_table(close.tolist(), dates)
is_uptrend260 = curr_slope260 > 0

# 狀態橫條
slope_col = '#4ade80' if is_uptrend260 else '#f87171'
slope_txt = '季線向上 ✓' if is_uptrend260 else '季線向下 ✗（加碼勝率不適用）'
st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
            padding:10px 16px;margin-bottom:12px;display:flex;gap:28px;align-items:center">
  <div>
    <div style="font-size:0.65rem;color:#64748b">EMA(60) 乖離率</div>
    <div style="font-size:1.3rem;font-weight:800;
                color:{'#fbbf24' if abs(curr_dev260)>10 else '#e2e8f0'}">{'+' if curr_dev260>=0 else ''}{curr_dev260:.2f}%</div>
    <div style="font-size:0.65rem;color:#64748b">桶：{curr_bkt260}</div>
  </div>
  <div>
    <div style="font-size:0.65rem;color:#64748b">EMA(60) 斜率</div>
    <div style="font-size:0.95rem;font-weight:700;color:{slope_col}">{slope_txt}</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not is_uptrend260:
    st.markdown(
        "<div class='warn-red'>季線目前向下，以下歷史統計僅供參考，不建議依此加碼。</div>",
        unsafe_allow_html=True)

if wr_tbl is None:
    st.markdown(
        "<div style='color:#475569;font-size:0.8rem'>各桶樣本數不足（< 5），無法統計勝率。"
        "請選擇更長的資料期間（建議 5y）。</div>",
        unsafe_allow_html=True)
else:
    rows_html = ""
    for _, row in wr_tbl.iterrows():
        is_curr = (row['乖離率區間'] == curr_bkt260) and is_uptrend260
        bg       = "background:rgba(99,102,241,0.15);border-left:3px solid #4f46e5" if is_curr else "border-left:3px solid transparent"
        wc       = wr_color(row['勝率'])
        avg_c    = '#4ade80' if row['平均報酬'] > 0 else '#f87171'
        ev_c     = '#4ade80' if row['期望值']   > 0 else '#f87171'
        now_tag  = (' <span style="background:#312e81;color:#a5b4fc;padding:1px 5px;'
                    'border-radius:6px;font-size:0.62rem;font-weight:700">現在</span>'
                    if is_curr else "")
        rows_html += f"""
        <tr style="{bg}">
          <td style="padding:7px 12px;font-weight:600;color:#e2e8f0">{row['乖離率區間']}{now_tag}</td>
          <td style="padding:7px 12px;text-align:right;font-size:1.05rem;font-weight:800;color:{wc}">{row['勝率']:.1f}%</td>
          <td style="padding:7px 12px;text-align:right;color:{avg_c}">{'+' if row['平均報酬']>0 else ''}{row['平均報酬']:.2f}%</td>
          <td style="padding:7px 12px;text-align:right;color:{ev_c}">{'+' if row['期望值']>0 else ''}{row['期望值']:.2f}%</td>
          <td style="padding:7px 12px;text-align:right;color:#64748b">{row['n']}</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:10px;overflow:hidden">
      <thead>
        <tr style="background:#0f172a">
          <th style="padding:7px 12px;text-align:left;color:#64748b;font-size:0.7em;text-transform:uppercase;letter-spacing:.05em">乖離率區間</th>
          <th style="padding:7px 12px;text-align:right;color:#64748b;font-size:0.7em;text-transform:uppercase;letter-spacing:.05em">20日勝率</th>
          <th style="padding:7px 12px;text-align:right;color:#64748b;font-size:0.7em;text-transform:uppercase;letter-spacing:.05em">平均報酬</th>
          <th style="padding:7px 12px;text-align:right;color:#64748b;font-size:0.7em;text-transform:uppercase;letter-spacing:.05em">期望值</th>
          <th style="padding:7px 12px;text-align:right;color:#64748b;font-size:0.7em;text-transform:uppercase;letter-spacing:.05em">n</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='color:#334155;font-size:0.65rem;margin-top:6px'>"
        "期望值 = 勝率 × 平均報酬 &nbsp;｜&nbsp; 僅計入 EMA(60) 斜率 &gt; 0 的交易日 &nbsp;｜&nbsp;"
        "含存活者偏差，n &lt; 20 時請謹慎解讀"
        "</div>",
        unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.68rem'>"
    "數據來源：Yahoo Finance（yfinance）｜ 台股代碼請加 .TW（例如 2330.TW）｜ 資料每小時快取更新"
    "</div>", unsafe_allow_html=True)
