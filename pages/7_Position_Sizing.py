import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import numpy as np
import math
import json
from datetime import datetime, timedelta

# ── CSS ───────────────────────────────────────────────────────────
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
  .tbl-hdr { display:grid; gap:4px; font-size:0.67rem; color:#475569;
             padding:4px 0; border-bottom:1px solid #1e293b; }
  .tbl-row { display:grid; gap:4px; font-size:0.73rem; padding:5px 0;
             border-bottom:1px solid #111827; }
</style>
""", unsafe_allow_html=True)

# ── Zigzag pivot detector ─────────────────────────────────────────
def zigzag(prices, min_move=0.08):
    """偵測顯著轉折點，min_move = 最小換向幅度"""
    if len(prices) < 20:
        return []
    pivots = []
    lp, li = float(prices[0]), 0
    direction = None

    for i in range(1, len(prices)):
        p = float(prices[i])
        if direction is None:
            if p >= lp * (1 + min_move):
                direction = 'up'
                pivots.append((li, lp, 'L'))
                lp, li = p, i
            elif p <= lp * (1 - min_move):
                direction = 'down'
                pivots.append((li, lp, 'H'))
                lp, li = p, i
        elif direction == 'up':
            if p > lp:
                lp, li = p, i
            elif p <= lp * (1 - min_move):
                pivots.append((li, lp, 'H'))
                direction = 'down'
                lp, li = p, i
        else:
            if p < lp:
                lp, li = p, i
            elif p >= lp * (1 + min_move):
                pivots.append((li, lp, 'L'))
                direction = 'up'
                lp, li = p, i

    if direction == 'up':
        pivots.append((li, lp, 'H'))
    elif direction == 'down':
        pivots.append((li, lp, 'L'))
    return pivots


def extract_swings(pivots, dates):
    """從轉折點萃取 L→H→L 波段"""
    swings = []
    for i in range(len(pivots) - 2):
        i0, p0, t0 = pivots[i]
        i1, p1, t1 = pivots[i+1]
        i2, p2, t2 = pivots[i+2]
        if t0 == 'L' and t1 == 'H' and t2 == 'L':
            gain     = (p1 - p0) / p0 * 100
            pullback = (p1 - p2) / p1 * 100
            swings.append({
                'from_date':   dates[i0],
                'peak_date':   dates[i1],
                'to_date':     dates[i2],
                'from_price':  round(p0, 2),
                'peak_price':  round(p1, 2),
                'to_price':    round(p2, 2),
                'gain':        round(gain, 1),
                'pullback':    round(pullback, 1),
            })
    return swings


BUCKET_ORDER = ['≤10%', '≤20%', '≤30%', '≤50%', '>50%']

def gain_bucket(g):
    if g <= 10: return '≤10%'
    if g <= 20: return '≤20%'
    if g <= 30: return '≤30%'
    if g <= 50: return '≤50%'
    return '>50%'


# ── Main analysis ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_analyze(ticker: str):
    end   = datetime.today()
    start = end - timedelta(days=730)
    df = yf.download(
        ticker,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        auto_adjust=True, progress=False,
    )
    if df.empty or len(df) < 60:
        return None

    prices = df['Close'].squeeze().dropna()
    dates  = [d.strftime('%Y-%m-%d') for d in prices.index]
    arr    = prices.values.astype(float)

    # Swing detection & extraction
    pivots = zigzag(arr, min_move=0.08)
    swings = extract_swings(pivots, dates)

    # Pullback statistics
    pbs      = [s['pullback'] for s in swings]
    avg_pb   = round(float(np.mean(pbs)),   1) if pbs else 0.0
    med_pb   = round(float(np.median(pbs)), 1) if pbs else 0.0
    max_pb   = round(float(np.max(pbs)),    1) if pbs else 0.0
    n_swings = len(swings)

    # Annualized volatility
    rets    = np.diff(arr) / arr[:-1]
    ann_vol = round(float(np.std(rets) * np.sqrt(252) * 100), 1)

    # Recommended threshold: max(median × 1.6, vol × 0.3), round up to 5
    raw_thr   = max(med_pb * 1.6, ann_vol * 0.3)
    threshold = int(math.ceil(raw_thr / 5) * 5)

    # Bucket breakdown
    bkt_data = {}
    for s in swings:
        bkt_data.setdefault(gain_bucket(s['gain']), []).append(s['pullback'])
    buckets = [
        {'label': bkt, 'avg': round(float(np.mean(bkt_data[bkt])), 1), 'n': len(bkt_data[bkt])}
        for bkt in BUCKET_ORDER if bkt in bkt_data
    ]

    # Current price vs recent high (60-day window)
    last60     = arr[-60:]
    high_idx   = int(np.argmax(last60))
    recent_high = round(float(last60[high_idx]), 2)
    current     = round(float(arr[-1]), 2)
    cur_dd      = round((recent_high - current) / recent_high * 100, 1) if recent_high > 0 else 0.0

    return {
        'ticker':       ticker.upper(),
        'as_of':        dates[-1],
        'current':      current,
        'recent_high':  recent_high,
        'cur_dd':       cur_dd,
        'avg_pb':       avg_pb,
        'med_pb':       med_pb,
        'max_pb':       max_pb,
        'ann_vol':      ann_vol,
        'n_swings':     n_swings,
        'threshold':    threshold,
        'buckets':      buckets,
        'swings':       swings[-12:],
    }


# ── Chart HTML ────────────────────────────────────────────────────
def build_chart_html(buckets, threshold):
    labels = json.dumps([b['label'] for b in buckets], ensure_ascii=False)
    data   = json.dumps([b['avg']   for b in buckets])
    counts = json.dumps([b['n']     for b in buckets])
    colors = json.dumps([
        '#22c55e' if b['avg'] <= 10 else
        '#fbbf24' if b['avg'] <= 20 else
        '#ef4444'
        for b in buckets
    ])
    n = len(buckets)
    return f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:-apple-system,sans-serif}}
.wrap{{padding:4px 0}}
</style></head><body>
<div class="wrap">
<canvas id="chart" height="180"></canvas>
</div>
<script>
const labels  = {labels};
const data    = {data};
const counts  = {counts};
const colors  = {colors};
const thr     = {threshold};
new Chart(document.getElementById('chart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [{{
      label: '平均回檔 %',
      data,
      backgroundColor: colors.map(c => c + 'bb'),
      borderColor: colors,
      borderWidth: 1.5,
      borderRadius: 5,
    }}]
  }},
  options: {{
    responsive: true,
    animation: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => `平均回檔：${{ctx.raw.toFixed(1)}}%`,
          afterLabel: ctx => `樣本：${{counts[ctx.dataIndex]}} 次`,
        }}
      }},
      annotation: {{ /* Chart.js annotation plugin not loaded, skip */ }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#64748b', font: {{ size: 12 }} }},
        grid:  {{ color: '#1e293b' }},
        title: {{ display: true, text: '波段漲幅區間', color: '#475569', font: {{ size: 11 }} }}
      }},
      y: {{
        ticks: {{ color: '#64748b', font: {{ size: 12 }}, callback: v => v + '%' }},
        grid:  {{ color: '#1e293b' }},
        title: {{ display: true, text: '平均回檔深度 %', color: '#475569', font: {{ size: 11 }} }},
        suggestedMin: 0,
      }}
    }}
  }}
}});
</script></body></html>"""


# ── Page ──────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🔁 股票加碼分析器")
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem'>"
        "輸入代號，自動計算回檔習性，給出建議加碼門檻 "
        "&nbsp;｜&nbsp; 台股請加 .TW（如 0050.TW）"
        "</span>",
        unsafe_allow_html=True,
    )
with col_refresh:
    if st.button("🔄 清除快取", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Controls
c1, c2 = st.columns([4, 1])
with c1:
    raw_input = st.text_input(
        "代碼", placeholder="NVDA / 2330.TW",
        label_visibility='collapsed', key='ps_raw',
    )
with c2:
    go_btn = st.button("查詢 →", use_container_width=True, type='primary')

# Quick chips
QUICK = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'META', 'CRWD', 'AMZN', 'MU', 'PLTR', '0050.TW', '2330.TW']
st.markdown('<div class="section-hdr" style="margin-top:8px">常用</div>', unsafe_allow_html=True)
chip_cols    = st.columns(len(QUICK))
chip_clicked = None
for col, t in zip(chip_cols, QUICK):
    with col:
        if st.button(t, key=f'ps_chip_{t}', use_container_width=True):
            chip_clicked = t

# Resolve ticker
if chip_clicked:
    st.session_state['ps_ticker'] = chip_clicked
elif go_btn and raw_input.strip():
    st.session_state['ps_ticker'] = raw_input.strip().upper()
ticker = st.session_state.get('ps_ticker', '')

if not ticker:
    st.markdown(
        "<div style='text-align:center;padding:4rem;color:#334155'>"
        "<div style='font-size:2.5rem'>🔁</div>"
        "<div style='font-size:1rem;margin-top:1rem'>輸入股票代碼後按查詢，或點選常用清單</div>"
        "</div>", unsafe_allow_html=True
    )
    st.stop()

# ── Fetch & analyze ───────────────────────────────────────────────
with st.spinner(f'分析 {ticker} 回檔習性中…'):
    D = fetch_and_analyze(ticker)

if D is None:
    st.error(f'找不到 **{ticker}** 的資料，或資料不足 60 天。請確認代碼正確（台股請加 .TW）。')
    st.stop()

# ── Stock title ───────────────────────────────────────────────────
st.markdown(
    f"### {D['ticker']} "
    f"<span style='color:#64748b;font-size:0.85rem'>截至 {D['as_of']}　"
    f"共偵測 {D['n_swings']} 個有效波段</span>",
    unsafe_allow_html=True,
)

# ── Current status bar ────────────────────────────────────────────
cur_dd   = D['cur_dd']
thr      = D['threshold']
dd_ratio = cur_dd / thr if thr > 0 else 0

if cur_dd <= 0:
    status_bg, status_border = '#052e16', '#16a34a'
    status_icon = '🟢'
    status_msg  = f'目前價格 ${D["current"]:,.2f} 位於近 60 日高點（${D["recent_high"]:,.2f}），尚無明顯回檔'
elif dd_ratio >= 1.0:
    status_bg, status_border = '#2d0000', '#dc2626'
    status_icon = '🔴'
    status_msg  = (f'目前已從近高回檔 <b style="color:#f87171">{cur_dd}%</b>，'
                   f'已達或超過建議門檻 {thr}%　→　可考慮執行加碼')
elif dd_ratio >= 0.6:
    status_bg, status_border = '#1c1500', '#d97706'
    status_icon = '🟡'
    status_msg  = (f'目前回檔 <b style="color:#fbbf24">{cur_dd}%</b>，'
                   f'接近門檻（{thr}%）{round(dd_ratio*100)}%　→　留意進場時機')
else:
    status_bg, status_border = '#0c1a2e', '#2563eb'
    status_icon = '🔵'
    status_msg  = (f'目前回檔 {cur_dd}%，距門檻 {thr}% 尚遠，持倉不急於加碼')

st.markdown(f"""
<div style="background:{status_bg};border:1px solid {status_border};border-radius:10px;
            padding:11px 18px;margin-bottom:14px;display:flex;align-items:center;gap:12px">
  <span style="font-size:1.4rem">{status_icon}</span>
  <div style="font-size:0.82rem;color:#94a3b8">{status_msg}</div>
</div>""", unsafe_allow_html=True)

# ── 5 Metric cards ────────────────────────────────────────────────
st.markdown('<div class="section-hdr">回檔統計指標（近兩年 · 8% 換向門檻）</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)

def mcard(col, title, val, sub=None, color='#e2e8f0'):
    with col:
        sub_html = f'<div style="font-size:0.63rem;color:#475569;margin-top:2px">{sub}</div>' if sub else ''
        st.markdown(f"""<div class="metric-card">
          <div style="font-size:0.68rem;color:#64748b">{title}</div>
          <div style="font-size:1.4rem;font-weight:700;color:{color};margin:3px 0">{val}</div>
          {sub_html}
        </div>""", unsafe_allow_html=True)

mcard(m1, '平均回檔深度', f"{D['avg_pb']}%",  f"共 {D['n_swings']} 個波段")
mcard(m2, '中位回檔深度', f"{D['med_pb']}%",  '50th percentile',
      '#fbbf24' if D['med_pb'] > 15 else '#4ade80')
mcard(m3, '最大回檔',    f"{D['max_pb']}%",  '歷史最深',
      '#f87171' if D['max_pb'] > 30 else '#fbbf24' if D['max_pb'] > 20 else '#4ade80')
mcard(m4, '年化波動率',  f"{D['ann_vol']}%",
      '高波動' if D['ann_vol'] > 50 else '中波動' if D['ann_vol'] > 25 else '低波動',
      '#f87171' if D['ann_vol'] > 50 else '#fbbf24' if D['ann_vol'] > 25 else '#4ade80')
mcard(m5, '有效波段數',  str(D['n_swings']),
      '建議 ≥ 5 才具統計意義',
      '#4ade80' if D['n_swings'] >= 5 else '#f87171')

# ── Recommendation box ────────────────────────────────────────────
vol_label = '高波動' if D['ann_vol'] > 50 else '中波動' if D['ann_vol'] > 25 else '低波動'
vol_color = '#f87171' if D['ann_vol'] > 50 else '#fbbf24' if D['ann_vol'] > 25 else '#4ade80'
vol_bg    = '#2d0000'  if D['ann_vol'] > 50 else '#1c1500'  if D['ann_vol'] > 25 else '#052e16'

st.markdown(f"""
<div style="background:{vol_bg};border:1px solid {vol_color}44;border-radius:12px;
            padding:18px 22px;margin:10px 0 14px">
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div>
      <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.05em">
        建議加碼門檻
        <span style="background:{vol_bg};border:1px solid {vol_color};color:{vol_color};
                     padding:1px 7px;border-radius:8px;font-size:0.65rem;margin-left:6px">
          {vol_label}
        </span>
      </div>
      <div style="font-size:2.8rem;font-weight:800;color:{vol_color};line-height:1.1;margin:4px 0">
        {thr}%
      </div>
      <div style="font-size:0.72rem;color:#64748b">
        公式：max（中位回檔 {D['med_pb']}% × 1.6，年化波動 {D['ann_vol']}% × 0.3）→ 無條件進位到 5 的倍數
      </div>
    </div>
    <div style="flex:1;min-width:220px;border-left:1px solid {vol_color}33;padding-left:18px">
      <div style="font-size:0.74rem;color:#94a3b8;line-height:1.8">
        <b style="color:#e2e8f0">邏輯：</b>門檻讓紙上獲利必須大於典型回檔，否則加碼那口會在震盪中被洗掉。<br>
        從低點漲超過 <b style="color:{vol_color}">{thr}%</b> 才加碼一倍，確保即使隨後回撤中位幅度（{D['med_pb']}%），仍保有足夠緩衝。
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Bucket bar chart + swing table ───────────────────────────────
chart_col, table_col = st.columns([3, 2])

with chart_col:
    st.markdown('<div class="section-hdr">按波段漲幅分類的平均回檔深度</div>', unsafe_allow_html=True)
    if D['buckets']:
        chart_html = build_chart_html(D['buckets'], thr)
        components.html(chart_html, height=240, scrolling=False)
        # Bucket legend table
        st.markdown(
            "<div class='tbl-hdr' style='grid-template-columns:80px 80px 80px 60px'>"
            "<span>漲幅區間</span><span>平均回檔</span><span>色標</span><span>次數</span></div>",
            unsafe_allow_html=True,
        )
        for b in D['buckets']:
            bar_color = '#22c55e' if b['avg'] <= 10 else '#fbbf24' if b['avg'] <= 20 else '#ef4444'
            st.markdown(
                f"<div class='tbl-row' style='grid-template-columns:80px 80px 80px 60px'>"
                f"<span style='color:#94a3b8'>{b['label']}</span>"
                f"<span style='color:{bar_color};font-weight:700'>{b['avg']}%</span>"
                f"<span><span style='background:{bar_color}44;color:{bar_color};padding:1px 8px;"
                f"border-radius:4px;font-size:0.65rem'>{'淺' if b['avg']<=10 else '中' if b['avg']<=20 else '深'}</span></span>"
                f"<span style='color:#475569'>{b['n']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info('波段數不足，無法產生分桶統計。')

with table_col:
    st.markdown('<div class="section-hdr">近期波段紀錄（最近 12 個）</div>', unsafe_allow_html=True)
    if D['swings']:
        st.markdown(
            "<div class='tbl-hdr' style='grid-template-columns:88px 55px 55px'>"
            "<span>波峰日期</span><span>漲幅</span><span>回檔</span></div>",
            unsafe_allow_html=True,
        )
        for s in reversed(D['swings']):
            gc = '#4ade80' if s['gain']     >= 20 else '#94a3b8'
            pc = '#f87171' if s['pullback'] >= 20 else '#fbbf24' if s['pullback'] >= 10 else '#4ade80'
            st.markdown(
                f"<div class='tbl-row' style='grid-template-columns:88px 55px 55px'>"
                f"<span style='color:#64748b'>{s['peak_date']}</span>"
                f"<span style='color:{gc};font-weight:600'>+{s['gain']}%</span>"
                f"<span style='color:{pc};font-weight:600'>-{s['pullback']}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<span style='color:#475569;font-size:0.8rem'>未偵測到有效波段（需股價出現 ≥8% 的換向走勢）</span>",
            unsafe_allow_html=True,
        )

# ── Warning ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="background:#1c1400;border:1px solid #854d0e;border-radius:8px;
            padding:10px 16px;color:#fde68a;font-size:0.78rem;font-weight:600">
  ⚠️ 建議門檻為歷史統計推導，過去回檔習性不代表未來表現。實際操作請搭配大盤環境與個人風險承受能力綜合判斷。
</div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.68rem;margin-top:12px'>"
    "數據來源：Yahoo Finance（yfinance）&nbsp;｜&nbsp;"
    "波段偵測：Zigzag 8% 換向門檻 &nbsp;｜&nbsp;"
    f"資料期間：近兩年日線 &nbsp;｜&nbsp; 截至 {D['as_of']}"
    "</div>",
    unsafe_allow_html=True,
)
