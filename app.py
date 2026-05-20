import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(
    page_title="牛市轉折偵測儀表板",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  .metric-card {
    background: #1a1f2e; border-radius: 10px; padding: 14px 16px;
    border: 1px solid #2d3748; margin-bottom: 8px;
  }
  .card-red   { border-color: #dc2626 !important; }
  .card-yellow{ border-color: #d97706 !important; }
  .card-green { border-color: #1e293b !important; }
  .val-green  { color: #4ade80; font-size: 1.5rem; font-weight: 700; }
  .val-yellow { color: #fbbf24; font-size: 1.5rem; font-weight: 700; }
  .val-red    { color: #f87171; font-size: 1.5rem; font-weight: 700; }
  .val-neutral{ color: #94a3b8; font-size: 1.5rem; font-weight: 700; }
  .badge-green { background:#14532d; color:#4ade80; padding:2px 8px; border-radius:10px; font-size:0.7rem; font-weight:700; }
  .badge-yellow{ background:#451a03; color:#fbbf24; padding:2px 8px; border-radius:10px; font-size:0.7rem; font-weight:700; }
  .badge-red   { background:#7f1d1d; color:#f87171; padding:2px 8px; border-radius:10px; font-size:0.7rem; font-weight:700; }
  .desc-text  { color: #64748b; font-size: 0.75rem; margin-top: 4px; line-height: 1.4; }
  .lift-tag   { color: #475569; font-size: 0.65rem; }
  .combo-active  { background:#1e1b4b; border:1px solid #4f46e5; border-radius:8px; padding:10px 14px; margin-bottom:6px; }
  .combo-inactive{ background:#111827; border:1px solid #1e293b; border-radius:8px; padding:10px 14px; margin-bottom:6px; opacity:0.6; }
  .uvxy-warn  { background:#451a03; border:1px solid #d97706; border-radius:8px; padding:10px 14px; color:#fcd34d; font-size:0.8rem; }
  .uvxy-ok    { background:#111827; border:1px solid #1e293b; border-radius:8px; padding:10px 14px; color:#475569; font-size:0.8rem; }
  .overall-green { background:linear-gradient(135deg,#052e16,#14532d); border:1px solid #16a34a; border-radius:12px; padding:16px 20px; }
  .overall-yellow{ background:linear-gradient(135deg,#1c1500,#3a2800); border:1px solid #d97706; border-radius:12px; padding:16px 20px; }
  .overall-red   { background:linear-gradient(135deg,#2d0000,#450a0a); border:1px solid #dc2626; border-radius:12px; padding:16px 20px; }
  .ov-green-title{ color:#4ade80; font-size:1.1rem; font-weight:700; }
  .ov-yellow-title{color:#fbbf24; font-size:1.1rem; font-weight:700;}
  .ov-red-title  { color:#f87171; font-size:1.1rem; font-weight:700; }
  .section-hdr{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#475569; margin-bottom:6px; margin-top:4px; }
</style>
""", unsafe_allow_html=True)


# ── Data fetch ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="載入最新市場數據中…")
def fetch_data():
    end   = datetime.today()
    start = end - timedelta(days=400)

    tickers = ['CME','SPY','IWM','IWF','IWD','^VIX','^VIX3M',
               'HYG','IEI','RSP','XLP','XLY',
               'XLK','XLF','XLV','XLE','XLI','XLB','XLU']

    raw   = yf.download(tickers, start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw['Close'].copy()
    close.columns = [c[1] if isinstance(c,tuple) else c for c in close.columns]
    close = close.dropna(how='all')

    sector_etfs = ['XLK','XLF','XLV','XLE','XLI','XLB','XLU','XLP','XLY']

    def s60(series):
        vals = series.iloc[-60:].tolist()
        return [round(v,3) if not (v!=v) else None for v in vals]

    spy    = close['SPY']
    ma200  = spy.rolling(200).mean()
    ma50   = spy.rolling(50).mean()

    cme_s  = (close['CME'].pct_change(10) - spy.pct_change(10)) * 100
    hyg_iei_s = (close['HYG']/close['IEI']).pct_change(20) * 100
    iwm_s  = (close['IWM']/spy).pct_change(60) * 100
    gv_s   = (close['IWF']/close['IWD']).pct_change(20) * 100
    vixr_s = close['^VIX'] / close['^VIX3M']
    xlp_s  = (close['XLP']/close['XLY']).pct_change(20) * 100
    rsp_s  = (close['RSP']/spy).pct_change(60) * 100

    # Sector breadth (vectorized)
    ma50_sec  = close[sector_etfs].rolling(50).mean()
    above     = close[sector_etfs] > ma50_sec
    breadth_s = above.sum(axis=1)

    latest = close.index[-1]

    return {
        'as_of':            str(latest.date()),
        'spy_price':        round(float(spy.iloc[-1]), 2),
        'spy_200ma_val':    round(float(ma200.iloc[-1]), 2),
        'spy_vs_200ma':     round((float(spy.iloc[-1])/float(ma200.iloc[-1])-1)*100, 2),
        'cme_excess_10d':   round(float(cme_s.iloc[-1]), 2),
        'hyg_iei_20d':      round(float(hyg_iei_s.iloc[-1]), 2),
        'iwm_spy_60d':      round(float(iwm_s.iloc[-1]), 2),
        'growth_value_20d': round(float(gv_s.iloc[-1]), 2),
        'vix':              round(float(close['^VIX'].iloc[-1]), 2),
        'vix3m':            round(float(close['^VIX3M'].iloc[-1]), 2),
        'vix_ratio':        round(float(vixr_s.iloc[-1]), 3),
        'xlp_xly_20d':      round(float(xlp_s.iloc[-1]), 2),
        'rsp_spy_60d':      round(float(rsp_s.iloc[-1]), 2),
        'sector_breadth':   int(breadth_s.iloc[-1]),
        # Series
        'cme_series':     s60(cme_s),
        'hyg_iei_series': s60(hyg_iei_s),
        'iwm_series':     s60(iwm_s),
        'gv_series':      s60(gv_s),
        'vixr_series':    s60(vixr_s),
        'xlp_series':     s60(xlp_s),
        'rsp_series':     s60(rsp_s),
        'breadth_series': [int(x) for x in breadth_s.iloc[-60:].tolist()],
    }


# ── Status functions ──────────────────────────────────────────────
def status(key, val):
    if key=='cme':   return 'red' if val>8 else 'yellow' if val>=5 else 'green'
    if key=='hyg':   return 'red' if val<-1 else 'yellow' if val<0 else 'green'
    if key=='iwm':   return 'red' if val<-5 else 'yellow' if val<-3 else 'green'
    if key=='gv':    return 'red' if val<-3 else 'yellow' if val<-1 else 'green'
    if key=='vixr':  return 'red' if val>=1.1 else 'yellow' if val>=1.0 else 'green'
    if key=='xlp':   return 'red' if val>2 else 'yellow' if val>1 else 'green'
    if key=='rsp':   return 'red' if val<-5 else 'yellow' if val<-3 else 'green'
    if key=='brdth': return 'red' if val<=3 else 'yellow' if val<=5 else 'green'
    return 'green'

BADGE = {
    'green':  '<span class="badge-green">正常</span>',
    'yellow': '<span class="badge-yellow">留意</span>',
    'red':    '<span class="badge-red">警示</span>',
}
VAL_CLASS = {'green':'val-green','yellow':'val-yellow','red':'val-red','neutral':'val-neutral'}


def hex_to_rgba(hex_color, alpha=0.13):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def sparkline(series, color, height=55):
    fig = go.Figure(go.Scatter(
        y=series, mode='lines',
        line=dict(color=color, width=1.5),
        fill='tozeroy', fillcolor=hex_to_rgba(color),
    ))
    fig.update_layout(
        height=height, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False
    )
    return fig


def color_for(key, st_val, inv=False):
    if st_val=='green':  return '#22c55e'
    if st_val=='yellow': return '#f59e0b'
    return '#ef4444'


def fmt(val):
    return f"+{val}%" if val > 0 else f"{val}%"


def card(title, val_str, st_val, desc, series, inv=False, lift=None, note=None):
    col_str = color_for('', st_val)
    note_html = f' <span style="font-size:0.6rem;color:#6366f1;background:#1e1b4b;padding:1px 5px;border-radius:4px">{note}</span>' if note else ''
    lift_html = f'<span class="lift-tag">實證倍率 {lift}</span>' if lift else ''
    st.markdown(f"""
    <div class="metric-card card-{st_val}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <span style="font-size:0.72rem;color:#94a3b8">{title}{note_html}</span>
        {BADGE[st_val]}
      </div>
      <div class="{VAL_CLASS[st_val]}" style="margin:4px 0 2px">{val_str}</div>
      {lift_html}
      <div class="desc-text">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(sparkline(series, col_str), use_container_width=True,
                    config={'displayModeBar':False}, key=f"chart_{title[:8]}")


# ── Sidebar navigation ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 導覽")
    st.page_link("app.py", label="📊 牛市轉折偵測儀表板")
    st.page_link("pages/2_EMA_Analysis.py", label="📐 季線乖離率分析")


# ── Main ──────────────────────────────────────────────────────────
D = fetch_data()

st_cme  = status('cme',   D['cme_excess_10d'])
st_hyg  = status('hyg',   D['hyg_iei_20d'])
st_iwm  = status('iwm',   D['iwm_spy_60d'])
st_vixr = status('vixr',  D['vix_ratio'])
st_xlp  = status('xlp',   D['xlp_xly_20d'])
st_rsp  = status('rsp',   D['rsp_spy_60d'])
st_br   = status('brdth', D['sector_breadth'])

# IWF/IWD 已從核心移除：60天回測倍率僅0.94x（低於基準），不具預測能力
core_statuses = [st_cme, st_hyg, st_iwm]
n_red    = core_statuses.count('red')
n_yellow = core_statuses.count('yellow')
overall  = 'red' if n_red>=2 else 'yellow' if n_red>=1 or n_yellow>=2 else 'green'

# Combo triggers
cme_triggered = st_cme in ('yellow','red')
combo1 = cme_triggered and st_iwm != 'green'
combo2 = cme_triggered and st_xlp in ('yellow','red')
combo3 = cme_triggered and st_hyg == 'red'

uvxy_warn = st_rsp == 'red' or st_br in ('yellow','red')

# ── Header ───────────────────────────────────────────────────────
col_title, col_nav, col_refresh = st.columns([4, 1.5, 1])
with col_title:
    st.markdown("## 📊 牛市轉折偵測儀表板")
    st.markdown(f"<span style='color:#64748b;font-size:0.78rem'>數據截至 {D['as_of']} &nbsp;｜&nbsp; SPY ${D['spy_price']} &nbsp;｜&nbsp; 200日均線 ${D['spy_200ma_val']} (+{D['spy_vs_200ma']}%，牛市確立)</span>", unsafe_allow_html=True)
with col_nav:
    st.page_link("pages/2_EMA_Analysis.py", label="📐 季線乖離率分析", use_container_width=True)
with col_refresh:
    if st.button("🔄 更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ── Overall ──────────────────────────────────────────────────────
ov_map = {
    'green':  ('核心指標全數正常', '牛市動能健康，高β倉位無需調整'),
    'yellow': ('核心指標出現警示', '建議收緊止損或留意槓桿敞口'),
    'red':    ('多項核心指標告警', '建議評估減碼高β倉位，啟動避險策略'),
}
ov_title, ov_desc = ov_map[overall]
st.markdown(f"""
<div class="overall-{overall}" style="margin-bottom:14px">
  <div class="ov-{overall}-title">{'🟢' if overall=='green' else '🟡' if overall=='yellow' else '🔴'} &nbsp;{ov_title}</div>
  <div style="color:#94a3b8;font-size:0.78rem;margin-top:4px">{ov_desc}</div>
</div>
""", unsafe_allow_html=True)

# ── Combo panel ──────────────────────────────────────────────────
st.markdown('<div class="section-hdr">關鍵信號組合</div>', unsafe_allow_html=True)

combos = [
    (combo1, "66.7%", "3.08x", "CME +5~8%  ＋  IWM/SPY 60日 <−5%", "最強組合，下跌>5%機率 66.7%（n=45）；Permutation test p<0.001，95% CI 52~79%"),
    (combo2, "54.3%", "2.50x", "CME +5~8%  ＋  防禦輪動 XLP/XLY >1%", "樣本最充足組合（n=129），超過五成下跌機率"),
    (combo3, "48.4%", "2.23x", "CME +5~8%  ＋  信用利差惡化 <−1%", "信用市場確認壓力，接近五成下跌機率（n=93）"),
]
for active, prob, lift, title, desc in combos:
    cls = "combo-active" if active else "combo-inactive"
    badge = '<span style="background:#312e81;color:#a5b4fc;padding:2px 8px;border-radius:10px;font-size:0.65rem;font-weight:700">觸發中</span>' if active else '<span style="background:#1e293b;color:#475569;padding:2px 8px;border-radius:10px;font-size:0.65rem">未觸發</span>'
    st.markdown(f"""
    <div class="{cls}">
      <div style="display:flex;align-items:center;gap:12px">
        <span style="font-size:1.1rem;font-weight:700;color:{'#a5b4fc' if active else '#475569'};width:50px">{prob}</span>
        <div style="flex:1">
          <div style="font-size:0.75rem;font-weight:600;color:{'#c7d2fe' if active else '#475569'}">{title}</div>
          <div style="font-size:0.68rem;color:{'#818cf8' if active else '#334155'};margin-top:2px">{desc}</div>
        </div>
        {badge}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

# ── UVXY bar ─────────────────────────────────────────────────────
if uvxy_warn:
    reasons = []
    if st_rsp == 'red':  reasons.append(f"RSP/SPY廣度 {D['rsp_spy_60d']}%")
    if st_br != 'green': reasons.append(f"板塊廣度 {D['sector_breadth']}/9")
    st.markdown(f'<div class="uvxy-warn">⚡ <b>UVXY早期預警觸發</b>——{("、").join(reasons)}，VIX仍低，可考慮小量預佈局</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="uvxy-ok">UVXY早期預警：未觸發（RSP/SPY與板塊廣度正常）</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

# ── Core indicators ───────────────────────────────────────────────
st.markdown('<div class="section-hdr">核心預警指標（影響整體燈號 · 實證有效）</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

cme_v = D['cme_excess_10d']
with c1:
    desc = (f"CME跑贏SPY {cme_v}%，警示區間（單獨倍率僅1.10x，須配合IWM/SPY同時觸發才有力）" if st_cme in ('yellow','red')
            else f"CME明顯跑輸（{cme_v}%），歷史看漲信號" if cme_v < -3
            else f"CME相對報酬中性，無明顯信號")
    card("CME超額報酬（10日）", fmt(cme_v), st_cme, desc, D['cme_series'], inv=True, lift="1.41x（單獨）/ 3.08x（+IWM）")

hyg_v = D['hyg_iei_20d']
with c2:
    desc = (f"信用利差擴大 {hyg_v}%，風險溢酬上升" if st_hyg=='red'
            else f"信用利差輕微惡化（{hyg_v}%）" if st_hyg=='yellow'
            else f"信用市場穩定（{fmt(hyg_v)}），輔助確認牛市")
    card("信用利差 HYG/IEI（20日）", fmt(hyg_v), st_hyg, desc, D['hyg_iei_series'], lift="1.60x")

iwm_v = D['iwm_spy_60d']
with c3:
    desc = (f"小型股60日跑輸大型股 {abs(iwm_v)}%，風險偏好惡化" if st_iwm=='red'
            else f"小型股相對弱勢（{iwm_v}%），需觀察" if st_iwm=='yellow'
            else f"小型股同步（{fmt(iwm_v)}），風險偏好正常")
    card("IWM/SPY 小型股（60日）", fmt(iwm_v), st_iwm, desc, D['iwm_series'], lift="1.55x")

# ── Context indicators ────────────────────────────────────────────
st.markdown('<div class="section-hdr">輔助情境指標（參考用途 · 不計入整體燈號）</div>', unsafe_allow_html=True)
x1, x2, x3, x4 = st.columns(4)

with x1:
    vr = D['vix_ratio']
    desc = (f"期限結構倒掛（{vr}），近月恐慌高於遠月" if st_vixr!='green'
            else f"期限結構正常 Contango（{vr}），情緒穩定")
    card("VIX/VIX3M 期限結構", str(vr), st_vixr, desc, D['vixr_series'], inv=True)

with x2:
    xl = D['xlp_xly_20d']
    desc = (f"防禦板塊跑贏景氣循環 {xl}%，資金轉向防禦" if st_xlp!='green'
            else f"景氣循環領先（{abs(xl)}%），情緒偏進取")
    card("防禦輪動 XLP/XLY（20日）", fmt(xl), st_xlp, desc, D['xlp_series'], inv=True)

with x3:
    rs = D['rsp_spy_60d']
    desc = (f"等權重跑輸市值加權 {abs(rs)}%，廣度惡化，UVXY早期預警" if st_rsp=='red'
            else f"廣度正常（{fmt(rs)}）")
    card("RSP/SPY 等權廣度（60日）", fmt(rs), st_rsp, desc, D['rsp_series'], note="UVXY訊號")

with x4:
    br = D['sector_breadth']
    desc = (f"僅 {br} 個板塊在50MA上，廣度嚴重惡化" if st_br=='red'
            else f"{br} 個板塊在50MA上，廣度偏窄" if st_br=='yellow'
            else f"{br} 個板塊在50MA上，廣度健康")
    card(f"板塊廣度（{br}/9 在50日均線上）", f"{br}/9", st_br, desc, D['breadth_series'], note="UVXY訊號")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#334155;font-size:0.68rem'>"
    f"數據來源：Yahoo Finance（yfinance）&nbsp;｜&nbsp;"
    f"數據每小時自動更新 &nbsp;｜&nbsp;"
    f"截至 {D['as_of']}"
    f"</div>",
    unsafe_allow_html=True
)
