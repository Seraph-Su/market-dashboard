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
  .ldr-chip-up   { background:#14532d; color:#4ade80; padding:2px 7px; border-radius:6px; font-size:0.68rem; font-weight:700; font-family:monospace; margin-right:4px; }
  .ldr-chip-down { background:#7f1d1d; color:#f87171; padding:2px 7px; border-radius:6px; font-size:0.68rem; font-weight:700; font-family:monospace; margin-right:4px; }
</style>
""", unsafe_allow_html=True)
# ── Data fetch ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="載入最新市場數據中…")
def fetch_data():
    # ── 批量下載（不含 CME）────────────────────────────────────────
    # CME 在批量下載時 Yahoo Finance 偶爾靜默失敗，改成單獨下載以確保可靠
    main_tickers = ['SPY','IWM','IWF','IWD','^VIX','^VIX3M',
                    'HYG','IEI','RSP','XLP','XLY',
                    'XLK','XLF','XLV','XLE','XLI','XLB','XLU']
    # period 相對參數避免 yfinance 內部 SQLite 快取回傳舊資料
    raw   = yf.download(main_tickers, period="400d", interval="1d",
                        auto_adjust=True, progress=False)
    close = raw['Close'].copy()
    # c[-1] 相容 ('Close','SPY') 及 ('SPY',) 等不同 yfinance 版本的 tuple 結構
    close.columns = [c[-1] if isinstance(c, tuple) else str(c) for c in close.columns]
    # ── CME 單獨下載 ───────────────────────────────────────────────
    cme_raw = yf.download("CME", period="400d", interval="1d",
                           auto_adjust=True, progress=False)
    if not cme_raw.empty:
        cme_close = cme_raw['Close'].squeeze()
        # 移除 timezone 差異後 reindex 對齊主資料日期
        if hasattr(cme_close.index, 'tz') and cme_close.index.tz is not None:
            cme_close.index = cme_close.index.tz_localize(None)
        if hasattr(close.index, 'tz') and close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        close['CME'] = cme_close.reindex(close.index)
    else:
        close['CME'] = float('nan')
    close = close.dropna(how='all')
    sector_etfs = ['XLK','XLF','XLV','XLE','XLI','XLB','XLU','XLP','XLY']
    def s60(series):
        vals = series.iloc[-60:].tolist()
        return [round(v, 3) if not (v != v) else None for v in vals]
    spy    = close['SPY']
    ma200  = spy.rolling(200).mean()
    ma50   = spy.rolling(50).mean()
    ma60   = spy.ewm(span=60,  adjust=False).mean()
    ma260  = spy.ewm(span=260, adjust=False).mean()
    spy60_s = (spy / ma60 - 1) * 100
    vix      = close['^VIX']
    vix_ma20 = vix.rolling(20).mean()
    vix_vs_ma20_s = (vix / vix_ma20 - 1) * 100
    # ffill() 確保 CME 的 pct_change 不因個別缺值而炸掉
    cme_s     = (close['CME'].ffill().pct_change(10) - spy.pct_change(10)) * 100
    hyg_iei_s = (close['HYG'] / close['IEI']).pct_change(20) * 100
    iwm_s     = (close['IWM'] / spy).pct_change(60) * 100
    gv_s      = (close['IWF'] / close['IWD']).pct_change(20) * 100
    vixr_s    = close['^VIX'] / close['^VIX3M']
    xlp_s     = (close['XLP'] / close['XLY']).pct_change(20) * 100
    rsp_s     = (close['RSP'] / spy).pct_change(60) * 100
    # SPY 距 252 日高（領頭股前哨的適用前提判斷用）
    spy_dist_hi = (spy / spy.rolling(252, min_periods=120).max() - 1) * 100
    # Sector breadth (vectorized)
    ma50_sec  = close[sector_etfs].rolling(50).mean()
    above     = close[sector_etfs] > ma50_sec
    breadth_s = above.sum(axis=1)
    latest = close.index[-1]
    def _safe(val, default=0.0, decimals=2):
        """NaN 安全轉換：避免 float(nan) 傳入後端導致顯示異常"""
        try:
            f = float(val)
            return default if pd.isna(f) else round(f, decimals)
        except Exception:
            return default
    return {
        'as_of':            str(latest.date()),
        'spy_price':        _safe(spy.iloc[-1]),
        'spy_200ma_val':    _safe(ma200.iloc[-1]),
        'spy_vs_200ma':     _safe((spy.iloc[-1] / ma200.iloc[-1] - 1) * 100),
        'cme_excess_10d':   _safe(cme_s.iloc[-1]),
        'hyg_iei_20d':      _safe(hyg_iei_s.iloc[-1]),
        'iwm_spy_60d':      _safe(iwm_s.iloc[-1]),
        'growth_value_20d': _safe(gv_s.iloc[-1]),
        'vix':              _safe(close['^VIX'].iloc[-1]),
        'vix3m':            _safe(close['^VIX3M'].iloc[-1]),
        'vix_ratio':        _safe(vixr_s.iloc[-1], decimals=3),
        'xlp_xly_20d':      _safe(xlp_s.iloc[-1]),
        'rsp_spy_60d':      _safe(rsp_s.iloc[-1]),
        'sector_breadth':   int(breadth_s.iloc[-1]),
        'spy_vs_60ma':      _safe(spy60_s.iloc[-1]),
        'spy_60ma_val':     _safe(ma60.iloc[-1]),
        'spy_vs_260ma':     _safe((spy.iloc[-1] / ma260.iloc[-1] - 1) * 100),
        'vix_vs_ma20':      _safe(vix_vs_ma20_s.iloc[-1]),
        'vix_ma20_val':     _safe(vix_ma20.iloc[-1]),
        'spy_dist_hi':      _safe(spy_dist_hi.iloc[-1]),
        # Series
        'cme_series':     s60(cme_s),
        'hyg_iei_series': s60(hyg_iei_s),
        'iwm_series':     s60(iwm_s),
        'gv_series':      s60(gv_s),
        'vixr_series':    s60(vixr_s),
        'xlp_series':     s60(xlp_s),
        'rsp_series':     s60(rsp_s),
        'breadth_series': [int(x) for x in breadth_s.iloc[-60:].tolist()],
        'spy60_series':      s60(spy60_s),
        'vix_ma20_series':   s60(vix_vs_ma20_s),
    }
# ── 領頭股前哨資料 ─────────────────────────────────────────────────
# 備援名單：Yahoo 篩選器失敗時使用（請偶爾手動核對）
FALLBACK_LEADERS = ["NVDA", "GOOGL", "AAPL", "MSFT", "AMZN", "META", "AVGO", "TSLA"]
# 同公司雙股別合併
_SHARE_CLASS = {"GOOG": "GOOGL", "BRK-A": "BRK-B"}
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_members() -> set:
    import requests
    from io import StringIO
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    return set(df["Symbol"].str.replace(".", "-", regex=False).tolist())
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_leader_list(n: int = 8) -> tuple:
    """每日自動抓 S&P 500 市值前 n 大（合併雙股別、過濾非成分股）。失敗回退備援名單。"""
    try:
        from yfinance import EquityQuery
        q = EquityQuery('and', [
            EquityQuery('gt', ['intradaymarketcap', 200_000_000_000]),
            EquityQuery('eq', ['region', 'us']),
        ])
        res = yf.screen(q, size=25, sortField='intradaymarketcap', sortAsc=False)
        try:
            members = fetch_sp500_members()
        except Exception:
            members = None   # 維基抓不到就不過濾成分股
        out, seen = [], set()
        for x in res.get('quotes', []):
            s = x.get('symbol', '')
            if not s or '.' in s:
                continue
            s = _SHARE_CLASS.get(s, s)
            if s in seen:
                continue
            if members is not None and s not in members:
                continue   # 排除 TSM、SPCX 等非 S&P 500 成分（外國發行人/新上市）
            seen.add(s)
            out.append(s)
            if len(out) >= n:
                break
        return tuple(out) if len(out) >= 6 else tuple(FALLBACK_LEADERS)
    except Exception:
        return tuple(FALLBACK_LEADERS)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_leaders(leaders: tuple):
    """領頭股健康度：收盤站上 EMA60 的檔數（無緩衝＝黃燈用；2% 緩衝＝紅燈用）"""
    raw = yf.download(list(leaders), period="400d", interval="1d",
                      auto_adjust=True, progress=False)
    close = raw['Close'].copy()
    close.columns = [c[-1] if isinstance(c, tuple) else str(c) for c in close.columns]
    close = close.dropna(how='all')
    detail, above0_cols = [], {}
    for t in leaders:
        if t not in close.columns:
            continue
        c = close[t].dropna()
        if len(c) < 80:
            continue
        e = c.ewm(span=60, adjust=False).mean()
        above0_cols[t] = c > e
        dist = float(c.iloc[-1] / e.iloc[-1] - 1) * 100
        detail.append({
            "t": t,
            "dist": round(dist, 1),
            "up0": dist > 0,          # 無緩衝：站上 EMA60
            "up2": dist > -2.0,       # 2% 緩衝：跌破 EMA60 達 2% 以上才算破
        })
    h0 = sum(d["up0"] for d in detail)
    h2 = sum(d["up2"] for d in detail)
    h_all = pd.DataFrame(above0_cols).sum(axis=1)
    h_ma10 = h_all.rolling(10).mean()
    # 方向：10日均線近兩週（10個交易日）的變化量
    delta10 = float(h_ma10.iloc[-1] - h_ma10.iloc[-11]) if len(h_ma10.dropna()) > 11 else 0.0
    return {
        "detail": detail, "h0": int(h0), "h2": int(h2), "n": len(detail),
        "series": [int(x) for x in h_all.iloc[-60:].tolist()],
        "ma_series": [round(float(x), 2) for x in h_ma10.iloc[-60:].tolist()],
        "delta10": round(delta10, 2),
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
    if key=='spy60':   return 'yellow' if val > 4.44 or val < -5 else 'green'
    if key=='vixma20': return 'yellow' if val > 0 else 'green'
    return 'green'
# ── Win-rate lookup table ─────────────────────────────────────────
_WR_TABLE = [
    {"dev":"5~10%","vix":"30~40","wr":100.0,"avg":8.11,"n":5},
    {"dev":"0~2%","vix":"30~40","wr":100.0,"avg":11.28,"n":7},
    {"dev":"5~10%","vix":"25~30","wr":100.0,"avg":8.1,"n":44},
    {"dev":"2~5%","vix":"30~40","wr":100.0,"avg":9.54,"n":18},
    {"dev":"2~5%","vix":"25~30","wr":100.0,"avg":9.47,"n":54},
    {"dev":"5~10%","vix":"20~25","wr":92.8,"avg":5.12,"n":111},
    {"dev":"2~5%","vix":"20~25","wr":88.0,"avg":5.3,"n":208},
    {"dev":"5~10%","vix":"18~20","wr":87.5,"avg":3.74,"n":32},
    {"dev":"0~2%","vix":"25~30","wr":83.9,"avg":7.99,"n":31},
    {"dev":"-5~-2%","vix":"15~18","wr":83.3,"avg":4.67,"n":12},
    {"dev":"-2~0%","vix":"15~18","wr":82.2,"avg":4.11,"n":174},
    {"dev":"-5~-2%","vix":"25~30","wr":81.2,"avg":5.3,"n":16},
    {"dev":"5~10%","vix":"15~18","wr":80.7,"avg":2.32,"n":57},
    {"dev":"-5~-2%","vix":"20~25","wr":79.3,"avg":3.65,"n":58},
    {"dev":"-2~0%","vix":"18~20","wr":78.4,"avg":3.72,"n":88},
    {"dev":"-2~0%","vix":"12~15","wr":77.8,"avg":2.63,"n":99},
    {"dev":"-2~0%","vix":"20~25","wr":76.9,"avg":3.4,"n":121},
    {"dev":"2~5%","vix":"18~20","wr":75.2,"avg":2.18,"n":222},
    {"dev":"0~2%","vix":"<12","wr":73.9,"avg":1.96,"n":176},
    {"dev":"2~5%","vix":"<12","wr":73.5,"avg":1.75,"n":377},
    {"dev":"0~2%","vix":"15~18","wr":71.7,"avg":2.17,"n":353},
    {"dev":"0~2%","vix":"12~15","wr":71.2,"avg":2.34,"n":489},
    {"dev":"2~5%","vix":"12~15","wr":69.6,"avg":1.05,"n":843},
    {"dev":"0~2%","vix":"20~25","wr":66.7,"avg":1.45,"n":102},
    {"dev":"0~2%","vix":"18~20","wr":64.9,"avg":1.46,"n":134},
    {"dev":"5~10%","vix":"12~15","wr":64.7,"avg":1.16,"n":68},
    {"dev":"2~5%","vix":"15~18","wr":61.7,"avg":0.65,"n":564},
    {"dev":"-2~0%","vix":"30~40","wr":60.0,"avg":4.75,"n":5},
    {"dev":"-2~0%","vix":"25~30","wr":53.8,"avg":2.54,"n":26},
    {"dev":"-5~-2%","vix":"18~20","wr":50.0,"avg":2.5,"n":34},
    {"dev":"-5~-2%","vix":"30~40","wr":27.3,"avg":1.33,"n":11},
    {"dev":"5~10%","vix":"<12","wr":0.0,"avg":-5.47,"n":8},
]
_WR_INDEX = {(r["dev"], r["vix"]): r for r in _WR_TABLE}
# 20 日（1個月）持有勝率表（2000–2026，年線之上）
_WR_TABLE_20D = [
    {"dev":"0~2%","vix":"30~40","wr":100.0,"avg":4.96,"n":7},
    {"dev":"-2~0%","vix":"30~40","wr":100.0,"avg":7.57,"n":5},
    {"dev":"2~5%","vix":"25~30","wr":87.3,"avg":3.56,"n":55},
    {"dev":"2~5%","vix":"30~40","wr":88.9,"avg":3.31,"n":18},
    {"dev":"-5~-2%","vix":"18~20","wr":85.3,"avg":3.31,"n":34},
    {"dev":"2~5%","vix":"20~25","wr":85.0,"avg":2.16,"n":207},
    {"dev":"-2~0%","vix":"15~18","wr":81.5,"avg":1.87,"n":173},
    {"dev":"0~2%","vix":"25~30","wr":90.3,"avg":4.14,"n":31},
    {"dev":"-5~-2%","vix":"25~30","wr":75.0,"avg":1.49,"n":16},
    {"dev":"5~10%","vix":"18~20","wr":77.4,"avg":1.52,"n":31},
    {"dev":"5~10%","vix":"20~25","wr":74.1,"avg":1.44,"n":112},
    {"dev":"-2~0%","vix":"18~20","wr":77.3,"avg":2.04,"n":88},
    {"dev":"-5~-2%","vix":"20~25","wr":69.0,"avg":1.31,"n":58},
    {"dev":"5~10%","vix":"25~30","wr":63.6,"avg":0.63,"n":44},
    {"dev":"0~2%","vix":"<12","wr":73.6,"avg":0.85,"n":174},
    {"dev":"2~5%","vix":"<12","wr":68.7,"avg":0.62,"n":374},
    {"dev":"0~2%","vix":"15~18","wr":62.8,"avg":0.54,"n":352},
    {"dev":"0~2%","vix":"12~15","wr":66.9,"avg":0.66,"n":490},
    {"dev":"2~5%","vix":"12~15","wr":63.0,"avg":0.40,"n":844},
    {"dev":"2~5%","vix":"18~20","wr":62.9,"avg":0.50,"n":224},
    {"dev":"0~2%","vix":"20~25","wr":64.1,"avg":0.67,"n":103},
    {"dev":"5~10%","vix":"15~18","wr":63.8,"avg":0.14,"n":58},
    {"dev":"2~5%","vix":"15~18","wr":63.7,"avg":0.13,"n":564},
    {"dev":"-2~0%","vix":"20~25","wr":69.4,"avg":1.45,"n":124},
    {"dev":"-2~0%","vix":"12~15","wr":51.5,"avg":-0.31,"n":99},
    {"dev":"-2~0%","vix":"25~30","wr":57.1,"avg":-0.07,"n":28},
    {"dev":"0~2%","vix":"18~20","wr":56.3,"avg":-0.39,"n":135},
    {"dev":"-5~-2%","vix":"15~18","wr":16.7,"avg":-0.99,"n":12},
    {"dev":"-5~-2%","vix":"30~40","wr":45.5,"avg":-1.80,"n":11},
    {"dev":"5~10%","vix":"12~15","wr":56.7,"avg":0.13,"n":67},
    {"dev":"5~10%","vix":"30~40","wr":20.0,"avg":-1.46,"n":5},
    {"dev":"5~10%","vix":"<12","wr":0.0,"avg":-3.92,"n":8},
]
_WR_INDEX_20D = {(r["dev"], r["vix"]): r for r in _WR_TABLE_20D}
def lookup_winrate_20d(dev, vix):
    return _WR_INDEX_20D.get((_dev_bucket(dev), _vix_bucket(vix)))
def _dev_bucket(dev):
    if dev < -5:  return "<-5%"
    if dev < -2:  return "-5~-2%"
    if dev < 0:   return "-2~0%"
    if dev < 2:   return "0~2%"
    if dev < 5:   return "2~5%"
    if dev < 10:  return "5~10%"
    return ">10%"
def _vix_bucket(vix):
    if vix < 12:  return "<12"
    if vix < 15:  return "12~15"
    if vix < 18:  return "15~18"
    if vix < 20:  return "18~20"
    if vix < 25:  return "20~25"
    if vix < 30:  return "25~30"
    if vix < 40:  return "30~40"
    return ">40"
def lookup_winrate(dev, vix):
    return _WR_INDEX.get((_dev_bucket(dev), _vix_bucket(vix)))
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
# ── Main ──────────────────────────────────────────────────────────
D = fetch_data()
st_cme  = status('cme',   D['cme_excess_10d'])
st_hyg  = status('hyg',   D['hyg_iei_20d'])
st_iwm  = status('iwm',   D['iwm_spy_60d'])
st_vixr = status('vixr',  D['vix_ratio'])
st_xlp  = status('xlp',   D['xlp_xly_20d'])
st_rsp  = status('rsp',   D['rsp_spy_60d'])
st_br    = status('brdth', D['sector_breadth'])
st_spy60   = status('spy60',   D['spy_vs_60ma'])
st_vixma20 = status('vixma20', D['vix_vs_ma20'])
# IWF/IWD 已從核心移除：60天回測倍率僅0.94x（低於基準），不具預測能力

# ── Combo triggers（先算，整體燈號要用）──
cme_triggered = st_cme in ('yellow','red')
combo1 = cme_triggered and st_iwm != 'green'                    # CME + 小型股弱勢
combo2 = cme_triggered and st_xlp in ('yellow','red')           # 訊號1：CME + 防禦輪動 → 壓力否決
combo3 = cme_triggered and st_iwm != 'green' and st_xlp in ('yellow','red')
uvxy_warn = st_rsp == 'red' or st_br in ('yellow','red')

# ── 整體燈號：否決邏輯，不數燈 ──
# 紅 = 任一 AND 組合成立（否決新倉）；黃 = 只有單燈亮（情境警示，不否決）；綠 = 全無
core_statuses = [st_cme, st_xlp, st_iwm]
n_lit = sum(1 for s in core_statuses if s != 'green')
veto_active = combo1 or combo2
if veto_active:
    overall = 'red'
elif n_lit >= 1:
    overall = 'yellow'
else:
    overall = 'green'
# ── Header ───────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 📊 大盤壓力儀表板")
    st.markdown(f"<span style='color:#64748b;font-size:0.78rem'>數據截至 {D['as_of']} &nbsp;｜&nbsp; SPY ${D['spy_price']} &nbsp;｜&nbsp; 200日均線 ${D['spy_200ma_val']} (+{D['spy_vs_200ma']}%，牛市確立)</span>", unsafe_allow_html=True)
with col_refresh:
    if st.button("🔄 更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
st.markdown("---")
# ── Overall ──────────────────────────────────────────────────────
ov_map = {
    'green':  ('無否決訊號',
               '核心壓力指標全數安靜。能否進場／加碼由個股觸發與領頭股燈號決定，見下方「進場許可」。'),
    'yellow': ('單一核心指標亮燈 — 情境警示，不構成否決',
               '單燈不否決。回測：單獨「防禦輪動」亮時 SPY 一個月報酬反而最好（爬憂慮之牆）；'
               '單獨 CME 亮對指數無害但對熱門動能股不利。動作：繫安全帶、不追延伸，但<b>不因單燈停止合規進場</b>。'),
    'red':    ('壓力否決成立 — AND 組合觸發',
               'CME 與防禦輪動／小型股弱勢同時亮。回測：此時新進場 EV 由 +15% 轉負、跌 5% 機率翻倍。'
               '動作：<b>不開新倉、不加碼</b>；既有持倉交給停損，不預先賣出（提前減碼已驗證負 EV）。'),
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
    (combo3, "46.2%", "3.05x", "CME +5~8%  ＋  IWM/SPY 60日 <−3%  ＋  防禦輪動 XLP/XLY >1%", "三核心同步觸發，下跌>5%機率 46.2%（n=39，2016後）；Permutation test p<0.001，95% CI 32~61%"),
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
# ── Core indicators ───────────────────────────────────────────────
st.markdown('<div class="section-hdr">核心預警指標（否決燈 · 需 AND 組合才否決）</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
cme_v = D['cme_excess_10d']
with c1:
    desc = (f"CME跑贏SPY {cme_v}%，警示區間（單獨倍率僅1.10x，須配合IWM/SPY同時觸發才有力）" if st_cme in ('yellow','red')
            else f"CME明顯跑輸（{cme_v}%），歷史看漲信號" if cme_v < -3
            else f"CME相對報酬中性，無明顯信號")
    card("CME超額報酬（10日）", fmt(cme_v), st_cme, desc, D['cme_series'], inv=True,
         lift="1.41x（單獨）/ 3.08x（+IWM）", note="否決燈 · 需與防禦輪動或小型股同亮")
xl_v = D['xlp_xly_20d']
with c2:
    if st_xlp in ('yellow','red') and not combo2:
        desc = (f"防禦輪動 {xl_v}%，但 CME 未同亮 → 單燈情境、不否決。"
                f"回測：單獨此燈亮時 SPY 一個月 +1.83%／勝率 73%（爬憂慮之牆），不是離場訊號")
    elif st_xlp in ('yellow','red') and combo2:
        desc = f"防禦輪動 {xl_v}% 且 CME 同亮 → 訊號1 成立，否決新倉"
    else:
        desc = f"景氣循環領先（{fmt(xl_v)}），市場情緒偏進取"
    card("防禦輪動 XLP/XLY（20日）", fmt(xl_v), st_xlp, desc, D['xlp_series'], inv=True,
         lift="2.50x", note="否決燈 · 需與 CME 同亮")
iwm_v = D['iwm_spy_60d']
with c3:
    desc = (f"小型股60日跑輸大型股 {abs(iwm_v)}%，風險偏好惡化" if st_iwm=='red'
            else f"小型股相對弱勢（{iwm_v}%），需觀察" if st_iwm=='yellow'
            else f"小型股同步（{fmt(iwm_v)}），風險偏好正常")
    card("IWM/SPY 小型股（60日）", fmt(iwm_v), st_iwm, desc, D['iwm_series'],
         lift="1.55x", note="否決燈 · 需與 CME 同亮")
# ── SPY 季線乖離率 ─────────────────────────────────────────────────
if st_spy60 == 'yellow':
    st.markdown(f'<div class="uvxy-warn">⚡ <b>SPY 季線乖離率示警</b>——距 EMA60 偏離 {fmt(D["spy_vs_60ma"])}，已超過 90 分位（+4.44%），留意過熱修正風險</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="uvxy-ok">SPY 季線乖離率：正常（距 EMA60 {fmt(D["spy_vs_60ma"])}，低於 90 分位 +4.44%）</div>', unsafe_allow_html=True)
y1, y2, y3, y4 = st.columns(4)
with y1:
    s60v = D['spy_vs_60ma']
    desc = (f"SPY 大幅超買，距季線 {s60v}%，歷史修正風險升高" if st_spy60=='red' and s60v>0
            else f"SPY 跌破季線 {abs(s60v)}%，短線走弱" if st_spy60=='red' and s60v<0
            else f"SPY 偏離季線 {fmt(s60v)}，留意過熱" if st_spy60=='yellow' and s60v>0
            else f"SPY 偏離季線 {fmt(s60v)}，接近季線支撐" if st_spy60=='yellow' and s60v<0
            else f"SPY 距季線 {fmt(s60v)}，位置健康（季線 ${D['spy_60ma_val']}）")
    card("SPY 季線乖離率（EMA 60）", fmt(s60v), st_spy60, desc, D['spy60_series'])
with y2:
    vmv = D['vix_vs_ma20']
    desc = (f"VIX 漲破 MA20，偏離 {fmt(vmv)}，恐慌情緒升溫" if st_vixma20 == 'yellow'
            else f"VIX 低於 MA20（{fmt(vmv)}），恐慌情緒平穩（MA20：{D['vix_ma20_val']}）")
    card("VIX 距 MA20", fmt(vmv), st_vixma20, desc, D['vix_ma20_series'], inv=True)
with y3:
    _dev_now   = D['spy_vs_60ma']
    _vix_now   = D['vix']
    _above260  = D['spy_vs_260ma'] > 0
    _wr_row    = lookup_winrate(_dev_now, _vix_now)
    _wr_row_20 = lookup_winrate_20d(_dev_now, _vix_now)
    _dev_bkt   = _dev_bucket(_dev_now)
    _vix_bkt   = _vix_bucket(_vix_now)
    def _wr_color(wr):
        if wr >= 90: return '#4ade80'
        if wr >= 70: return '#86efac'
        if wr >= 60: return '#fb923c'
        return '#f87171'
    if _above260:
        _all_wr_20 = [r['wr'] for r in _WR_TABLE_20D]
        _all_wr_60 = [r['wr'] for r in _WR_TABLE]
        def _percentile_in(wr, all_wrs):
            return round(sum(1 for w in all_wrs if w <= wr) / len(all_wrs) * 100)
        def _period_row(label, row, all_wrs):
            if not row:
                return (f"<div style='flex:1;opacity:0.4'>"
                        f"<div style='font-size:0.6rem;color:#64748b'>{label}</div>"
                        f"<div style='font-size:1.1rem;font-weight:700;color:#475569'>—</div></div>")
            wc   = _wr_color(row['wr'])
            pctl = _percentile_in(row['wr'], all_wrs)
            pctl_col = '#4ade80' if pctl >= 75 else '#fbbf24' if pctl >= 50 else '#f87171'
            return (f"<div style='flex:1'>"
                    f"<div style='font-size:0.6rem;color:#64748b;margin-bottom:2px'>{label}</div>"
                    f"<div style='font-size:1.3rem;font-weight:800;color:{wc}'>{row['wr']:.1f}%</div>"
                    f"<div style='font-size:0.65rem;color:{pctl_col}'>P{pctl} 百分位</div>"
                    f"<div style='font-size:0.62rem;color:#475569'>n={row['n']}</div></div>")
        row1 = _period_row("1個月勝率", _wr_row_20, _all_wr_20)
        row2 = _period_row("3個月勝率", _wr_row,    _all_wr_60)
        border_col = _wr_color(_wr_row['wr']) if _wr_row else '#334155'
        st.markdown(f"""
        <div class="metric-card" style="border-color:{border_col}55">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
            <span style="font-size:0.72rem;color:#94a3b8">進場勝率（EMA260 之上）</span>
            <span style="background:#14532d;color:#4ade80;padding:2px 7px;border-radius:8px;font-size:0.62rem;font-weight:700">✓</span>
          </div>
          <div style="display:flex;gap:12px">{row1}{row2}</div>
          <div class="desc-text" style="margin-top:6px">乖離率桶 {_dev_bkt} × VIX桶 {_vix_bkt}<br>資料期間 2000–2026，含存活者偏差</div>
        </div>""", unsafe_allow_html=True)
    elif not _above260:
        st.markdown(f"""
        <div class="metric-card card-red">
          <div style="font-size:0.72rem;color:#94a3b8">進場勝率</div>
          <div style="font-size:1rem;font-weight:700;color:#f87171;margin:4px 0 2px">年線以下，勝率不適用</div>
          <div class="desc-text">SPY 位於 EMA260 下方 {fmt(D['spy_vs_260ma'])}，回測條件不成立</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:0.72rem;color:#94a3b8">進場勝率</div>
          <div style="font-size:1rem;font-weight:700;color:#475569;margin:4px 0 2px">查無資料</div>
          <div class="desc-text">乖離率桶 {_dev_bkt} × VIX桶 {_vix_bkt}</div>
        </div>""", unsafe_allow_html=True)
with y4:
    if _above260 and (_wr_row or _wr_row_20):
        def _ev_block(label, row):
            if not row:
                return (f"<div style='flex:1;opacity:0.4'>"
                        f"<div style='font-size:0.6rem;color:#64748b'>{label}</div>"
                        f"<div style='font-size:0.95rem;font-weight:700;color:#475569'>—</div></div>")
            avg = row['avg']; wr = row['wr']; ev = round(wr/100*avg, 2)
            ac = '#4ade80' if avg>0 else '#f87171'
            ec = '#4ade80' if ev>0  else '#f87171'
            return (f"<div style='flex:1'>"
                    f"<div style='font-size:0.6rem;color:#64748b;margin-bottom:2px'>{label}</div>"
                    f"<div style='font-size:0.7rem;color:#475569'>期望值</div>"
                    f"<div style='font-size:1.15rem;font-weight:800;color:{ec}'>{'+' if ev>0 else ''}{ev:.2f}%</div>"
                    f"<div style='font-size:0.7rem;color:#475569;margin-top:3px'>平均報酬</div>"
                    f"<div style='font-size:1.15rem;font-weight:800;color:{ac}'>{'+' if avg>0 else ''}{avg:.2f}%</div>"
                    f"</div>")
        b1 = _ev_block("1個月", _wr_row_20)
        b2 = _ev_block("3個月", _wr_row)
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:0.72rem;color:#94a3b8;margin-bottom:6px">期望值 &amp; 平均報酬</div>
          <div style="display:flex;gap:12px">{b1}{b2}</div>
          <div class="desc-text" style="margin-top:6px">期望值 = 勝率 × 平均報酬</div>
        </div>""", unsafe_allow_html=True)
    elif not _above260:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:0.72rem;color:#94a3b8">期望值 &amp; 平均報酬</div>
          <div style="font-size:1rem;font-weight:700;color:#475569;margin:6px 0 4px">—</div>
          <div class="desc-text">年線以下，數據不適用</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:0.72rem;color:#94a3b8">期望值 &amp; 平均報酬</div>
          <div style="font-size:1rem;font-weight:700;color:#475569;margin:6px 0 4px">查無資料</div>
        </div>""", unsafe_allow_html=True)
# ── 領頭股前哨 ────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">領頭股前哨（權值股隊形 · 不計入整體燈號）</div>', unsafe_allow_html=True)
ldr_status = None
L = None
try:
    _leader_list = fetch_leader_list()
    _list_is_live = tuple(_leader_list) != tuple(FALLBACK_LEADERS)
    L = fetch_leaders(_leader_list)
    # 兩級判定：紅 = 2% 緩衝後仍 ≤4/8（歷史上僅對應大型頭部）；黃 = 無緩衝 ≤4/8
    ldr_status = 'red' if L['h2'] <= 4 else 'yellow' if L['h0'] <= 4 else 'green'
    spy_far_from_high = D['spy_dist_hi'] < -3.0
    chips = "".join(
        f'<span class="{"ldr-chip-up" if d["up0"] else "ldr-chip-down"}">'
        f'{d["t"]} {"✓" if d["up0"] else "✗"} {d["dist"]:+.1f}%</span>'
        for d in L['detail'])
    if ldr_status == 'red':
        ldr_title, ldr_desc = (
            f"紅燈：隊形嚴重瓦解（無緩衝 {L['h0']}/{L['n']}，2%緩衝 {L['h2']}/{L['n']}）",
            "2% 緩衝後仍過半破線——2016 年以來此狀態僅出現於 2022/1、2025/3、2026/3 三次大型頭部前。"
            "建議執行部分減碼並準備避險，等待核心壓力訊號確認。")
    elif ldr_status == 'yellow':
        ldr_title, ldr_desc = (
            f"黃燈：隊形出現裂痕（無緩衝 {L['h0']}/{L['n']} 站上 EMA60）",
            "半數以上權值股跌破 EMA60 而指數仍在高位。回測（2016–2026 滾動權值名單）：此狀態下 60 日內"
            "SPY 跌 5%+ 機率約 51%（基準 27%），誤報率約五成——對應動作：暫停新加碼、收緊移動停損。")
    else:
        ldr_title, ldr_desc = (
            f"隊形完整（{L['h0']}/{L['n']} 站上 EMA60）",
            "權值股中期趨勢健康，指數上漲有實質支撐。裂痕定義：無緩衝 ≤4 亮黃、2% 緩衝 ≤4 亮紅。")
    if spy_far_from_high:
        ldr_desc += f"　⚠ SPY 已距 252 日高 {D['spy_dist_hi']}%，前哨統計以高點附近為前提，此時參考壓力訊號為主。"
    lc1, lc2 = st.columns([1.6, 1])
    with lc1:
        st.markdown(f"""
        <div class="metric-card card-{ldr_status}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <span style="font-size:0.72rem;color:#94a3b8">領頭股健康度（收盤 vs EMA60）
              <span style="font-size:0.6rem;color:#6366f1;background:#1e1b4b;padding:1px 5px;border-radius:4px">
              {'名單每日自動更新' if _list_is_live else '⚠ 使用備援名單'}</span></span>
            {BADGE[ldr_status]}
          </div>
          <div class="{VAL_CLASS[ldr_status]}" style="margin:4px 0 6px">{ldr_title}</div>
          <div style="margin-bottom:6px">{chips}</div>
          <div class="desc-text">{ldr_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    with lc2:
        # 方向標籤：10日均線近兩週變化
        _d10 = L['delta10']
        if _d10 <= -0.5:
            dir_txt, dir_col = f"↘ 惡化中（10日均兩週 {_d10:+.1f} 檔）", '#f87171'
        elif _d10 >= 0.5:
            dir_txt, dir_col = f"↗ 修復中（10日均兩週 {_d10:+.1f} 檔）", '#4ade80'
        else:
            dir_txt, dir_col = f"→ 持平（10日均兩週 {_d10:+.1f} 檔）", '#94a3b8'
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">'
            f'<span style="font-size:0.65rem;color:#475569">健康度近 60 日（灰＝每日 · 粗線＝10日均 · 虛線＝裂痕線）</span>'
            f'<span style="font-size:0.72rem;font-weight:700;color:{dir_col}">{dir_txt}</span></div>',
            unsafe_allow_html=True)
        figL = go.Figure()
        figL.add_trace(go.Scatter(
            y=L['series'], mode='lines',
            line=dict(color='#334155', width=1, shape='hv')))
        figL.add_trace(go.Scatter(
            y=L['ma_series'], mode='lines',
            line=dict(color=dir_col, width=2.2)))
        figL.add_hline(y=4, line=dict(color='#7f1d1d', width=1, dash='dot'))
        figL.update_layout(
            height=96, margin=dict(l=0, r=0, t=2, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, range=[-0.3, L['n'] + 0.3]),
            showlegend=False)
        st.plotly_chart(figL, use_container_width=True,
                        config={'displayModeBar': False}, key="chart_leaders")
except Exception as _e:
    st.markdown(f'<div class="uvxy-ok">領頭股前哨載入失敗（{_e}），不影響其他指標。</div>',
                unsafe_allow_html=True)

# ── 進場許可總結：否決燈（AND）＋ 領頭股燈號（帶記憶）合成一個答案 ──
st.markdown('<div class="section-hdr">進場許可（否決燈 × 領頭股燈號）</div>', unsafe_allow_html=True)
if ldr_status is not None and L is not None:
    _d10 = L['delta10']
    if ldr_status == 'green':
        ldr_open, ldr_reason = True,  f"領頭股綠燈（{L['h0']}/{L['n']}）"
    elif ldr_status == 'yellow' and _d10 <= -0.5:
        ldr_open, ldr_reason = True,  "惡化黃（綠→黃）：回測 EV ≈ 綠燈，仍屬開門狀態"
    elif ldr_status == 'yellow':
        ldr_open, ldr_reason = False, "修復黃（紅→黃）：回測 EV ≈ 紅燈，禁區，等真正轉綠"
    else:
        ldr_open, ldr_reason = False, f"領頭股紅燈（{L['h0']}/{L['n']}）"

    can_enter = ldr_open and not veto_active
    if can_enter:
        box_cls, title = 'overall-green', '🟢 可開新倉／可評估加碼'
        body = (f"領頭股：{ldr_reason}。壓力否決：未成立。"
                f"→ 進場與否回到個股觸發（創 63 日新高／合規加碼點）與 R、Heat、保本閘門。")
    elif veto_active:
        _which = "、".join(x for x, ok in [("CME＋防禦輪動", combo2), ("CME＋小型股弱勢", combo1)] if ok)
        box_cls, title = 'overall-red', '🔴 不開新倉、不加碼 — 壓力否決成立'
        body = f"AND 組合觸發（{_which}）。持倉交給停損，不預先賣出。領頭股：{ldr_reason}。"
    else:
        box_cls, title = 'overall-yellow', '🟡 不開新倉 — 領頭股燈號未開門'
        body = f"{ldr_reason}。壓力否決未成立，但領頭股燈號不在開門狀態（綠或惡化黃）。等燈號回綠。"
    if n_lit >= 1 and not veto_active:
        body += " 　※ 目前有單一壓力燈亮：屬情境警示，不否決，勿追延伸。"
    st.markdown(f"""
    <div class="{box_cls}" style="margin-bottom:14px">
      <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0">{title}</div>
      <div style="color:#94a3b8;font-size:0.78rem;margin-top:6px">{body}</div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="uvxy-ok">領頭股資料未載入，無法合成進場許可；請以壓力否決與個股觸發判斷。</div>',
                unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
# ── Context indicators ────────────────────────────────────────────
st.markdown('<div class="section-hdr">輔助情境指標（參考用途 · 不計入整體燈號）</div>', unsafe_allow_html=True)
x1, x2, x3 = st.columns(3)
with x1:
    vr = D['vix_ratio']
    desc = (f"期限結構倒掛（{vr}），近月恐慌高於遠月" if st_vixr!='green'
            else f"期限結構正常 Contango（{vr}），情緒穩定")
    card("VIX/VIX3M 期限結構", str(vr), st_vixr, desc, D['vixr_series'], inv=True)
with x2:
    rs = D['rsp_spy_60d']
    desc = (f"等權重跑輸市值加權 {abs(rs)}%，廣度惡化" if st_rsp=='red'
            else f"廣度正常（{fmt(rs)}）")
    card("RSP/SPY 等權廣度（60日）", fmt(rs), st_rsp, desc, D['rsp_series'])
with x3:
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
