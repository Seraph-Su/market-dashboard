import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
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
  .insight-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
  }
</style>
""", unsafe_allow_html=True)

# ── Complete datasets ─────────────────────────────────────────────
DATA = {
    "3m_上": [
        {"dev":"-5~-2%","vix":"17~20","wr":59.1,"avg":3.2,"n":44},
        {"dev":"-5~-2%","vix":"20~23","wr":77.5,"avg":3.43,"n":40},
        {"dev":"-5~-2%","vix":"23~25","wr":83.3,"avg":4.13,"n":18},
        {"dev":"-5~-2%","vix":"25~28","wr":75.0,"avg":5.28,"n":12},
        {"dev":"-5~-2%","vix":"28~30","wr":100.0,"avg":5.37,"n":4},
        {"dev":"-5~-2%","vix":"30~35","wr":14.3,"avg":-1.42,"n":7},
        {"dev":"-5~-2%","vix":">35","wr":66.7,"avg":7.19,"n":6},
        {"dev":"-2~0%","vix":"11~13","wr":70.0,"avg":1.87,"n":10},
        {"dev":"-2~0%","vix":"13~15","wr":79.1,"avg":2.68,"n":91},
        {"dev":"-2~0%","vix":"15~17","wr":83.3,"avg":4.43,"n":114},
        {"dev":"-2~0%","vix":"17~20","wr":79.1,"avg":3.63,"n":148},
        {"dev":"-2~0%","vix":"20~23","wr":81.6,"avg":4.1,"n":87},
        {"dev":"-2~0%","vix":"23~25","wr":64.7,"avg":1.59,"n":34},
        {"dev":"-2~0%","vix":"25~28","wr":55.0,"avg":1.6,"n":20},
        {"dev":"-2~0%","vix":"28~30","wr":50.0,"avg":5.69,"n":6},
        {"dev":"-2~0%","vix":"30~35","wr":33.3,"avg":-0.59,"n":3},
        {"dev":"-2~0%","vix":">35","wr":100.0,"avg":12.76,"n":2},
        {"dev":"0~2%","vix":"<11","wr":95.5,"avg":3.66,"n":67},
        {"dev":"0~2%","vix":"11~13","wr":64.0,"avg":1.25,"n":292},
        {"dev":"0~2%","vix":"13~15","wr":74.2,"avg":2.87,"n":306},
        {"dev":"0~2%","vix":"15~17","wr":71.3,"avg":2.12,"n":261},
        {"dev":"0~2%","vix":"17~20","wr":68.1,"avg":1.81,"n":226},
        {"dev":"0~2%","vix":"20~23","wr":65.6,"avg":1.25,"n":93},
        {"dev":"0~2%","vix":"23~25","wr":77.8,"avg":3.5,"n":9},
        {"dev":"0~2%","vix":"25~28","wr":81.8,"avg":7.78,"n":22},
        {"dev":"0~2%","vix":"28~30","wr":88.9,"avg":8.51,"n":9},
        {"dev":"0~2%","vix":"30~35","wr":100.0,"avg":11.1,"n":5},
        {"dev":"0~2%","vix":">35","wr":100.0,"avg":11.73,"n":2},
        {"dev":"2~5%","vix":"<11","wr":79.4,"avg":2.33,"n":170},
        {"dev":"2~5%","vix":"11~13","wr":69.2,"avg":1.02,"n":535},
        {"dev":"2~5%","vix":"13~15","wr":69.7,"avg":1.17,"n":515},
        {"dev":"2~5%","vix":"15~17","wr":60.5,"avg":0.29,"n":400},
        {"dev":"2~5%","vix":"17~20","wr":70.7,"avg":1.9,"n":386},
        {"dev":"2~5%","vix":"20~23","wr":88.1,"avg":4.99,"n":168},
        {"dev":"2~5%","vix":"23~25","wr":87.5,"avg":6.58,"n":40},
        {"dev":"2~5%","vix":"25~28","wr":100.0,"avg":9.56,"n":39},
        {"dev":"2~5%","vix":"28~30","wr":100.0,"avg":9.24,"n":15},
        {"dev":"2~5%","vix":"30~35","wr":100.0,"avg":9.53,"n":16},
        {"dev":"2~5%","vix":">35","wr":100.0,"avg":9.67,"n":2},
        {"dev":"5~8%","vix":"11~13","wr":64.0,"avg":1.37,"n":25},
        {"dev":"5~8%","vix":"13~15","wr":56.0,"avg":0.13,"n":50},
        {"dev":"5~8%","vix":"15~17","wr":75.0,"avg":1.51,"n":40},
        {"dev":"5~8%","vix":"17~20","wr":89.8,"avg":3.9,"n":49},
        {"dev":"5~8%","vix":"20~23","wr":88.3,"avg":4.4,"n":60},
        {"dev":"5~8%","vix":"23~25","wr":97.7,"avg":6.42,"n":43},
        {"dev":"5~8%","vix":"25~28","wr":100.0,"avg":8.17,"n":30},
        {"dev":"5~8%","vix":"28~30","wr":100.0,"avg":8.32,"n":9},
        {"dev":"5~8%","vix":"30~35","wr":100.0,"avg":7.88,"n":4},
    ],
    "1m_上": [
        {"dev":"-5~-2%","vix":"17~20","wr":70.5,"avg":2.33,"n":44},
        {"dev":"-5~-2%","vix":"20~23","wr":75.0,"avg":2.05,"n":40},
        {"dev":"-5~-2%","vix":"23~25","wr":57.9,"avg":-0.12,"n":19},
        {"dev":"-5~-2%","vix":"25~28","wr":80.0,"avg":1.28,"n":15},
        {"dev":"-5~-2%","vix":"30~35","wr":14.3,"avg":-5.75,"n":7},
        {"dev":"-5~-2%","vix":">35","wr":83.3,"avg":4.33,"n":6},
        {"dev":"-2~0%","vix":"11~13","wr":20.0,"avg":-2.13,"n":10},
        {"dev":"-2~0%","vix":"13~15","wr":53.8,"avg":-0.12,"n":91},
        {"dev":"-2~0%","vix":"15~17","wr":82.5,"avg":1.91,"n":114},
        {"dev":"-2~0%","vix":"17~20","wr":78.4,"avg":1.99,"n":148},
        {"dev":"-2~0%","vix":"20~23","wr":70.5,"avg":1.69,"n":88},
        {"dev":"-2~0%","vix":"23~25","wr":67.6,"avg":0.76,"n":37},
        {"dev":"-2~0%","vix":"25~28","wr":57.1,"avg":-0.48,"n":21},
        {"dev":"-2~0%","vix":"28~30","wr":57.1,"avg":1.17,"n":7},
        {"dev":"-2~0%","vix":">35","wr":100.0,"avg":10.28,"n":2},
        {"dev":"0~2%","vix":"<11","wr":76.1,"avg":0.99,"n":67},
        {"dev":"0~2%","vix":"11~13","wr":64.0,"avg":0.37,"n":292},
        {"dev":"0~2%","vix":"13~15","wr":71.6,"avg":0.97,"n":306},
        {"dev":"0~2%","vix":"15~17","wr":63.6,"avg":0.57,"n":261},
        {"dev":"0~2%","vix":"17~20","wr":58.3,"avg":0.04,"n":228},
        {"dev":"0~2%","vix":"20~23","wr":62.8,"avg":0.55,"n":94},
        {"dev":"0~2%","vix":"23~25","wr":77.8,"avg":2.73,"n":9},
        {"dev":"0~2%","vix":"25~28","wr":86.4,"avg":3.87,"n":22},
        {"dev":"0~2%","vix":"28~30","wr":100.0,"avg":4.81,"n":9},
        {"dev":"0~2%","vix":"30~35","wr":100.0,"avg":4.75,"n":5},
        {"dev":"2~5%","vix":"<11","wr":71.8,"avg":0.78,"n":170},
        {"dev":"2~5%","vix":"11~13","wr":65.2,"avg":0.49,"n":535},
        {"dev":"2~5%","vix":"13~15","wr":61.9,"avg":0.34,"n":515},
        {"dev":"2~5%","vix":"15~17","wr":59.1,"avg":-0.07,"n":401},
        {"dev":"2~5%","vix":"17~20","wr":69.1,"avg":0.71,"n":398},
        {"dev":"2~5%","vix":"20~23","wr":82.7,"avg":1.87,"n":168},
        {"dev":"2~5%","vix":"23~25","wr":95.0,"avg":3.32,"n":40},
        {"dev":"2~5%","vix":"25~28","wr":84.6,"avg":3.49,"n":39},
        {"dev":"2~5%","vix":"28~30","wr":93.3,"avg":3.94,"n":15},
        {"dev":"2~5%","vix":"30~35","wr":87.5,"avg":3.17,"n":16},
        {"dev":"5~8%","vix":"11~13","wr":48.0,"avg":-1.01,"n":25},
        {"dev":"5~8%","vix":"13~15","wr":54.0,"avg":0.18,"n":50},
        {"dev":"5~8%","vix":"15~17","wr":57.5,"avg":-0.28,"n":40},
        {"dev":"5~8%","vix":"17~20","wr":78.0,"avg":1.5,"n":50},
        {"dev":"5~8%","vix":"20~23","wr":68.3,"avg":0.94,"n":60},
        {"dev":"5~8%","vix":"23~25","wr":90.7,"avg":2.34,"n":43},
        {"dev":"5~8%","vix":"25~28","wr":83.3,"avg":2.19,"n":30},
        {"dev":"5~8%","vix":"28~30","wr":22.2,"avg":-3.23,"n":9},
        {"dev":"5~8%","vix":"30~35","wr":25.0,"avg":-1.4,"n":4},
    ],
    "3m_下": [
        {"dev":"<-10%","vix":"30~35","wr":78.6,"avg":4.74,"n":14},
        {"dev":"<-10%","vix":">35","wr":60.3,"avg":6.82,"n":116},
        {"dev":"-10~-5%","vix":"23~25","wr":35.7,"avg":-3.85,"n":28},
        {"dev":"-10~-5%","vix":"25~28","wr":60.8,"avg":0.3,"n":74},
        {"dev":"-10~-5%","vix":"28~30","wr":68.0,"avg":2.0,"n":50},
        {"dev":"-10~-5%","vix":"30~35","wr":78.7,"avg":4.87,"n":89},
        {"dev":"-10~-5%","vix":">35","wr":69.3,"avg":1.14,"n":101},
        {"dev":"-5~-2%","vix":"15~17","wr":100.0,"avg":7.91,"n":22},
        {"dev":"-5~-2%","vix":"17~20","wr":81.6,"avg":5.09,"n":49},
        {"dev":"-5~-2%","vix":"20~23","wr":44.5,"avg":-4.14,"n":119},
        {"dev":"-5~-2%","vix":"23~25","wr":35.5,"avg":-3.79,"n":76},
        {"dev":"-5~-2%","vix":"25~28","wr":53.6,"avg":-0.05,"n":125},
        {"dev":"-5~-2%","vix":"28~30","wr":50.0,"avg":-1.13,"n":36},
        {"dev":"-5~-2%","vix":"30~35","wr":72.9,"avg":4.65,"n":70},
        {"dev":"-5~-2%","vix":">35","wr":80.0,"avg":5.43,"n":40},
        {"dev":"-2~0%","vix":"13~15","wr":100.0,"avg":6.99,"n":16},
        {"dev":"-2~0%","vix":"15~17","wr":93.5,"avg":5.04,"n":31},
        {"dev":"-2~0%","vix":"17~20","wr":32.1,"avg":-8.29,"n":56},
        {"dev":"-2~0%","vix":"20~23","wr":23.5,"avg":-7.98,"n":98},
        {"dev":"-2~0%","vix":"23~25","wr":41.7,"avg":-1.09,"n":60},
        {"dev":"-2~0%","vix":"25~28","wr":48.6,"avg":0.85,"n":37},
        {"dev":"-2~0%","vix":"28~30","wr":46.7,"avg":1.75,"n":15},
        {"dev":"-2~0%","vix":"30~35","wr":72.0,"avg":3.02,"n":25},
        {"dev":"-2~0%","vix":">35","wr":85.7,"avg":10.9,"n":14},
        {"dev":"0~2%","vix":"13~15","wr":91.7,"avg":6.47,"n":12},
        {"dev":"0~2%","vix":"17~20","wr":38.5,"avg":-2.93,"n":39},
        {"dev":"0~2%","vix":"20~23","wr":30.0,"avg":-3.31,"n":60},
        {"dev":"0~2%","vix":"23~25","wr":41.5,"avg":-0.17,"n":41},
        {"dev":"0~2%","vix":"25~28","wr":41.9,"avg":0.69,"n":31},
        {"dev":"0~2%","vix":"28~30","wr":25.0,"avg":-0.8,"n":16},
        {"dev":"0~2%","vix":"30~35","wr":77.1,"avg":7.85,"n":35},
        {"dev":"0~2%","vix":">35","wr":90.0,"avg":9.13,"n":10},
        {"dev":"2~5%","vix":"17~20","wr":13.3,"avg":-7.85,"n":15},
        {"dev":"2~5%","vix":"20~23","wr":18.9,"avg":-4.48,"n":37},
        {"dev":"2~5%","vix":"23~25","wr":23.5,"avg":-2.77,"n":34},
        {"dev":"2~5%","vix":"25~28","wr":40.0,"avg":0.52,"n":20},
        {"dev":"2~5%","vix":"28~30","wr":77.8,"avg":5.53,"n":18},
        {"dev":"2~5%","vix":"30~35","wr":87.0,"avg":7.16,"n":23},
        {"dev":"2~5%","vix":">35","wr":100.0,"avg":9.66,"n":12},
        {"dev":"5~8%","vix":"17~20","wr":0.0,"avg":-9.82,"n":5},
        {"dev":"5~8%","vix":"28~30","wr":100.0,"avg":11.41,"n":3},
        {"dev":"5~8%","vix":"30~35","wr":100.0,"avg":10.08,"n":8},
        {"dev":"5~8%","vix":">35","wr":100.0,"avg":7.98,"n":7},
    ],
    "1m_下": [
        {"dev":"<-10%","vix":"30~35","wr":78.6,"avg":5.87,"n":14},
        {"dev":"<-10%","vix":">35","wr":70.7,"avg":5.07,"n":116},
        {"dev":"-10~-5%","vix":"23~25","wr":32.1,"avg":-2.02,"n":28},
        {"dev":"-10~-5%","vix":"25~28","wr":47.3,"avg":-1.58,"n":74},
        {"dev":"-10~-5%","vix":"28~30","wr":64.0,"avg":1.79,"n":50},
        {"dev":"-10~-5%","vix":"30~35","wr":69.2,"avg":2.63,"n":91},
        {"dev":"-10~-5%","vix":">35","wr":43.6,"avg":-1.9,"n":101},
        {"dev":"-5~-2%","vix":"15~17","wr":95.5,"avg":3.53,"n":22},
        {"dev":"-5~-2%","vix":"17~20","wr":65.3,"avg":1.13,"n":49},
        {"dev":"-5~-2%","vix":"20~23","wr":33.6,"avg":-2.08,"n":119},
        {"dev":"-5~-2%","vix":"23~25","wr":49.4,"avg":-1.03,"n":79},
        {"dev":"-5~-2%","vix":"25~28","wr":59.5,"avg":0.97,"n":131},
        {"dev":"-5~-2%","vix":"28~30","wr":63.9,"avg":1.16,"n":36},
        {"dev":"-5~-2%","vix":"30~35","wr":64.3,"avg":0.83,"n":70},
        {"dev":"-5~-2%","vix":">35","wr":55.0,"avg":-0.98,"n":40},
        {"dev":"-2~0%","vix":"13~15","wr":81.2,"avg":2.83,"n":16},
        {"dev":"-2~0%","vix":"15~17","wr":80.6,"avg":2.12,"n":31},
        {"dev":"-2~0%","vix":"17~20","wr":23.2,"avg":-3.04,"n":56},
        {"dev":"-2~0%","vix":"20~23","wr":33.7,"avg":-2.29,"n":98},
        {"dev":"-2~0%","vix":"23~25","wr":45.9,"avg":-0.14,"n":61},
        {"dev":"-2~0%","vix":"25~28","wr":57.9,"avg":0.8,"n":38},
        {"dev":"-2~0%","vix":"28~30","wr":60.0,"avg":1.65,"n":15},
        {"dev":"-2~0%","vix":"30~35","wr":80.0,"avg":2.07,"n":25},
        {"dev":"-2~0%","vix":">35","wr":85.7,"avg":3.42,"n":14},
        {"dev":"0~2%","vix":"13~15","wr":75.0,"avg":0.99,"n":12},
        {"dev":"0~2%","vix":"15~17","wr":76.9,"avg":0.77,"n":13},
        {"dev":"0~2%","vix":"17~20","wr":51.3,"avg":-0.66,"n":39},
        {"dev":"0~2%","vix":"20~23","wr":38.3,"avg":-1.68,"n":60},
        {"dev":"0~2%","vix":"23~25","wr":48.8,"avg":-0.43,"n":41},
        {"dev":"0~2%","vix":"25~28","wr":80.6,"avg":1.16,"n":31},
        {"dev":"0~2%","vix":"28~30","wr":75.0,"avg":0.97,"n":16},
        {"dev":"0~2%","vix":"30~35","wr":97.1,"avg":4.29,"n":35},
        {"dev":"0~2%","vix":">35","wr":80.0,"avg":4.46,"n":10},
        {"dev":"2~5%","vix":"15~17","wr":80.0,"avg":1.27,"n":10},
        {"dev":"2~5%","vix":"17~20","wr":13.3,"avg":-2.49,"n":15},
        {"dev":"2~5%","vix":"20~23","wr":16.2,"avg":-3.17,"n":37},
        {"dev":"2~5%","vix":"23~25","wr":38.2,"avg":-1.87,"n":34},
        {"dev":"2~5%","vix":"25~28","wr":50.0,"avg":-0.08,"n":20},
        {"dev":"2~5%","vix":"28~30","wr":61.1,"avg":1.48,"n":18},
        {"dev":"2~5%","vix":"30~35","wr":69.6,"avg":0.98,"n":23},
        {"dev":"2~5%","vix":">35","wr":100.0,"avg":5.73,"n":12},
        {"dev":"5~8%","vix":"17~20","wr":0.0,"avg":-7.82,"n":5},
        {"dev":"5~8%","vix":"30~35","wr":87.5,"avg":2.27,"n":8},
        {"dev":"5~8%","vix":">35","wr":100.0,"avg":5.22,"n":7},
    ],
}

# ── Bin order & helpers ───────────────────────────────────────────
VIX_BINS_ORDER = ["<11","11~13","13~15","15~17","17~20","20~23","23~25","25~28","28~30","30~35",">35"]
DEV_BINS_ORDER = ["<-10%","-10~-5%","-5~-2%","-2~0%","0~2%","2~5%","5~8%","8~10%","10~15%",">15%"]

def get_dev_bin(dev):
    if dev < -10: return "<-10%"
    if dev < -5:  return "-10~-5%"
    if dev < -2:  return "-5~-2%"
    if dev < 0:   return "-2~0%"
    if dev < 2:   return "0~2%"
    if dev < 5:   return "2~5%"
    if dev < 8:   return "5~8%"
    if dev < 10:  return "8~10%"
    if dev < 15:  return "10~15%"
    return ">15%"

def get_vix_bin(vix):
    if vix < 11: return "<11"
    if vix < 13: return "11~13"
    if vix < 15: return "13~15"
    if vix < 17: return "15~17"
    if vix < 20: return "17~20"
    if vix < 23: return "20~23"
    if vix < 25: return "23~25"
    if vix < 28: return "25~28"
    if vix < 30: return "28~30"
    if vix < 35: return "30~35"
    return ">35"

# ── Live data ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="載入市場數據中…")
def fetch_live():
    end   = datetime.today()
    start = end - timedelta(days=320)
    spy = yf.download("SPY",
                      start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    vix = yf.download("^VIX",
                      start=(end - timedelta(days=10)).strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    ema60  = spy.ewm(span=60,  adjust=False).mean()
    sma200 = spy.rolling(200).mean()
    price    = round(float(spy.iloc[-1]),    2)
    ema60_v  = round(float(ema60.iloc[-1]),  2)
    sma200_v = round(float(sma200.iloc[-1]), 2)
    vix_v    = round(float(vix.iloc[-1]),    2)
    dev      = round((price - ema60_v) / ema60_v * 100, 2)
    return {
        "price":        price,
        "ema60":        ema60_v,
        "sma200":       sma200_v,
        "vix":          vix_v,
        "dev":          dev,
        "above_sma200": price > sma200_v,
        "as_of":        str(spy.index[-1].date()),
        "dev_bin":      get_dev_bin(dev),
        "vix_bin":      get_vix_bin(vix_v),
    }

# ── Cell colour ───────────────────────────────────────────────────
def cell_bg_text(wr, n):
    if n < 5:    return "#111827", "#374151"
    if wr >= 95: return "#064e3b", "#6ee7b7"
    if wr >= 90: return "#14532d", "#4ade80"
    if wr >= 80: return "#1c3a1a", "#86efac"
    if wr >= 70: return "#2d2a05", "#fbbf24"
    if wr >= 60: return "#1e3a5f", "#93c5fd"
    if wr >= 50: return "#1a1a3a", "#a5b4fc"
    if wr >= 40: return "#2d1a05", "#fdba74"
    if wr >= 30: return "#3b1515", "#fca5a5"
    return "#7f1d1d", "#fecaca"

# ── Matrix HTML builder ───────────────────────────────────────────
def build_matrix_html(data_list, curr_dev_bin, curr_vix_bin):
    lookup = {(r["dev"], r["vix"]): r for r in data_list}
    present_devs = [d for d in DEV_BINS_ORDER if any(r["dev"] == d for r in data_list)]
    present_vix  = [v for v in VIX_BINS_ORDER if any(r["vix"] == v for r in data_list)]

    # Legend
    legend_items = [
        ("≥95%", "#064e3b", "#6ee7b7"),
        ("≥90%", "#14532d", "#4ade80"),
        ("≥80%", "#1c3a1a", "#86efac"),
        ("≥70%", "#2d2a05", "#fbbf24"),
        ("≥60%", "#1e3a5f", "#93c5fd"),
        ("≥50%", "#1a1a3a", "#a5b4fc"),
        ("≥40%", "#2d1a05", "#fdba74"),
        ("≥30%", "#3b1515", "#fca5a5"),
        ("<30%", "#7f1d1d", "#fecaca"),
        ("n<5",  "#111827", "#374151"),
    ]
    legend_html = "".join(
        f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
        f'font-size:0.65rem;font-weight:600;border:1px solid {fg}22">{lbl}</span>'
        for lbl, bg, fg in legend_items
    )

    # Header
    th = ("padding:8px 10px;background:#0f172a;border:1px solid #1e293b;"
          "white-space:nowrap;font-size:0.68em;letter-spacing:.04em;text-transform:uppercase")
    header = (f'<th style="{th};text-align:left;min-width:78px;color:#334155;'
              f'position:sticky;left:0;z-index:2">VIX \\ 乖離</th>')
    for d in present_devs:
        is_curr = (d == curr_dev_bin)
        c  = "#f59e0b" if is_curr else "#475569"
        fw = "700"     if is_curr else "400"
        mk = ' <span style="color:#f59e0b;font-size:0.8em">◀</span>' if is_curr else ""
        header += (f'<th style="{th};text-align:center;min-width:95px;color:{c};font-weight:{fw}">'
                   f'{d}{mk}</th>')

    # Rows
    rows = ""
    for vix_b in present_vix:
        is_curr_row = (vix_b == curr_vix_bin)
        vc  = "#f59e0b" if is_curr_row else "#334155"
        vfw = "700"     if is_curr_row else "400"
        vmk = " ◀"      if is_curr_row else ""
        vix_td = (f'<td style="padding:8px 10px;background:#0f172a;border:1px solid #1e293b;'
                  f'color:{vc};font-weight:{vfw};font-size:0.68em;white-space:nowrap;'
                  f'position:sticky;left:0;z-index:1">{vix_b}{vmk}</td>')
        row_cells = vix_td
        for dev_b in present_devs:
            key = (dev_b, vix_b)
            is_curr_cell = (dev_b == curr_dev_bin and vix_b == curr_vix_bin)
            if key in lookup:
                r = lookup[key]
                bg, fg = cell_bg_text(r["wr"], r["n"])
                if r["n"] < 5:
                    inner = f'<span style="color:#374151;font-size:0.9rem">—</span>'
                else:
                    avg_s = f"+{r['avg']:.2f}%" if r["avg"] >= 0 else f"{r['avg']:.2f}%"
                    inner = (
                        f'<div style="font-size:1.05rem;font-weight:800;line-height:1.2">{r["wr"]:.0f}%</div>'
                        f'<div style="font-size:0.62rem;margin-top:1px;opacity:0.85">{avg_s}</div>'
                        f'<div style="font-size:0.58rem;opacity:0.45;margin-top:1px">n={r["n"]}</div>'
                    )
            else:
                bg, fg = "#0a0f1a", "#1e293b"
                inner = '<span style="color:#1e293b">—</span>'

            if is_curr_cell:
                border = "border:2px solid #f59e0b"
                shadow = "box-shadow:inset 0 0 0 2px #f59e0b66,0 0 10px #f59e0b44"
            else:
                border = "border:1px solid #1e293b"
                shadow = ""

            row_cells += (
                f'<td style="padding:8px 10px;text-align:center;background:{bg};'
                f'{border};{shadow};min-width:95px">'
                f'<div style="color:{fg}">{inner}</div></td>'
            )
        rows += f'<tr>{row_cells}</tr>'

    table_h = 48 + len(present_vix) * 62

    return f"""<!DOCTYPE html><html lang="zh-TW"><head>
<meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0f1a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
.legend{{display:flex;flex-wrap:wrap;gap:5px;padding:6px 0 10px;align-items:center}}
.legend-lbl{{font-size:0.65rem;color:#334155;margin-right:4px}}
.wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:max-content;min-width:100%}}
tr:hover td:not([style*="sticky"]){{filter:brightness(1.25);transition:filter .1s}}
</style></head><body>
<div class="legend">
  <span class="legend-lbl">勝率色標：</span>{legend_html}
  <span style="margin-left:8px;font-size:0.62rem;color:#334155">橘框 = 當前位置</span>
</div>
<div class="wrap" style="max-height:{table_h}px">
<table>
  <thead><tr>{header}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</body></html>"""

# ── Page header ───────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🎯 S&P500 勝率矩陣")
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem'>"
        "EMA60 乖離率 × VIX → 未來 1個月 / 3個月進場勝率 &nbsp;｜&nbsp;"
        "回測區間：2000年1月 ～ 2026年5月"
        "</span>",
        unsafe_allow_html=True
    )
with col_refresh:
    if st.button("🔄 更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

try:
    L = fetch_live()

    # ── Live metric cards ─────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    dev_color = "#4ade80" if L["dev"] >= 0 else "#f87171"
    with c1:
        st.markdown(f"""<div class="metric-card">
          <div style="font-size:0.68rem;color:#64748b">SPY 最新收盤</div>
          <div style="font-size:1.45rem;font-weight:700;color:#e2e8f0;margin:2px 0">${L['price']:,.2f}</div>
          <div style="font-size:0.63rem;color:#475569">截至 {L['as_of']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="metric-card">
          <div style="font-size:0.68rem;color:#64748b">EMA60 季線</div>
          <div style="font-size:1.45rem;font-weight:700;color:#e2e8f0;margin:2px 0">${L['ema60']:,.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="metric-card">
          <div style="font-size:0.68rem;color:#64748b">乖離率（EMA60）</div>
          <div style="font-size:1.45rem;font-weight:700;color:{dev_color};margin:2px 0">{'+' if L['dev']>=0 else ''}{L['dev']:.2f}%</div>
          <div style="font-size:0.63rem;color:#475569">桶：<b style="color:#f59e0b">{L['dev_bin']}</b></div>
        </div>""", unsafe_allow_html=True)

    vix_color = "#f87171" if L["vix"] > 25 else "#fbbf24" if L["vix"] > 18 else "#4ade80"
    with c4:
        st.markdown(f"""<div class="metric-card">
          <div style="font-size:0.68rem;color:#64748b">VIX 恐慌指數</div>
          <div style="font-size:1.45rem;font-weight:700;color:{vix_color};margin:2px 0">{L['vix']:.2f}</div>
          <div style="font-size:0.63rem;color:#475569">桶：<b style="color:#f59e0b">{L['vix_bin']}</b></div>
        </div>""", unsafe_allow_html=True)

    above    = L["above_sma200"]
    ma_color = "#4ade80" if above else "#f87171"
    ma_text  = "年線之上 ✓" if above else "年線之下 ✗"
    sma_dev  = round((L["price"] / L["sma200"] - 1) * 100, 2)
    with c5:
        st.markdown(f"""<div class="metric-card">
          <div style="font-size:0.68rem;color:#64748b">SMA200 年線狀態</div>
          <div style="font-size:1.1rem;font-weight:700;color:{ma_color};margin:4px 0">{ma_text}</div>
          <div style="font-size:0.63rem;color:#475569">${L['sma200']:,.2f}（{'+' if sma_dev>=0 else ''}{sma_dev:.2f}%）</div>
        </div>""", unsafe_allow_html=True)

    # ── Current position banner ───────────────────────────────────
    above_icon  = "🟢" if above else "🔴"
    above_label = "年線之上" if above else "年線之下"
    rec_tabs    = "📅/⚡ 年線之上 分頁" if above else "📅/⚡ 年線之下 分頁（注意：熊市勝率大幅下降）"
    st.markdown(f"""
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                padding:11px 18px;margin-bottom:14px;display:flex;align-items:center;gap:14px">
      <span style="font-size:1.5rem">{above_icon}</span>
      <div>
        <div style="font-size:0.82rem;color:#94a3b8">
          <b style="color:#e2e8f0">{above_label}</b>
          &nbsp;｜&nbsp; 乖離率桶 <b style="color:#f59e0b">{L['dev_bin']}</b>
          &nbsp;×&nbsp; VIX桶 <b style="color:#f59e0b">{L['vix_bin']}</b>
        </div>
        <div style="font-size:0.68rem;color:#475569;margin-top:3px">
          橘色邊框 = 目前所在格子 ｜ 請優先參考 {rec_tabs}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Tabs & matrices ───────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 3個月 年線之上",
        "⚡ 1個月 年線之上",
        "📅 3個月 年線之下 ⚠️",
        "⚡ 1個月 年線之下 ⚠️",
    ])

    def _matrix_height(key):
        pv = [v for v in VIX_BINS_ORDER if any(r["vix"] == v for r in DATA[key])]
        return 80 + len(pv) * 63   # legend(45) + header(45) + rows

    BEAR_WARN = """<div style="background:#1c0a00;border:1px solid #854d0e;border-radius:8px;
                   padding:8px 14px;color:#fde68a;font-size:0.78rem;margin-bottom:10px">
                   ⚠️ 年線之下為熊市環境，整體勝率大幅下降。此分頁僅供極端情況參考，不建議在熊市中依此加碼。
                   </div>"""

    with tab1:
        components.html(
            build_matrix_html(DATA["3m_上"], L["dev_bin"], L["vix_bin"]),
            height=_matrix_height("3m_上"), scrolling=False
        )
    with tab2:
        components.html(
            build_matrix_html(DATA["1m_上"], L["dev_bin"], L["vix_bin"]),
            height=_matrix_height("1m_上"), scrolling=False
        )
    with tab3:
        st.markdown(BEAR_WARN, unsafe_allow_html=True)
        components.html(
            build_matrix_html(DATA["3m_下"], L["dev_bin"], L["vix_bin"]),
            height=_matrix_height("3m_下"), scrolling=False
        )
    with tab4:
        st.markdown(BEAR_WARN, unsafe_allow_html=True)
        components.html(
            build_matrix_html(DATA["1m_下"], L["dev_bin"], L["vix_bin"]),
            height=_matrix_height("1m_下"), scrolling=False
        )

    # ── Insights ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-hdr">核心洞察（2000–2026 回測）</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="insight-card">
          <div style="font-size:0.8rem;font-weight:700;color:#4ade80;margin-bottom:8px">
            🟢 進場甜蜜點（年線之上）
          </div>
          <div style="font-size:0.75rem;color:#94a3b8;line-height:1.7">
            <b style="color:#86efac">VIX 25~30 整欄幾乎全綠</b>，任何乖離率搭配都表現良好<br>
            <b style="color:#86efac">乖離率 2~5% × VIX 20+</b> 是最穩定的高勝率區間（樣本充足）<br>
            <b style="color:#86efac">乖離率 5~8% × VIX 17~25</b> 3個月勝率達 88~98%
          </div>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight-card">
          <div style="font-size:0.8rem;font-weight:700;color:#f87171;margin-bottom:8px">
            🔴 減碼警示（年線之上）
          </div>
          <div style="font-size:0.75rem;color:#94a3b8;line-height:1.7">
            <b style="color:#fca5a5">乖離率 5~8% × VIX &lt; 13</b>
            → 勝率僅 48~64%，平均報酬接近 0 甚至負<br>
            <b style="color:#fca5a5">乖離率 -5~-2% × VIX 30~35</b>
            → 特別差（14%），恐慌＋弱勢無法反彈<br>
            <b style="color:#fca5a5">VIX 11~13 整欄</b>在乖離率偏正時表現普遍偏弱
          </div>
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="insight-card">
          <div style="font-size:0.8rem;font-weight:700;color:#fbbf24;margin-bottom:8px">
            ⚠️ 熊市環境（年線之下）
          </div>
          <div style="font-size:0.75rem;color:#94a3b8;line-height:1.7">
            整體勝率大幅下降，尤其
            <b style="color:#fdba74">乖離率 -2~2% × VIX 17~25</b>
            是最危險區（20~35%）<br>
            唯一機會窗口：
            <b style="color:#86efac">VIX &gt; 30 ＋ 乖離率負向</b>
            （極度恐慌的均值回歸反彈）
          </div>
        </div>""", unsafe_allow_html=True)

    # Caveats
    st.markdown("""
    <div style="background:#1e293b;border:1px solid #1e3a5f;border-radius:8px;
                padding:10px 16px;margin-top:4px">
      <div style="font-size:0.7rem;color:#475569;line-height:1.8">
        📌 <b style="color:#64748b">統計說明</b>
        &nbsp;｜&nbsp; ① 相鄰日期樣本高度重疊（非獨立事件），實際信賴區間比 n 看起來更寬
        &nbsp;｜&nbsp; ② n &lt; 5 的格子統計意義有限，顯示「—」
        &nbsp;｜&nbsp; ③ 歷史表現不代表未來績效，建議搭配大盤壓力儀表板的燈號綜合判斷
        &nbsp;｜&nbsp; ④ 數據含存活者偏差，S&P500 歷史成分股隨時間更換
      </div>
    </div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"載入失敗：{e}")
    import traceback
    st.text(traceback.format_exc())

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.68rem'>"
    "數據來源：Yahoo Finance（yfinance）&nbsp;｜&nbsp;"
    "回測區間：2000年1月 ～ 2026年5月 &nbsp;｜&nbsp;"
    "即時數據每小時自動更新"
    "</div>",
    unsafe_allow_html=True
)
