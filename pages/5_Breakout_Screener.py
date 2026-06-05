import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from io import StringIO

# ── 股票清單 ──────────────────────────────────────────────────────
NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","PEP","CSCO","ADBE","INTC","CMCSA","HON","AMGN",
    "TXN","QCOM","INTU","AMAT","ISRG","BKNG","ADP","SBUX","GILD","MU",
    "LRCX","REGN","ADI","PANW","KLAC","MDLZ","SNPS","CDNS","MELI","FTNT",
    "CTAS","CSX","PAYX","ORLY","MRVL","IDXX","ROST","CPRT","PCAR","KDP",
    "DXCM","BIIB","TEAM","ILMN","MRNA","ZS","CRWD","OKTA","DDOG","SNOW",
    "APP","PLTR","CEG","GEHC","TTD","ARM","DASH","MSTR","RBLX","ON",
]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_largecap_us_tickers(min_market_cap: int = 1_000_000_000) -> list:
    """使用 yfinance EquityQuery 抓取市值 > min_market_cap 的全美股清單"""
    from yfinance import EquityQuery
    q = EquityQuery('and', [
        EquityQuery('gt', ['intradaymarketcap', min_market_cap]),
        EquityQuery('eq', ['region', 'us']),
    ])
    tickers = []
    for offset in range(0, 7000, 250):   # 全美股約 6400+ 支
        try:
            result = yf.screen(q, size=250, offset=offset)
            quotes  = result.get('quotes', [])
            tickers.extend(x['symbol'] for x in quotes if '.' not in x['symbol'])
            if len(quotes) < 250:
                break
        except Exception:
            break
    return tickers


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()

# ── 核心選股邏輯 ──────────────────────────────────────────────────
CFG = dict(
    consolidation_days   = 14,
    consolidation_range  = 0.16,
    breakout_buffer      = 0.003,
    ema_spread_threshold = 0.05,
    ema_lookback         = 20,
    volume_mult          = 1.3,
)

def add_emas(df):
    df = df.copy()
    df["ema20"]  = df["Close"].ewm(span=20,  adjust=False).mean()
    df["ema60"]  = df["Close"].ewm(span=60,  adjust=False).mean()
    df["ema260"] = df["Close"].ewm(span=260, adjust=False).mean()
    if len(df) >= 300:
        cols = ["ema20", "ema60", "ema260"]
        df["ema_mode"] = "三線"
    else:
        cols = ["ema20", "ema60"]
        df["ema_mode"] = "雙線"
    df["ema_spread"] = (df[cols].max(axis=1) - df[cols].min(axis=1)) / df[cols].median(axis=1)
    return df

def detect_signal(df, offset=0):
    """
    offset=0 → 最新一天；offset=1 → 前一個交易日，以此類推。
    """
    cfg = CFG
    n, rng, buf = cfg["consolidation_days"], cfg["consolidation_range"], cfg["breakout_buffer"]
    vmult = cfg["volume_mult"]
    spread_thr, ema_lb = cfg["ema_spread_threshold"], cfg["ema_lookback"]

    # 把 df 截到目標日期
    if offset > 0:
        df = df.iloc[:len(df) - offset]

    if len(df) < max(n + 2, 65):
        return None

    df = add_emas(df)
    today  = df.iloc[-1]
    consol = df.iloc[-(n + 1):-1]

    close     = float(today["Close"])
    avg_vol   = float(consol["Volume"].mean())
    if avg_vol == 0:
        return None
    vol_ratio = float(today["Volume"]) / avg_vol

    signal_type = None
    day_gain    = (close - float(df.iloc[-2]["Close"])) / float(df.iloc[-2]["Close"])
    consol_high = float(consol["High"].max())
    consol_low  = float(consol["Low"].min())
    consol_rng  = (consol_high - consol_low) / consol_low

    def make_record(signal_type):
        return {
            "觸發日期":    str(df.index[-1].date()),
            "訊號類型":    signal_type,
            "收盤價":      round(close, 2),
            "突破幅度":    f"{round((close / consol_high - 1) * 100, 2):+.2f}%",
            "量比":        f"{vol_ratio:.1f}x",
            "EMA收斂度":   f"{round(float(today['ema_spread']) * 100, 2):.2f}%",
            "20日最小收斂":f"{round(float(df['ema_spread'].iloc[-ema_lb:].min()) * 100, 2):.2f}%",
            "EMA20":       round(float(today["ema20"]), 2),
            "均線模式":    str(today["ema_mode"]),
            "_day_gain":   round(day_gain * 100, 2),
        }

    # 訊號 A：K棒盤整突破
    cond_a = (consol_rng <= rng
              and close > consol_high * (1 + buf)
              and vol_ratio >= vmult)

    # 訊號 B / C：EMA 收斂突破
    recent     = df.iloc[-(ema_lb + 1):-1]
    prev_close = float(df.iloc[-2]["Close"])
    day_gain   = (close - prev_close) / prev_close

    ema_was_tight  = recent["ema_spread"].min() < spread_thr
    spread2 = abs(df["ema20"] - df["ema60"]) / df[["ema20","ema60"]].mean(axis=1)
    ema2_was_tight = spread2.iloc[-(ema_lb + 1):-1].min() < spread_thr

    recent_high = float(recent["High"].max())
    broke_high  = close > recent_high * (1 + buf)
    big_move    = day_gain > 0.03

    cond_bc = (ema_was_tight or ema2_was_tight) and broke_high and big_move and vol_ratio >= vmult
    ema_type = "B｜EMA三線收斂突破" if ema_was_tight else "C｜EMA雙線收斂突破"

    if not cond_a and not cond_bc:
        return None

    # 兩個條件同時成立 → 兩筆獨立記錄
    if cond_a and cond_bc:
        return [make_record("A｜K棒盤整突破"), make_record(ema_type)]

    # 單一條件
    return make_record("A｜K棒盤整突破" if cond_a else ema_type)

def _download_batch(tickers, start, end):
    """批次下載，回傳 {ticker: df} 字典"""
    stock_data = {}
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    df = pd.DataFrame({
                        "Close":  raw["Close"][ticker],
                        "High":   raw["High"][ticker],
                        "Low":    raw["Low"][ticker],
                        "Volume": raw["Volume"][ticker],
                    }).dropna()
                    if len(df) >= 65:
                        stock_data[ticker] = df
                except Exception:
                    pass
        else:
            # 單一股票
            df = raw[["Close","High","Low","Volume"]].dropna()
            if len(df) >= 65:
                stock_data[tickers[0]] = df
    except Exception:
        pass
    return stock_data


@st.cache_data(ttl=86400, show_spinner=False)
def run_screener(universe: str):
    if universe == "S&P 500":
        tickers = fetch_sp500_tickers()
    elif universe == "全美股":
        tickers = fetch_largecap_us_tickers(min_market_cap=1_000_000_000)
    else:
        tickers = NASDAQ100

    end_str   = datetime.today().strftime("%Y-%m-%d")
    start_str = (datetime.today() - timedelta(days=460)).strftime("%Y-%m-%d")

    # 全美股分批下載（每批 150 支，避免逾時）
    BATCH = 150 if universe == "全美股" else len(tickers)
    stock_data = {}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        stock_data.update(_download_batch(batch, start_str, end_str))

    SCAN_DAYS = 5   # 往回掃描幾個交易日
    rows = []
    seen = set()    # 同一支股票同一天同一訊號只記一次
    for ticker, df in stock_data.items():
        for offset in range(SCAN_DAYS):
            result = detect_signal(df, offset=offset)
            if result is None:
                continue
            records = result if isinstance(result, list) else [result]
            for sig in records:
                key = (ticker, sig["觸發日期"], sig["訊號類型"])
                if key not in seen:
                    seen.add(key)
                    sig["代號"] = ticker
                    rows.append(sig)

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    # 排序：先觸發日期（新→舊），再訊號類型（B > C > A），再突破幅度
    order = {"B｜EMA三線收斂突破": 0, "C｜EMA雙線收斂突破": 1, "A｜K棒盤整突破": 2}
    df_out["_order"] = df_out["訊號類型"].map(order)
    df_out = df_out.sort_values(["觸發日期", "_order", "_day_gain"],
                                ascending=[False, True, False])
    df_out = df_out.drop(columns=["_order", "_day_gain"]).reset_index(drop=True)
    df_out.index += 1
    cols = ["觸發日期", "代號", "訊號類型", "收盤價", "突破幅度", "量比",
            "EMA收斂度", "20日最小收斂", "EMA20", "均線模式"]
    return df_out[cols]

# ── Chart data for hover tooltips ────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_chart_data(tickers: tuple) -> dict:
    """只針對結果股票下載 130 天資料，計算三條均線"""
    end   = datetime.today()
    start = end - timedelta(days=400)   # 需足夠讓 EMA260 收斂
    raw   = yf.download(list(tickers), start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        auto_adjust=True, progress=False)
    result = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"][ticker].dropna()
            else:
                closes = raw["Close"].dropna()
            df = closes.to_frame("Close")
            df["EMA20"]  = df["Close"].ewm(span=20,  adjust=False).mean()
            df["EMA60"]  = df["Close"].ewm(span=60,  adjust=False).mean()
            df["EMA260"] = df["Close"].ewm(span=260, adjust=False).mean()
            df = df.tail(90)
            result[ticker] = {
                "dates": [str(d.date()) for d in df.index],
                "close": [round(v, 2) for v in df["Close"].tolist()],
                "ema20": [round(v, 2) for v in df["EMA20"].tolist()],
                "ema60": [round(v, 2) for v in df["EMA60"].tolist()],
                "ema260":[round(v, 2) for v in df["EMA260"].tolist()],
            }
        except Exception:
            pass
    return result


def build_hover_table(result_df: pd.DataFrame, chart_data_dict: dict) -> str:
    sig_colors = {
        "B｜EMA三線收斂突破": "#a5b4fc",
        "C｜EMA雙線收斂突破": "#67e8f9",
        "A｜K棒盤整突破":    "#94a3b8",
    }
    rows_html = ""
    for _, row in result_df.iterrows():
        ticker    = row["代號"]
        sig_color = sig_colors.get(row["訊號類型"], "#94a3b8")
        has_chart = ticker in chart_data_dict
        bp        = str(row["突破幅度"])
        bp_color  = "#4ade80" if "+" in bp else "#f87171"
        ticker_td = (
            f'<td class="ticker-cell" data-ticker="{ticker}" '
            f'style="color:#60a5fa;font-weight:700;font-family:monospace;cursor:crosshair">{ticker}</td>'
            if has_chart else
            f'<td style="color:#60a5fa;font-weight:700;font-family:monospace">{ticker}</td>'
        )
        rows_html += f"""<tr>
          <td style="color:#64748b">{row['觸發日期']}</td>
          {ticker_td}
          <td style="color:{sig_color};font-size:0.8em;font-weight:600">{row['訊號類型']}</td>
          <td style="text-align:right">{row['收盤價']}</td>
          <td style="text-align:right;color:{bp_color}">{bp}</td>
          <td style="text-align:right;color:#fbbf24">{row['量比']}</td>
          <td style="text-align:right;color:#94a3b8">{row['EMA收斂度']}</td>
          <td style="text-align:right;color:#64748b">{row['20日最小收斂']}</td>
          <td style="text-align:right;color:#64748b">{row['EMA20']}</td>
          <td style="text-align:right;color:#475569;font-size:0.78em">{row['均線模式']}</td>
        </tr>"""

    chart_json = json.dumps(chart_data_dict, ensure_ascii=False)
    table_height = min(700, 80 + len(result_df) * 37)

    return f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:transparent; font-family:-apple-system,sans-serif; color:#e2e8f0; font-size:0.84em; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ padding:7px 10px; text-align:left; background:#0f172a; color:#64748b;
        font-size:0.72em; text-transform:uppercase; letter-spacing:.05em; white-space:nowrap; }}
  td {{ padding:7px 10px; border-bottom:1px solid #1e293b; white-space:nowrap; }}
  tr:hover td {{ background:rgba(255,255,255,0.02); }}
  #chartTooltip {{
    display:none; position:fixed; z-index:9999;
    background:#1e293b; border:1px solid #334155; border-radius:12px;
    padding:14px; width:390px;
    box-shadow:0 8px 32px rgba(0,0,0,0.6); pointer-events:none;
  }}
  #tooltipTitle {{ color:#93c5fd; font-size:0.95em; font-weight:700;
                   margin-bottom:8px; font-family:monospace; letter-spacing:.05em; }}
  #tooltipLegend {{ color:#64748b; font-size:0.72em; margin-bottom:6px; }}
</style>
</head><body>
<div id="chartTooltip">
  <div id="tooltipTitle"></div>
  <div id="tooltipLegend">收盤 &nbsp;│&nbsp; <span style="color:#60a5fa">EMA20</span> &nbsp;│&nbsp; <span style="color:#f59e0b">EMA60</span> &nbsp;│&nbsp; <span style="color:#f87171">EMA260</span></div>
  <canvas id="tooltipCanvas" width="362" height="200"></canvas>
</div>
<div style="height:{table_height}px;overflow-y:auto">
<table>
  <thead><tr>
    <th>觸發日期</th><th>代號 ↑ hover 看圖</th><th>訊號類型</th>
    <th style="text-align:right">收盤價</th><th style="text-align:right">突破幅度</th>
    <th style="text-align:right">量比</th><th style="text-align:right">EMA收斂度</th>
    <th style="text-align:right">20日最小收斂</th><th style="text-align:right">EMA20</th>
    <th style="text-align:right">均線模式</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
<script>
const chartData = {chart_json};
let currentChart = null;
const tooltip  = document.getElementById('chartTooltip');
const ttTitle  = document.getElementById('tooltipTitle');
const canvas   = document.getElementById('tooltipCanvas');

document.querySelectorAll('.ticker-cell').forEach(cell => {{
  cell.addEventListener('mouseenter', function(e) {{
    const ticker = this.dataset.ticker;
    const data   = chartData[ticker];
    if (!data) return;
    ttTitle.textContent = ticker;
    tooltip.style.display = 'block';
    positionTooltip(e);
    if (currentChart) {{ currentChart.destroy(); currentChart = null; }}
    currentChart = new Chart(canvas, {{
      type: 'line',
      data: {{
        labels: data.dates.map(d => d.slice(5)),
        datasets: [
          {{ label:'收盤',  data:data.close,  borderColor:'#e2e8f0', borderWidth:1.5, pointRadius:0, tension:0.2 }},
          {{ label:'EMA20', data:data.ema20,  borderColor:'#60a5fa', borderWidth:1.5, pointRadius:0, tension:0.2 }},
          {{ label:'EMA60', data:data.ema60,  borderColor:'#f59e0b', borderWidth:1.5, pointRadius:0, tension:0.2 }},
          {{ label:'EMA260',data:data.ema260, borderColor:'#f87171', borderWidth:1.5, pointRadius:0, tension:0.2, borderDash:[4,3] }},
        ]
      }},
      options: {{
        responsive:false, animation:false,
        plugins: {{ legend:{{display:false}}, tooltip:{{enabled:false}} }},
        scales: {{
          x: {{ ticks:{{color:'#475569',font:{{size:9}},maxTicksLimit:8,maxRotation:0}}, grid:{{color:'#1e293b'}} }},
          y: {{ ticks:{{color:'#475569',font:{{size:9}}}}, grid:{{color:'#1e293b'}} }}
        }}
      }}
    }});
  }});
  cell.addEventListener('mouseleave', () => {{ tooltip.style.display='none'; }});
  cell.addEventListener('mousemove',  e => positionTooltip(e));
}});

function positionTooltip(e) {{
  const tw=390, th=260;
  let x = e.clientX + 24;
  let y = e.clientY - 100;
  if (x + tw > window.innerWidth)  x = e.clientX - tw - 10;
  if (y + th > window.innerHeight) y = window.innerHeight - th - 10;
  if (y < 0) y = 10;
  tooltip.style.left = x + 'px';
  tooltip.style.top  = y + 'px';
}}
</script></body></html>"""


# ── Page ──────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 📡 均線收斂突破選股")
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem'>"
        "偵測 EMA 三線/雙線收斂後放量突破訊號 &nbsp;｜&nbsp; "
        "回測：EMA收斂訊號 QQQ 成分股 20日平均報酬 +10.7%，勝率 82%（2020–2026，去除熊市）"
        "&nbsp;｜&nbsp; 資料每日自動更新"
        "</span>",
        unsafe_allow_html=True,
    )
with col_refresh:
    if st.button("🔄 重新掃描", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# 股票池選擇
universe = st.radio(
    "掃描股票池",
    ["Nasdaq 100（建議）", "S&P 500", "全美股（市值 > 10億，約需 3–5 分鐘）"],
    horizontal=True,
    help="EMA 收斂訊號在 Nasdaq 100 科技成長股中效果最顯著；全美股掃描首次載入較慢",
)
if universe.startswith("Nasdaq"):
    universe_key = "Nasdaq 100"
elif universe.startswith("S&P"):
    universe_key = "S&P 500"
else:
    universe_key = "全美股"

st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

# 策略說明摺疊區
with st.expander("📖 三種訊號說明"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**A｜K棒盤整突破**

短期緊密橫盤後直接向上突破。

- 過去 14 日高低差 ≤ 16%
- 今日收盤 > 盤整最高點 × 1.003
- 量比 > 1.3×
        """)
    with c2:
        st.markdown("""
**B｜EMA 三線收斂突破** ⭐

月線/季線/年線三線充分靠近後爆量突破。

- 近 20 日內三線收斂度曾 < 5%
- 今日漲幅 > 3%，突破近 20 日高點
- 量比 > 1.3×（僅適用上市 300 日以上）
        """)
    with c3:
        st.markdown("""
**C｜EMA 雙線收斂突破**

EMA20 / EMA60 在回調期間重新糾結後突破。

- 近 20 日內雙線收斂度曾 < 5%
- 今日漲幅 > 3%，突破近 20 日高點
- 量比 > 1.3×
        """)

# 掃描
try:
    wait = "約需 3–5 分鐘" if universe_key == "全美股" else "約需 30–60 秒"
    with st.spinner(f"掃描 {universe_key} 中，{wait}（結果會快取至明日）…"):
        result = run_screener(universe_key)

    as_of = datetime.today().strftime("%Y-%m-%d")

    if result.empty:
        st.info(f"今日 {universe_key} 無符合條件的訊號。")
    else:
        # 各類型數量
        n_b = (result["訊號類型"].str.startswith("B")).sum()
        n_c = (result["訊號類型"].str.startswith("C")).sum()
        n_a = (result["訊號類型"].str.startswith("A")).sum()

        st.markdown(
            f"<span style='color:#4ade80;font-size:0.9rem;font-weight:700'>"
            f"✅ 今日訊號：{len(result)} 個</span>"
            f"<span style='color:#475569;font-size:0.75rem'>"
            f" &nbsp;（B 三線收斂 {n_b} 個 ／ C 雙線收斂 {n_c} 個 ／ A K棒盤整 {n_a} 個）"
            f" &nbsp;截至 {as_of}"
            f"</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        # 取得結果股票的 EMA 線圖資料（hover 用）
        result_tickers = tuple(result["代號"].unique().tolist())
        with st.spinner("載入線圖資料…"):
            chart_data = fetch_chart_data(result_tickers)

        table_html = build_hover_table(result, chart_data)
        table_height = min(720, 100 + len(result) * 37)
        components.html(table_html, height=table_height, scrolling=False)

        st.markdown(
            "<div style='color:#334155;font-size:0.68rem;margin-top:8px'>"
            "⚠️ 策略在多頭環境表現顯著優於熊市，建議搭配大盤壓力儀表板的燈號判斷是否進場。"
            "</div>",
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"掃描失敗：{e}")
