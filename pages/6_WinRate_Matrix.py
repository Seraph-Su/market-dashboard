import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── CSS ────────────────────────────────────────────────────────────────
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

# ── Bin helpers ────────────────────────────────────────────────────────
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

def get_vix_bin(v):
    if v < 11: return "<11"
    if v < 13: return "11~13"
    if v < 15: return "13~15"
    if v < 17: return "15~17"
    if v < 20: return "17~20"
    if v < 23: return "20~23"
    if v < 25: return "23~25"
    if v < 28: return "25~28"
    if v < 30: return "28~30"
    if v < 35: return "30~35"
    return ">35"

# ── Index config ───────────────────────────────────────────────────────
INDEX_CONFIG = [
    {"key": "SPY",  "label": "S&P 500",       "icon": "🇺🇸", "tab": "🇺🇸 S&P 500"},
    {"key": "QQQ",  "label": "Nasdaq 100",     "icon": "💻", "tab": "💻 Nasdaq 100"},
    {"key": "DIA",  "label": "道瓊工業指數",     "icon": "🏭", "tab": "🏭 道瓊工業"},
    {"key": "SOXX", "label": "費城半導體指數",   "icon": "🔬", "tab": "🔬 費城半導體"},
]

# ── Backtest computation (cached 7 days) ───────────────────────────────
@st.cache_data(ttl=7 * 24 * 3600, show_spinner="計算歷史回測矩陣中（首次約需 5 秒）…")
def compute_backtest():
    tickers = ["SPY", "QQQ", "DIA", "SOXX"]

    raw = yf.download(tickers, period="max", interval="1d",
                      auto_adjust=True, progress=False)
    vix_raw = yf.download("^VIX", period="max", interval="1d",
                           auto_adjust=False, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = pd.to_datetime(vix.index).tz_localize(None)

    close = raw["Close"].copy()
    close.columns = [c[-1] if isinstance(c, tuple) else str(c) for c in close.columns]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    # date range info
    start_date = close.index[0].date()
    end_date   = close.index[-1].date()

    all_data = {}
    for ticker in tickers:
        price = close[ticker].dropna()
        shared = price.index.intersection(vix.index)
        price  = price.reindex(shared)
        v      = vix.reindex(shared)

        ema60  = price.ewm(span=60, adjust=False).mean()
        sma200 = price.rolling(200).mean()
        dev_pct = (price - ema60) / ema60 * 100
        fwd21  = price.shift(-21) / price - 1
        fwd63  = price.shift(-63) / price - 1
        above  = price > sma200

        df = pd.DataFrame({
            "vix": v, "dev": dev_pct, "above": above,
            "fwd21": fwd21, "fwd63": fwd63,
        }).dropna()

        ticker_data = {}
        for (label_above, horizon, col) in [
            ("上", "3m", "fwd63"), ("上", "1m", "fwd21"),
            ("下", "3m", "fwd63"), ("下", "1m", "fwd21"),
        ]:
            key = f"{horizon}_{label_above}"
            sub = df[df["above"] == (label_above == "上")]
            agg = []
            for dev_b, dg in sub.groupby(sub["dev"].map(get_dev_bin)):
                for vix_b, cell in dg.groupby(dg["vix"].map(get_vix_bin)):
                    rets = cell[col].dropna()
                    n    = len(rets)
                    if n < 3:
                        continue
                    wr  = round(float((rets > 0).mean() * 100), 1)
                    avg = round(float(rets.mean() * 100), 2)
                    # Binomial test vs 50%
                    successes = int((rets > 0).sum())
                    binom_p   = float(stats.binomtest(
                        successes, n, p=0.5, alternative="two-sided").pvalue)
                    agg.append({
                        "dev": dev_b, "vix": vix_b,
                        "wr": wr, "avg": avg, "n": n,
                        "sig": int(binom_p < 0.05),
                        "binom_p": round(binom_p, 4),
                    })
            ticker_data[key] = agg
        all_data[ticker] = ticker_data

    return all_data, str(start_date), str(end_date)

# ── Live data (cached 1 hour) ──────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="載入即時行情中…")
def fetch_live():
    tickers = ["SPY", "QQQ", "DIA", "SOXX"]
    vix = yf.download("^VIX", period="5d", interval="1d",
                      auto_adjust=False, progress=False)["Close"].squeeze().dropna()
    vix_v = round(float(vix.iloc[-1]), 2)

    result = {"vix": vix_v}
    for ticker in tickers:
        try:
            raw   = yf.download(ticker, period="400d", interval="1d",
                                 auto_adjust=True, progress=False)
            price_s  = raw["Close"].squeeze().dropna()
            ema60    = price_s.ewm(span=60, adjust=False).mean()
            sma200   = price_s.rolling(200).mean()
            price    = round(float(price_s.iloc[-1]), 2)
            ema60_v  = round(float(ema60.iloc[-1]), 2)
            sma200_v = round(float(sma200.iloc[-1]), 2)
            dev      = round((price - ema60_v) / ema60_v * 100, 2)
            result[ticker] = {
                "price": price, "ema60": ema60_v, "sma200": sma200_v,
                "dev": dev, "above_sma200": bool(price > sma200_v),
                "as_of": str(price_s.index[-1].date()),
                "dev_bin": get_dev_bin(dev),
                "vix_bin": get_vix_bin(vix_v),
            }
        except Exception:
            result[ticker] = None
    return result

# ── Cell colour ────────────────────────────────────────────────────────
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

# ── Matrix HTML builder ────────────────────────────────────────────────
def build_matrix_html(data_list, curr_dev_bin, curr_vix_bin):
    lookup = {(r["dev"], r["vix"]): r for r in data_list}
    present_devs = [d for d in DEV_BINS_ORDER if any(r["dev"] == d for r in data_list)]
    present_vix  = [v for v in VIX_BINS_ORDER if any(r["vix"] == v for r in data_list)]

    legend_items = [
        ("≥95%","#064e3b","#6ee7b7"),("≥90%","#14532d","#4ade80"),
        ("≥80%","#1c3a1a","#86efac"),("≥70%","#2d2a05","#fbbf24"),
        ("≥60%","#1e3a5f","#93c5fd"),("≥50%","#1a1a3a","#a5b4fc"),
        ("≥40%","#2d1a05","#fdba74"),("≥30%","#3b1515","#fca5a5"),
        ("<30%","#7f1d1d","#fecaca"),("n<5","#111827","#374151"),
    ]
    legend_html = "".join(
        f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
        f'font-size:0.65rem;font-weight:600;border:1px solid {fg}22">{lbl}</span>'
        for lbl, bg, fg in legend_items
    )

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
                r  = lookup[key]
                bg, fg = cell_bg_text(r["wr"], r["n"])
                if r["n"] < 5:
                    inner = '<span style="color:#374151;font-size:0.9rem">—</span>'
                else:
                    avg_s     = f"+{r['avg']:.2f}%" if r["avg"] >= 0 else f"{r['avg']:.2f}%"
                    sig_badge = ('<span style="color:#f59e0b;font-size:0.55rem;'
                                 'position:absolute;top:3px;right:4px">★</span>'
                                 if r.get("sig") else "")
                    inner = (
                        f'<div style="font-size:1.05rem;font-weight:800;line-height:1.2">{r["wr"]:.0f}%</div>'
                        f'<div style="font-size:0.62rem;margin-top:1px;opacity:0.85">{avg_s}</div>'
                        f'<div style="font-size:0.58rem;opacity:0.45;margin-top:1px">n={r["n"]}</div>'
                        f'{sig_badge}'
                    )
            else:
                bg, fg = "#0a0f1a", "#1e293b"
                inner  = '<span style="color:#1e293b">—</span>'

            if is_curr_cell:
                border = "border:2px solid #f59e0b"
                shadow = "box-shadow:inset 0 0 0 2px #f59e0b66,0 0 10px #f59e0b44"
            else:
                border = "border:1px solid #1e293b"
                shadow = ""

            row_cells += (
                f'<td style="padding:8px 10px;text-align:center;background:{bg};'
                f'{border};{shadow};min-width:95px;position:relative">'
                f'<div style="color:{fg}">{inner}</div></td>'
            )
        rows += f"<tr>{row_cells}</tr>"

    table_h = 80 + len(present_vix) * 62

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
  <span style="margin-left:8px;font-size:0.62rem;color:#334155">橘框=當前位置 &nbsp;★=二項檢定 p&lt;0.05</span>
</div>
<div class="wrap" style="max-height:{table_h}px">
<table>
  <thead><tr>{header}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</body></html>"""

# ────────────────────────────────────────────────────────────────────────
# Page header
# ────────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 📊 多指數勝率矩陣")
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem'>"
        "EMA60 乖離率 × VIX → 未來 1個月 / 3個月進場勝率 &nbsp;｜&nbsp;"
        "回測：2000年起至今 &nbsp;｜&nbsp;"
        "S&P500 &nbsp;·&nbsp; Nasdaq 100 &nbsp;·&nbsp; 道瓊工業 &nbsp;·&nbsp; 費城半導體"
        "</span>",
        unsafe_allow_html=True
    )
with col_refresh:
    if st.button("🔄 更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

try:
    ALL_DATA, bt_start, bt_end = compute_backtest()
    LIVE = fetch_live()
    vix_v     = LIVE["vix"]
    vix_color = "#f87171" if vix_v > 25 else "#fbbf24" if vix_v > 18 else "#4ade80"

    BEAR_WARN = """<div style="background:#1c0a00;border:1px solid #854d0e;border-radius:8px;
                   padding:8px 14px;color:#fde68a;font-size:0.78rem;margin-bottom:10px">
                   ⚠️ 年線之下為熊市環境，整體勝率大幅下降。此分頁僅供極端情況參考，不建議在熊市中依此加碼。
                   </div>"""

    def _matrix_height(key, ticker):
        pv = [v for v in VIX_BINS_ORDER
              if any(r["vix"] == v for r in ALL_DATA[ticker][key])]
        return 80 + len(pv) * 63

    # ── Index tabs ─────────────────────────────────────────────────────
    idx_tabs = st.tabs([c["tab"] for c in INDEX_CONFIG])

    for tab_ui, cfg in zip(idx_tabs, INDEX_CONFIG):
        ticker = cfg["key"]
        with tab_ui:
            live = LIVE.get(ticker)
            if live is None:
                st.error(f"無法載入 {ticker} 數據")
                continue

            dev_bin = live["dev_bin"]
            vix_bin = live["vix_bin"]
            above   = live["above_sma200"]

            # ── Metric cards ───────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            dev_color = "#4ade80" if live["dev"] >= 0 else "#f87171"

            with c1:
                st.markdown(f"""<div class="metric-card">
                  <div style="font-size:0.68rem;color:#64748b">{ticker} 最新收盤</div>
                  <div style="font-size:1.45rem;font-weight:700;color:#e2e8f0;margin:2px 0">${live['price']:,.2f}</div>
                  <div style="font-size:0.63rem;color:#475569">截至 {live['as_of']}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                  <div style="font-size:0.68rem;color:#64748b">EMA60 季線</div>
                  <div style="font-size:1.45rem;font-weight:700;color:#e2e8f0;margin:2px 0">${live['ema60']:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                  <div style="font-size:0.68rem;color:#64748b">乖離率（EMA60）</div>
                  <div style="font-size:1.45rem;font-weight:700;color:{dev_color};margin:2px 0">{'+' if live['dev']>=0 else ''}{live['dev']:.2f}%</div>
                  <div style="font-size:0.63rem;color:#475569">桶：<b style="color:#f59e0b">{dev_bin}</b></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                  <div style="font-size:0.68rem;color:#64748b">VIX 恐慌指數</div>
                  <div style="font-size:1.45rem;font-weight:700;color:{vix_color};margin:2px 0">{vix_v:.2f}</div>
                  <div style="font-size:0.63rem;color:#475569">桶：<b style="color:#f59e0b">{vix_bin}</b></div>
                </div>""", unsafe_allow_html=True)
            ma_color = "#4ade80" if above else "#f87171"
            ma_text  = "年線之上 ✓" if above else "年線之下 ✗"
            sma_dev  = round((live["price"] / live["sma200"] - 1) * 100, 2)
            with c5:
                st.markdown(f"""<div class="metric-card">
                  <div style="font-size:0.68rem;color:#64748b">SMA200 年線狀態</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{ma_color};margin:4px 0">{ma_text}</div>
                  <div style="font-size:0.63rem;color:#475569">${live['sma200']:,.2f}（{'+' if sma_dev>=0 else ''}{sma_dev:.2f}%）</div>
                </div>""", unsafe_allow_html=True)

            # ── Position banner ────────────────────────────────────────
            above_icon  = "🟢" if above else "🔴"
            above_label = "年線之上" if above else "年線之下"
            rec_note    = "📅/⚡ 年線之上 分頁" if above else "📅/⚡ 年線之下 分頁（注意：熊市勝率大幅下降）"
            st.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                        padding:11px 18px;margin-bottom:14px;display:flex;align-items:center;gap:14px">
              <span style="font-size:1.5rem">{above_icon}</span>
              <div>
                <div style="font-size:0.82rem;color:#94a3b8">
                  <b style="color:#e2e8f0">{ticker} {above_label}</b>
                  &nbsp;｜&nbsp; 乖離率桶 <b style="color:#f59e0b">{dev_bin}</b>
                  &nbsp;×&nbsp; VIX桶 <b style="color:#f59e0b">{vix_bin}</b>
                </div>
                <div style="font-size:0.68rem;color:#475569;margin-top:3px">
                  橘色邊框 = 目前所在格子 ｜ 請優先參考 {rec_note}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Sub-tabs (4 scenarios) ─────────────────────────────────
            sub1, sub2, sub3, sub4 = st.tabs([
                "📅 3個月 年線之上", "⚡ 1個月 年線之上",
                "📅 3個月 年線之下 ⚠️", "⚡ 1個月 年線之下 ⚠️",
            ])
            with sub1:
                components.html(
                    build_matrix_html(ALL_DATA[ticker]["3m_上"], dev_bin, vix_bin),
                    height=_matrix_height("3m_上", ticker), scrolling=False)
            with sub2:
                components.html(
                    build_matrix_html(ALL_DATA[ticker]["1m_上"], dev_bin, vix_bin),
                    height=_matrix_height("1m_上", ticker), scrolling=False)
            with sub3:
                st.markdown(BEAR_WARN, unsafe_allow_html=True)
                components.html(
                    build_matrix_html(ALL_DATA[ticker]["3m_下"], dev_bin, vix_bin),
                    height=_matrix_height("3m_下", ticker), scrolling=False)
            with sub4:
                st.markdown(BEAR_WARN, unsafe_allow_html=True)
                components.html(
                    build_matrix_html(ALL_DATA[ticker]["1m_下"], dev_bin, vix_bin),
                    height=_matrix_height("1m_下", ticker), scrolling=False)

    # ── Statistical significance section ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-hdr">統計顯著性說明</div>', unsafe_allow_html=True)

    sig_c1, sig_c2 = st.columns(2)
    with sig_c1:
        st.markdown("""<div class="insight-card">
          <div style="font-size:0.8rem;font-weight:700;color:#fbbf24;margin-bottom:8px">
            📐 ★ 標記 = 二項檢定顯著（p &lt; 0.05）
          </div>
          <div style="font-size:0.75rem;color:#94a3b8;line-height:1.8">
            格子右上角 <b style="color:#f59e0b">★</b> 表示相對於 50% 基準線，在二項分布檢定下 <b>p &lt; 0.05</b>。<br><br>
            <b style="color:#f87171">重要限制：重疊窗口問題</b><br>
            日線資料的前向報酬高度重疊（3個月視窗有 62/63 天互相重疊），相鄰日幾乎是同一個事件。
            真實有效樣本數約為：<br>
            &nbsp;&nbsp;• 3個月視窗：n_eff ≈ n ÷ 63<br>
            &nbsp;&nbsp;• 1個月視窗：n_eff ≈ n ÷ 21<br><br>
            以 n_eff 校正後，幾乎所有格子的統計顯著性都消失。
            <b>★ 標記反映的是歷史一致性，而非統計保證。</b>
          </div>
        </div>""", unsafe_allow_html=True)
    with sig_c2:
        st.markdown("""<div class="insight-card">
          <div style="font-size:0.8rem;font-weight:700;color:#86efac;margin-bottom:8px">
            ✅ 使用建議
          </div>
          <div style="font-size:0.75rem;color:#94a3b8;line-height:1.8">
            <b style="color:#86efac">適合做的事</b><br>
            ① 判斷目前環境的大方向（高VIX+負乖離歷史上普遍表現較好）<br>
            ② 跨指數比較：同一環境下，哪個指數歷史反彈幅度更強？<br>
            ③ 搭配大盤壓力儀表板燈號做綜合判斷<br><br>
            <b style="color:#f87171">不適合做的事</b><br>
            ① 以精確勝率數字做機械化決策（重疊樣本使數字不可靠）<br>
            ② 熊市（年線之下）中跟著矩陣加碼（整體勝率大幅降低）<br>
            ③ 把 n 當成獨立樣本數（真實有效 N 遠小於顯示值）
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Cross-index comparison ─────────────────────────────────────────
    st.markdown('<div class="section-hdr">四指數特性比較</div>', unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    comparisons = [
        ("🇺🇸 S&P 500 (SPY)", "#86efac",
         "年化波動約 <b>15%</b>，乖離±5% 為核心區間。",
         "VIX > 20 時均值回歸最穩定，是最佳基準參照指數。"),
        ("💻 Nasdaq 100 (QQQ)", "#93c5fd",
         "年化波動約 <b>20%</b>，乖離區間更寬（±10% 常見）。",
         "高VIX × 大幅負乖離時反彈力道通常最強，但也最不穩定。"),
        ("🏭 道瓊工業 (DIA)", "#fbbf24",
         "年化波動約 <b>13%</b>，波動最小，格子分布最集中。",
         "低波動使乖離不易拉大，超過±8% 的格子樣本偏少。"),
        ("🔬 費城半導體 (SOXX)", "#f87171",
         "年化波動約 <b>30%+</b>，乖離可達±15% 以上。",
         "VIX > 25 × 深度負乖離的反彈幅度遠大於其他指數，但失敗時跌幅亦更深。"),
    ]
    for col_ui, (title, color, line1, line2) in zip([cc1, cc2, cc3, cc4], comparisons):
        with col_ui:
            st.markdown(f"""<div class="insight-card">
              <div style="font-size:0.78rem;font-weight:700;color:{color};margin-bottom:8px">{title}</div>
              <div style="font-size:0.72rem;color:#94a3b8;line-height:1.7">{line1}<br>{line2}</div>
            </div>""", unsafe_allow_html=True)

    # ── Caveats ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#1c1400;border:1px solid #854d0e;border-radius:8px;
                padding:10px 16px;margin-top:4px;margin-bottom:6px;
                color:#fde68a;font-size:0.82rem;font-weight:600">
      ⚠️ 勝率僅為參考，過去統計數字不代表未來表現
    </div>
    <div style="background:#1e293b;border:1px solid #1e3a5f;border-radius:8px;
                padding:10px 16px;margin-top:0">
      <div style="font-size:0.7rem;color:#475569;line-height:1.8">
        📌 <b style="color:#64748b">統計說明</b>
        &nbsp;｜&nbsp; ① 相鄰日期樣本高度重疊（非獨立事件），實際信賴區間比 n 看起來更寬
        &nbsp;｜&nbsp; ② n &lt; 5 的格子統計意義有限，顯示「—」
        &nbsp;｜&nbsp; ③ 歷史表現不代表未來績效，建議搭配大盤壓力儀表板的燈號綜合判斷
        &nbsp;｜&nbsp; ④ 數據含存活者偏差，各指數歷史成分股隨時間更換
        &nbsp;｜&nbsp; ⑤ SOXX 自 2001 年起；其餘三指數自 2000 年起
        &nbsp;｜&nbsp; ⑥ 回測區間：{bt_start} ～ {bt_end}（每週自動更新）
      </div>
    </div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"載入失敗：{e}")
    import traceback
    st.text(traceback.format_exc())

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.68rem'>"
    "數據來源：Yahoo Finance（yfinance）&nbsp;｜&nbsp;"
    "回測每週自動更新 &nbsp;｜&nbsp; 即時行情每小時更新 &nbsp;｜&nbsp;"
    "ETF：SPY · QQQ · DIA · SOXX"
    "</div>",
    unsafe_allow_html=True
)
