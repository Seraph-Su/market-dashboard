import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
# ═══════════════════════════════════════════════════════════════════
# 🩹 修復觸發選股（順勢交易系統 第三引擎）
#   觸發：過去 120 個交易日內至少一天收盤 < 年線(EMA260)，
#         且今日是「最後一次收年線下之後的第一個多頭排列日」（收盤 > EMA20 > EMA60 > EMA260）。
#   出場：收盤跌破年線。尺寸：R ÷ max(進場價 − 年線, 10%×進場價)。
#   不看量、不看當日漲幅。原本的 A/B/C/D 四種均線收斂突破訊號已於 2026-09-15 移除。
# ═══════════════════════════════════════════════════════════════════
BELOW_LOOKBACK = 120     # 幾個交易日內曾收在年線下
SCAN_DAYS      = 10      # 往回列出幾個交易日的觸發
MONSTER_TH     = 1.20    # 「曾怪物」：過去 250 日內半年漲幅曾 ≥120%
MONSTER_LB     = 250
MIN_RISK       = 0.10    # 尺寸下限：停損距離至少抓 10%
NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","PEP","CSCO","ADBE","INTC","CMCSA","HON","AMGN",
    "TXN","QCOM","INTU","AMAT","ISRG","BKNG","ADP","SBUX","GILD","MU",
    "LRCX","REGN","ADI","PANW","KLAC","MDLZ","SNPS","CDNS","MELI","FTNT",
    "CTAS","CSX","PAYX","ORLY","MRVL","IDXX","ROST","CPRT","PCAR","KDP",
    "DXCM","BIIB","TEAM","ILMN","MRNA","ZS","CRWD","OKTA","DDOG","SNOW",
    "APP","PLTR","CEG","GEHC","TTD","ARM","DASH","MSTR","RBLX","ON",
]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasdaq100() -> tuple:
    """Nasdaq-100 成分股，每日自動更新（Wikipedia）；失敗回退備援名單。"""
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
                         headers=UA, timeout=30)
        r.raise_for_status()
        for tb in pd.read_html(StringIO(r.text)):
            cols = [str(c) for c in tb.columns]
            hit = [c for c in cols if "Ticker" in c or "Symbol" in c]
            if not hit or len(tb) < 90:
                continue
            ser = tb[hit[0]].astype(str).str.strip().str.replace(".", "-", regex=False)
            out = [t for t in dict.fromkeys(ser.tolist())
                   if t and t.upper() != "NAN" and 1 <= len(t) <= 6 and t.replace("-", "").isalpha()]
            if len(out) >= 90:
                return tuple(out), f"Wikipedia（{len(out)} 檔，每日更新）"
    except Exception:
        pass
    return tuple(NASDAQ100_FALLBACK), f"內建備援名單（{len(NASDAQ100_FALLBACK)} 檔，可能已過期）"
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500() -> tuple:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_html(StringIO(r.text))[0]
    return tuple(df["Symbol"].str.replace(".", "-", regex=False).tolist()), f"S&P 500（{len(df)} 檔）"
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_semis(min_cap: int = 1_000_000_000) -> tuple:
    """費城半導體（SOX）概念宇宙：Yahoo 產業分類「Semiconductors」＋「Semiconductor Equipment & Materials」。
    Wikipedia 沒有 SOX 成分表，改用產業篩選（自動更新，涵蓋 SOX 全部成分並略寬）。"""
    from yfinance import EquityQuery
    out = []
    for ind in ("Semiconductors", "Semiconductor Equipment & Materials"):
        try:
            q = EquityQuery("and", [EquityQuery("eq", ["region", "us"]),
                                    EquityQuery("eq", ["industry", ind]),
                                    EquityQuery("gt", ["intradaymarketcap", min_cap])])
            res = yf.screen(q, size=250, sortField="intradaymarketcap", sortAsc=False)
            for x in res.get("quotes", []):
                sym = x.get("symbol", "")
                # 排除 OTC 掛牌的外國股（ASMLF、TOELF 等五碼 F/Y 結尾代碼）
                if not sym or "." in sym or len(sym) > 5:
                    continue
                if x.get("exchange") not in {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS"}:
                    continue
                out.append(sym)
        except Exception:
            pass
    # Yahoo 把少數 SOX 成分歸到別的產業（光通訊、儲存），手動補回
    out += ["COHR", "SNDK", "LITE", "AAOI"]
    out = list(dict.fromkeys(out))
    return tuple(out), f"半導體產業（{len(out)} 檔，Yahoo 產業分類＋SOX 補漏，每日更新）"
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_largecap(min_cap: int = 1_000_000_000) -> tuple:
    from yfinance import EquityQuery
    q = EquityQuery("and", [EquityQuery("gt", ["intradaymarketcap", min_cap]),
                            EquityQuery("eq", ["region", "us"])])
    out = []
    for off in range(0, 7000, 250):
        try:
            res = yf.screen(q, size=250, offset=off)
            qs = res.get("quotes", [])
            out += [x["symbol"] for x in qs if "." not in x.get("symbol", ".")]
            if len(qs) < 250:
                break
        except Exception:
            break
    return tuple(dict.fromkeys(out)), f"全美股市值 >$1B（{len(out)} 檔）"
MIN_BARS = 800           # EMA260 收斂需要的最少根數（500 根會嚴重低估年線）
@st.cache_resource(ttl=86400, show_spinner=False)   # ⚠️ 不可改回 cache_data：
def download(tickers: tuple) -> dict:               #    cache_data 每個 session 複製一份，人一多就爆記憶體
    """5 年日線，分批下載。EMA260 是無限記憶平均，資料太短會讓年線失真：
    例 MAAS 2026-09-14，取 500 根算出年線 11.61（價在其上、誤觸發），
    取 800 根以上為 18.31（價在其下、不該觸發）。"""
    out, tk = {}, list(tickers)
    BATCH = 150
    for i in range(0, len(tk), BATCH):
        ch = tk[i:i + BATCH]
        try:
            raw = yf.download(ch, period="5y", interval="1d", auto_adjust=True, progress=False)
        except Exception:
            continue
        for t in ch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    d = pd.DataFrame({"Close": raw["Close"][t]}).dropna()
                else:
                    d = raw[["Close"]].dropna()
                if len(d) >= MIN_BARS:
                    out[t] = d["Close"]
            except Exception:
                pass
    return out
def detect_recovery(c: pd.Series, offset: int = 0):
    """修復觸發：120 日內曾收年線下，且今日是『最後一次收在年線下之後的第一個多頭排列日』。
    嚴格版（2026-09-15）：光看「昨日尚未排列」會讓早已修復的股票每次跌破月線再站回就重新觸發
    （例 MAAS 4/16 最後一次收年線下，之後觸發了 7 次）。回測（2015–2026）：嚴格版筆數 875 → 797/年，
    EV +0.53R → +0.54R、勝率 33% 不變；曾怪物子集 EV +1.44R → +1.49R、勝率 46% → 47%。
    數字幾乎相同，但名單乾淨很多。"""
    if offset > 0:
        c = c.iloc[:len(c) - offset]
    if len(c) < MIN_BARS:
        return None
    e20 = c.ewm(span=20, adjust=False).mean()
    e60 = c.ewm(span=60, adjust=False).mean()
    e260 = c.ewm(span=260, adjust=False).mean()
    al = (c > e20) & (e20 > e60) & (e60 > e260)
    if not bool(al.iloc[-1]) or bool(al.iloc[-2]):
        return None
    below = (c < e260).iloc[-(BELOW_LOOKBACK + 1):-1]
    if not below.any():
        return None
    # 嚴格條件：最後一次收年線下之後，今天以前不曾出現過多頭排列
    last_below_ts = below[below].index[-1]
    if bool(al.loc[(al.index > last_below_ts) & (al.index < c.index[-1])].any()):
        return None
    px, yr = float(c.iloc[-1]), float(e260.iloc[-1])
    r126 = c / c.shift(126) - 1
    r_max = float(r126.iloc[-MONSTER_LB:].max()) if len(r126.dropna()) else np.nan
    hi63 = float(c.iloc[-64:-1].max())
    return {
        "觸發日": str(c.index[-1].date()),
        "收盤": round(px, 2),
        "年線": round(yr, 2),
        "距年線%": round((px / yr - 1) * 100, 1),
        "最後收年線下": str(below[below].index[-1].date()),
        "曾怪物": (f"🦖 {r_max*100:+.0f}%" if r_max >= MONSTER_TH else f"{r_max*100:+.0f}%") if r_max == r_max else "—",
        "距63日高%": round((px / hi63 - 1) * 100, 1),
        "_risk": max(px - yr, MIN_RISK * px),
        "_monster": bool(r_max >= MONSTER_TH) if r_max == r_max else False,
    }
@st.cache_data(ttl=86400, show_spinner=False)
def scan(tickers: tuple) -> tuple:
    """下載 → 偵測 → 只回傳結果表（幾十列）與資料日期。
    快取的是這張小表，不是上千檔價格：每個讀者複製的成本從幾百 MB 降到幾十 KB。"""
    PX = download(tickers)
    rows = []
    for t, c in PX.items():
        for off in range(SCAN_DAYS):
            r = detect_recovery(c, offset=off)
            if r is not None:
                r["代號"] = t
                rows.append(r)
    as_of = max((c.index[-1] for c in PX.values()), default=None)
    return pd.DataFrame(rows), (str(as_of.date()) if as_of is not None else "—"), len(PX)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_industry(tickers: tuple) -> dict:
    out = {}
    for t in tickers:
        try:
            i = yf.Ticker(t).info
            out[t] = (i.get("industry") or i.get("sector") or "", (i.get("marketCap") or 0) / 1e9)
        except Exception:
            out[t] = ("", 0.0)
    return out
# ── 全頁字級放大（適合 50 歲以上閱讀）2026-09-15 ────────────────────
#   與 pages/1_Dashboard.py、6_WinRate_Matrix.py 同一套：根字級 16px → 20px。
#   結果表改用 HTML 表格（見下方 table_html）：st.dataframe 是 canvas 繪製，
#   字級不吃 CSS，只有換成 HTML 表格才能跟著放大。
st.markdown("""
<style>
  html { font-size: 20px; }
  body, .stApp, [data-testid="stAppViewContainer"] { font-size: 1rem; line-height: 1.65; }
  [data-testid="stMarkdownContainer"] p  { font-size: 1rem; line-height: 1.7; }
  [data-testid="stMarkdownContainer"] li { font-size: 1rem; line-height: 1.7; }
  [data-testid="stMarkdownContainer"] h1 { font-size: 2.1rem; }
  [data-testid="stMarkdownContainer"] h2 { font-size: 1.75rem; }
  [data-testid="stMarkdownContainer"] h3 { font-size: 1.4rem; }
  [data-testid="stMarkdownContainer"] h4 { font-size: 1.2rem; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { font-size: 1rem !important; }
  [data-testid="stCaptionContainer"] p { font-size: 0.92rem !important; }
  .stButton button, .stDownloadButton button { font-size: 1rem; padding: 0.5rem 0.9rem; }
  .stButton button p { font-size: 1rem; }
  .stTextInput input, .stNumberInput input,
  [data-baseweb="select"] div, [data-baseweb="tag"] span { font-size: 1rem; }
  [data-testid="stCheckbox"] label p { font-size: 1rem !important; }
  [data-testid="stExpander"] summary p, details summary { font-size: 1.1rem; font-weight: 600; }
  [data-testid="stAlert"] p { font-size: 1rem; }
  [data-testid="stSidebar"] * { font-size: 1rem; }
  [data-testid="stSidebarNav"] a span, [data-testid="stSidebarNavLink"] span { font-size: 1.02rem; }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

  /* 修復觸發結果表（取代 st.dataframe，字級才放得大） */
  .rec-wrap { overflow-x: auto; max-height: 620px; border: 1px solid #1e293b; border-radius: 8px; }
  table.rec { border-collapse: collapse; width: max-content; min-width: 100%; }
  table.rec th {
    position: sticky; top: 0; z-index: 2;
    background: #0f172a; color: #64748b; font-size: 0.88rem; font-weight: 600;
    text-align: right; white-space: nowrap; padding: 10px 14px;
    border-bottom: 1px solid #1e293b;
  }
  table.rec td {
    font-size: 1rem; color: #cbd5e1; text-align: right; white-space: nowrap;
    padding: 9px 14px; border-bottom: 1px solid #16202f;
  }
  table.rec th.l, table.rec td.l { text-align: left; }
  table.rec tr:hover td { background: #131c2b; }
  table.rec td.code { font-weight: 700; color: #e2e8f0; font-size: 1.08rem; }
</style>
""", unsafe_allow_html=True)


# ── 管理員模式 ─────────────────────────────────────────────────────
#   高成本功能（重新掃描＝清全域快取、全美股掃描）只給管理員，避免讀者一多就把
#   Community Cloud 的 1 GB 記憶體與 Yahoo 限流打爆。
#   密碼放 .streamlit/secrets.toml 的 admin_password；
#   Community Cloud 在 App settings → Secrets 貼上，改完會自動重啟。
def is_admin() -> bool:
    if st.session_state.get("_is_admin"):
        return True
    try:
        real = st.secrets.get("admin_password", "")
    except Exception:
        real = ""
    if not real:
        return False          # 沒設 admin_password → 管理功能一律關閉（讀者看不到任何提示）
    with st.sidebar:
        with st.expander("🔑 管理員"):
            pw = st.text_input("管理密碼", type="password", key="_adminpw")
            if pw:
                if pw == real:
                    st.session_state["_is_admin"] = True
                    st.rerun()
                st.caption("密碼不正確")
    return False


ADMIN = is_admin()


# ── Page ──────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🩹 修復觸發選股")
    st.markdown(
        "<span style='color:#64748b;font-size:0.9rem'>"
        "<b>觸發</b>：過去 120 日內曾收在年線（EMA260）下，且今日是<b>最後一次收年線下之後的第一個</b>「收盤 &gt; 月線 &gt; 季線 &gt; 年線」日　｜　"
        "<b>出場</b>：收盤跌破年線　｜　<b>尺寸</b>：R ÷ max(收盤 − 年線, 10%)　｜　"
        "不看量、不看當日漲幅"
        "</span>", unsafe_allow_html=True)
with col_refresh:
    if ADMIN and st.button("🔄 重新掃描", use_container_width=True):
        # 清全域快取，所有讀者一起重抓 → 只開放給管理員
        download.clear(); scan.clear()
        st.rerun()
st.markdown("---")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    _pools = ["S&P 500", "Nasdaq 100", "費城半導體（SOX 概念）"]
    if ADMIN:
        _pools.append("全美股（市值 >$1B，約 3–5 分鐘）")   # 成本最高，讀者看不到
    picked = st.multiselect("掃描股票池（可複選，自動去重）", _pools, default=["S&P 500"])
with c2:
    only_monster = st.checkbox("只看 🦖 曾怪物", value=False,
                               help="過去 250 日內半年漲幅曾 ≥120%。回測：加此濾網 EV +0.66R → +1.52R、勝率 33% → 44%、≥3R 10% → 17%。")
with c3:
    r_usd = st.number_input("R（美元）", min_value=1.0, value=930.0, step=10.0)
if not picked:
    st.info("請至少選一個股票池。")
    st.stop()
_all, _srcs = [], []
for p in picked:
    if p.startswith("S&P"):
        t, sc = fetch_sp500()
    elif p.startswith("Nasdaq"):
        t, sc = fetch_nasdaq100()
    elif p.startswith("費城"):
        t, sc = fetch_semis()
    else:
        t, sc = fetch_largecap()
    _all += list(t); _srcs.append(sc)
tickers = tuple(dict.fromkeys(_all))
src = "　＋　".join(_srcs)
fallback = "備援" in src
st.markdown(f"<span style='color:{'#fbbf24' if fallback else '#475569'};font-size:0.9rem'>"
            f"成分股來源：{src}　→　去重後共 <b>{len(tickers)}</b> 檔"
            f"{'　⚠️ Wikipedia 抓取失敗，名單可能未反映最近調整。' if fallback else ''}</span>",
            unsafe_allow_html=True)
try:
    wait = "約需 3–5 分鐘" if any(p.startswith("全美股") for p in picked) else "約需 30–60 秒"
    with st.spinner(f"掃描 {len(tickers)} 檔中，{wait}（每日只跑一次，之後所有人共用結果）…"):
        rec, as_of, n_px = scan(tuple(tickers))
    rec = rec.copy()                      # 快取物件不可就地修改
    if len(rec) and only_monster:
        rec = rec[rec["_monster"]]
    st.markdown(f"<span style='color:#94a3b8;font-size:0.9rem'>取得價格 {n_px} 檔｜資料截至 {as_of}"
                f"｜每日盤後更新一次</span>", unsafe_allow_html=True)
    if rec.empty:
        st.info(f"近 {SCAN_DAYS} 個交易日無修復觸發。修復股在崩盤後的修復年（2016、2020、2023、2025）最密集，"
                f"延續年與慢牛年本來就少。")
    else:
        with st.spinner("載入產業資料…"):
            ind = fetch_industry(tuple(rec["代號"].unique().tolist()))
        rec["產業"] = rec["代號"].map(lambda t: ind.get(t, ("", 0))[0][:24])
        rec["市值B"] = rec["代號"].map(lambda t: round(ind.get(t, ("", 0))[1], 1))
        rec["股數_1R"] = (r_usd / rec["_risk"]).astype(int)
        rec["名目"] = (rec["股數_1R"] * rec["收盤"]).map(lambda v: f"{v:,.0f}")
        rec = rec.sort_values(["觸發日", "_monster", "市值B"], ascending=[False, False, False])
        n_m = int(rec["_monster"].sum())
        st.markdown(
            f"<span style='color:#4ade80;font-size:0.99rem;font-weight:700'>🩹 修復觸發：{len(rec)} 筆</span>"
            f"<span style='color:#475569;font-size:0.9rem'>　（其中 🦖 曾怪物 {n_m} 筆｜近 {SCAN_DAYS} 個交易日）</span>",
            unsafe_allow_html=True)
        show = ["觸發日", "代號", "收盤", "年線", "距年線%", "最後收年線下", "曾怪物",
                "距63日高%", "市值B", "產業", "股數_1R", "名目"]
        # 靠左欄位（文字類），其餘靠右對齊
        LEFT = {"觸發日", "代號", "最後收年線下", "曾怪物", "產業"}
        def _cell(col, v):
            cls = "l" if col in LEFT else ""
            if col == "代號":
                cls = "l code"
            if col == "距年線%":
                c = "#4ade80" if float(v) >= 0 else "#f87171"
                return f'<td style="color:{c};font-weight:700">{v:+.1f}%</td>'
            if col == "距63日高%":
                return f'<td>{v:+.1f}%</td>'
            return f'<td class="{cls}">{v}</td>'
        head = "".join(f'<th class="{"l" if c in LEFT else ""}">{c}</th>' for c in show)
        body = "".join(
            "<tr>" + "".join(_cell(c, r[c]) for c in show) + "</tr>"
            for _, r in rec[show].reset_index(drop=True).iterrows()
        )
        st.markdown(f'<div class="rec-wrap"><table class="rec">'
                    f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#334155;font-size:0.9rem;margin-top:6px'>"
            "停損＝收盤跌破年線（隔日開盤出）；股數 = R ÷ max(收盤 − 年線, 10%×收盤)，為系統規則之計算示例。"
            "觸發日以外的日期進場不在回測統計內。<b>質化門檻「產業龍頭」（子產業市值第一）請自行判斷</b>——"
            "本表僅列產業與市值。本表為客觀條件標記，不構成任何投資建議。</div>",
            unsafe_allow_html=True)
except Exception as e:
    st.error(f"掃描失敗：{e}")
with st.expander("📖 規則與依據"):
    st.markdown(f"""
**觸發**：過去 {BELOW_LOOKBACK} 個交易日內至少一天收盤在年線（EMA260）之下，且今日是
**「最後一次收在年線下之後的第一個多頭排列日」**（收盤 > EMA20 > EMA60 > EMA260）。不看量、不看當日漲幅。

　嚴格版（2026-09-15 改）：原本只檢查「昨日尚未排列」，導致早已修復的股票每次跌破月線再站回就重新入選——
　例 MAAS 最後一次收年線下是 4/16，之後又觸發了 7 次。回測（2015–2026、≥$5B 美股）：
　嚴格版筆數 875 → 797 筆/年，EV +0.53R → **+0.54R**、勝率 33% 不變；曾怪物子集 EV +1.44R → **+1.49R**、勝率 46% → 47%。
　績效差異在雜訊內，但名單乾淨很多。

**出場**：收盤跌破年線，隔日開盤出。**尺寸**：股數 = R ÷ max(收盤 − 年線, 10%×收盤)，部位 ≤ 帳戶 10%。

**質化門檻：產業龍頭**（子產業市值第一）。回測：加此門檻後尾部由 −31R 縮到 −8R。本頁不自動判定，只列產業與市值。

**🦖 曾怪物**：過去 {MONSTER_LB} 日內半年漲幅曾 ≥{MONSTER_TH*100:.0f}%。
回測（2015–2026、目前市值 ≥$5B 美股、倖存者偏差）：同一條修復規則加上此濾網，
EV **+0.66R → +1.52R**、勝率 **33% → 44%**、≥3R 比例 **10% → 17%**、中位持有 47 → 87 日。
前十大贏家全在此類（LITE 2025/5 +90R、BE、PLTR 2023/8、MSTR 2023/10、GME 2020/9）。

**資料長度**：本頁抓 **5 年日線、至少 800 根**才計算。EMA260 是無限記憶平均，資料太短年線會失真——
例：MAAS 2026-09-14，用 500 根算出年線 11.61（股價 +41% 在其上 → 誤觸發），
用 800 根以上為 18.31（股價 −11% 在其下 → 正確地不觸發）。看盤軟體用完整歷史，所以會跟短視窗的結果相反。
副作用：上市未滿約 3.2 年的新股會因資料不足被排除，這與規則一「上市滿一年」的精神一致。

**額度**：修復 ≤6 檔、動能 ≤9 檔，兩者獨立；總曝險合計以規則四為準。

**適用季節**：崩盤後的修復年（2016、2020、2023、2025）部位級 +0.65R／勝率 37%；
崩盤年本身（2018、2022）EV ≈ 0。與動能策略年度相關僅 +0.22，合併 CAGR 4.8% → 12.3%。
""")
