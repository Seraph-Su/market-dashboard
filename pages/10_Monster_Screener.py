import streamlit as st
import pandas as pd
import numpy as np
import time
import yfinance as yf
from yfinance import EquityQuery as Q

# ── 全頁字級放大（適合 50 歲以上閱讀）2026-09-15 ────────────────────
#   與 1_Dashboard.py / 6_WinRate_Matrix.py / 5_Breakout_Screener.py 同一套：
#   根字級 16px → 20px，本頁所有 rem 一次放大 1.25 倍。
#   ⚠️ st.data_editor（可編輯表格）是 canvas 繪製，字級不吃 CSS，
#      需在專案根目錄 .streamlit/config.toml 設 [theme] baseFontSize = 20。
#      唯讀表格已改用 big_table() 的 HTML 表格，字級才放得大。
st.markdown("""
<style>
  html { font-size: 20px; }
  body, .stApp, [data-testid="stAppViewContainer"] { font-size: 1rem; line-height: 1.65; }
  [data-testid="stMarkdownContainer"] p  { font-size: 1rem; line-height: 1.7; }
  [data-testid="stMarkdownContainer"] li { font-size: 1rem; line-height: 1.7; }
  [data-testid="stMarkdownContainer"] h1 { font-size: 2.1rem; }
  [data-testid="stMarkdownContainer"] h2 { font-size: 1.75rem; }
  [data-testid="stMarkdownContainer"] h3 { font-size: 1.4rem; }
  [data-testid="stMarkdownContainer"] h4 { font-size: 1.25rem; }
  [data-testid="stMarkdownContainer"] h5 { font-size: 1.1rem; }
  [data-testid="stMetricValue"] { font-size: 1.85rem !important; }
  [data-testid="stMetricLabel"] p { font-size: 0.98rem !important; }
  [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div { font-size: 0.95rem !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { font-size: 1rem !important; }
  [data-testid="stCaptionContainer"] p { font-size: 0.92rem !important; }
  [data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p { font-size: 1rem !important; }
  [data-testid="stAlert"] p { font-size: 1rem; }
  .stButton button, .stDownloadButton button { font-size: 1rem; padding: 0.5rem 0.9rem; }
  .stButton button p { font-size: 1rem; }
  .stTextInput input, .stNumberInput input, .stDateInput input,
  [data-baseweb="select"] div, [data-baseweb="tag"] span, [data-baseweb="tab"] p { font-size: 1rem; }
  [data-testid="stExpander"] summary p, details summary { font-size: 1.08rem; font-weight: 600; }
  [data-testid="stSidebar"] * { font-size: 1rem; }
  [data-testid="stSidebarNav"] a span, [data-testid="stSidebarNavLink"] span { font-size: 1.02rem; }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

  /* 唯讀表格（big_table）：取代 st.dataframe，字級才能跟著放大 */
  .bigtbl-wrap { overflow: auto; border: 1px solid #1e293b; border-radius: 8px; }
  table.bigtbl { border-collapse: collapse; width: max-content; min-width: 100%; }
  table.bigtbl th {
    position: sticky; top: 0; z-index: 2; background: #0f172a; color: #64748b;
    font-size: 0.88rem; font-weight: 600; text-align: right; white-space: nowrap;
    padding: 10px 14px; border-bottom: 1px solid #1e293b;
  }
  table.bigtbl td {
    font-size: 1rem; color: #cbd5e1; text-align: right; white-space: nowrap;
    padding: 9px 14px; border-bottom: 1px solid #16202f;
  }
  table.bigtbl th.l, table.bigtbl td.l { text-align: left; }
  table.bigtbl td.code { font-weight: 700; color: #e2e8f0; font-size: 1.08rem; }
  table.bigtbl tr:hover td { background: #131c2b; }
</style>
""", unsafe_allow_html=True)


def big_table(df, height=520, show_index=True, fmt=None, left=()):
    """大字級唯讀表格：st.dataframe 是 canvas 繪製、字級吃不到 CSS，改輸出 HTML 表格。
    fmt：{欄名: "{:,.0f}"} 明確指定格式；未指定的浮點欄自動套千分位。
    left：靠左對齊的欄名集合（文字欄）；index 一律靠左並加粗。"""
    d = df.copy()
    fmt = fmt or {}
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: "—" if pd.isna(v) else f.format(v))
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "—" if pd.isna(v)
                            else (f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}"))
    head = (f'<th class="l">{d.index.name or ""}</th>' if show_index else "")
    head += "".join(f'<th class="{"l" if c in left else ""}">{c}</th>' for c in d.columns)
    body = ""
    for i, row in d.iterrows():
        tds = f'<td class="l code">{i}</td>' if show_index else ""
        tds += "".join(f'<td class="{"l" if c in left else ""}">'
                       f'{"" if pd.isna(row[c]) else row[c]}</td>' for c in d.columns)
        body += f"<tr>{tds}</tr>"
    st.markdown(f'<div class="bigtbl-wrap" style="max-height:{height}px">'
                f'<table class="bigtbl"><thead><tr>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 🦖 怪物股選股器（順勢交易系統 規則一＋規則二）
#   宇宙：美股普通股、市值 > $1B、過去半年漲幅 > 150%、上市滿一年、剔除能源／礦業金屬／生技製藥／加密貨幣相關
#   觸發：今日收盤創 63 日新高；許可：領頭股燈號綠燈（≥5/8）且壓力否決未成立
#   輸出：合格名單＋前波支撐停損＋1R 股數（修復突破候選在「均線收斂突破選股」頁）
# ═══════════════════════════════════════════════════════════════════

TOP8_FALLBACK = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA"]
MACRO = ["SPY", "CME", "XLP", "XLY", "IWM"]
EXCL_SECTOR = {"Energy"}
EXCL_INDUSTRY_KW = ["Biotech", "Drug Manufacturers", "Pharmaceutical", "Gold", "Silver", "Copper", "Steel",
                    "Aluminum", "Other Industrial Metals", "Other Precious Metals", "Coking Coal", "Thermal Coal",
                    "Uranium", "Oil & Gas"]
# 加密貨幣相關（礦機商、持幣公司、交易所）與迷因股：產業分類抓不到，用名單剔除
EXCL_TICKERS_DEFAULT = "MSTR, MARA, RIOT, HUT, CLSK, BITF, CIFR, WULF, IREN, CORZ, GREE, BTBT, HIVE, SDIG, BTCS, COIN, BKKT, GLXY, SBET, BMNR, DFDV, CEP, CAN, EBON, NCTY, BTDR, SLNH, GRYP, APLD, GME, AMC, KOSS, MULN"
EXCH_OK = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS"}
BATCH = 25


# ── 資料 ─────────────────────────────────────────────────────────
# 備援宇宙用的半導體名單（Yahoo 產業篩選同樣會被擋，只能寫死）
SEMIS_FALLBACK = ["NVDA", "AVGO", "AMD", "TSM", "MU", "INTC", "QCOM", "TXN", "ADI", "LRCX",
                  "AMAT", "KLAC", "MRVL", "NXPI", "MCHP", "ON", "SWKS", "QRVO", "MPWR", "TER",
                  "ENTG", "ASML", "ARM", "ALAB", "CRDO", "RMBS", "LSCC", "SITM", "POWI", "AOSL",
                  "COHR", "SNDK", "LITE", "AAOI", "WOLF", "AMKR", "FORM", "ACLS", "UCTT", "ICHR"]


@st.cache_data(ttl=86400, show_spinner=False)
def _index_universe() -> pd.DataFrame:
    """備援宇宙：S&P 500 ＋ Nasdaq 100 ＋ 半導體（維基百科，不經 Yahoo 篩選器）。
    沒有市值欄位（cap = NaN），但這些成分股本來就都 >$1B，市值門檻等同已滿足。"""
    import requests
    from io import StringIO
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    out = []
    for url, minlen in (("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 400),
                        ("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", 90)):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            for tb in pd.read_html(StringIO(r.text)):
                cols = [str(c) for c in tb.columns]
                hit = [c for c in cols if "Ticker" in c or "Symbol" in c]
                if not hit or len(tb) < minlen:
                    continue
                ser = tb[hit[0]].astype(str).str.strip().str.replace(".", "-", regex=False)
                out += [t for t in ser.tolist()
                        if t and t.upper() != "NAN" and 1 <= len(t) <= 6 and t.replace("-", "").isalpha()]
                break
        except Exception:
            pass
    out += SEMIS_FALLBACK
    out = list(dict.fromkeys(out))
    return pd.DataFrame([{"t": t, "cap": float("nan"), "name": ""} for t in out])


UNIV_FILE = "data/monster_universe.json"      # 由排程 sync_universe.py 產生後 commit 上來


def _univ_path():
    """找 repo 裡的名單檔。st.navigation 下 __file__ 與工作目錄的關係不一定固定，
    所以多試幾個位置：pages/ 的上一層、目前工作目錄、相對路徑。"""
    from pathlib import Path as _P
    cands = []
    try:
        cands.append(_P(__file__).resolve().parent.parent / UNIV_FILE)
    except Exception:
        pass
    cands += [_P.cwd() / UNIV_FILE, _P(UNIV_FILE)]
    for f in cands:
        try:
            if f.exists():
                return f
        except Exception:
            pass
    return None


def _univ_sig() -> str:
    """檔案簽章（路徑＋修改時間＋大小），當作快取 key 的一部分：
    排程推上新名單、或檔案從無到有時，快取自動失效，不用等 24 小時或手動重啟。"""
    f = _univ_path()
    if f is None:
        return "none"
    try:
        st_ = f.stat()
        return f"{f}:{int(st_.st_mtime)}:{st_.st_size}"
    except Exception:
        return "err"


def _file_universe(min_cap_b: float) -> tuple:
    """讀 repo 裡由排程產生的宇宙名單。雲端 IP 打不到 Yahoo screener，
    但沙盒／住宅 IP 打得到——所以名單由排程算好 commit 上來，網頁只負責讀。
    檔案不存在就回 (None, "")，自動往下一層走。"""
    import json, datetime as _dt
    f = _univ_path()
    if f is None:
        return None, ""
    try:
        blob = json.loads(f.read_text(encoding="utf-8"))
        rows = [x for x in blob["rows"] if (x.get("cap") or 0) >= min_cap_b]
        if not rows:
            return None, ""
        built = blob.get("built_at", "?")
        age = (_dt.date.today() - _dt.date.fromisoformat(built[:10])).days
        note = f"排程名單（{built[:10]}，{len(rows)} 檔候選）"
        if age >= 3:
            note = f"⚠️ 排程名單已經 {age} 天沒更新（{built[:10]}，{len(rows)} 檔）——請在本機重跑 build_universe.py 並 commit。"
        return pd.DataFrame(rows), note
    except Exception:
        return None, ""


@st.cache_data(ttl=86400, show_spinner=False)
def screen_universe(min_cap_b: float, min_52w: float, _sig: str = "") -> tuple:
    """Yahoo 篩選器粗篩：市值 > min_cap、52 週漲幅 > min_52w（半年 >150% 的必要條件近似）。
    回傳 (DataFrame, 來源說明)。

    ⚠️ yf.screen 需要 Yahoo 的 cookie/crumb 認證，雲端機房 IP 很常被回 401/429
    （本機跑得動、部署到 Community Cloud 就掛）。所以這裡：
      ① 每頁重試 3 次、指數退避；② 整段失敗不再往外丟例外，改用指數成分股當備援宇宙。"""
    df_file, note_file = _file_universe(min_cap_b)
    if df_file is not None:                       # 第一層：排程產生的名單
        return df_file, note_file
    q = Q("and", [Q("gt", ["intradaymarketcap", min_cap_b * 1e9]),
                  Q("gt", ["fiftytwowkpercentchange", min_52w]),
                  Q("eq", ["region", "us"])])
    rows, off, last_err = [], 0, None
    while off < 3000:
        r = None
        for attempt in range(3):
            try:
                r = yf.screen(q, size=250, offset=off, sortField="intradaymarketcap", sortAsc=False)
                break
            except Exception as e:                 # 401／429／連線中斷都在這裡吸收
                last_err = e
                # Yahoo 對機房 IP 封的是 screener endpoint 本身，不是限流：
                # 回「User is unable to access this feature」時重試永遠不會成功，直接放棄省 12 秒。
                if "unable to access this feature" in str(e).lower():
                    break
                time.sleep(2 * (attempt + 1))
        if r is None:
            break
        qs = r.get("quotes", [])
        rows += qs
        off += 250
        if not qs or off >= r.get("total", 0):
            break
    keep = []
    for x in rows:
        if x.get("quoteType") != "EQUITY" or x.get("exchange") not in EXCH_OK:
            continue
        keep.append({"t": x["symbol"], "cap": (x.get("marketCap") or 0) / 1e9,
                     "name": x.get("shortName", "")})
    if keep:
        return pd.DataFrame(keep), f"Yahoo 篩選器（{len(keep)} 檔候選）"
    fb = _index_universe()
    why = "（Yahoo 封鎖機房 IP 的 screener endpoint）" if last_err is not None else "（無回傳資料）"
    # 備援結果不該鎖 24 小時：把 TTL 改短做不到（裝飾器固定），改用簽章讓新檔一到就失效
    return fb, ("⚠️ Yahoo 篩選器連線失敗 " + why +
                f"——雲端 IP 常被限流。已改用備援宇宙：S&P 500 ＋ Nasdaq 100 ＋ 半導體共 {len(fb)} 檔。"
                "指數外的中小型怪物股這時候會漏掉，市值欄位也會是「—」。")


@st.cache_resource(ttl=86400, show_spinner=False)   # ⚠️ 不可改回 cache_data：
def fetch_prices(tickers: tuple, period: str = "1y") -> dict:   # cache_data 每 session 複製一份 → 爆記憶體
    """分批下載 OHLC（被限流時暫停重試一次）；缺太多時不逐檔補抓，避免卡死。回傳 {ticker: DataFrame}。"""
    out = {}
    tk = list(tickers)
    for i in range(0, len(tk), BATCH):
        ch = tk[i:i + BATCH]
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(ch, period=period, interval="1d", auto_adjust=True,
                                  progress=False, threads=False, group_by="ticker")
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(3)          # 疑似限流：等一下再試
        if raw is None or raw.empty:
            continue
        for t in ch:
            try:
                d = raw[t] if len(ch) > 1 else raw
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Open", "High", "Low", "Close"]].dropna(how="all")
                if len(d) >= 130:
                    out[t] = d
            except Exception:
                pass
    missing = [t for t in tk if t not in out]
    if 0 < len(missing) <= 40:      # 少量缺漏才逐檔補抓
        for t in missing:
            try:
                d = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Open", "High", "Low", "Close"]].dropna(how="all")
                if len(d) >= 130:
                    out[t] = d
            except Exception:
                pass
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sector(t: str) -> tuple:
    try:
        i = yf.Ticker(t).info
        return (i.get("sector") or "", i.get("industry") or "")
    except Exception:
        return ("", "")


# 同公司雙股別合併（與大盤壓力儀表板一致）
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
def top8_by_cap(n: int = 8) -> tuple:
    """S&P 500 市值前 n 大（合併雙股別、排除非成分股）。
    ⚠️ 必須與『大盤壓力儀表板』的 fetch_leader_list 完全相同，否則兩頁燈號會不一致：
    門檻 $2000 億（非 $3000 億）、且要過 S&P 500 成分股濾網（排除 TSM 等外國發行人）。"""
    try:
        r = yf.screen(Q("and", [Q("eq", ["region", "us"]), Q("gt", ["intradaymarketcap", 2e11])]),
                      size=25, sortField="intradaymarketcap", sortAsc=False)
        try:
            members = fetch_sp500_members()
        except Exception:
            members = None                      # 維基抓不到就不過濾
        seen, out = set(), []
        for x in r.get("quotes", []):
            sym = x.get("symbol", "")
            if not sym or "." in sym:
                continue
            sym = _SHARE_CLASS.get(sym, sym)
            if sym in seen:
                continue
            if members is not None and sym not in members:
                continue
            seen.add(sym); out.append(sym)
            if len(out) >= n:
                break
        return tuple(out) if len(out) >= 6 else tuple(TOP8_FALLBACK)
    except Exception:
        return tuple(TOP8_FALLBACK)


def atr14(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def swing_stop(df: pd.DataFrame, px: float) -> tuple:
    """最近已確認擺盪低點（前後 3 日最低、低於現價 5% 以上），距離上限 25%。回傳 (停損, 日期或 None)。"""
    lo = df["Low"]
    for i in range(len(lo) - 4, max(len(lo) - 120, 3), -1):
        if lo.iloc[i] == lo.iloc[i - 3:i + 4].min() and lo.iloc[i] < px * 0.95:
            return max(float(lo.iloc[i]), px * 0.75), lo.index[i].date()
    return px * 0.80, None



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
    st.markdown("## 🦖 怪物股選股器")
    st.markdown(
        "<span style='color:#64748b;font-size:0.9rem'>"
        "宇宙＝市值 > $1B、半年漲幅 > 150%、上市滿一年、非能源／礦業／生技／加密　｜　觸發＝今日創 63 日新高　｜　"
        "許可＝領頭股綠燈且壓力否決未成立　｜　資料每日快取"
        "</span>", unsafe_allow_html=True)
with col_refresh:
    if ADMIN and st.button("🔄 重新掃描", use_container_width=True):
        # 清全域快取，所有讀者一起重抓 → 只開放給管理員
        fetch_prices.clear(); st.cache_data.clear()
        st.rerun()
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
with c1:
    mom_th = st.slider("半年漲幅門檻（%）", 100, 300, 150, 10,
                       help="回測：>150% 均 +0.44R／勝率 48%；100～150% 反而最弱（勝率 34%）。門檻不建議下修。")
with c2:
    min_cap = st.selectbox("最低市值（$B）", [1.0, 2.0, 5.0, 10.0], index=0)
with c3:
    r_usd = st.number_input("R（美元，帳戶 1%）", min_value=1.0, value=930.0, step=10.0)
with c4:
    excl_ipo = st.checkbox("剔除上市未滿一年", value=True,
                           help="2019～2025 年 IPO 回測：上市 <1 年進場均 −0.10R、勝率 26%、怪物率 ~4%；老牌股 +0.44R／48%／10.6%。")
# 「例外」與「排除名單」兩個輸入框移到頁面最下方（2026-09-15）。
# Streamlit 是由上往下執行，掃描時就要用到這兩個值，所以先從 session_state 讀（第一次用預設值），
# 輸入框本身放在頁尾、綁同一個 key；使用者一改，rerun 時這裡就拿到新值。
IPO_EXEMPT_DEFAULT = "SNDK, GEV, SOLV, SOLS, Q, VSNT"
st.session_state.setdefault("ipo_exempt_txt", IPO_EXEMPT_DEFAULT)
st.session_state.setdefault("excl_txt", EXCL_TICKERS_DEFAULT)
ipo_exempt = {s.strip().upper() for s in st.session_state["ipo_exempt_txt"].split(",") if s.strip()}
excl_tickers = {s.strip().upper() for s in st.session_state["excl_txt"].split(",") if s.strip()}

# ── 1. 宇宙 ──
with st.spinner("Yahoo 篩選器粗篩中…"):
    univ, univ_note = screen_universe(min_cap, 50.0, _univ_sig())   # 簽章變 → 快取失效
if univ.empty:
    st.error("篩選器與備援名單都取不到資料（Yahoo 與 Wikipedia 同時失敗），請稍後再試。")
    st.stop()
if univ_note.startswith("⚠️"):
    st.warning(univ_note)

# ── 2. 價格 ──
top8 = list(top8_by_cap())
tickers = tuple(dict.fromkeys(univ["t"].tolist() + top8 + TOP8_FALLBACK + MACRO))
prog = st.progress(0, text=f"下載 {len(tickers)} 檔價格資料（首次約 2～4 分鐘，之後快取）…")
PX = fetch_prices(tickers)
prog.progress(100, text=f"價格資料完成：{len(PX)} 檔")
prog.empty()
cov = sum(1 for t in univ["t"] if t in PX) / max(len(univ), 1)
st.markdown(f"<span style='color:#475569;font-size:0.9rem'>篩選器候選 {len(univ)} 檔｜取得價格 {sum(1 for t in univ['t'] if t in PX)} 檔（{cov*100:.0f}%）｜資料截至 {max((PX[t].index[-1] for t in PX), default='—')}</span>",
            unsafe_allow_html=True)
if cov < 0.6:
    st.warning(f"⚠️ 只取得 {cov*100:.0f}% 候選股的價格，很可能被 Yahoo 暫時限流——名單會不完整。請等 1～2 分鐘後按「🔄 重新掃描」。")
    if cov < 0.2:
        fetch_prices.clear()   # 幾乎全空的結果不要快取一整天

# ── 3. 閘門 ──
def close_of(t):
    return PX[t]["Close"].ffill() if t in PX else None

# 判定與大盤壓力儀表板一致：紅＝2% 緩衝後仍 ≤4；黃＝無緩衝 ≤4；其餘為綠
n_above, n_buf, have = 0, 0, 0
_above_cols = {}
for t in list(top8) + [x for x in TOP8_FALLBACK if x not in top8]:   # 缺資料時用備援名單補到 8 檔
    if have >= 8:
        break
    c = close_of(t)
    if c is None or len(c) < 80:
        continue
    e60 = c.ewm(span=60, adjust=False).mean()
    have += 1
    dist = float(c.iloc[-1] / e60.iloc[-1] - 1) * 100
    n_above += int(dist > 0)          # 無緩衝
    n_buf += int(dist > -2.0)         # 2% 緩衝
    _above_cols[t] = (c > e60)
# 方向：每日站上家數的 10 日均線，近兩週變化（與儀表板相同）
delta10 = 0.0
try:
    _h_all = pd.DataFrame(_above_cols).sum(axis=1)
    _h_ma10 = _h_all.rolling(10).mean()
    if len(_h_ma10.dropna()) > 11:
        delta10 = float(_h_ma10.iloc[-1] - _h_ma10.iloc[-11])
except Exception:
    pass
if have >= 6:
    if n_buf <= 4:
        light = "紅"
    elif n_above <= 4:
        light = "惡化黃" if delta10 <= -0.5 else "修復黃"
    else:
        light = "綠"
else:
    light = "資料不足"
spy, cme, xlp, xly = close_of("SPY"), close_of("CME"), close_of("XLP"), close_of("XLY")
cme10 = float((cme / spy).pct_change(10).iloc[-1] * 100) if cme is not None and spy is not None else float("nan")
xl20 = float((xlp / xly).pct_change(20).iloc[-1] * 100) if xlp is not None and xly is not None else float("nan")
sig1 = (cme10 >= 5) and (xl20 > 1)
gate_open = light in ("綠", "惡化黃") and not sig1
asof = spy.index[-1].date() if spy is not None else "—"

lc = {"綠": "#4ade80", "惡化黃": "#fbbf24", "修復黃": "#fb923c", "紅": "#f87171", "資料不足": "#94a3b8"}[light]
g1, g2, g3, g4 = st.columns(4)
g1.metric("領頭股燈號", f"{n_above}/{have}", light, delta_color="off",
          help=f"S&P 500 市值前八大：{', '.join(top8)}（與大盤壓力儀表板同一份名單與判定）。"
               f"紅＝2% 緩衝後仍 ≤4／黃＝無緩衝 ≤4（惡化黃＝10 日均近兩週下滑 ≥0.5 檔，開門；修復黃＝關門）／其餘為綠。"
               f"目前：無緩衝 {n_above}/{have}、2% 緩衝 {n_buf}/{have}、方向 {delta10:+.1f}。")
g2.metric("CME/SPY 10 日", f"{cme10:+.1f}%", "否決燈 · 需與 XLP/XLY 同亮", delta_color="off")
g3.metric("XLP/XLY 20 日", f"{xl20:+.1f}%", "單燈亮不否決", delta_color="off")
g4.metric("新倉許可", "✅ 開" if gate_open else "⛔ 關",
          "壓力否決成立" if sig1 else ("燈號未開門" if not gate_open else f"截至 {asof}"), delta_color="off")
if not gate_open:
    st.warning("閘門關：下方名單僅供觀察，不開新倉。等燈號轉綠／否決解除後，再看當日有無 🔔 觸發。")

# ── 4. 名單 ──
rows, excluded, excluded_ipo = [], [], []
for _, x in univ.iterrows():
    t = x["t"]
    if t not in PX:
        continue
    df = PX[t]
    c, l, h = df["Close"].ffill(), df["Low"].ffill(), df["High"].ffill()
    px = float(c.iloc[-1])
    if px < 5 or len(c) < 130:
        continue
    r6 = px / float(c.iloc[-126]) - 1
    e60 = c.ewm(span=60, adjust=False).mean()
    if r6 < mom_th / 100:
        continue
    # 上市未滿一年：抓 1 年資料卻不足 ~240 個交易日（回測：<1 年 IPO 均 −0.10R／勝率 26%，老牌股 +0.44R／48%）
    if excl_ipo and len(c) < 240 and t not in ipo_exempt:
        excluded_ipo.append(f"{t}（{len(c)} 日）")
        continue
    if t in excl_tickers:
        excluded.append(f"{t}（加密／迷因）")
        continue
    sec, ind = fetch_sector(t)
    if sec in EXCL_SECTOR or any(k.lower() in ind.lower() for k in EXCL_INDUSTRY_KW):
        excluded.append(f"{t}（{ind or sec}）")
        continue
    a14 = float(atr14(df).iloc[-1])
    hi63 = float(c.iloc[-64:-1].max())
    stop, stop_dt = swing_stop(df, px)
    sh = int(r_usd / (px - stop)) if px > stop else 0
    _cap = x["cap"]
    if _cap != _cap:                    # NaN：備援宇宙沒有市值資料
        band = "—"
    else:
        band = ">10B" if _cap > 10 else ("2-10B" if _cap > 2 else "1-2B")
    base = dict(代碼=t, 名稱=x["name"][:18], 半年=f"{r6*100:+.0f}%", 市值B=(round(_cap, 1) if _cap == _cap else float("nan")), 帶=band, 價=round(px, 2),
                距63日高=f"{(px/hi63-1)*100:+.1f}%", 觸發="🔔" if px >= hi63 else "",
                距季線=f"{(px/e60.iloc[-1]-1)*100:+.0f}%", ATR=f"{a14/px*100:.1f}%",
                停損=round(stop, 2), 停損距=f"{(px/stop-1)*100:.0f}%", 停損日=str(stop_dt) if stop_dt else "—",
                股數_1R=sh, 名目=f"{sh*px:,.0f}", 產業=ind[:22])
    rows.append(base)

def order(df_):
    if df_.empty:
        return df_
    a = df_[df_["觸發"] == "🔔"].sort_values("市值B", ascending=False)
    b = df_[df_["觸發"] == ""].sort_values("市值B", ascending=False)
    return pd.concat([a, b])

mon = order(pd.DataFrame(rows))
st.markdown(f"#### 🦖 怪物股宇宙：{len(mon)} 檔（半年 > {mom_th}%）　🔔 觸發 {int((mon['觸發']=='🔔').sum()) if len(mon) else 0} 檔")
if len(mon):
    big_table(mon.set_index("代碼"), height=min(72 + 46 * len(mon), 680),
              left={"名稱", "帶", "觸發", "停損日", "產業"},
              fmt={"市值B": "{:,.1f}", "價": "{:,.2f}", "停損": "{:,.2f}"})
    st.markdown(
        "<div style='color:#334155;font-size:0.9rem'>"
        "停損＝最近已確認擺盪低點（前後 3 日最低、低於現價 5% 以上），距離上限 25%；太近（<8%）的停損請改看更前一個結構低點——"
        "股數 = R ÷（現價 − 停損），為系統規則之計算示例。🔔 僅表示當日收盤創 63 日新高，為客觀條件標記，不構成任何投資建議。"
        "</div>", unsafe_allow_html=True)
else:
    st.markdown("<span style='color:#64748b'>目前沒有合格的怪物股——這在慢牛年很正常，不是系統壞了。</span>", unsafe_allow_html=True)

if excluded:
    st.markdown(f"<div style='color:#475569;font-size:0.9rem;margin-top:8px'>剔除（能源／礦業金屬／生技製藥／加密迷因）：{'、'.join(excluded)}</div>",
                unsafe_allow_html=True)
if excluded_ipo:
    st.markdown(f"<div style='color:#475569;font-size:0.9rem;margin-top:4px'>上市未滿一年剔除（括號＝可用交易日；若為分拆／重新上市的老公司，請加進下方例外欄）：{'、'.join(excluded_ipo)}</div>",
                unsafe_allow_html=True)

# ── 名單設定（例外／排除）──放在頁尾，改了會自動重新套用到上方名單 ──
st.markdown("---")
st.markdown("#### ⚙️ 名單設定")
st.text_input("例外：分拆／重新上市的老公司（逗號分隔，不視為新股）", key="ipo_exempt_txt",
              help="回測樣本只含真正的 IPO，不含分拆與重新掛牌；這些公司有完整營運歷史，不適用新股結論。DELL 2018 年重新上市、資料已滿一年，不受影響。")
st.text_input("排除名單：加密貨幣相關／迷因股（逗號分隔）", key="excl_txt",
              help="回測：剔除加密與迷因股後最大回撤由 −25.8% 收到 −20.0%，加碼貢獻由 −7R 轉為 +8～12R。產業分類抓不到這類公司，只能用名單。")

with st.expander("📖 規則與依據"):
    st.markdown(f"""
**規則一（買什麼）**：市值 > ${min_cap:g}B、半年漲幅 > {mom_th}%、上市滿一年、非能源／礦業／生技製藥／加密貨幣相關。依據：半年 >150% 的股票，六個月內再漲 >100% 的機率 7.4%（隨機 0.7%）；100～150% 區間勝率 34%、EV +0.10R，>150% 勝率 48%、EV +0.44R、怪物率 10.6%。高動能生技 EV −0.17R、勝率 25%，所有產業最差。上市未滿一年的 IPO（2019～2025 年 1,003 檔）觸發後均 −0.10R、勝率 26%、怪物率約 4%——前波低點未經驗證、閉鎖期解禁供給，故剔除。

**規則二（何時買）**：收盤創 63 日新高那天觸發，隔日開盤進；領頭股 ≥5/8 站上季線為綠燈才開新倉。綠燈 EV +17.7%、紅燈 +3.2%，差在怪物率（8.9% vs 4.7%）不在勝率。壓力否決＝CME/SPY 10 日 ≥+5% **且** XLP/XLY 20 日 >+1% 同時成立，綠燈也不開；單燈亮不否決。

**規則三（多大、停損）**：R＝帳戶 1%，停損＝最新前波支撐低點，股數＝R ÷ 停損距離。停損只往上移。破了就走、不破就抱。停損出場後再創新高＝重進場。

**規則四（幾檔）**：最多 9 檔，本金曝險 ≤ 9%。同一天多檔觸發時各給 1R，讓停損去篩；名額不夠用題材籠子分（一個題材 3～4 檔）。

⚠️ 篩選器用 52 週漲幅 >50% 粗篩，極端情況（半年漲 150% 但 52 週仍 <50%）會漏掉；歷史回測有倖存者偏差；本頁不構成投資建議。
""")
