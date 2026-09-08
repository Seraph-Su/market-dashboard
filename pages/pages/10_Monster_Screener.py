import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance import EquityQuery as Q

# ═══════════════════════════════════════════════════════════════════
# 🦖 怪物股選股器（順勢交易系統 規則一＋規則二）
#   宇宙：美股普通股、市值 > $1B、過去半年漲幅 > 150%、剔除能源／礦業金屬／生技製藥
#   觸發：今日收盤創 63 日新高；許可：領頭股燈號綠燈（≥5/8）且壓力否決未成立
#   輸出：合格名單＋前波支撐停損＋1R 股數；另附「修復突破」候選（年線下→多頭排列）
# ═══════════════════════════════════════════════════════════════════

TOP8_FALLBACK = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA"]
MACRO = ["SPY", "CME", "XLP", "XLY", "IWM"]
EXCL_SECTOR = {"Energy"}
EXCL_INDUSTRY_KW = ["Biotech", "Drug Manufacturers", "Pharmaceutical", "Gold", "Silver", "Copper", "Steel",
                    "Aluminum", "Other Industrial Metals", "Other Precious Metals", "Coking Coal", "Thermal Coal",
                    "Uranium", "Oil & Gas"]
EXCH_OK = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS"}
BATCH = 40


# ── 資料 ─────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def screen_universe(min_cap_b: float, min_52w: float) -> pd.DataFrame:
    """Yahoo 篩選器粗篩：市值 > min_cap、52 週漲幅 > min_52w（半年 >150% 的必要條件近似）。"""
    q = Q("and", [Q("gt", ["intradaymarketcap", min_cap_b * 1e9]),
                  Q("gt", ["fiftytwowkpercentchange", min_52w]),
                  Q("eq", ["region", "us"])])
    rows, off = [], 0
    while off < 3000:
        r = yf.screen(q, size=250, offset=off, sortField="intradaymarketcap", sortAsc=False)
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
    return pd.DataFrame(keep)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_prices(tickers: tuple, period: str = "2y") -> dict:
    """分批下載 OHLC，失敗單檔補抓；回傳 {ticker: DataFrame}。"""
    out = {}
    tk = list(tickers)
    for i in range(0, len(tk), BATCH):
        ch = tk[i:i + BATCH]
        try:
            raw = yf.download(ch, period=period, interval="1d", auto_adjust=True,
                              progress=False, threads=False, group_by="ticker")
        except Exception:
            raw = None
        for t in ch:
            try:
                d = raw[t] if (raw is not None and len(ch) > 1) else raw
                d = d[["Open", "High", "Low", "Close"]].dropna(how="all")
                if len(d) >= 130:
                    out[t] = d
            except Exception:
                pass
    missing = [t for t in tk if t not in out]
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


@st.cache_data(ttl=86400, show_spinner=False)
def top8_by_cap() -> tuple:
    """即時市值前八大普通股（抓不到則用備援名單）。"""
    try:
        r = yf.screen(Q("and", [Q("eq", ["region", "us"]), Q("gt", ["intradaymarketcap", 3e11])]),
                      size=30, sortField="intradaymarketcap", sortAsc=False)
        seen, out = set(), []
        for x in r.get("quotes", []):
            if x.get("quoteType") != "EQUITY":
                continue
            base = x["symbol"].replace("GOOG", "GOOGL") if x["symbol"] == "GOOG" else x["symbol"]
            if base in seen or x["symbol"] in ("BRK-A",):
                continue
            seen.add(base); out.append(x["symbol"])
            if len(out) == 8:
                break
        for t in TOP8_FALLBACK:          # 不足八檔時用備援名單補齊
            if len(out) >= 8:
                break
            if t not in out:
                out.append(t)
        return tuple(out[:8])
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


# ── Page ──────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🦖 怪物股選股器")
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem'>"
        "宇宙＝市值 > $1B、半年漲幅 > 150%、非能源／礦業／生技　｜　觸發＝今日創 63 日新高　｜　"
        "許可＝領頭股綠燈且壓力否決未成立　｜　資料每日快取"
        "</span>", unsafe_allow_html=True)
with col_refresh:
    if st.button("🔄 重新掃描", use_container_width=True):
        st.cache_data.clear()
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
    show_recovery = st.checkbox("同時列出修復突破候選", value=True,
                                help="120 日內曾收在年線下、今日站回 價>月>季>年 多頭排列（第三引擎；需再過「產業龍頭」質化門檻）")

# ── 1. 宇宙 ──
with st.spinner("Yahoo 篩選器粗篩中…"):
    univ = screen_universe(min_cap, 50.0)
if univ.empty:
    st.error("篩選器沒有回傳資料（可能被 Yahoo 暫時限流），請稍後按「重新掃描」。")
    st.stop()

# ── 2. 價格 ──
top8 = list(top8_by_cap())
tickers = tuple(dict.fromkeys(univ["t"].tolist() + top8 + MACRO))
prog = st.progress(0, text=f"下載 {len(tickers)} 檔價格資料（首次約 2～4 分鐘，之後快取）…")
PX = fetch_prices(tickers)
prog.progress(100, text=f"價格資料完成：{len(PX)} 檔")
prog.empty()

# ── 3. 閘門 ──
def close_of(t):
    return PX[t]["Close"].ffill() if t in PX else None

n_above, n_prev, have = 0, 0, 0
for t in top8:
    c = close_of(t)
    if c is None or len(c) < 80:
        continue
    e60 = c.ewm(span=60, adjust=False).mean()
    have += 1
    n_above += int(c.iloc[-1] > e60.iloc[-1])
    n_prev += int(c.iloc[-11] > e60.iloc[-11])
if have >= 6:
    light = "綠" if n_above >= 5 else ("紅" if n_above <= 3 else ("惡化黃" if n_prev > n_above else "修復黃"))
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
          help=f"前八大：{', '.join(top8)}。≥5 綠、4 黃（惡化黃＝從綠掉下來，開門；修復黃＝從紅爬上來，關門）、≤3 紅。")
g2.metric("CME/SPY 10 日", f"{cme10:+.1f}%", "否決燈 · 需與 XLP/XLY 同亮", delta_color="off")
g3.metric("XLP/XLY 20 日", f"{xl20:+.1f}%", "單燈亮不否決", delta_color="off")
g4.metric("新倉許可", "✅ 開" if gate_open else "⛔ 關",
          "壓力否決成立" if sig1 else ("燈號未開門" if not gate_open else f"截至 {asof}"), delta_color="off")
if not gate_open:
    st.warning("閘門關：下方名單僅供觀察，不開新倉。等燈號轉綠／否決解除後，再看當日有無 🔔 觸發。")

# ── 4. 名單 ──
rows, rec_rows, excluded = [], [], []
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
    e20 = c.ewm(span=20, adjust=False).mean(); e60 = c.ewm(span=60, adjust=False).mean()
    e260 = c.ewm(span=260, adjust=False).mean() if len(c) >= 300 else None
    is_monster = r6 >= mom_th / 100
    is_recovery = False
    if show_recovery and e260 is not None:
        stack = (c > e20) & (e20 > e60) & (e60 > e260)
        was_below = bool((c.iloc[-121:-1] < e260.iloc[-121:-1]).any())
        is_recovery = bool(stack.iloc[-1]) and was_below and r6 >= 0.3
    if not (is_monster or is_recovery):
        continue
    sec, ind = fetch_sector(t)
    if sec in EXCL_SECTOR or any(k.lower() in ind.lower() for k in EXCL_INDUSTRY_KW):
        excluded.append(f"{t}（{ind or sec}）")
        continue
    a14 = float(atr14(df).iloc[-1])
    hi63 = float(c.iloc[-64:-1].max())
    stop, stop_dt = swing_stop(df, px)
    sh = int(r_usd / (px - stop)) if px > stop else 0
    band = ">10B" if x["cap"] > 10 else ("2-10B" if x["cap"] > 2 else "1-2B")
    base = dict(代碼=t, 名稱=x["name"][:18], 半年=f"{r6*100:+.0f}%", 市值B=round(x["cap"], 1), 帶=band, 價=round(px, 2),
                距63日高=f"{(px/hi63-1)*100:+.1f}%", 觸發="🔔" if px >= hi63 else "",
                距季線=f"{(px/e60.iloc[-1]-1)*100:+.0f}%", ATR=f"{a14/px*100:.1f}%",
                停損=round(stop, 2), 停損距=f"{(px/stop-1)*100:.0f}%", 停損日=str(stop_dt) if stop_dt else "—",
                股數_1R=sh, 名目=f"{sh*px:,.0f}", 產業=ind[:22])
    if is_monster:
        rows.append(base)
    if is_recovery and not is_monster:
        yl = float(e260.iloc[-1])
        sd = max(px / yl - 1, 0.10)
        rec_rows.append(dict(代碼=t, 名稱=x["name"][:18], 半年=f"{r6*100:+.0f}%", 市值B=round(x["cap"], 1), 價=round(px, 2),
                             距年線=f"{(px/yl-1)*100:+.0f}%", 距季線=f"{(px/e60.iloc[-1]-1)*100:+.0f}%",
                             距63日高=f"{(px/hi63-1)*100:+.1f}%", 觸發="🔔" if px >= hi63 else "",
                             年線停損=round(yl, 2), 股數_1R=int(r_usd / (px * sd)), 名目=f"{int(r_usd/(px*sd))*px:,.0f}",
                             產業=ind[:22]))

def order(df_):
    if df_.empty:
        return df_
    a = df_[df_["觸發"] == "🔔"].sort_values("市值B", ascending=False)
    b = df_[df_["觸發"] == ""].sort_values("市值B", ascending=False)
    return pd.concat([a, b])

mon = order(pd.DataFrame(rows))
st.markdown(f"#### 🦖 怪物股宇宙：{len(mon)} 檔（半年 > {mom_th}%）　🔔 觸發 {int((mon['觸發']=='🔔').sum()) if len(mon) else 0} 檔")
if len(mon):
    st.dataframe(mon.set_index("代碼"), use_container_width=True, height=min(60 + 35 * len(mon), 600))
    if gate_open and (mon["觸發"] == "🔔").any():
        st.success("🔔 今日有觸發且閘門開：隔日開盤市價進。進場前——質化五題 ≥3 分、股數照表、本金 Heat ≤ 9%、單股 ≤ 帳戶 10%（怪物股加碼上限 20%）、同題材 ≤ 50～60%。")
    st.markdown(
        "<div style='color:#334155;font-size:0.68rem'>"
        "停損＝最近已確認擺盪低點（前後 3 日最低、低於現價 5% 以上），距離上限 25%；太近（<8%）的停損請改看更前一個結構低點——"
        "停損寬度比任何加碼規則重要。股數 = R ÷（現價 − 停損）。名單只認突破，不做回調進場。"
        "</div>", unsafe_allow_html=True)
else:
    st.markdown("<span style='color:#64748b'>目前沒有合格的怪物股——這在慢牛年很正常，不是系統壞了。</span>", unsafe_allow_html=True)

if show_recovery:
    rec = order(pd.DataFrame(rec_rows))
    st.markdown("---")
    st.markdown(f"#### 🔁 修復突破候選：{len(rec)} 檔（年線下 → 站回多頭排列，半年 ≥30%，未達怪物門檻）")
    if len(rec):
        st.dataframe(rec.set_index("代碼"), use_container_width=True, height=min(60 + 35 * len(rec), 500))
        st.markdown(
            "<div style='color:#334155;font-size:0.68rem'>"
            "第三引擎：出場＝收盤跌破年線；股數 = R ÷ max(現價 − 年線, 10%×現價)（部位 ≤ 帳戶 10%）；"
            "質化門檻＝產業龍頭（尾部風險 −31R → −8R）；額度與怪物股獨立（修復 ≤6 檔）。燈號對此類無鑑別力，不套用。"
            "此候選表用「當日多頭排列且 120 日內曾破年線」判定，不要求今日剛站上，請自行確認站回日期。"
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#64748b'>目前沒有修復突破候選。</span>", unsafe_allow_html=True)

if excluded:
    st.markdown(f"<div style='color:#475569;font-size:0.72rem;margin-top:8px'>產業剔除（能源／礦業金屬／生技製藥）：{'、'.join(excluded)}</div>",
                unsafe_allow_html=True)

with st.expander("📖 規則與依據"):
    st.markdown(f"""
**規則一（買什麼）**：市值 > ${min_cap:g}B、半年漲幅 > {mom_th}%、非能源／礦業／生技製藥。依據：半年 >150% 的股票，六個月內再漲 >100% 的機率 7.4%（隨機 0.7%）；100～150% 區間勝率 34%、EV +0.10R，>150% 勝率 48%、EV +0.44R、怪物率 10.6%。高動能生技 EV −0.17R、勝率 25%，所有產業最差。

**規則二（何時買）**：收盤創 63 日新高那天觸發，隔日開盤進；領頭股 ≥5/8 站上季線為綠燈才開新倉。綠燈 EV +17.7%、紅燈 +3.2%，差在怪物率（8.9% vs 4.7%）不在勝率。壓力否決＝CME/SPY 10 日 ≥+5% **且** XLP/XLY 20 日 >+1% 同時成立，綠燈也不開；單燈亮不否決。

**規則三（多大、停損）**：R＝帳戶 1%，停損＝最新前波支撐低點，股數＝R ÷ 停損距離。停損只往上移。破了就走、不破就抱。停損出場後再創新高＝重進場。

**規則四（幾檔）**：最多 9 檔，本金 Heat ≤ 9%。同一天多檔觸發時各給 1R，讓停損去篩；名額不夠用題材籠子分（一個題材 3～4 檔）。

**修復突破（第三引擎）**：年線下 → 站回 價>月>季>年；跌破年線出；產業龍頭限定；與怪物股額度獨立。崩盤後的修復年主場（2016、2020、2023、2025）。

⚠️ 篩選器用 52 週漲幅 >50% 粗篩，極端情況（半年漲 150% 但 52 週仍 <50%）會漏掉；歷史回測有倖存者偏差；本頁不構成投資建議。
""")
