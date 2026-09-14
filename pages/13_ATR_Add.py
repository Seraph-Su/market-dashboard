import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from yfinance import EquityQuery as Q
# ═══════════════════════════════════════════════════════════════════
# 🎯 ATR 加碼計算器（A＋T 自動檢測＋加碼比例）
#   取代舊的「EMA 季線計算器」與「ATR 乖離計算器」兩頁。
#   A 回季線：距季線 −3%～+6% 且 ATR5 < ATR14 且閘門開 → 0.5R
#   T 機械　：收盤 ≥ 上次進場價 + 2×ATR14 → 0.5R（只套怪物股與修復龍頭）
#   加碼單停損：加碼價 − 3×ATR14（初始停損，不追蹤；之後只跟基本倉結構停損上移）
#   一律收盤判定，隔日開盤市價進。加碼比例欄位用來控制單股名目與總曝險。
# ═══════════════════════════════════════════════════════════════════
TOP8_FALLBACK = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA"]
COLS = ["代碼", "類型", "進場日", "進場價", "目前股數", "上次加碼價"]
STORE = Path(__file__).with_name(".atr_add_meta.json")      # 本頁自動存檔（與舊頁分開，試用期互不干擾）
STORE_POS = Path(__file__).with_name(".positions.json")     # 總曝險計算器的持倉檔
def clean_code(v) -> str:
    t = str(v).strip().upper()
    return "" if t in ("", "NAN", "NONE", "<NA>") else t
def blank_row() -> pd.DataFrame:
    return pd.DataFrame([["", "一般", "", 0.0, 0, 0.0]], columns=COLS)
def load_add() -> pd.DataFrame:
    try:
        if STORE.exists():
            df = pd.read_json(STORE, dtype={"進場日": str})
            for c in COLS:
                if c not in df.columns:
                    df[c] = "" if c in ("代碼", "類型", "進場日") else 0
            df = df[COLS]
            if len(df):
                return pd.concat([df, blank_row()], ignore_index=True)
    except Exception:
        pass
    return blank_row()
def save_add(df: pd.DataFrame) -> None:
    try:
        keep = df[df["代碼"].map(clean_code) != ""]
        STORE.write_text(keep.to_json(orient="records", force_ascii=False), encoding="utf-8")
    except Exception:
        pass
def sync_from_exposure(cur: pd.DataFrame) -> pd.DataFrame:
    """從總曝險計算器帶入代碼、進場價、股數；已填過的類型／進場日／上次加碼價保留。"""
    try:
        pos = pd.read_json(STORE_POS)
    except Exception:
        return cur
    if not len(pos) or "代碼" not in pos:
        return cur
    prev = {clean_code(r["代碼"]): r for _, r in cur.iterrows() if clean_code(r["代碼"])}
    rows = []
    for code, g in pos.groupby(pos["代碼"].map(clean_code)):
        if not code:
            continue
        base_px = float(pd.to_numeric(g["進場價"], errors="coerce").fillna(0).iloc[0])
        shares = float(pd.to_numeric(g["股數"], errors="coerce").fillna(0).sum())
        old = prev.get(code)
        rows.append([code,
                     str(old["類型"]) if old is not None else "一般",
                     str(old["進場日"]) if old is not None and str(old["進場日"]).strip() else "",
                     float(old["進場價"]) if old is not None and float(old["進場價"] or 0) > 0 else base_px,
                     shares,
                     float(old["上次加碼價"]) if old is not None else 0.0])
    return pd.concat([pd.DataFrame(rows, columns=COLS), blank_row()], ignore_index=True) if rows else cur
@st.cache_data(ttl=3600, show_spinner=False)
def top8_by_cap() -> tuple:
    try:
        r = yf.screen(Q("and", [Q("eq", ["region", "us"]), Q("gt", ["intradaymarketcap", 3e11])]),
                      size=30, sortField="intradaymarketcap", sortAsc=False)
        seen, out = set(), []
        for x in r.get("quotes", []):
            if x.get("quoteType") != "EQUITY":
                continue
            base = "GOOGL" if x["symbol"] == "GOOG" else x["symbol"]
            if base in seen or x["symbol"] == "BRK-A":
                continue
            seen.add(base); out.append(x["symbol"])
            if len(out) == 8:
                break
        for t in TOP8_FALLBACK:
            if len(out) >= 8:
                break
            if t not in out:
                out.append(t)
        return tuple(out[:8])
    except Exception:
        return tuple(TOP8_FALLBACK)
@st.cache_data(ttl=1800, show_spinner=False)
def fetch(tickers: tuple) -> dict:
    """2 年日線，30 分鐘快取。"""
    out = {}
    if not tickers:
        return out
    try:
        raw = yf.download(list(tickers), period="2y", interval="1d", auto_adjust=True,
                          progress=False, threads=False, group_by="ticker")
        for t in tickers:
            try:
                d = raw[t] if len(tickers) > 1 else raw
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Open", "High", "Low", "Close"]].dropna(how="all").ffill()
                if len(d) >= 130:
                    out[t] = d
            except Exception:
                pass
    except Exception:
        pass
    return out
def atr_series(df: pd.DataFrame, n: int) -> pd.Series:
    c = df["Close"]
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()          # 回測同款：14 日簡單平均，非 Wilder
def swing_low(df: pd.DataFrame, px: float):
    lo = df["Low"]
    for i in range(len(lo) - 4, max(len(lo) - 120, 3), -1):
        if lo.iloc[i] == lo.iloc[i - 3:i + 4].min() and lo.iloc[i] < px * 0.95:
            return float(lo.iloc[i]), lo.index[i].date()
    return np.nan, None
def t_history(df: pd.DataFrame, entry_date, entry_px: float, step_n: float, stop_n: float):
    """收盤 ≥ 上次起算價 + step_n×ATR14 → 一次觸發。回傳 (歷史清單, 目前起算價)。"""
    c = df["Close"]; a14 = atr_series(df, 14)
    since = c[c.index >= pd.Timestamp(entry_date)]
    if not len(since):
        return [], entry_px
    last = entry_px; hist = []
    for dt, cl in since.items():
        a = a14.loc[dt]
        if pd.notna(a) and cl >= last + step_n * float(a):
            init_stop = float(cl) - stop_n * float(a)
            swept = bool((df["Low"].loc[df.index > dt] <= init_stop).any())
            hist.append({"訊號日": dt.date(), "收盤": round(float(cl), 2),
                         "當日ATR14": round(float(a), 2), "該筆初始停損": round(init_stop, 2),
                         "狀態": "已被掃" if swept else "持有中"})
            last = float(cl)
    return hist, last
def metrics_of(df: pd.DataFrame) -> dict:
    """單檔的均線／ATR／乖離指標（取代舊的 EMA 季線與 ATR 乖離計算器）。"""
    c = df["Close"]
    px = float(c.iloc[-1])
    a14 = atr_series(df, 14); a5 = atr_series(df, 5)
    atr = float(a14.iloc[-1]); ratio = float(a5.iloc[-1] / a14.iloc[-1]) if atr > 0 else np.nan
    e20 = c.ewm(span=20, adjust=False).mean(); e60 = c.ewm(span=60, adjust=False).mean()
    e260 = c.ewm(span=260, adjust=False).mean()
    dev60 = px / float(e60.iloc[-1]) - 1
    dev_pct = float(((c / e60).iloc[-500:] - 1 <= dev60).mean() * 100)
    return dict(px=px, atr=atr, ratio=ratio, e20=float(e20.iloc[-1]), e60=float(e60.iloc[-1]),
                e260=float(e260.iloc[-1]), dev20=px / float(e20.iloc[-1]) - 1, dev60=dev60,
                dev260=px / float(e260.iloc[-1]) - 1, dev_pct=dev_pct,
                r126=float(px / c.iloc[-127] - 1) if len(c) > 127 else np.nan,
                hi63=float(c.iloc[-64:-1].max()), asof=c.index[-1].date())
# ── Page ──────────────────────────────────────────────────────────
st.markdown("## 🎯 ATR 加碼計算器（A＋T 自動檢測）")
st.markdown(
    "<span style='color:#64748b;font-size:0.78rem'>"
    "輸入代碼與進場成本，每天自動檢測兩條加碼規則：<b>A 回季線</b>（距季線 −3%～+6% 且 ATR5 < ATR14）與 "
    "<b>T 機械</b>（收盤 ≥ 上次進場價 + 2×ATR14，只套怪物股與修復龍頭，且 ATR% ≤7%）。"
    "加碼單初始停損＝加碼價 − 3×ATR14，之後不追蹤、只跟基本倉結構停損上移。"
    "一律<b>收盤</b>判定、隔日開盤市價進。"
    "</span>", unsafe_allow_html=True)
st.markdown("---")
with st.sidebar:
    st.markdown("### 帳戶")
    acct = st.number_input("帳戶總資金 $", min_value=1000.0, value=93333.0, step=1000.0)
    r_usd = st.number_input("R（美元，帳戶 1%）", min_value=1.0, value=930.0, step=10.0)
    add_r = st.number_input("每筆加碼幾 R", min_value=0.25, value=0.5, step=0.25)
    st.markdown("### 名目上限（加碼比例）")
    cap_normal = st.number_input("一般部位上限（帳戶 %）", min_value=1.0, value=10.0, step=1.0) / 100
    cap_monster = st.number_input("怪物股上限（帳戶 %）", min_value=1.0, value=20.0, step=1.0) / 100
    st.markdown("### 加碼參數")
    step_n = st.select_slider("T 加碼間距（×ATR14）", options=[1.5, 2.0, 2.5, 3.0, 4.0], value=2.0)
    stop_n = st.select_slider("加碼單初始停損（×ATR14）", options=[2.0, 2.5, 3.0, 3.5, 4.0], value=3.0)
    a_lo = st.number_input("A 距季線下限 %", value=-3.0, step=0.5) / 100
    a_hi = st.number_input("A 距季線上限 %", value=6.0, step=0.5) / 100
    a_ratio = st.number_input("A 需 ATR5/ATR14 <", value=1.0, step=0.05)
    a_veto = st.number_input("急擴張否決（比值 ≥）", value=1.3, step=0.05)
    atr_max = st.number_input("T 波動上限：ATR% >", value=7.0, step=0.5,
                              help="回測：ATR% >7% 的加碼 EV −0.31R；設 7% 上限後年化 4.71%→5.30%、每筆加碼 EV +0.053R→+0.125R。基本倉不受此限。") / 100
st.markdown("#### 持倉")
st.markdown("<span style='color:#64748b;font-size:0.72rem'>"
            "只需要填<b>代碼</b>、<b>類型</b>、<b>進場日</b>、<b>進場價</b>（股數用來算加碼比例，可留 0）。"
            "「類型」決定 T 是否適用：怪物股（半年 ≥150%）與修復龍頭才開 T，一般動能股只看 A。"
            "「進場價」填 0 = 用進場日開盤價；「上次加碼價」填 0 = 由進場日自動推算。每次修改自動存檔。</span>",
            unsafe_allow_html=True)
if "add_pos" not in st.session_state:
    st.session_state.add_pos = load_add()
b1, b2, b3 = st.columns([2, 1, 1])
with b1:
    if st.button("↺ 從總曝險計算器帶入持倉", use_container_width=True,
                 help="讀取總曝險頁存的持倉，自動填代碼、進場價與股數；已填過的類型與進場日會保留。"):
        st.session_state.add_pos = sync_from_exposure(st.session_state.add_pos)
        save_add(st.session_state.add_pos)
        st.rerun()
with b2:
    if st.button("➕ 加一列", use_container_width=True):
        st.session_state.add_pos = pd.concat([st.session_state.add_pos, blank_row()], ignore_index=True)
        st.rerun()
with b3:
    if st.button("🗑 清空", use_container_width=True):
        st.session_state.add_pos = blank_row()
        save_add(st.session_state.add_pos)
        st.rerun()
edited = st.data_editor(
    st.session_state.add_pos, num_rows="dynamic", use_container_width=True, key="add_editor",
    column_config={
        "類型": st.column_config.SelectboxColumn(options=["怪物股", "修復龍頭", "一般"], width="small"),
        "進場日": st.column_config.TextColumn(help="YYYY-MM-DD"),
        "進場價": st.column_config.NumberColumn(format="%.2f", help="0 = 用進場日開盤價"),
        "目前股數": st.column_config.NumberColumn(format="%.0f", help="用來算目前名目與加碼後比例"),
        "上次加碼價": st.column_config.NumberColumn(format="%.2f", help="0 = 自動推算"),
    })
if not edited.equals(st.session_state.add_pos):
    save_add(edited)
st.session_state.add_pos = edited
tickers = tuple(sorted({c for c in edited["代碼"].map(clean_code) if c}))
top8 = list(top8_by_cap())
with st.spinner("下載價格資料…"):
    PX = fetch(tuple(dict.fromkeys(list(tickers) + top8 + TOP8_FALLBACK + ["SPY", "CME", "XLP", "XLY"])))
if "SPY" not in PX:
    st.error("抓不到大盤資料（可能被 Yahoo 限流），請稍後重新整理。")
    st.stop()
# ── 閘門 ──
n_above = n_prev = have = 0
for t in top8 + [x for x in TOP8_FALLBACK if x not in top8]:
    if have >= 8 or t not in PX:
        continue
    c = PX[t]["Close"]; e = c.ewm(span=60, adjust=False).mean()
    have += 1
    n_above += int(c.iloc[-1] > e.iloc[-1]); n_prev += int(c.iloc[-11] > e.iloc[-11])
light = "綠" if n_above >= 5 else ("紅" if n_above <= 3 else ("惡化黃" if n_prev > n_above else "修復黃"))
cme10 = float((PX["CME"]["Close"] / PX["SPY"]["Close"]).pct_change(10).iloc[-1] * 100) if "CME" in PX else np.nan
xl20 = float((PX["XLP"]["Close"] / PX["XLY"]["Close"]).pct_change(20).iloc[-1] * 100) if "XLP" in PX and "XLY" in PX else np.nan
sig1 = (cme10 >= 5) and (xl20 > 1)
allow = light in ("綠", "惡化黃") and not sig1
asof = PX["SPY"]["Close"].index[-1].date()
g1, g2, g3, g4 = st.columns(4)
g1.metric("領頭股燈號", f"{n_above}/{have}", light, delta_color="off",
          help="≥5 綠、4 黃（惡化黃開門／修復黃關門）、≤3 紅")
g2.metric("CME/SPY 10 日", f"{cme10:+.1f}%", "否決燈 · 需雙亮", delta_color="off")
g3.metric("XLP/XLY 20 日", f"{xl20:+.1f}%", "單燈不否決", delta_color="off")
g4.metric("加碼許可", "✅ 是" if allow else "⛔ 否",
          "壓力否決成立" if sig1 else ("燈號未開門" if not allow else f"截至 {asof}"), delta_color="off")
if not allow:
    st.warning("閘門關：今日不加碼。錯過的觸發不補，等閘門開之後的下一次收盤訊號。")
# ── 逐檔判定 ──
rows, details, missing = [], {}, []
for _, x in edited.iterrows():
    t = clean_code(x["代碼"])
    if not t:
        continue
    if t not in PX:
        missing.append(t); continue
    df = PX[t]; m = metrics_of(df)
    px, atr, ratio = m["px"], m["atr"], m["ratio"]
    kind = str(x["類型"]).strip()
    shares = float(pd.to_numeric(x["目前股數"], errors="coerce") or 0)
    # 進場資料
    try:
        ed_ts = pd.Timestamp(str(x["進場日"]).strip())
    except Exception:
        ed_ts = df.index[-1]
    ep_in = float(pd.to_numeric(x["進場價"], errors="coerce") or 0)
    opens = df["Open"][df.index >= ed_ts]
    ep = ep_in if ep_in > 0 else (float(opens.iloc[0]) if len(opens) else px)
    hist, auto_last = t_history(df, ed_ts, ep, step_n, stop_n)
    manual = float(pd.to_numeric(x["上次加碼價"], errors="coerce") or 0)
    ref = manual if manual > 0 else auto_last
    nxt = ref + step_n * atr
    t_today = bool(hist) and hist[-1]["訊號日"] == df.index[-1].date()
    t_ok_kind = kind in ("怪物股", "修復龍頭")
    vol_ok = (atr / px) <= atr_max
    t_fire = t_today and t_ok_kind and allow and vol_ok
    a_pos = a_lo <= m["dev60"] <= a_hi
    a_atr = ratio < a_ratio
    a_fire = a_pos and a_atr and allow
    # 加碼比例
    add_sh = int(add_r * r_usd / (stop_n * atr)) if atr > 0 else 0
    cap_pct = cap_monster if kind == "怪物股" else cap_normal
    now_notional = shares * px
    after_notional = now_notional + add_sh * px
    room_sh = int(max(acct * cap_pct - now_notional, 0) / px) if px > 0 else 0
    over = after_notional > acct * cap_pct
    if over:                      # 名目上限優先：加碼股數砍到上限為止
        add_sh = min(add_sh, room_sh)
        after_notional = now_notional + add_sh * px
    flag = "🔔 A" if a_fire else ""
    flag = (flag + "＋T" if flag else "🔔 T") if t_fire else flag
    rows.append(dict(訊號=flag, 代碼=t, 類型=kind, 價=round(px, 2),
                     半年=f"{m['r126']*100:+.0f}%" if not np.isnan(m["r126"]) else "—",
                     距季線=f"{m['dev60']*100:+.1f}%", 乖離百分位=f"{m['dev_pct']:.0f}",
                     **{"ATR5/14": f"{ratio:.2f}", "ATR%": f"{atr/px*100:.1f}%"},
                     A條件=("✅ 成立" if a_fire else ("位置✓ ATR✗" if a_pos and not a_atr else
                                                    ("位置✗" if not a_pos else "閘門✗"))),
                     T起算價=round(ref, 2), T下一觸發=round(nxt, 2), 距觸發=f"{(nxt/px-1)*100:+.1f}%",
                     T狀態=("🔔 今日觸發" if t_today else "未觸發") + ("" if t_ok_kind else "（類型不適用）")
                           + ("" if vol_ok else f"（ATR {atr/px*100:.1f}% >{atr_max*100:g}%，不加碼）"),
                     加碼股數=add_sh, 加碼名目=f"{add_sh*px:,.0f}",
                     加碼停損=round(px - stop_n * atr, 2),
                     目前名目=f"{now_notional:,.0f}", 目前占比=f"{now_notional/acct*100:.1f}%",
                     加碼後占比=f"{after_notional/acct*100:.1f}%",
                     上限=f"{cap_pct*100:.0f}%" + ("　⛔ 已滿" if room_sh == 0 else ""),
                     備註=("波動過大，T 不加" if not vol_ok else
                           ("近季線但 ATR 急擴張" if (a_pos and ratio >= a_veto) else ""))))
    details[t] = dict(hist=hist, ep=ep, ed=str(ed_ts.date()), m=m,
                      sw=swing_low(df, px), add_sh=add_sh, kind=kind)
if missing:
    st.warning(f"⚠️ 抓不到資料：{'、'.join(missing)}")
if not rows:
    st.info("請先在上表輸入持倉（代碼、類型、進場日、進場價）。")
    st.stop()
res = pd.DataFrame(rows).sort_values("訊號", ascending=False)
fired = [r for r in rows if r["訊號"]]
n_a = sum(1 for r in fired if "A" in r["訊號"]); n_t = sum(1 for r in fired if "T" in r["訊號"])
add_cost = sum(r["加碼股數"] * r["價"] for r in fired)
now_total = sum(float(str(r["目前名目"]).replace(",", "")) for r in rows)
s1, s2, s3, s4 = st.columns(4)
s1.metric("今日訊號", f"A {n_a}／T {n_t}", "收盤判定・隔日開盤進", delta_color="off")
s2.metric("目前名目曝險", f"${now_total:,.0f}", f"帳戶 {now_total/acct*100:.1f}%", delta_color="off")
s3.metric("今日加碼金額", f"${add_cost:,.0f}", f"帳戶 {add_cost/acct*100:.1f}%", delta_color="off")
s4.metric("加碼後名目曝險", f"${now_total+add_cost:,.0f}",
          f"帳戶 {(now_total+add_cost)/acct*100:.1f}%", delta_color="off")
if fired:
    st.success(f"🔔 今日成立：A {n_a} 檔／T {n_t} 檔　→　隔日開盤市價進，每筆 {add_r:g}R、"
               f"初始停損＝加碼價 − {stop_n:g}×ATR14（不追蹤，之後只跟結構停損上移）。"
               f"加碼後總名目 {(now_total+add_cost)/acct*100:.1f}%——請再到總曝險計算器確認本金曝險 ≤ 9%。")
else:
    st.markdown("<span style='color:#64748b'>今日沒有加碼訊號。等下一次收盤觸發，不追。</span>",
                unsafe_allow_html=True)
st.dataframe(res.set_index("代碼"), use_container_width=True, height=min(60 + 38 * len(res), 420))
st.markdown("<div style='color:#334155;font-size:0.68rem'>"
            "「加碼股數」＝加碼 R ÷（停損寬度 × ATR14），並受單股名目上限（一般 10%、怪物股 20%）壓縮；"
            "上限已滿時顯示 ⛔。財報前只砍加碼單。本表為系統規則之計算，不構成投資建議。</div>",
            unsafe_allow_html=True)
# ── 逐檔明細（含均線與乖離，取代舊的兩個計算器）──
st.markdown("#### 逐檔明細")
for t, dd in details.items():
    m = dd["m"]; sw, sw_dt = dd["sw"]
    with st.expander(f"{t}　{m['px']:.2f}　｜　進場 {dd['ed']} @ {dd['ep']:.2f}　｜　T 歷史觸發 {len(dd['hist'])} 次"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("月線 EMA20", f"{m['e20']:.2f}", f"{m['dev20']*100:+.1f}%", delta_color="off")
        c2.metric("季線 EMA60", f"{m['e60']:.2f}", f"{m['dev60']*100:+.1f}%（百分位 {m['dev_pct']:.0f}）",
                  delta_color="off")
        c3.metric("年線 EMA260", f"{m['e260']:.2f}", f"{m['dev260']*100:+.1f}%", delta_color="off")
        c4.metric("ATR14", f"{m['atr']:.2f}", f"{m['atr']/m['px']*100:.1f}% of price　ATR5/14 {m['ratio']:.2f}",
                  delta_color="off")
        d1, d2, d3 = st.columns(3)
        d1.metric(f"若今日加碼：初始停損（−{stop_n:g}ATR）", f"{m['px'] - stop_n*m['atr']:.2f}",
                  f"{-stop_n*m['atr']/m['px']*100:.1f}%", delta_color="off")
        d2.metric("最近前波支撐（基本倉停損參考）", f"{sw:.2f}" if not np.isnan(sw) else "—",
                  f"{(sw/m['px']-1)*100:+.1f}%（{sw_dt}）" if not np.isnan(sw) else "近 120 日無合格擺盪低點",
                  delta_color="off")
        d3.metric("距 63 日高", f"{(m['px']/m['hi63']-1)*100:+.1f}%", f"63 日高 {m['hi63']:.2f}", delta_color="off")
        if dd["hist"]:
            st.dataframe(pd.DataFrame(dd["hist"]).set_index("訊號日"), use_container_width=True)
            st.markdown("<span style='color:#64748b;font-size:0.72rem'>"
                        "「已被掃」＝該筆加碼單的初始停損之後曾被觸及（基本倉不受影響）；"
                        "被掃不影響起算價，下一筆仍需站上「上次加碼價 + 間距」。</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#64748b;font-size:0.78rem'>進場後尚無 T 觸發。</span>",
                        unsafe_allow_html=True)
# ── 單檔快查（不必是持倉）──
st.markdown("---")
st.markdown("#### 🔍 單檔快查（均線／乖離／ATR／加碼股數）")
q1, q2, q3 = st.columns([1, 1, 2])
with q1:
    qt = st.text_input("代碼", value="").strip().upper()
with q2:
    q_kind = st.selectbox("類型（決定名目上限）", ["一般", "怪物股", "修復龍頭"])
if qt:
    qpx = fetch((qt,))
    if qt not in qpx:
        st.warning(f"抓不到 {qt} 的資料。")
    else:
        m = metrics_of(qpx[qt]); sw, sw_dt = swing_low(qpx[qt], m["px"])
        sh = int(add_r * r_usd / (stop_n * m["atr"])) if m["atr"] > 0 else 0
        base_sh = int(r_usd / (m["px"] - sw)) if (not np.isnan(sw) and m["px"] > sw) else 0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("收盤", f"{m['px']:.2f}", f"半年 {m['r126']*100:+.0f}%" if not np.isnan(m["r126"]) else "—",
                  delta_color="off")
        k2.metric("距季線", f"{m['dev60']*100:+.1f}%", f"季線 {m['e60']:.2f}（百分位 {m['dev_pct']:.0f}）",
                  delta_color="off")
        k3.metric("ATR14", f"{m['atr']:.2f}", f"{m['atr']/m['px']*100:.1f}%　ATR5/14 {m['ratio']:.2f}",
                  delta_color="off")
        k4.metric(f"加碼停損（−{stop_n:g}ATR）", f"{m['px'] - stop_n*m['atr']:.2f}",
                  f"加碼 {sh} 股＝${sh*m['px']:,.0f}", delta_color="off")
        cap_pct = cap_monster if q_kind == "怪物股" else cap_normal
        a_pos = a_lo <= m["dev60"] <= a_hi; a_atr = m["ratio"] < a_ratio
        st.markdown(
            f"<div style='font-size:0.85rem;line-height:1.9'>"
            f"月線 {m['e20']:.2f}（{m['dev20']*100:+.1f}%）｜季線 {m['e60']:.2f}（{m['dev60']*100:+.1f}%）｜"
            f"年線 {m['e260']:.2f}（{m['dev260']*100:+.1f}%）｜多頭排列 "
            f"{'✅' if m['px'] > m['e20'] > m['e60'] > m['e260'] else '✗'}<br>"
            f"A 條件：位置 {'✓' if a_pos else '✗'}（需 {a_lo*100:+.0f}%～{a_hi*100:+.0f}%）　"
            f"ATR 收縮 {'✓' if a_atr else '✗'}（需 <{a_ratio:g}）　閘門 {'✓' if allow else '✗'}　→ "
            f"<b>{'成立' if (a_pos and a_atr and allow) else '不成立'}</b>"
            f"{'（近季線但 ATR 急擴張，不加）' if (a_pos and m['ratio'] >= a_veto) else ''}<br>"
            f"T 條件：需要進場價才能算觸發鏈——若持有請加到上表。單股名目上限 {cap_pct*100:.0f}% = "
            f"${acct*cap_pct:,.0f}（約 {int(acct*cap_pct/m['px'])} 股）<br>"
            f"基本倉參考：前波支撐 {sw:.2f}（{sw_dt}）→ 1R 股數 {base_sh} 股、名目 ${base_sh*m['px']:,.0f}"
            f"</div>" if not np.isnan(sw) else
            f"<div style='font-size:0.85rem'>近 120 日無合格擺盪低點，基本倉停損請看更前面的結構。</div>",
            unsafe_allow_html=True)
csv = st.session_state.add_pos.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ 下載監控清單 CSV（備份用）", csv, "加碼監控清單.csv", "text/csv")
with st.expander("📖 規則與依據"):
    st.markdown(f"""
**A 回季線加碼**：距季線 {a_lo*100:+.0f}%～{a_hi*100:+.0f}% 且 ATR5/ATR14 < {a_ratio:g} 且閘門開 → 加 {add_r:g}R。
比值 ≥ {a_veto:g} 時即使位置對也不加。十檔怪物上 27 筆只被掃 6 筆；但怪物股很少回季線（在季線附近的日子 2～8%），
修復股常觸發（92%）。

**T 機械加碼**：收盤 ≥ 上次進場價 + {step_n:g}×ATR14 → 加 {add_r:g}R。**只套怪物股（半年 ≥150%）與修復龍頭，且 ATR% ≤ {atr_max*100:g}%**
（波動上限，2026-09-14 新增：ATR% 7～10% 的加碼 EV −0.31R、>10% 為 −0.50R；設限後年化 4.71%→5.30%、每筆加碼 EV 翻 2.4 倍。基本倉不受此限——高波動股的基本倉 EV 反而最高）。
次數不設上限，用名目上限控制（一般 {cap_normal*100:.0f}%、怪物股 {cap_monster*100:.0f}%）；財報前只砍加碼單。
起算價＝上一次實際加碼的收盤價；錯過的不補、被掃的不下修。

**加碼單停損**：初始停損＝加碼價 − {stop_n:g}×ATR14，之後不追蹤、不因漲多收緊；只在基本倉結構停損上移超過它時跟著走。
股數＝{add_r:g}R ÷ ({stop_n:g}×ATR14)。依據：十檔怪物三種停損——高點追蹤 +32R、階梯棘輪 +36R、**固定不調 +197R**；
母體每筆加碼 +0.040R／+0.066R／**+0.077R**。

**判定時點**：一律收盤判定、隔日開盤市價進，盤中摸到不算（回測即以此執行）。唯一例外是基本倉的結構停損：盤中跌破就走。

**ATR14**：TR = max(高−低, |高−昨收|, |低−昨收|)，取 14 日**簡單平均**（非 Wilder），與回測一致。

**加碼比例**：本頁只管單股名目上限；總曝險（本金曝險 ≤ 帳戶 9%、檔數 ≤9、題材 ≤50～60%、保本地板）請到「總曝險計算器」確認。
""")
