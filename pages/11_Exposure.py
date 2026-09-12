import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
# ═══════════════════════════════════════════════════════════════════
# 🌡️ 總曝險計算器（順勢交易系統 規則四：曝險與部位上限）
#   本金曝險＝所有部位同時打到停損、會傷到「本金」的總額（閘門依據，≤ 帳戶 9%）
#   利潤曝險＝停損在進場價之上、回吐的只是帳面獲利（監控，不設限）
#   另檢查：檔數 ≤9、單股 ≤10%、題材 ≤50~60%、保本地板
# ═══════════════════════════════════════════════════════════════════
COLS = ["代碼", "題材", "股數", "進場價", "停損價"]        # 現價由 yfinance 自動帶入，不手動輸入
STORE = Path(__file__).with_name(".positions.json")      # 自動存檔（與本頁同目錄）
def blank_row() -> pd.DataFrame:
    return pd.DataFrame([["", "", 0, 0.0, 0.0]], columns=COLS)
def load_positions() -> pd.DataFrame:
    try:
        if STORE.exists():
            df = pd.read_json(STORE)
            for c in COLS:
                if c not in df.columns:
                    df[c] = "" if c in ("代碼", "題材") else 0
            df = df[COLS]
            return pd.concat([df, blank_row()], ignore_index=True) if len(df) else blank_row()
    except Exception:
        pass
    return blank_row()
def save_positions(df: pd.DataFrame) -> None:
    try:
        keep = df[df["代碼"].map(clean_code) != ""]
        STORE.write_text(keep.to_json(orient="records", force_ascii=False), encoding="utf-8")
    except Exception:
        pass          # 唯讀環境就只靠 session（重新整理會回到上次存檔）
@st.cache_data(ttl=900, show_spinner=False)
def fetch_last(tickers: tuple) -> dict:
    """最新收盤價（15 分鐘快取）。"""
    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period="5d", interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        out = {}
        for t in tickers:
            try:
                s = raw["Close"][t] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
                s = s.dropna()
                if len(s):
                    out[t] = float(s.iloc[-1])
            except Exception:
                pass
        return out
    except Exception:
        return {}
def clean_code(v) -> str:
    t = str(v).strip().upper()
    return "" if t in ("", "NAN", "NONE", "<NA>") else t
def compute(df: pd.DataFrame, px_map: dict) -> pd.DataFrame:
    d = df.copy()
    d["代碼"] = d["代碼"].map(clean_code)
    d = d[d["代碼"] != ""]
    if d.empty:
        return d
    for c in ["股數", "進場價", "停損價"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["現價"] = [px_map.get(t, np.nan) for t in d["代碼"]]
    d["現價"] = pd.to_numeric(d["現價"], errors="coerce").fillna(d["進場價"])   # 抓不到價就用進場價，避免整頁掛掉
    d["市值"] = d["股數"] * d["現價"]
    d["單批曝險"] = (d["股數"] * (d["現價"] - d["停損價"])).clip(lower=0)   # 停損已在現價之上＝0
    d["停損處損益"] = d["股數"] * (d["停損價"] - d["進場價"])
    # 本金曝險＝單批曝險中「低於進場價」的那一段（已經發生的帳面虧損不重複計入）
    d["本金曝險"] = (d["股數"] * (np.minimum(d["現價"], d["進場價"]) - d["停損價"])).clip(lower=0)
    d["利潤曝險"] = (d["單批曝險"] - d["本金曝險"]).clip(lower=0)
    d["距停損%"] = np.where(d["現價"] > 0, (d["停損價"] / d["現價"] - 1) * 100, 0)
    d["未實現"] = d["股數"] * (d["現價"] - d["進場價"])
    return d
def bar(label, value, cap, unit="$", help_txt=None):
    """一行進度條：value / cap。"""
    pct = value / cap if cap else 0
    color = "#4ade80" if pct <= 0.8 else ("#fbbf24" if pct <= 1.0 else "#f87171")
    txt = (f"{value:,.0f} / {cap:,.0f}" if unit == "$" else f"{value:.1f}% / {cap:.1f}%")
    st.markdown(
        f"<div style='margin:2px 0 8px'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.78rem'>"
        f"<span style='color:#94a3b8'>{label}</span>"
        f"<span style='color:{color};font-weight:600'>{txt}　({pct*100:.0f}%)</span></div>"
        f"<div style='height:7px;background:#1e293b;border-radius:4px;overflow:hidden;margin-top:3px'>"
        f"<div style='height:100%;width:{min(pct,1)*100:.1f}%;background:{color}'></div></div></div>",
        unsafe_allow_html=True)
    if help_txt:
        st.markdown(f"<div style='color:#475569;font-size:0.68rem;margin:-6px 0 8px'>{help_txt}</div>",
                    unsafe_allow_html=True)
# ── Page ──────────────────────────────────────────────────────────
st.markdown("## 🌡️ 總曝險計算器")
st.markdown(
    "<span style='color:#64748b;font-size:0.78rem'>"
    "本金曝險＝所有停損同時被打到、會傷到本金的總額（開新倉的閘門，≤ 帳戶 9%）　｜　"
    "利潤曝險＝停損已在進場價之上，回吐的只是帳面獲利（監控，不設限）　｜　"
    "另檢查：檔數 ≤9、單股 ≤10%、題材 ≤50~60%、保本地板"
    "</span>", unsafe_allow_html=True)
st.markdown("---")
with st.sidebar:
    st.markdown("### 帳戶參數")
    acct = st.number_input("帳戶總資金 $", min_value=1000.0, value=93333.0, step=1000.0)
    r_pct = st.number_input("單筆風險 R（帳戶 %）", min_value=0.1, value=1.0, step=0.1) / 100
    max_pos = st.number_input("最大同時持倉數", min_value=1, value=9, step=1)
    heat_cap_pct = st.number_input("本金曝險上限（帳戶 %）", min_value=1.0, value=9.0, step=0.5) / 100
    name_cap_pct = st.number_input("單一標的上限（帳戶 %）", min_value=1.0, value=10.0, step=1.0) / 100
    theme_cap_pct = st.number_input("單一題材上限（帳戶 %）", min_value=10.0, value=60.0, step=5.0) / 100
    st.markdown("### 保本閘門")
    start_cap = st.number_input("起始本金 $", min_value=0.0, value=61900.0, step=1000.0)
    sleeve_cash = st.number_input("袖內現金 $", min_value=0.0, value=0.0, step=500.0)
    tol_pct = st.number_input("可容忍本金損失 %", min_value=0.0, value=0.0, step=1.0) / 100
r_usd = acct * r_pct
st.markdown(f"<span style='color:#94a3b8;font-size:0.8rem'>R = 帳戶 {r_pct*100:.1f}% = "
            f"<b style='color:#e2e8f0'>${r_usd:,.0f}</b>　｜　本金曝險上限 = ${acct*heat_cap_pct:,.0f}"
            f"　｜　單股上限 = ${acct*name_cap_pct:,.0f}　｜　題材上限 = ${acct*theme_cap_pct:,.0f}</span>",
            unsafe_allow_html=True)
# ── 持倉輸入（全部在網頁上手動輸入；自動存檔，重新整理不會不見）──
if "pos" not in st.session_state:
    st.session_state.pos = load_positions()
hdr, btn1, btn2 = st.columns([4, 1, 1])
with hdr:
    st.markdown("#### 持倉明細")
    st.markdown("<span style='color:#64748b;font-size:0.72rem'>"
                "直接在表格輸入，最後一列是空白列——填進去就會自動長出新的一列。"
                "「現價」與「距停損%」自動帶入最新收盤（15 分鐘快取），不用也不能手動填。加碼單另開一列（同一個代碼可以有多列），填該筆自己的進場價與停損價。"
                "每次修改會自動存檔。</span>", unsafe_allow_html=True)
with btn1:
    if st.button("➕ 加一列", use_container_width=True):
        st.session_state.pos = pd.concat([st.session_state.pos, blank_row()], ignore_index=True)
        st.rerun()
with btn2:
    if st.button("🗑 全部清空", use_container_width=True):
        st.session_state.pos = blank_row()
        save_positions(st.session_state.pos)
        st.rerun()
# 先用目前存的代碼抓價，帶進表格當唯讀欄位
cur_tickers = tuple(sorted({c for c in st.session_state.pos["代碼"].map(clean_code) if c}))
with st.spinner("抓取最新收盤價…"):
    px_map = fetch_last(cur_tickers)
view = st.session_state.pos.copy()
view["現價"] = [px_map.get(clean_code(t), np.nan) for t in view["代碼"]]
view["距停損%"] = [(sp / pxv - 1) * 100 if (pxv and pxv == pxv and sp) else np.nan
                   for pxv, sp in zip(view["現價"], pd.to_numeric(view["停損價"], errors="coerce"))]
edited_view = st.data_editor(
    view, num_rows="dynamic", use_container_width=True, key="editor",
    disabled=["現價", "距停損%"],
    column_config={
        "股數": st.column_config.NumberColumn(format="%.0f"),
        "進場價": st.column_config.NumberColumn(format="%.2f"),
        "停損價": st.column_config.NumberColumn(format="%.2f"),
        "現價": st.column_config.NumberColumn(format="%.2f", help="自動帶入最新收盤（15 分鐘快取），不可編輯"),
        "距停損%": st.column_config.NumberColumn(format="%+.1f%%", help="停損價 ÷ 現價 − 1"),
    })
edited = edited_view[COLS]
if not edited.equals(st.session_state.pos):      # 有改動才寫檔，並重跑以帶出新代碼的現價
    save_positions(edited)
    st.session_state.pos = edited
    new_t = {c for c in edited["代碼"].map(clean_code) if c}
    if new_t - set(cur_tickers):
        st.rerun()
st.session_state.pos = edited
tickers = tuple(sorted({c for c in edited["代碼"].map(clean_code) if c}))
if set(tickers) - set(px_map):
    px_map.update(fetch_last(tuple(sorted(set(tickers) - set(px_map)))))
d = compute(edited, px_map)
if d.empty:
    st.info("請先在上表輸入持倉。")
    st.stop()
missing = [t for t in tickers if t not in px_map]
if missing:
    st.warning(f"⚠️ 抓不到現價（已用進場價替代，請確認代碼是否正確）：{'、'.join(missing)}")
# ── 彙總 ──
n_names = d["代碼"].nunique()
principal_heat = d["本金曝險"].sum()
profit_heat = d["利潤曝險"].sum()
total_heat = d["單批曝險"].sum()
notional = d["市值"].sum()
unreal = d["未實現"].sum()
sleeve_eq = sleeve_cash + notional
worst_eq = sleeve_eq - total_heat
floor = start_cap * (1 - tol_pct)
gap = worst_eq - floor
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("持倉檔數", f"{n_names} / {max_pos}", f"{max_pos - n_names} 個名額", delta_color="off")
c2.metric("名目曝險", f"${notional:,.0f}", f"帳戶 {notional/acct*100:.0f}%", delta_color="off")
c3.metric("本金曝險", f"${principal_heat:,.0f}", f"帳戶 {principal_heat/acct*100:.2f}%（上限 {heat_cap_pct*100:.0f}%）",
          delta_color="off")
c4.metric("利潤曝險", f"${profit_heat:,.0f}", "監控不設限", delta_color="off")
c5.metric("未實現損益", f"${unreal:,.0f}", f"{unreal/max(notional-unreal,1)*100:+.1f}%", delta_color="off")
st.markdown("#### 額度使用")
bar("本金曝險（開新倉閘門）", principal_heat, acct * heat_cap_pct,
    help_txt=f"剩餘額度 ${max(acct*heat_cap_pct - principal_heat, 0):,.0f}"
             f"　≈ 還能開 {int(max(acct*heat_cap_pct - principal_heat, 0) // r_usd)} 筆 1R 新倉")
bar("持倉檔數", n_names, max_pos, unit="n")
bar("名目曝險 / 帳戶", notional / acct * 100, 100.0, unit="%")
# 單股
by_name = d.groupby("代碼").agg(市值=("市值", "sum"), 本金曝險=("本金曝險", "sum"),
                                單批曝險=("單批曝險", "sum"), 筆數=("代碼", "size")).sort_values("市值", ascending=False)
by_name["佔帳戶%"] = by_name["市值"] / acct * 100
over_name = by_name[by_name["佔帳戶%"] > name_cap_pct * 100]
# 題材
by_theme = d.groupby("題材").agg(市值=("市值", "sum"), 檔數=("代碼", "nunique")).sort_values("市值", ascending=False)
by_theme["佔帳戶%"] = by_theme["市值"] / acct * 100
over_theme = by_theme[by_theme["佔帳戶%"] > theme_cap_pct * 100]
cA, cB = st.columns(2)
with cA:
    st.markdown("##### 單一標的曝險")
    st.dataframe(by_name.round(2), use_container_width=True, height=min(60 + 35 * len(by_name), 300))
    if len(over_name):
        st.error(f"⛔ 超過單股上限 {name_cap_pct*100:.0f}%：{'、'.join(over_name.index)}")
with cB:
    st.markdown("##### 題材曝險")
    st.dataframe(by_theme.round(2), use_container_width=True, height=min(60 + 35 * len(by_theme), 300))
    if len(over_theme):
        st.error(f"⛔ 超過題材上限 {theme_cap_pct*100:.0f}%：{'、'.join(over_theme.index)}")
# ── 保本閘門 ──
st.markdown("#### 保本閘門")
g1, g2, g3, g4 = st.columns(4)
g1.metric("袖權益（現金＋市值）", f"${sleeve_eq:,.0f}", f"起始本金 ${start_cap:,.0f}", delta_color="off")
g2.metric("獲利緩衝", f"${sleeve_eq - start_cap:,.0f}", "袖權益 − 起始本金", delta_color="off")
g3.metric("最壞情況權益（全停損）", f"${worst_eq:,.0f}", "袖權益 − 總曝險", delta_color="off")
g4.metric("保本缺口", f"${gap:,.0f}", "✅ 守住地板" if gap >= 0 else "✗ 破地板，需降曝險", delta_color="off")
if gap < 0:
    st.error(f"⛔ 全部停損同時被打到會跌破保本地板 ${floor:,.0f}，缺口 ${-gap:,.0f}。"
             f"降低曝險的方法只有：出掉部位、或等停損自然上移——**不可為了降低曝險把停損移到結構之外**。")
# ── 明細 ──
st.markdown("#### 逐筆明細")
show = d[["代碼", "題材", "股數", "進場價", "停損價", "現價", "市值",
          "距停損%", "單批曝險", "本金曝險", "利潤曝險", "停損處損益", "未實現"]].copy()
show = show.sort_values("市值", ascending=False)
st.dataframe(show.style.format({"進場價": "{:.2f}", "停損價": "{:.2f}", "現價": "{:.2f}",
                                "市值": "{:,.0f}", "距停損%": "{:+.1f}", "單批曝險": "{:,.0f}",
                                "本金曝險": "{:,.0f}", "利潤曝險": "{:,.0f}",
                                "停損處損益": "{:+,.0f}", "未實現": "{:+,.0f}"}),
             use_container_width=True, height=min(60 + 35 * len(show), 500))
csv = st.session_state.pos.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ 下載持倉 CSV（備份用，非必要）", csv, "持倉.csv", "text/csv")
# ── 新倉試算 ──
st.markdown("---")
st.markdown("#### 🧪 新倉試算（開之前先過閘門）")
s1, s2, s3, s4 = st.columns(4)
with s1:
    n_t = st.text_input("代碼", value="").strip().upper()
with s2:
    n_entry = st.number_input("預計進場價", min_value=0.0, value=0.0, step=0.01, format="%.2f")
with s3:
    n_stop = st.number_input("停損價（前波支撐）", min_value=0.0, value=0.0, step=0.01, format="%.2f")
with s4:
    n_r = st.number_input("幾 R", min_value=0.5, value=1.0, step=0.5)
if n_entry > 0 and n_stop > 0 and n_stop < n_entry:
    risk_per_sh = n_entry - n_stop
    sh = int(n_r * r_usd / risk_per_sh)
    add_heat = sh * risk_per_sh
    new_heat = principal_heat + add_heat
    new_notional = notional + sh * n_entry
    cap_ok = new_heat <= acct * heat_cap_pct
    cnt_ok = (n_t in set(d["代碼"])) or (n_names + 1 <= max_pos)
    name_now = float(by_name["市值"].get(n_t, 0)) + sh * n_entry
    name_ok = name_now <= acct * name_cap_pct
    floor_ok = (sleeve_eq + 0) - (total_heat + add_heat) >= floor
    st.markdown(
        f"<div style='font-size:0.85rem;line-height:1.8'>"
        f"停損距離 <b>{risk_per_sh/n_entry*100:.1f}%</b>　→　股數 <b>{sh}</b> 股"
        f"（{n_r:g}R = ${n_r*r_usd:,.0f} ÷ {risk_per_sh:.2f}）　名目 <b>${sh*n_entry:,.0f}</b>"
        f"（帳戶 {sh*n_entry/acct*100:.1f}%）<br>"
        f"{'✅' if cap_ok else '⛔'} 本金曝險：${principal_heat:,.0f} → <b>${new_heat:,.0f}</b>"
        f"（{new_heat/acct*100:.2f}%，上限 {heat_cap_pct*100:.0f}%）<br>"
        f"{'✅' if cnt_ok else '⛔'} 檔數：{n_names} → {n_names + (0 if n_t in set(d['代碼']) else 1)} / {max_pos}<br>"
        f"{'✅' if name_ok else '⛔'} 單股曝險：${name_now:,.0f}（{name_now/acct*100:.1f}%，上限 {name_cap_pct*100:.0f}%）<br>"
        f"{'✅' if floor_ok else '⛔'} 保本地板：最壞情況權益 ${sleeve_eq - total_heat - add_heat:,.0f} vs 地板 ${floor:,.0f}"
        f"</div>", unsafe_allow_html=True)
    if cap_ok and cnt_ok and name_ok and floor_ok:
        st.success("四項額度都過。仍需確認：燈號綠燈、壓力否決未成立、該股當日確實觸發（創 63 日新高／修復站回）。")
    else:
        st.error("有額度未過 → 不開新倉。規則四：開新倉前先算曝險，超了就不開。")
elif n_entry > 0 and n_stop >= n_entry:
    st.warning("停損價需低於進場價。")
with st.expander("📖 定義與規則"):
    st.markdown(f"""
- **單批曝險** = 股數 ×（現價 − 停損價），停損已在現價之上時為 0。代表「現在全部停損被打到，帳面會少掉多少」。
- **本金曝險** = 停損價低於進場價的那部分損失＝股數 ×（進場價 − 停損價），只算還沒鎖住獲利的部位。**這是開新倉的唯一閘門，上限帳戶 {heat_cap_pct*100:.0f}%。**
- **利潤曝險** = 單批曝險 − 本金曝險。停損已上移到成本之上，被打到只是回吐獲利，不傷本金，**監控但不設限**。
- **保本地板** = 起始本金 × (1 − 可容忍損失%)；最壞情況權益 = 袖權益 − 總曝險。破地板時只能出部位或等停損自然上移，**不可為了降低曝險把停損移到結構之外**（禁止事項）。
- 曝險是針對整桶金計算，不分策略；同一檔股票（含加碼單）合計 ≤ 帳戶 {name_cap_pct*100:.0f}%；同一題材 ≤ {theme_cap_pct*100:.0f}%。
- 加碼單請獨立列出一列，初始停損＝加碼價 − 3×ATR14，之後只跟基本倉結構停損上移。
- 持倉全部在本頁手動輸入，每次修改自動存檔於伺服器（重新整理、關掉再開都還在）。若儀表板重新部署或容器重啟，檔案可能被清掉，需要長期保存可用下方 CSV 備份。
""")
