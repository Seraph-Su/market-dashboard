import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

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
# 🌡️ 總曝險計算器（順勢交易系統 規則四）
#   總曝險 = Σ 股數 ×（停損價 − 進場價）÷ 帳戶總資金
#   ＝「所有停損同時被打到時，帳戶會賺或賠多少」
#   例：用 10% 資金買、停損 −10% → 曝險 −1%；停損移到成本 → 0%；停損 +10% → +1%
#   只涵蓋主動選股部位（每月換股輪動不設停損，不列入）
# ═══════════════════════════════════════════════════════════════════
COLS = ["代碼", "題材", "股數", "進場價", "停損價"]        # 現價由 yfinance 自動帶入
STORE = Path(__file__).with_name(".positions.json")      # 自動存檔（與本頁同目錄）
def clean_code(v) -> str:
    t = str(v).strip().upper()
    return "" if t in ("", "NAN", "NONE", "<NA>") else t
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
        pass          # 唯讀環境就只靠 session
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
def compute(df: pd.DataFrame, px_map: dict, acct: float) -> pd.DataFrame:
    d = df.copy()
    d["代碼"] = d["代碼"].map(clean_code)
    d = d[d["代碼"] != ""]
    if d.empty:
        return d
    for c in ["股數", "進場價", "停損價"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["現價"] = [px_map.get(t, np.nan) for t in d["代碼"]]
    d["現價"] = pd.to_numeric(d["現價"], errors="coerce").fillna(d["進場價"])
    d["成本"] = d["股數"] * d["進場價"]
    d["市值"] = d["股數"] * d["現價"]
    d["曝險$"] = d["股數"] * (d["停損價"] - d["進場價"])      # 打到停損時的損益（可正可負）
    d["曝險%"] = d["曝險$"] / acct * 100
    d["距停損%"] = np.where(d["現價"] > 0, (d["停損價"] / d["現價"] - 1) * 100, 0)
    d["停損 vs 成本%"] = np.where(d["進場價"] > 0, (d["停損價"] / d["進場價"] - 1) * 100, 0)
    d["未實現"] = d["股數"] * (d["現價"] - d["進場價"])
    return d
def bar(label, value, cap, unit="$", help_txt=None):
    pct = value / cap if cap else 0
    color = "#4ade80" if pct <= 0.8 else ("#fbbf24" if pct <= 1.0 else "#f87171")
    txt = (f"{value:,.0f} / {cap:,.0f}" if unit == "$" else f"{value:.1f} / {cap:.1f}")
    st.markdown(
        f"<div style='margin:2px 0 10px'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.9rem'>"
        f"<span style='color:#94a3b8'>{label}</span>"
        f"<span style='color:{color};font-weight:600'>{txt}　({pct*100:.0f}%)</span></div>"
        f"<div style='height:7px;background:#1e293b;border-radius:4px;overflow:hidden;margin-top:3px'>"
        f"<div style='height:100%;width:{min(max(pct,0),1)*100:.1f}%;background:{color}'></div></div>"
        + (f"<div style='color:#475569;font-size:0.9rem;margin-top:3px'>{help_txt}</div>" if help_txt else "")
        + "</div>", unsafe_allow_html=True)
# ── Page ──────────────────────────────────────────────────────────
st.markdown("## 🌡️ 總曝險計算器")
st.markdown(
    "<span style='color:#64748b;font-size:0.9rem'>"
    "<b>總曝險 = 所有停損同時被打到時，帳戶會賺或賠多少</b>　＝ Σ 股數 ×（停損價 − 進場價）÷ 帳戶總資金<br>"
    "例：用 10% 資金買、停損設在 −10% → 曝險 −1%；停損移到成本價 → 0%；停損移到 +10% → +1%。"
    "只涵蓋主動選股部位。"
    "</span>", unsafe_allow_html=True)
st.markdown("---")
with st.sidebar:
    st.markdown("### 帳戶參數")
    acct = st.number_input("帳戶總資金 $", min_value=1000.0, value=93333.0, step=1000.0,
                           help="R 與各項上限皆以帳戶總資金為分母，與規則書一致。")
    r_pct = st.number_input("單筆風險 R（帳戶 %）", min_value=0.1, value=1.0, step=0.1) / 100
    max_pos = st.number_input("最大同時持倉數", min_value=1, value=9, step=1)
    heat_cap_pct = st.number_input("總曝險上限（帳戶 %）", min_value=1.0, value=9.0, step=0.5) / 100
    name_cap_pct = st.number_input("單一標的上限（帳戶 %）", min_value=1.0, value=10.0, step=1.0) / 100
    theme_cap_pct = st.number_input("單一題材上限（帳戶 %）", min_value=10.0, value=60.0, step=5.0) / 100
r_usd = acct * r_pct
st.markdown(f"<span style='color:#94a3b8;font-size:0.9rem'>R = 帳戶 {r_pct*100:.1f}% = "
            f"<b style='color:#e2e8f0'>${r_usd:,.0f}</b>　｜　總曝險上限 = 帳戶 −{heat_cap_pct*100:.0f}% "
            f"（−${acct*heat_cap_pct:,.0f}）　｜　單股上限 = ${acct*name_cap_pct:,.0f}　｜　"
            f"題材上限 = ${acct*theme_cap_pct:,.0f}</span>", unsafe_allow_html=True)
# ── 持倉輸入 ──
if "pos" not in st.session_state:
    st.session_state.pos = load_positions()
hdr, btn1, btn2 = st.columns([4, 1, 1])
with hdr:
    st.markdown("#### 持倉明細")
    st.markdown("<span style='color:#64748b;font-size:0.9rem'>"
                "直接在表格輸入，最後一列是空白列——填進去就會長出新的一列。"
                "「現價」「距停損%」自動帶入（15 分鐘快取）。加碼單另開一列（同代碼可多列）。每次修改自動存檔。"
                "</span>", unsafe_allow_html=True)
with btn1:
    if st.button("➕ 加一列", use_container_width=True):
        st.session_state.pos = pd.concat([st.session_state.pos, blank_row()], ignore_index=True)
        st.rerun()
with btn2:
    if st.button("🗑 全部清空", use_container_width=True):
        st.session_state.pos = blank_row()
        save_positions(st.session_state.pos)
        st.rerun()
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
        "現價": st.column_config.NumberColumn(format="%.2f", help="自動帶入最新收盤，不可編輯"),
        "距停損%": st.column_config.NumberColumn(format="%+.1f%%", help="停損價 ÷ 現價 − 1"),
    })
edited = edited_view[COLS]
if not edited.equals(st.session_state.pos):
    save_positions(edited)
    st.session_state.pos = edited
    if {c for c in edited["代碼"].map(clean_code) if c} - set(cur_tickers):
        st.rerun()
st.session_state.pos = edited
tickers = tuple(sorted({c for c in edited["代碼"].map(clean_code) if c}))
if set(tickers) - set(px_map):
    px_map.update(fetch_last(tuple(sorted(set(tickers) - set(px_map)))))
d = compute(edited, px_map, acct)
if d.empty:
    st.info("請先在上表輸入持倉。")
    st.stop()
missing = [t for t in tickers if t not in px_map]
if missing:
    st.warning(f"⚠️ 抓不到現價（已用進場價替代，請確認代碼）：{'、'.join(missing)}")
# ── 總曝險（主角）──
exposure = d["曝險$"].sum()
exp_pct = exposure / acct * 100
notional = d["市值"].sum()
cost = d["成本"].sum()
unreal = d["未實現"].sum()
n_names = d["代碼"].nunique()
neg = exposure < 0
st.markdown(
    f"<div style='margin:14px 0 6px;padding:18px 22px;border-radius:12px;"
    f"background:{'#450a0a' if neg else '#052e16'};border:1px solid {'#7f1d1d' if neg else '#166534'}'>"
    f"<div style='font-size:0.9rem;color:#94a3b8;margin-bottom:6px'>總曝險　"
    f"<span style='font-size:0.9rem'>（所有停損同時打到 → 帳戶{'賠' if neg else '賺'}這麼多）</span></div>"
    f"<div style='font-size:2.73rem;font-weight:800;line-height:1;"
    f"color:{'#f87171' if neg else '#4ade80'}'>{exp_pct:+.2f}%</div>"
    f"<div style='font-size:1.1rem;color:#cbd5e1;margin-top:6px'>{'−' if neg else '+'}${abs(exposure):,.0f}"
    f"　<span style='color:#64748b;font-size:0.9rem'>＝ {exposure/r_usd:+.1f}R　｜　"
    f"進場成本 ${cost:,.0f} 的 {exposure/max(cost,1)*100:+.1f}%</span></div>"
    f"</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("持倉檔數", f"{n_names} / {max_pos}", f"還能開 {max(max_pos - n_names, 0)} 檔", delta_color="off")
c2.metric("名目部位", f"${notional:,.0f}", f"帳戶 {notional/acct*100:.0f}%", delta_color="off")
c3.metric("未實現損益", f"${unreal:,.0f}", f"{unreal/max(cost,1)*100:+.1f}%", delta_color="off")
c4.metric("曝險額度剩餘", f"${max(acct*heat_cap_pct + min(exposure, 0), 0):,.0f}",
          f"≈ 還能開 {int(max(acct*heat_cap_pct + min(exposure, 0), 0) // r_usd)} 筆 1R", delta_color="off")
st.markdown("#### 額度使用")
if neg:
    bar(f"總曝險（上限 −{heat_cap_pct*100:.0f}%）", -exposure, acct * heat_cap_pct,
        help_txt="超過上限就不開新倉。降低曝險的方法只有出部位或等停損自然上移——不可為了降曝險把停損移到結構之外。")
else:
    st.markdown(
        "<div style='margin:2px 0 10px;padding:8px 12px;border-radius:6px;background:#052e16;"
        "color:#4ade80;font-size:0.9rem'>✅ 全部停損都已在成本之上，打到停損仍是獲利——曝險額度全滿可用。</div>",
        unsafe_allow_html=True)
bar("持倉檔數", n_names, max_pos, unit="n")
bar("名目部位 / 帳戶", notional / acct * 100, 100.0, unit="n")
by_name = d.groupby("代碼").agg(市值=("市值", "sum"), **{"曝險$": ("曝險$", "sum")},
                                筆數=("代碼", "size")).sort_values("市值", ascending=False)
by_name["佔帳戶%"] = (by_name["市值"] / acct * 100).round(1)
by_name["曝險%"] = (by_name["曝險$"] / acct * 100).round(2)
by_theme = d.groupby("題材").agg(市值=("市值", "sum"), **{"曝險$": ("曝險$", "sum")},
                                 檔數=("代碼", "nunique")).sort_values("市值", ascending=False)
by_theme["佔帳戶%"] = (by_theme["市值"] / acct * 100).round(1)
cA, cB = st.columns(2)
with cA:
    st.markdown("##### 單一標的")
    big_table(by_name, height=min(72 + 46 * len(by_name), 340),
              fmt={"市值": "{:,.0f}", "曝險$": "{:+,.0f}", "佔帳戶%": "{:.1f}", "曝險%": "{:+.2f}"})
    over = by_name[by_name["佔帳戶%"] > name_cap_pct * 100]
    if len(over):
        st.error(f"⛔ 超過單股上限 {name_cap_pct*100:.0f}%：{'、'.join(over.index)}")
with cB:
    st.markdown("##### 題材")
    big_table(by_theme, height=min(72 + 46 * len(by_theme), 340),
              fmt={"市值": "{:,.0f}", "曝險$": "{:+,.0f}", "佔帳戶%": "{:.1f}"})
    overt = by_theme[by_theme["佔帳戶%"] > theme_cap_pct * 100]
    if len(overt):
        st.error(f"⛔ 超過題材上限 {theme_cap_pct*100:.0f}%：{'、'.join(overt.index)}")
st.markdown("#### 逐筆明細")
show = d[["代碼", "題材", "股數", "進場價", "停損價", "現價", "成本", "市值",
          "停損 vs 成本%", "距停損%", "曝險$", "曝險%", "未實現"]].sort_values("曝險$")
big_table(show, show_index=False, height=min(72 + 46 * len(show), 540),
          left={"代碼", "題材"},
          fmt={"進場價": "{:.2f}", "停損價": "{:.2f}", "現價": "{:.2f}",
               "成本": "{:,.0f}", "市值": "{:,.0f}", "停損 vs 成本%": "{:+.1f}",
               "距停損%": "{:+.1f}", "曝險$": "{:+,.0f}", "曝險%": "{:+.2f}",
               "未實現": "{:+,.0f}"})
st.markdown("<div style='color:#334155;font-size:0.9rem'>"
            "曝險$ = 股數 ×（停損價 − 進場價）：停損在成本之下為負（會傷本金），在成本之上為正（已鎖利）。"
            "整頁的總曝險就是這一欄的加總。</div>", unsafe_allow_html=True)
# ── 新倉試算 ──
st.markdown("---")
st.markdown("#### 🧪 新倉試算")
s1, s2, s3, s4 = st.columns(4)
with s1:
    n_t = st.text_input("代碼", value="").strip().upper()
with s2:
    n_entry = st.number_input("預計進場價", min_value=0.0, value=0.0, step=0.01, format="%.2f")
with s3:
    n_stop = st.number_input("停損價（前波支撐）", min_value=0.0, value=0.0, step=0.01, format="%.2f")
with s4:
    n_r = st.number_input("幾 R", min_value=0.5, value=1.0, step=0.5)
if n_entry > 0 and 0 < n_stop < n_entry:
    risk_ps = n_entry - n_stop
    sh = int(n_r * r_usd / risk_ps)
    add_exp = -sh * risk_ps
    new_exp = exposure + add_exp
    cap_ok = -min(new_exp, 0) <= acct * heat_cap_pct
    cnt_ok = (n_t in set(d["代碼"])) or (n_names + 1 <= max_pos)
    name_now = float(by_name["市值"].get(n_t, 0)) + sh * n_entry
    name_ok = name_now <= acct * name_cap_pct
    st.markdown(
        f"<div style='font-size:0.9rem;line-height:1.9'>"
        f"停損距離 <b>{risk_ps/n_entry*100:.1f}%</b>　→　<b>{sh}</b> 股　名目 <b>${sh*n_entry:,.0f}</b>"
        f"（帳戶 {sh*n_entry/acct*100:.1f}%）<br>"
        f"{'✅' if cap_ok else '⛔'} 總曝險：{exp_pct:+.2f}% → <b>{new_exp/acct*100:+.2f}%</b>"
        f"（上限 −{heat_cap_pct*100:.0f}%）<br>"
        f"{'✅' if cnt_ok else '⛔'} 檔數：{n_names} → {n_names + (0 if n_t in set(d['代碼']) else 1)} / {max_pos}<br>"
        f"{'✅' if name_ok else '⛔'} 單股：${name_now:,.0f}（{name_now/acct*100:.1f}%，上限 {name_cap_pct*100:.0f}%）"
        f"</div>", unsafe_allow_html=True)
    if cap_ok and cnt_ok and name_ok:
        st.success("三項額度都過。仍需確認：燈號綠燈、壓力否決未成立、該股當日確實觸發。")
    else:
        st.error("有額度未過 → 不開新倉。")
elif n_entry > 0 and n_stop >= n_entry:
    st.warning("停損價需低於進場價。")
csv = st.session_state.pos.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ 下載持倉 CSV（備份用）", csv, "持倉.csv", "text/csv")
with st.expander("📖 定義"):
    st.markdown(f"""
**總曝險** = Σ 股數 ×（停損價 − 進場價）÷ 帳戶總資金。就是「現在所有停損同時被打到，帳戶會賺或賠多少」。

- 用 10% 資金買、停損 −10% → 曝險 **−1%**
- 停損上移到成本價 → 曝險 **0%**（打到停損不賺不賠）
- 停損再上移到 +10% → 曝險 **+1%**（打到停損還賺 1%）

**上限**：總曝險不得低於 −{heat_cap_pct*100:.0f}%（規則四）。開新倉前先算，超了就不開。
曝險為正時代表全部部位都已鎖利，額度全滿可用。

**降低曝險只有兩種合法方式**：出掉部位、或等圖上出現新支撐讓停損自然上移。
**不可為了降曝險把停損移到結構之外**（禁止事項）。

其他限制：同時最多 {max_pos} 檔、同一檔 ≤ 帳戶 {name_cap_pct*100:.0f}%、同一題材 ≤ {theme_cap_pct*100:.0f}%。
加碼單獨立列出（同代碼多列），各自填自己的進場價與停損價。
""")
