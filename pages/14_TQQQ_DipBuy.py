import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime as _dt
from dipbuy_backtest import Params, run, summarize, benchmark, load_yf

# ═══════════════════════════════════════════════════════════════════
# 🪜 TQQQ 分批低接回測
#   訊號看 QQQ 距高點回撤，買 TQQQ；2010 前用合成 TQQQ（3×QQQ − 費用 − 借款成本）。
#   回測引擎在 dipbuy_backtest.py，本頁只負責參數、圖表與表格。
# ═══════════════════════════════════════════════════════════════════

# ── 全頁字級放大（與其他頁同一套：16px → 20px） ─────────────────────
st.markdown("""
<style>
  html { font-size: 20px; }
  body, .stApp, [data-testid="stAppViewContainer"] { font-size: 1rem; line-height: 1.65; }
  [data-testid="stMarkdownContainer"] p  { font-size: 1rem; line-height: 1.7; }
  [data-testid="stMarkdownContainer"] li { font-size: 1rem; line-height: 1.7; }
  [data-testid="stMarkdownContainer"] h2 { font-size: 1.75rem; }
  [data-testid="stMarkdownContainer"] h3 { font-size: 1.4rem; }
  [data-testid="stMarkdownContainer"] h4 { font-size: 1.25rem; }
  [data-testid="stMetricValue"] { font-size: 1.85rem !important; }
  [data-testid="stMetricLabel"] p { font-size: 0.98rem !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { font-size: 1rem !important; }
  [data-testid="stCaptionContainer"] p { font-size: 0.92rem !important; }
  [data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p { font-size: 1rem !important; }
  [data-baseweb="select"] div, [data-baseweb="tab"] p { font-size: 1rem; }
  [data-testid="stExpander"] summary p, details summary { font-size: 1.08rem; font-weight: 600; }
  [data-testid="stSidebar"] * { font-size: 1rem; }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

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
  table.bigtbl td.code { font-weight: 700; color: #e2e8f0; }
  table.bigtbl tr:hover td { background: #131c2b; }
  .note { background:#0f2027; border:1px solid #1e3a5f; border-radius:8px;
          padding:10px 16px; color:#94a3b8; font-size:0.95rem; margin:6px 0 12px; }
  .warn { background:#1c0a00; border:1px solid #854d0e; border-radius:8px;
          padding:10px 16px; color:#fbbf24; font-size:0.95rem; margin:6px 0 12px; }
</style>
""", unsafe_allow_html=True)


def big_table(df, height=520, fmt=None, left=(), show_index=False):
    """大字級唯讀表格（與 13_ATR_Add.py 同款）。"""
    d = df.copy()
    fmt = fmt or {}
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: "—" if pd.isna(v) else f.format(v))
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


PCT = "{:+.1%}"
PERIODS = {
    "全期 1999~（含 2000 泡沫，2010 前為合成 TQQQ）": "1999-01-01",
    "2003~（排除 2000 泡沫破裂）": "2003-03-01",
    "2010~（真實 TQQQ）": "2010-02-11",
}
EXIT_LABEL = {"recover": "QQQ 回到前高", "target": "TQQQ 獲利目標", "hold": "持有固定天數"}


@st.cache_data(ttl=3600, show_spinner="下載 QQQ / TQQQ / VIX 歷史資料中…")
def get_data(hour_key: str):
    return load_yf()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_run(hour_key: str, start: str, p_dict: dict):
    df = get_data(hour_key)
    return run(df[df.index >= start], Params(**p_dict))


# ────────────────────────────────────────────────────────────────────
# 參數（側欄）
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 🪜 低接參數")
    period_lbl = st.selectbox("回測期間", list(PERIODS.keys()), index=0)
    ref = st.radio("回撤參考高點", ["hi252", "ath"], horizontal=True,
                   format_func=lambda x: "52 週高" if x == "hi252" else "歷史高")
    start = st.slider("第一批：QQQ 回撤 ≥（%）", 3, 40, 10) / 100
    n = st.slider("分幾批", 1, 8, 3)
    step = st.slider("每批間隔：QQQ 再跌（%）", 1, 15, 5, disabled=(n == 1)) / 100
    weights = st.radio("每批金額", ["equal", "pyramid"], horizontal=True,
                       format_func=lambda x: "平均" if x == "equal" else "越跌越多 1:2:3")
    exit_ = st.selectbox("出場方式", list(EXIT_LABEL), index=2, format_func=EXIT_LABEL.get)
    target, hold_days = 0.5, 126
    if exit_ == "target":
        target = st.slider("TQQQ 持倉報酬達（%）出場", 10, 200, 50, step=10) / 100
    if exit_ == "hold":
        hold_days = st.select_slider("首批後持有（交易日）", [21, 42, 63, 126, 189, 252, 378, 504], 126)
    max_days = st.select_slider("時間停損（交易日，0=不設）", [0, 126, 252, 378, 504, 756], 0)
    stop_loss = st.slider("持倉停損（%，0=不設）", 0, 90, 0, step=5) / 100
    vix_min = st.slider("觸發時 VIX ≥（0=不過濾）", 0, 40, 0)
    trend = st.radio("年線過濾（EMA260）", ["any", "above"], horizontal=True,
                     format_func=lambda x: "不過濾" if x == "any" else "觸發時須在年線上")
    asset = st.radio("買進標的", ["tqqq", "qqq"], horizontal=True, format_func=str.upper)

P = Params(ref=ref, start=start, n=n, step=step, weights=weights, exit=exit_, target=target,
           hold_days=hold_days, max_days=max_days, stop_loss=stop_loss, vix_min=vix_min,
           trend=trend, asset=asset)
period_start = PERIODS[period_lbl]

# ────────────────────────────────────────────────────────────────────
# 標題
# ────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🪜 TQQQ 分批低接回測")
    st.markdown("<span style='color:#64748b;font-size:0.95rem'>"
                "QQQ 距高點回撤觸發 → 分 N 批買 TQQQ → 依出場規則賣出 ｜ "
                "資金池 = 專為低接保留的現金，閒置時賺國庫券利率</span>", unsafe_allow_html=True)
with col_refresh:
    if st.button("🔄 更新數據", width="stretch"):
        st.cache_data.clear()
        st.rerun()

hour_key = _dt.now().strftime("%Y-%m-%d-%H")
try:
    DF = get_data(hour_key)
except Exception as e:
    st.error(f"資料下載失敗：{e}")
    st.stop()

# ────────────────────────────────────────────────────────────────────
# ① 目前位置與掛單價位
# ────────────────────────────────────────────────────────────────────
last = DF.iloc[-1]
ref_hi = last[ref]
dd_now = last["qqq"] / ref_hi - 1
dev60 = last["qqq"] / last["ema60"] - 1
above = last["qqq"] > last["ema260"]

st.markdown("---")
st.markdown(f"#### ① 目前位置（{DF.index[-1].date()} 收盤）")
c1, c2, c3 = st.columns(3)
c1.metric("QQQ ／ TQQQ", f"{last['qqq']:.2f} ／ {last['tqqq']:.2f}")
c2.metric("VIX", f"{last['vix']:.2f}")
c3.metric(f"距{'52週高' if ref == 'hi252' else '歷史高'}", f"{dd_now:+.1%}", f"高點 {ref_hi:.2f}", delta_color="off")
st.markdown(f"<span style='color:#64748b'>EMA60 乖離 <b>{dev60:+.1%}</b> ｜ "
            f"{'🟢 年線之上' if above else '🔴 年線之下'}（EMA260 {last['ema260']:.2f}）</span>",
            unsafe_allow_html=True)

w = P.tranche_weights()
rows = []
for k in range(n):
    lvl_dd = -(start + k * step)
    q_px = ref_hi * (1 + lvl_dd)
    move = q_px / last["qqq"] - 1
    rows.append({
        "批次": f"第 {k + 1} 批", "QQQ 回撤": lvl_dd, "QQQ 價位": q_px,
        "距現價": move, "TQQQ 粗估價": last["tqqq"] * max(1 + 3 * move, 0.01),
        "資金池比例": w[k], "狀態": "✅ 已達" if last["qqq"] <= q_px else "⏳ 等待",
    })
big_table(pd.DataFrame(rows), height=420, left=("批次", "狀態"),
          fmt={"QQQ 回撤": "{:.0%}", "QQQ 價位": "{:.2f}", "距現價": PCT,
               "TQQQ 粗估價": "{:.2f}", "資金池比例": "{:.0%}"})
st.caption("價位以目前參考高點計算；觸發後高點即鎖定。TQQQ 粗估價 = 現價 × (1 + 3 × QQQ 距離)，"
           "實際會因路徑與波動耗損而偏低，掛單請以 QQQ 價位為準。")

# ────────────────────────────────────────────────────────────────────
# ② 分頁
# ────────────────────────────────────────────────────────────────────
tab_bt, tab_grid, tab_depth, tab_help = st.tabs(
    ["📈 單一策略回測", "🧮 參數掃描", "🎯 回撤深度 × TQQQ 未來報酬", "📖 說明"])

# ── 單一策略 ─────────────────────────────────────────────────────────
with tab_bt:
    eq, eps = cached_run(hour_key, period_start, P.__dict__)
    s = summarize(eq, eps)
    sub = DF[DF.index >= period_start]
    yrs = (sub.index[-1] - sub.index[0]).days / 365.25

    def bh(col):
        b = benchmark(sub, col)
        return b.iloc[-1] ** (1 / yrs) - 1, float((b / b.cummax() - 1).min())

    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)
    m1.metric("資金池 CAGR", f"{s['CAGR']:+.1%}")
    m2.metric("資金池最大回撤", f"{s['最大回撤']:.1%}")
    m3.metric("事件數 ／ 勝率", f"{s['事件數']} ／ {s['勝率']:.0%}" if s["事件數"] else "0")
    m4.metric("每次平均 ／ 中位", f"{s['平均報酬']:+.1%} ／ {s['中位報酬']:+.1%}" if s["事件數"] else "—")
    m5.metric("最差一次", f"{s['最差事件']:+.1%}" if s["事件數"] else "—")
    m6.metric("平均投入 ／ 天數", f"{s['平均投入']:.0%} ／ {s['平均天數']:.0f}" if s["事件數"] else "—")

    q_c, q_m = bh("qqq")
    t_c, t_m = bh("tqqq")
    st.markdown(f"<div class='note'>對照（同期間）：QQQ 買進持有 CAGR <b>{q_c:+.1%}</b>／最大回撤 {q_m:.0%} ｜ "
                f"TQQQ 買進持有 CAGR <b>{t_c:+.1%}</b>／最大回撤 {t_m:.0%} ｜ "
                f"資金池閒置時賺國庫券，所以 CAGR 要和「現金 + 偶爾出手」比，不是和全倉比。</div>",
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="低接資金池", line=dict(color="#f59e0b", width=2.5)))
    fig.add_trace(go.Scatter(x=sub.index, y=benchmark(sub, "qqq"), name="QQQ 持有",
                             line=dict(color="#60a5fa", width=1.3)))
    fig.add_trace(go.Scatter(x=sub.index, y=benchmark(sub, "tqqq"), name="TQQQ 持有",
                             line=dict(color="#64748b", width=1, dash="dot")))
    for _, e in eps.iterrows():
        fig.add_vrect(x0=str(e["觸發日"]), x1=str(e["出場日"]),
                      fillcolor="#4ade80" if e["資金池報酬"] > 0 else "#f87171",
                      opacity=0.12, line_width=0)
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=15, color="#cbd5e1"),
                      yaxis=dict(type="log", title="淨值（起始 = 1，對數）", gridcolor="#1e293b"),
                      xaxis=dict(gridcolor="#1e293b"),
                      legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("綠底 = 獲利事件、紅底 = 虧損事件（持倉期間）。")

    if len(eps):
        st.markdown("##### 每次事件明細")
        show = eps.copy()
        show["合成資料"] = show["合成資料"].map({True: "合成", False: ""})
        big_table(show.iloc[::-1], height=560, left=("觸發日", "出場日", "出場原因", "合成資料"),
                  fmt={"投入比例": "{:.0%}", "持倉報酬": PCT, "資金池報酬": PCT,
                       "期間最大回撤": "{:.1%}", "QQQ最深回撤": "{:.1%}"})
        st.caption("持倉報酬 = 已買部位的報酬；資金池報酬 = 含未動用現金的整體報酬（未成交批次留在現金）。"
                   "期間最大回撤 = 該事件中資金池從高點的最大跌幅。")

# ── 參數掃描 ─────────────────────────────────────────────────────────
with tab_grid:
    st.markdown("固定側欄的其他參數，只掃 **第一批回撤 × 分幾批**，看哪一區穩定、哪一格只是運氣。")
    g1, g2 = st.columns(2)
    metric = g1.selectbox("顯示指標", ["CAGR", "最差事件", "最大回撤", "勝率", "平均報酬", "事件數"])
    g_period = g2.selectbox("掃描期間", list(PERIODS.keys()), index=0, key="gp")
    starts = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    ns = [1, 2, 3, 4, 5, 6]

    @st.cache_data(ttl=3600, show_spinner="掃描 48 組參數中…")
    def sweep(hour_key, period, p_dict):
        df = get_data(hour_key)
        df = df[df.index >= period]
        out = {}
        for s_ in starts:
            for n_ in ns:
                p = Params(**{**p_dict, "start": s_, "n": n_})
                out[(s_, n_)] = summarize(*run(df, p))
        return out

    res = sweep(hour_key, PERIODS[g_period], P.__dict__)
    z = np.array([[res[(s_, n_)][metric] for n_ in ns] for s_ in starts], dtype=float)
    txt = [[("—" if np.isnan(v) else (f"{v:+.0%}" if metric not in ("勝率", "事件數") else
             (f"{v:.0%}" if metric == "勝率" else f"{v:.0f}"))) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{n_} 批" for n_ in ns], y=[f"−{s_:.0%}" for s_ in starts],
        text=txt, texttemplate="%{text}", textfont=dict(size=16),
        colorscale="RdYlGn", reversescale=False,
        zmid=0 if metric in ("CAGR", "平均報酬", "最差事件") else None,
        colorbar=dict(title=metric)))
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=15, color="#cbd5e1"),
                      yaxis=dict(title="第一批 QQQ 回撤", autorange="reversed"),
                      xaxis=dict(title=f"分幾批（間隔 {step:.0%}）", side="top"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("<div class='warn'>⚠️ 1999 年以來 QQQ 跌超過 10% 的獨立事件只有約 20 次，"
                "相鄰格子差很多就代表是單一事件（例如 2000、2008、2022）決定的結果，不要挑最亮的那一格。"
                "看「最差事件」與「最大回撤」是否能接受，比追最高 CAGR 重要。</div>", unsafe_allow_html=True)

# ── 回撤深度 × 未來報酬 ───────────────────────────────────────────────
with tab_depth:
    st.markdown("在 QQQ 位於某個回撤深度的**任一天**買進 TQQQ，之後 N 天的報酬分布（類似勝率矩陣，但標的換成 TQQQ）。")
    d1, d2 = st.columns(2)
    d_period = d1.selectbox("期間", list(PERIODS.keys()), index=0, key="dp")
    split = d2.radio("依年線拆分", ["不拆", "年線之上", "年線之下"], horizontal=True)
    h = st.radio("持有期間", [63, 126, 252], index=1, horizontal=True,
                 format_func=lambda x: {63: "3 個月", 126: "6 個月", 252: "1 年"}[x])

    @st.cache_data(ttl=3600, show_spinner=False)
    def depth_table(hour_key, period, ref_, split_, h_):
        df = get_data(hour_key).copy()
        df["dd"] = df["qqq"] / df[ref_] - 1
        df["fwd"] = df["tqqq"].shift(-h_) / df["tqqq"] - 1
        df["mae"] = df["tqqq"].rolling(h_).min().shift(-h_) / df["tqqq"] - 1
        df = df[df.index >= period].dropna(subset=["fwd"])
        if split_ == "年線之上":
            df = df[df["qqq"] > df["ema260"]]
        elif split_ == "年線之下":
            df = df[df["qqq"] <= df["ema260"]]
        bins = [-1, -.40, -.30, -.25, -.20, -.15, -.10, -.05, 0.0001]
        labels = ["< −40%", "−30~−40%", "−25~−30%", "−20~−25%", "−15~−20%", "−10~−15%", "−5~−10%", "0~−5%"]
        df["b"] = pd.cut(df["dd"], bins, labels=labels)
        rows = []
        for lb in labels[::-1]:
            x = df[df["b"] == lb]
            if len(x) < 5:
                continue
            ep = (x.index.to_series().diff().dt.days > 30).sum() + 1
            rows.append({"QQQ 回撤": lb, "勝率": (x["fwd"] > 0).mean(), "中位報酬": x["fwd"].median(),
                         "最差 10%": x["fwd"].quantile(0.1), "最好 10%": x["fwd"].quantile(0.9),
                         "期間最大續跌（中位）": x["mae"].median(), "樣本天數": len(x), "約略獨立段數": ep})
        return pd.DataFrame(rows)

    dt = depth_table(hour_key, PERIODS[d_period], ref, split, h)
    big_table(dt, height=480, left=("QQQ 回撤",),
              fmt={"勝率": "{:.0%}", "中位報酬": PCT, "最差 10%": PCT, "最好 10%": PCT,
                   "期間最大續跌（中位）": "{:.0%}", "樣本天數": "{:.0f}", "約略獨立段數": "{:.0f}"})
    st.caption("樣本為每日重疊資料，相鄰日子高度相關；「約略獨立段數」才是真正的事件數（間隔 >30 天算新的一段）。"
               "回撤參考高點跟側欄設定。")

# ── 說明 ─────────────────────────────────────────────────────────────
with tab_help:
    st.markdown("""
**規則**
1. QQQ 收盤距參考高點回撤 ≥ 第一批門檻（且通過 VIX／年線過濾）→ 事件開始，高點鎖定。
2. 資金池（事件開始時的全部現金）依「平均」或「1:2:3…」切成 N 批；第 k 批在 QQQ 回撤 = 門檻 + (k−1)×間隔 時以收盤價買 TQQQ。跳空大跌可能同一天成交好幾批。
3. 出場：QQQ 回到鎖定的高點／TQQQ 持倉報酬達標／首批後滿 N 天，或觸發時間停損、持倉停損。全部一次賣出。
4. 出場後 QQQ 回撤要先縮小到門檻的一半以內，才會重新觸發（避免同一波跌勢一直進出）。
5. 未動用現金以 13 週國庫券利率計息；不含交易成本、稅與滑價。

**合成 TQQQ（2010-02 以前）**：每日報酬 = 3 × QQQ 報酬 − (0.95% 費用 + 2 × (國庫券利率 + 0.5%)) / 252。
2010~2026 與真實 TQQQ 比對，年化報酬差 < 0.1%。2000~2002 合成 TQQQ 最大跌幅約 −99.9%，這就是槓桿 ETF 的尾部風險。

**判讀重點**
- 「回到前高才出場」對 QQQ 合理，對 TQQQ 很危險：2000 年 QQQ 花了 15 年回到前高，TQQQ 早已歸零。務必搭配時間停損或改用固定持有天數。
- 批數多 = 平均成本較低、最差情境較輕，但多數小回檔只會成交第一批，報酬也變少。批數是**風險旋鈕**，不是報酬放大器。
- 本頁是歷史統計，不保證未來；事件數少，請看參數區域是否穩定，而非單一最佳值。
""")
