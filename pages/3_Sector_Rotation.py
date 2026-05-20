import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
import json


# ── Data fetch ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="抓取 S&P 500 成分股與週線資料中…")
def load_data():
    # 1. S&P 500 成分股
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df_components = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    df_components["Symbol"] = df_components["Symbol"].str.replace(".", "-", regex=False)

    # 2. 週線價格
    end   = datetime.today()
    start = end - timedelta(days=10)
    tickers_str = " ".join(df_components["Symbol"].tolist())
    data = yf.download(tickers_str, start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    close = data["Close"] if "Close" in data.columns else data.xs("Close", axis=1, level=0)
    close = close.dropna(how="all")
    if len(close) < 2:
        raise ValueError("價格資料不足")
    latest   = close.iloc[-1]
    week_ago = close.iloc[-6] if len(close) >= 6 else close.iloc[0]
    weekly_return = ((latest - week_ago) / week_ago * 100).dropna()
    end_date   = close.index[-1].date()
    start_date = close.index[(-6 if len(close) >= 6 else 0)].date()

    # 3. 細分產業統計
    df = df_components.copy()
    df["weekly_return"] = df["Symbol"].map(weekly_return)
    df = df.dropna(subset=["weekly_return"])
    stats = (
        df.groupby(["GICS Sector", "GICS Sub-Industry"])
        .agg(
            avg_return=("weekly_return", "mean"),
            stock_count=("Symbol", "count"),
            top_stock=("weekly_return", lambda x: df.loc[x.idxmax(), "Symbol"]),
            top_stock_ret=("weekly_return", "max"),
            worst_stock=("weekly_return", lambda x: df.loc[x.idxmin(), "Symbol"]),
            worst_stock_ret=("weekly_return", "min"),
        )
        .reset_index()
        .sort_values("avg_return", ascending=False)
        .reset_index(drop=True)
    )
    stats["rank"] = range(1, len(stats) + 1)
    sector_avg = (
        df.groupby("GICS Sector")["weekly_return"]
        .mean().sort_values(ascending=False).reset_index()
    )
    return stats, df, sector_avg, end_date, start_date


# ── HTML report ───────────────────────────────────────────────────
def build_html(stats, df_detail, sector_avg, end_date, start_date):
    now_str    = datetime.now().strftime("%Y/%m/%d %H:%M")
    date_range = f"{start_date} → {end_date}"
    UP, DOWN   = "▲", "▼"

    def color(v):
        return "#22c55e" if v > 0 else ("#ef4444" if v < 0 else "#94a3b8")

    def arrow(v):
        return UP if v > 0 else (DOWN if v < 0 else "—")

    def fmt(v):
        return f"{arrow(v)} {abs(v):.2f}%"

    def make_rank_item(row, is_up):
        badge_cls = "badge-up" if is_up else "badge-down"
        ar = UP if is_up else DOWN
        ret_str = "{:.2f}".format(abs(row["avg_return"]))
        return (
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:8px 0;border-bottom:1px solid #1e293b">'
            '<div>'
            '<div style="font-weight:600;font-size:0.9em">' + row["GICS Sub-Industry"] + '</div>'
            '<div style="color:#64748b;font-size:0.75em">' + row["GICS Sector"] + '</div>'
            '</div>'
            '<span class="badge ' + badge_cls + '" style="font-size:0.9em">' + ar + ' ' + ret_str + '%</span>'
            '</div>'
        )

    top5_html = "".join(make_rank_item(row, True)  for _, row in stats.head(5).iterrows())
    bot5_html = "".join(make_rank_item(row, False) for _, row in stats.tail(5).iterrows())

    sector_rows = ""
    for _, row in sector_avg.iterrows():
        bg = "rgba(34,197,94,0.08)" if row["weekly_return"] > 0 else "rgba(239,68,68,0.08)"
        sector_rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:10px 14px;font-weight:600">{row["GICS Sector"]}</td>'
            f'<td style="padding:10px 14px;text-align:right;color:{color(row["weekly_return"])};font-weight:700;font-size:1.05em">'
            f'{fmt(row["weekly_return"])}</td></tr>'
        )

    stats_json   = json.dumps(stats.to_dict(orient="records"), ensure_ascii=False)
    stats_count  = len(stats)

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing:border-box;margin:0;padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0; }}
  .header {{ background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:22px 28px;border-bottom:1px solid #1e40af44; }}
  .header h1 {{ font-size:1.4em;font-weight:700;color:#fff; }}
  .header .meta {{ color:#94a3b8;font-size:0.82em;margin-top:5px; }}
  .date-range {{ background:#1e40af33;border:1px solid #3b82f644;border-radius:8px;display:inline-block;padding:3px 10px;margin-top:6px;font-size:0.8em;color:#93c5fd; }}
  .container {{ max-width:1160px;margin:0 auto;padding:20px 16px; }}
  .grid2 {{ display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px; }}
  .card {{ background:#1e293b;border:1px solid #334155;border-radius:12px;overflow:hidden; }}
  .card-title {{ padding:14px 18px 10px;font-size:0.88em;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #334155; }}
  .sector-table {{ width:100%;border-collapse:collapse; }}
  .sector-table tr:hover {{ background:rgba(255,255,255,.03); }}
  .sector-table td {{ border-bottom:1px solid #1e293b; }}
  .badge {{ display:inline-block;padding:2px 8px;border-radius:999px;font-size:.75em;font-weight:600; }}
  .badge-up   {{ background:rgba(34,197,94,.15);color:#22c55e; }}
  .badge-down {{ background:rgba(239,68,68,.15);color:#ef4444; }}
  .toolbar {{ display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;align-items:center; }}
  .search-box {{ background:#1e293b;border:1px solid #334155;border-radius:8px;padding:7px 12px;color:#e2e8f0;font-size:.88em;flex:1;min-width:180px; }}
  .search-box:focus {{ outline:none;border-color:#3b82f6; }}
  .filter-btn {{ background:#1e293b;border:1px solid #334155;border-radius:8px;padding:7px 12px;color:#94a3b8;font-size:.82em;cursor:pointer;white-space:nowrap; }}
  .filter-btn.active {{ background:#1e40af;border-color:#3b82f6;color:#fff; }}
  .full-table {{ width:100%;border-collapse:collapse;font-size:.86em; }}
  .full-table th {{ padding:9px 11px;text-align:left;background:#0f172a;color:#64748b;font-weight:600;font-size:.78em;text-transform:uppercase;letter-spacing:.04em;cursor:pointer;white-space:nowrap; }}
  .full-table th:hover {{ color:#94a3b8; }}
  .full-table td {{ padding:9px 11px;border-bottom:1px solid #1e293b; }}
  .full-table tr:hover td {{ background:rgba(255,255,255,.02); }}
  .rank-badge {{ display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;font-size:.75em;font-weight:700; }}
  .rank-top {{ background:rgba(34,197,94,.2);color:#22c55e; }}
  .rank-bot {{ background:rgba(239,68,68,.2);color:#ef4444; }}
  .rank-mid {{ background:#1e293b;color:#64748b; }}
  .stock-chip {{ display:inline-block;background:#0f172a;border:1px solid #334155;border-radius:5px;padding:2px 6px;font-size:.76em;font-family:monospace;color:#93c5fd; }}
  .pagination {{ display:flex;gap:6px;justify-content:center;margin-top:14px;align-items:center; }}
  .page-btn {{ background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 11px;color:#94a3b8;cursor:pointer;font-size:.82em; }}
  .page-btn.active {{ background:#1e40af;color:#fff;border-color:#3b82f6; }}
  .page-btn:hover:not(.active) {{ background:#334155; }}
  .page-info {{ color:#64748b;font-size:.82em; }}
  @media(max-width:700px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>🇺🇸 美股細分產業週報</h1>
  <div class="meta">產生時間：{now_str} &nbsp;｜&nbsp; 資料來源：Yahoo Finance / Wikipedia S&amp;P 500</div>
  <div class="date-range">📅 週期：{date_range}</div>
</div>
<div class="container">
  <div class="grid2">
    <div class="card">
      <div class="card-title">🏛️ 十一大板塊漲跌</div>
      <table class="sector-table">{sector_rows}</table>
    </div>
    <div style="display:flex;flex-direction:column;gap:14px">
      <div class="card">
        <div class="card-title">🚀 本週最強細分產業 Top 5</div>
        <div style="padding:10px 14px">{top5_html}</div>
      </div>
      <div class="card">
        <div class="card-title">📉 本週最弱細分產業 Bottom 5</div>
        <div style="padding:10px 14px">{bot5_html}</div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">🔍 全部 {stats_count} 個細分產業排名</div>
    <div style="padding:14px 14px 8px">
      <div class="toolbar">
        <input class="search-box" type="text" id="searchInput" placeholder="搜尋產業名稱（如 Software、Bank、Pharma…）" oninput="filterTable()">
        <button class="filter-btn active" id="btn-all"  onclick="setFilter('all')">全部</button>
        <button class="filter-btn"        id="btn-up"   onclick="setFilter('up')">▲ 上漲</button>
        <button class="filter-btn"        id="btn-down" onclick="setFilter('down')">▼ 下跌</button>
      </div>
      <table class="full-table" id="mainTable">
        <thead><tr>
          <th onclick="sortTable(0)">排名 ↕</th>
          <th onclick="sortTable(1)">細分產業 ↕</th>
          <th onclick="sortTable(2)">大板塊 ↕</th>
          <th onclick="sortTable(3)" style="text-align:right">週報酬率 ↕</th>
          <th style="text-align:right">股票數</th>
          <th>本週最強</th>
          <th>本週最弱</th>
        </tr></thead>
        <tbody id="tableBody"></tbody>
      </table>
      <div class="pagination" id="pagination"></div>
    </div>
  </div>
</div>
<script>
const allData = {stats_json};
let filtered = [...allData];
let currentFilter = 'all';
let currentSort = {{col:3, asc:false}};
let currentPage = 1;
const perPage = 20;
function setFilter(f) {{
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + f).classList.add('active');
  filterTable();
}}
function filterTable() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  filtered = allData.filter(r => {{
    const matchText = r['GICS Sub-Industry'].toLowerCase().includes(q) || r['GICS Sector'].toLowerCase().includes(q);
    const matchFilter = currentFilter==='all' || (currentFilter==='up' && r.avg_return>0) || (currentFilter==='down' && r.avg_return<0);
    return matchText && matchFilter;
  }});
  sortData(); currentPage = 1; renderTable();
}}
function sortTable(col) {{
  if (currentSort.col===col) currentSort.asc = !currentSort.asc;
  else {{ currentSort.col=col; currentSort.asc = col!==3; }}
  sortData(); renderTable();
}}
function sortData() {{
  const keys = ['rank','GICS Sub-Industry','GICS Sector','avg_return','stock_count'];
  const key = keys[currentSort.col];
  filtered.sort((a,b) => {{
    const va=a[key], vb=b[key];
    if (typeof va==='string') return currentSort.asc ? va.localeCompare(vb) : vb.localeCompare(va);
    return currentSort.asc ? va-vb : vb-va;
  }});
}}
function renderTable() {{
  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total/perPage));
  if (currentPage > totalPages) currentPage = totalPages;
  const start = (currentPage-1)*perPage;
  const slice = filtered.slice(start, start+perPage);
  document.getElementById('tableBody').innerHTML = slice.map(r => {{
    const ret = r.avg_return;
    const retColor = ret>0 ? '#22c55e' : (ret<0 ? '#ef4444' : '#94a3b8');
    const retSign  = ret>0 ? '▲' : (ret<0 ? '▼' : '—');
    const rankClass = r.rank<=10 ? 'rank-top' : (r.rank>allData.length-10 ? 'rank-bot' : 'rank-mid');
    return `<tr>
      <td><span class="rank-badge ${{rankClass}}">${{r.rank}}</span></td>
      <td style="font-weight:600">${{r['GICS Sub-Industry']}}</td>
      <td style="color:#94a3b8;font-size:.85em">${{r['GICS Sector']}}</td>
      <td style="text-align:right;color:${{retColor}};font-weight:700;font-size:1.05em">${{retSign}} ${{Math.abs(ret).toFixed(2)}}%</td>
      <td style="text-align:right;color:#64748b">${{r.stock_count}}</td>
      <td><span class="stock-chip">${{r.top_stock}}</span> <span style="color:#22c55e;font-size:.76em">+${{r.top_stock_ret.toFixed(1)}}%</span></td>
      <td><span class="stock-chip">${{r.worst_stock}}</span> <span style="color:#ef4444;font-size:.76em">${{r.worst_stock_ret.toFixed(1)}}%</span></td>
    </tr>`;
  }}).join('');
  const pg = document.getElementById('pagination');
  if (totalPages<=1) {{ pg.innerHTML=''; return; }}
  let html = `<button class="page-btn" onclick="goPage(${{currentPage-1}})" ${{currentPage===1?'disabled':''}}>‹</button>`;
  for (let i=1; i<=totalPages; i++) {{
    if (i===1 || i===totalPages || Math.abs(i-currentPage)<=2)
      html += `<button class="page-btn ${{i===currentPage?'active':''}}" onclick="goPage(${{i}})">${{i}}</button>`;
    else if (Math.abs(i-currentPage)===3)
      html += `<span class="page-info">…</span>`;
  }}
  html += `<button class="page-btn" onclick="goPage(${{currentPage+1}})" ${{currentPage===totalPages?'disabled':''}}>›</button>`;
  html += `<span class="page-info">&nbsp;共 ${{total}} 個產業</span>`;
  pg.innerHTML = html;
}}
function goPage(p) {{
  const totalPages = Math.ceil(filtered.length/perPage);
  if (p<1||p>totalPages) return;
  currentPage = p; renderTable();
}}
sortData(); renderTable();
</script>
</body></html>"""


# ── Page ──────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("## 🇺🇸 美股細分產業週報")
with col_refresh:
    if st.button("🔄 更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    stats, df_detail, sector_avg, end_date, start_date = load_data()
    html_report = build_html(stats, df_detail, sector_avg, end_date, start_date)
    components.html(html_report, height=1400, scrolling=True)
except Exception as e:
    st.error(f"資料載入失敗：{e}")
