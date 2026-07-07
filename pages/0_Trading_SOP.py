"""
第 0 頁 · 交易 SOP 流程圖
把 trading_SOP_flowchart.html 內嵌到儀表板。

放置方式：
1. 這支檔放到 pages/0_Trading_SOP.py
2. trading_SOP_flowchart.html 放到「專案根目錄」（跟你的目錄入口檔同一層），
   或放到 assets/ 資料夾（下面路徑會自動找）。
3. 在目錄入口檔的 pages 清單「最前面」加一行（見對話說明）。
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

_here = Path(__file__).resolve().parent
_candidates = [
    _here.parent / "trading_SOP_flowchart.html",           # 專案根目錄
    _here.parent / "assets" / "trading_SOP_flowchart.html", # assets/
    _here / "trading_SOP_flowchart.html",                   # pages/ 同層
]
html_path = next((p for p in _candidates if p.exists()), None)

if html_path is None:
    st.error(
        "找不到 trading_SOP_flowchart.html。"
        "請把它放到專案根目錄，或 assets/ 資料夾。"
    )
else:
    html = html_path.read_text(encoding="utf-8")
    # 內嵌到儀表板時隱藏右上角的浮動編輯按鈕（要編輯請直接開 HTML 檔）
    html = html.replace(
        "</head>",
        "<style>.editbar,.edithint{display:none!important;}</style></head>",
        1,
    )
    # height 給足避免被截斷；字級放大後偏高，不夠就往上調
    components.html(html, height=4200, scrolling=True)
