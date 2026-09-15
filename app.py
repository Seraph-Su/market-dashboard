import streamlit as st

st.set_page_config(page_title="美股儀表板", page_icon="📊", layout="wide")

# ── 導覽：照順勢交易系統的動作順序　① 進場時機 → ② 選股 → ③ 加碼＆倉位管理 ──
home = st.Page("pages/00_Home.py", title="首頁｜每日三分鐘", icon="🏠", default=True)

timing = [
    st.Page("pages/1_Dashboard.py",       title="大盤壓力儀表板", icon="📊"),
    st.Page("pages/6_WinRate_Matrix.py",  title="四大指數勝率矩陣", icon="🎯"),
]
picking = [
    st.Page("pages/5_Breakout_Screener.py", title="均線收斂突破選股", icon="📡"),
    st.Page("pages/10_Monster_Screener.py", title="怪物股選股器", icon="🦖"),
]
sizing = [
    st.Page("pages/11_Exposure.py", title="總曝險計算器", icon="🌡️"),
    st.Page("pages/13_ATR_Add.py",  title="ATR 加碼計算器", icon="🎯"),
]
reference = [
    st.Page("pages/0_Trading_SOP.py", title="交易 SOP 流程圖", icon="🗺️"),
]
# 封存：檔案留著、不顯示；側欄開關打開才出現
archived = [
    st.Page("pages/2_EMA_Analysis.py",      title="[封存] EMA 分析", icon="📈"),          # 封存 2026-09-12
    st.Page("pages/3_Sector_Rotation.py",   title="[封存] 細分產業週報", icon="🇺🇸"),
    st.Page("pages/7_Position_Sizing.py",   title="[封存] 加碼比例計算器", icon="📉"),    # 封存 2026-09-12
    st.Page("pages/9_ATR_Pullback_Stats.py", title="[封存] ATR 回檔機率", icon="🌡️"),    # 封存 2026-09-12
]

with st.sidebar:
    st.markdown("### 📊 美股儀表板")
    st.markdown("<span style='color:#64748b;font-size:0.75rem'>順勢交易系統　進場 → 選股 → 加碼與倉位</span>",
                unsafe_allow_html=True)
    show_archived = st.toggle("顯示封存頁面", value=False)

sections = {
    "": [home],
    "① 進場時機": timing,
    "② 選股": picking,
    "③ 加碼＆倉位管理": sizing,
    "參考": reference,
}
if show_archived:
    sections["封存"] = archived

pg = st.navigation(sections)
pg.run()
