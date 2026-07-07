import streamlit as st
st.set_page_config(
    page_title="美股儀表板",
    page_icon="📊",
    layout="wide"
)
pages = [
    st.Page("pages/0_Trading_SOP.py",     title="交易 SOP 流程圖", icon="🗺️"),
    st.Page("pages/1_Dashboard.py",       title="大盤壓力儀表板", icon="📊"),
    st.Page("pages/2_EMA_Analysis.py",    title="EMA 分析",       icon="📈"),
    st.Page("pages/3_Sector_Rotation.py", title="細分產業週報",   icon="🇺🇸"),
    st.Page("pages/5_Breakout_Screener.py", title="均線收斂突破選股", icon="📡"),
    st.Page("pages/6_WinRate_Matrix.py",    title="四大指數勝率矩陣", icon="🎯"),
    st.Page("pages/7_Position_Sizing.py",  title="加碼比例計算器", icon="📉"),
    st.Page("pages/9_ATR_Pullback_Stats.py", title="ATR 回檔機率", icon="🌡️"),
]
pg = st.navigation(pages)
pg.run()
