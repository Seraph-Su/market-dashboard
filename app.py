import streamlit as st

st.set_page_config(
    page_title="牛市轉折偵測儀表板",
    page_icon="📊",
    layout="wide"
)

pages = [
    st.Page("pages/1_Dashboard.py",    title="大盤儀表板", icon="📊"),
    st.Page("pages/2_EMA_Analysis.py", title="EMA 分析",   icon="📈"),
]

pg = st.navigation(pages)
pg.run()
