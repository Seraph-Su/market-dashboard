import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# 🏠 首頁：每日三分鐘
#   照規則書「看燈號 → 看持倉停損 → 看名單 🔔」的順序排三張卡，每張卡直接連到對應頁面。
#   放在 pages/00_Home.py；app.py 設為 default 頁。
# ═══════════════════════════════════════════════════════════════════

st.markdown("## 📈 美股儀表板")
st.markdown("<span style='color:#64748b;font-size:0.85rem'>順勢交易系統：宇宙決定「什麼」、新高決定「何時」、"
            "R 決定「多大」、前波低點決定「何時走」、曝險決定「還能不能再開」。</span>", unsafe_allow_html=True)
st.markdown("---")

CARD = ("<div style='padding:18px 20px;border-radius:12px;background:#0f172a;border:1px solid #1e293b;"
        "min-height:150px'>"
        "<div style='font-size:0.72rem;color:#64748b;letter-spacing:.08em'>{step}</div>"
        "<div style='font-size:1.25rem;font-weight:700;color:#e2e8f0;margin:4px 0 8px'>{title}</div>"
        "<div style='font-size:0.82rem;color:#94a3b8;line-height:1.6'>{desc}</div></div>")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(CARD.format(step="STEP ①　進場時機", title="🚦 今天能不能開新倉？",
                            desc="領頭股燈號（≥5/8 綠）＋壓力否決（CME/SPY 與 XLP/XLY 雙亮才擋）。"
                                 "紅燈或否決成立 → 名單只看不做。"), unsafe_allow_html=True)
    st.page_link("pages/1_Dashboard.py", label="大盤壓力儀表板", icon="📊")
    st.page_link("pages/6_WinRate_Matrix.py", label="四大指數勝率矩陣", icon="🎯")
with c2:
    st.markdown(CARD.format(step="STEP ②　選股", title="🦖 今天有沒有 🔔 觸發？",
                            desc="怪物股：半年 >150% 創 63 日新高。修復股：年線下站回多頭排列的產業龍頭。"
                                 "只有這兩條路，錯過觸發日就等下一次。"), unsafe_allow_html=True)
    st.page_link("pages/10_Monster_Screener.py", label="怪物股選股器", icon="🦖")
    st.page_link("pages/5_Breakout_Screener.py", label="均線收斂突破選股（含修復觸發）", icon="📡")
with c3:
    st.markdown(CARD.format(step="STEP ③　加碼＆倉位管理", title="🌡️ 還能開幾檔？該加碼嗎？",
                            desc="總曝險 ≥ −9% 才開新倉；持倉打到停損就走。"
                                 "A 回季線／T 機械加碼每天自動檢測，ATR% >7% 不加。"), unsafe_allow_html=True)
    st.page_link("pages/11_Exposure.py", label="總曝險計算器", icon="🌡️")
    st.page_link("pages/13_ATR_Add.py", label="ATR 加碼計算器", icon="🎯")

st.markdown("---")
st.page_link("pages/0_Trading_SOP.py", label="交易 SOP 流程圖", icon="🗺️")
st.markdown("<div style='color:#475569;font-size:0.75rem'>"
            "每日三分鐘：看燈號 → 看持倉有沒有打到停損 → 看名單有沒有 🔔。沒事就關掉。<br>"
            "禁止：紅燈開新倉｜停損收得比結構緊｜為降曝險把停損移到結構外｜看上一檔股票之後回頭改規則｜系統外的進場。"
            "</div>", unsafe_allow_html=True)
