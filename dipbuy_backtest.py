"""
TQQQ 分批低接回測引擎（供 pages/14_TQQQ_DipBuy.py 使用，也可獨立執行）

訊號用 QQQ（TQQQ 的標的），下單買 TQQQ（或 QQQ 作對照）。
2010-02 之前 TQQQ 不存在 → 以「3 × QQQ 日報酬 − 費用 − 2 倍借款成本」合成，
校準結果：2010~2026 合成 CAGR 與真實 TQQQ 相差 <0.1%/年。

策略邏輯（一次「事件」）：
  1. 觸發：QQQ 距參考高點（52週高 或 歷史高）回撤 ≥ start，且通過 VIX / 年線過濾
     → 鎖定該參考高點 peak，資金池平分（或金字塔）成 N 批
  2. 第 k 批掛在 QQQ = peak × (1 − start − (k−1)×step)，收盤價觸及即成交（跳空可一次成交多批）
  3. 出場（任一成立）：QQQ 回到 peak（解套）／ 持倉報酬達 target ／ 首批後滿 hold 天 ／ 停損
  4. 出場後需 QQQ 回撤縮小到 start/2 以內才重新武裝（避免同一波跌勢連續觸發）
  未投入的現金以 13 週國庫券利率（^IRX）計息。
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

TQQQ_FEE = 0.0095      # TQQQ 費用率
BORROW_SPREAD = 0.005  # 借款利差（校準後最貼近真實 TQQQ）


# ── 資料 ─────────────────────────────────────────────────────────────────
def build_frame(qqq: pd.Series, tqqq: pd.Series, vix: pd.Series, irx: pd.Series) -> pd.DataFrame:
    """qqq 需為還原權值收盤價；回傳含合成/真實拼接 TQQQ 的日資料。"""
    q = qqq.dropna()
    rf = (irx.reindex(q.index).ffill().bfill().fillna(0) / 100).clip(lower=0)
    rq = q.pct_change().fillna(0)
    synth = (1 + 3 * rq - (TQQQ_FEE + 2 * (rf + BORROW_SPREAD)) / 252).cumprod()

    t = tqqq.dropna()
    t = t[t.index >= q.index[0]]
    if len(t):
        first = t.index[0]
        # 以真實上市日接合：合成序列縮放到真實首日價，其後用真實價
        scaled = synth / synth.loc[first] * t.iloc[0]
        tq = pd.concat([scaled[scaled.index < first], t]).reindex(q.index).ffill()
    else:
        tq = synth

    df = pd.DataFrame({
        "qqq": q,
        "tqqq": tq,
        "vix": vix.reindex(q.index).ffill(),
        "rf": rf,
    })
    df["ema60"] = df["qqq"].ewm(span=60, adjust=False).mean()
    df["ema260"] = df["qqq"].ewm(span=260, adjust=False).mean()
    df["hi252"] = df["qqq"].rolling(252, min_periods=1).max()
    df["ath"] = df["qqq"].cummax()
    df["is_synth"] = df.index < (t.index[0] if len(t) else df.index[-1] + pd.Timedelta(days=1))
    return df.dropna(subset=["qqq", "tqqq"])


def load_yf(period: str = "9999d") -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(["QQQ", "TQQQ", "^VIX", "^IRX"], period=period, interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    raw.columns = [c[-1] if isinstance(c, tuple) else str(c) for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return build_frame(raw["QQQ"], raw["TQQQ"], raw["^VIX"], raw["^IRX"])


# ── 參數 ─────────────────────────────────────────────────────────────────
@dataclass
class Params:
    ref: str = "hi252"           # 回撤參考高點："hi252"（52週高）或 "ath"（歷史高）
    start: float = 0.10          # 第一批：QQQ 回撤 ≥ 10%
    n: int = 3                   # 分幾批
    step: float = 0.05           # 每批間隔：QQQ 再跌 5%
    weights: str = "equal"       # "equal" 平均 ／ "pyramid" 越跌買越多（1:2:3…）
    exit: str = "recover"        # "recover" QQQ 回前高 ／ "target" 獲利目標 ／ "hold" 固定天數
    target: float = 0.5          # exit="target"：TQQQ 持倉報酬 ≥ 50% 出場
    hold_days: int = 252         # exit="hold"：首批後持有天數
    max_days: int = 0            # 時間停損（交易日，0=不設）
    stop_loss: float = 0.0       # 持倉報酬 ≤ −stop_loss 停損（0=不設）
    vix_min: float = 0.0         # 觸發時 VIX 需 ≥（0=不過濾）
    trend: str = "any"           # "any" ／ "above"：觸發時 QQQ 需在 EMA260 之上
    asset: str = "tqqq"          # "tqqq" 或 "qqq"（對照組）

    def tranche_weights(self):
        w = np.arange(1, self.n + 1, dtype=float) if self.weights == "pyramid" else np.ones(self.n)
        return w / w.sum()


# ── 回測 ─────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame, p: Params):
    """回傳 (equity: pd.Series, episodes: pd.DataFrame)。equity 以資金池 = 1 起算。"""
    qqq = df["qqq"].to_numpy()
    px = df[p.asset].to_numpy()
    ref = df[p.ref].to_numpy()
    vix = df["vix"].to_numpy()
    ema260 = df["ema260"].to_numpy()
    rf_d = df["rf"].to_numpy() / 252
    dates = df.index
    w = p.tranche_weights()

    cash, shares, cost = 1.0, 0.0, 0.0
    in_ep, armed = False, True
    eq = np.empty(len(df))
    eps = []
    peak = budget = 0.0
    filled = 0
    i0 = 0
    ep_peak_eq = ep_mdd = 0.0
    min_dd = 0.0

    for i in range(len(df)):
        cash *= 1 + rf_d[i]
        dd = qqq[i] / ref[i] - 1

        if not in_ep:
            if not armed and dd > -p.start / 2:
                armed = True
            if (armed and dd <= -p.start
                    and vix[i] >= p.vix_min
                    and (p.trend == "any" or qqq[i] > ema260[i])):
                in_ep, armed = True, False
                peak, budget, filled = ref[i], cash, 0
                i0, ep_peak_eq, ep_mdd, min_dd = i, cash, 0.0, dd

        if in_ep:
            ep_dd = qqq[i] / peak - 1
            min_dd = min(min_dd, ep_dd)
            while filled < p.n and ep_dd <= -(p.start + filled * p.step) + 1e-12:
                amt = min(budget * w[filled], cash)
                shares += amt / px[i]
                cost += amt
                cash -= amt
                filled += 1

            val = shares * px[i]
            days = i - i0
            pos_ret = val / cost - 1 if cost > 0 else 0.0
            reason = None
            if p.exit == "recover" and qqq[i] >= peak:
                reason = "回前高"
            elif p.exit == "target" and pos_ret >= p.target:
                reason = "達標"
            elif p.exit == "hold" and days >= p.hold_days:
                reason = "持滿"
            if reason is None and p.stop_loss > 0 and pos_ret <= -p.stop_loss:
                reason = "停損"
            if reason is None and p.max_days > 0 and days >= p.max_days:
                reason = "時間停損"
            if reason is None and i == len(df) - 1:
                reason = "未結束"

            tot = cash + val
            ep_peak_eq = max(ep_peak_eq, tot)
            ep_mdd = min(ep_mdd, tot / ep_peak_eq - 1)

            if reason:
                eps.append({
                    "觸發日": dates[i0].date(), "出場日": dates[i].date(), "天數": days,
                    "批數": filled, "投入比例": cost / budget,
                    "持倉報酬": pos_ret, "資金池報酬": tot / budget - 1,
                    "期間最大回撤": ep_mdd, "QQQ最深回撤": min_dd,
                    "出場原因": reason, "合成資料": bool(df["is_synth"].iloc[i0]),
                })
                cash, shares, cost, in_ep = tot, 0.0, 0.0, False
                val = 0.0
        else:
            val = 0.0
        eq[i] = cash + val

    return pd.Series(eq, index=dates), pd.DataFrame(eps)


def summarize(equity: pd.Series, eps: pd.DataFrame) -> dict:
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / yrs) - 1 if equity.iloc[-1] > 0 else -1.0
    mdd = float((equity / equity.cummax() - 1).min())
    done = eps[eps["出場原因"] != "未結束"] if len(eps) else eps
    r = done["資金池報酬"] if len(done) else pd.Series(dtype=float)
    return {
        "CAGR": cagr, "最大回撤": mdd, "期末倍數": float(equity.iloc[-1]),
        "事件數": len(done), "勝率": float((r > 0).mean()) if len(r) else np.nan,
        "平均報酬": float(r.mean()) if len(r) else np.nan,
        "中位報酬": float(r.median()) if len(r) else np.nan,
        "最差事件": float(r.min()) if len(r) else np.nan,
        "平均天數": float(done["天數"].mean()) if len(done) else np.nan,
        "平均投入": float(done["投入比例"].mean()) if len(done) else np.nan,
    }


def benchmark(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    return s / s.iloc[0]


if __name__ == "__main__":
    data = load_yf()
    eq, ep = run(data, Params())
    print(summarize(eq, ep))
    print(ep.to_string())
