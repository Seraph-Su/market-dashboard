"""
breakout_screener.py
--------------------
每日掃描股票，偵測「EMA收斂後放量突破」訊號，輸出 CSV。

核心邏輯（兩個條件同時滿足其一即觸發）：
  A. K棒條件：前N天盤整幅度 < X%，今日收盤突破最高點，且量比 > 門檻
  B. EMA條件：近期三線（月/季/年 EMA）曾高度收斂，今日收盤突破 EMA20，且量比 > 門檻

使用方式：
    python breakout_screener.py                  # 掃描預設清單 (Nasdaq 100)
    python breakout_screener.py --tickers PANW NVDA APP
    python breakout_screener.py --list sp500

輸出：
    signals_YYYY-MM-DD.csv  (與腳本同目錄)
"""

import argparse
import datetime
import os

import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────
# 參數設定
# ─────────────────────────────────────────────
DEFAULTS = dict(
    # K棒盤整條件
    consolidation_days   = 14,    # 盤整期長度（交易日）
    consolidation_range  = 0.16,  # 盤整幅度上限（16%）
    breakout_buffer      = 0.003, # 收盤需超過最高點 0.3%

    # EMA收斂條件
    ema_spread_threshold = 0.05,  # 三線最大收斂度（spread < 5% 視為收斂）
    ema_lookback         = 20,    # 往前看幾天內曾收斂過

    # 共用
    volume_mult          = 1.3,   # 成交量需 > 均量 × 1.3
    lookback_days        = 400,   # 下載歷史天數（EMA260 需要足夠資料）
    cooldown_days        = 10,    # 同支股票兩次訊號最短間隔（交易日）
)

# 預設 Nasdaq 100 精簡清單（前50大）
DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","PEP","CSCO","ADBE","INTC","CMCSA","HON","AMGN",
    "TXN","QCOM","INTU","AMAT","ISRG","BKNG","ADP","SBUX","GILD","MU",
    "LRCX","REGN","ADI","PANW","KLAC","MDLZ","SNPS","CDNS","MELI","FTNT",
    "CTAS","CSX","PAYX","ORLY","MRVL","IDXX","ROST","CPRT","PCAR","KDP",
]


def get_sp500_tickers() -> list[str]:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_data(tickers: list[str], lookback: int) -> dict[str, pd.DataFrame]:
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=lookback + 60)
    print(f"[*] 下載 {len(tickers)} 支股票資料…")
    raw = yf.download(
        tickers, start=str(start), end=str(end),
        group_by="ticker", auto_adjust=True, progress=False, threads=True,
    )
    result = {}
    for t in tickers:
        try:
            df = raw.copy() if len(tickers) == 1 else raw[t].copy()
            df = df.dropna(subset=["Close"])
            if len(df) >= 60:
                result[t] = df
        except Exception:
            pass
    return result


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增 EMA20/60/260 與收斂度欄位。
    - 歷史 >= 300 天：三線收斂（月/季/年）
    - 歷史 < 300 天（新股）：雙線收斂（月/季），並標記 ema_mode
    """
    df = df.copy()
    df["ema20"]  = df["Close"].ewm(span=20,  adjust=False).mean()
    df["ema60"]  = df["Close"].ewm(span=60,  adjust=False).mean()
    df["ema260"] = df["Close"].ewm(span=260, adjust=False).mean()

    if len(df) >= 300:
        # 三線收斂
        cols = ["ema20","ema60","ema260"]
        df["ema_mode"] = "3ema"
    else:
        # 新股：只用 EMA20 & EMA60
        cols = ["ema20","ema60"]
        df["ema_mode"] = "2ema"

    df["ema_spread"] = (
        df[cols].max(axis=1) - df[cols].min(axis=1)
    ) / df[cols].median(axis=1)
    return df


def detect_breakout(df: pd.DataFrame, cfg: dict) -> dict | None:
    """
    判斷最新交易日是否出現突破訊號。
    條件 A（K棒盤整）或 條件 B（EMA收斂）滿足其一即回傳訊號。
    """
    n      = cfg["consolidation_days"]
    rng    = cfg["consolidation_range"]
    buf    = cfg["breakout_buffer"]
    vmult  = cfg["volume_mult"]
    spread_thr  = cfg["ema_spread_threshold"]
    ema_lb      = cfg["ema_lookback"]

    min_rows = max(n + 2, 65)   # 新股至少需要 EMA60 穩定
    if len(df) < min_rows:
        return None

    df = add_emas(df)
    today        = df.iloc[-1]
    consolidation = df.iloc[-(n + 1):-1]

    close    = float(today["Close"])
    avg_vol  = float(consolidation["Volume"].mean())
    if avg_vol == 0:
        return None
    vol_ratio = float(today["Volume"]) / avg_vol

    # ── 條件 A：K棒盤整突破 ──────────────────────
    signal_type = None
    consol_high = float(consolidation["High"].max())
    consol_low  = float(consolidation["Low"].min())
    consol_range_pct = (consol_high - consol_low) / consol_low

    if (consol_range_pct <= rng
            and close > consol_high * (1 + buf)
            and vol_ratio >= vmult):
        signal_type = "K棒盤整突破"

    # ── 條件 B：EMA收斂後突破（三線或雙線）──────────
    if signal_type is None:
        recent        = df.iloc[-(ema_lb + 1):-1]
        recent_spread = recent["ema_spread"]
        prev_close    = float(df.iloc[-2]["Close"])
        day_gain      = (close - prev_close) / prev_close

        # 三線收斂（原有邏輯）
        ema_was_tight = recent_spread.min() < spread_thr

        # 雙線收斂（EMA20/60，適用於 EMA260 因前段大漲而位移的情況）
        spread2 = abs(df["ema20"] - df["ema60"]) / df[["ema20","ema60"]].mean(axis=1)
        ema2_was_tight = spread2.iloc[-(ema_lb + 1):-1].min() < spread_thr

        recent_high       = float(recent["High"].max())
        broke_recent_high = close > recent_high * (1 + buf)
        big_move          = day_gain > 0.03

        if (ema_was_tight or ema2_was_tight) and broke_recent_high and big_move and vol_ratio >= vmult:
            signal_type = "EMA收斂突破" if ema_was_tight else "EMA雙線收斂突破"

    if signal_type is None:
        return None

    return {
        "date":             str(df.index[-1].date()),
        "signal_type":      signal_type,
        "ema_mode":         str(today["ema_mode"]),
        "close":            round(close, 2),
        "ema20":            round(float(today["ema20"]), 2),
        "ema_spread":       round(float(today["ema_spread"]), 4),
        "min_spread_20d":   round(float(df["ema_spread"].iloc[-ema_lb:].min()), 4),
        "consol_high":      round(consol_high, 2),
        "consol_range_pct": round(consol_range_pct * 100, 2),
        "breakout_pct":     round((close / consol_high - 1) * 100, 2),
        "volume_ratio":     round(vol_ratio, 2),
    }


def run(tickers: list[str], cfg: dict, output_dir: str = ".") -> pd.DataFrame:
    data    = fetch_data(tickers, cfg["lookback_days"])
    signals = []

    for ticker, df in data.items():
        result = detect_breakout(df, cfg)
        if result:
            result["ticker"] = ticker
            signals.append(result)
            print(f"  ✓ {ticker:6s}  [{result['signal_type']}]  "
                  f"close={result['close']}  "
                  f"breakout={result['breakout_pct']:+.2f}%  "
                  f"vol={result['volume_ratio']:.1f}x  "
                  f"ema_spread={result['ema_spread']:.3f}")

    cols = ["ticker","date","signal_type","ema_mode","close","ema20","ema_spread",
            "min_spread_20d","consol_high","consol_range_pct","breakout_pct","volume_ratio"]
    out_df = (pd.DataFrame(signals)[cols].sort_values("breakout_pct", ascending=False)
              if signals else pd.DataFrame())

    date_str = datetime.date.today().strftime("%Y-%m-%d")
    out_path = os.path.join(output_dir, f"signals_{date_str}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[+] 找到 {len(signals)} 個訊號 → {out_path}")
    return out_df


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EMA收斂突破選股器")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--list", choices=["sp500","nasdaq100"], default="nasdaq100")
    parser.add_argument("--consol-days",   type=int,   default=DEFAULTS["consolidation_days"])
    parser.add_argument("--consol-range",  type=float, default=DEFAULTS["consolidation_range"])
    parser.add_argument("--ema-spread",    type=float, default=DEFAULTS["ema_spread_threshold"],
                        help="EMA收斂門檻（預設 0.05）")
    parser.add_argument("--vol-mult",      type=float, default=DEFAULTS["volume_mult"])
    parser.add_argument("--output-dir",    default=".")
    args = parser.parse_args()

    tickers = (
        [t.upper() for t in args.tickers] if args.tickers
        else (get_sp500_tickers() if args.list == "sp500" else DEFAULT_TICKERS)
    )

    cfg = {**DEFAULTS,
           "consolidation_days":  args.consol_days,
           "consolidation_range": args.consol_range,
           "ema_spread_threshold": args.ema_spread,
           "volume_mult":          args.vol_mult}

    print(f"[*] 參數: 盤整={cfg['consolidation_days']}天 幅度<{cfg['consolidation_range']*100:.0f}% "
          f"EMA收斂<{cfg['ema_spread_threshold']*100:.0f}% 量比>{cfg['volume_mult']}x")
    run(tickers, cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
