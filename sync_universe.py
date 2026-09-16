#!/usr/bin/env python3
"""
怪物股宇宙名單：抓取 → 直接 commit 到 GitHub。

背景：Yahoo 封鎖 Streamlit Community Cloud 那段 IP 的 screener endpoint
（401 "User is unable to access this feature"），所以網頁端拿不到宇宙名單，
但沙盒／本機打得通。這支腳本把名單抓好後用 GitHub Contents API 直接寫進 repo，
Community Cloud 偵測到 commit 會自動重新部署，選股器就讀得到最新名單。

Token：
  只從環境變數 GITHUB_TOKEN 或 .gh_token 檔讀取，腳本不會印出、不會寫進任何輸出。
  建議用 fine-grained PAT，範圍只給 seraph-su/market-dashboard 這一個 repo，
  權限只勾 Contents: Read and write，並設到期日。

用法：
  python3 sync_universe.py --local    # 抓 → 寫到 ./data/monster_universe.json（GitHub Actions 用，之後由 git 提交）
  python3 sync_universe.py            # 抓 → 用 GitHub API 直接推（本機手動用，需要 .gh_token）
  python3 sync_universe.py --dry-run  # 只抓不推，印出摘要

正式排程走 GitHub Actions（.github/workflows/universe.yml）：每個交易日收盤後在 GitHub 的機器上跑
--local 模式，用 GitHub 內建身分 commit，不需要任何 PAT、不經過 Cowork。
"""
import base64
import datetime as dt
import glob
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

import yfinance as yf
from yfinance import EquityQuery as Q

REPO = "seraph-su/market-dashboard"
BRANCH = "main"
PATH = "data/monster_universe.json"

MIN_CAP_B = 1.0      # 粗篩最寬門檻；網頁端再依使用者選的市值過濾
MIN_52W = 50.0       # 52 週漲幅 >50%：半年 >150% 的必要條件近似
EXCH_OK = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS"}
DRY = "--dry-run" in sys.argv
LOCAL = "--local" in sys.argv      # 寫檔不推：交給 GitHub Actions 的 git commit


def get_token() -> str:
    """環境變數優先；否則找任何已連結資料夾裡的 .gh_token。取不到就結束。

    ⚠️ 不要放在 outputs：那個資料夾跟著每次對話走，排程隔天會找不到。
       要放在固定的連結資料夾（例：~/美股儀表板/.gh_token）。"""
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    for p in sorted(glob.glob("/sessions/*/mnt/*/.gh_token")):
        try:
            tok = open(p, encoding="utf-8").read().strip()
            if tok:
                return tok
        except Exception:
            pass
    raise SystemExit("找不到 GitHub token：請在 outputs 資料夾建立 .gh_token（內容只有 token 一行）。")


def api(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "market-dashboard-universe-sync",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def build() -> list:
    q = Q("and", [Q("gt", ["intradaymarketcap", MIN_CAP_B * 1e9]),
                  Q("gt", ["fiftytwowkpercentchange", MIN_52W]),
                  Q("eq", ["region", "us"])])
    rows, off = [], 0
    while off < 3000:
        r = None
        for attempt in range(3):
            try:
                r = yf.screen(q, size=250, offset=off,
                              sortField="intradaymarketcap", sortAsc=False)
                break
            except Exception as e:
                print(f"  offset {off} 第 {attempt + 1} 次失敗：{str(e)[:120]}")
                time.sleep(3 * (attempt + 1))
        if r is None:
            raise SystemExit("screener 連續三次失敗，這次不更新名單（保留 repo 裡的舊檔）。")
        qs = r.get("quotes", [])
        rows += qs
        off += 250
        if not qs or off >= r.get("total", 0):
            break
        time.sleep(1)

    keep, seen = [], set()
    for x in rows:
        t = x.get("symbol", "")
        if x.get("quoteType") != "EQUITY" or x.get("exchange") not in EXCH_OK:
            continue
        if not t or t in seen:
            continue
        seen.add(t)
        keep.append({"t": t,
                     "cap": round((x.get("marketCap") or 0) / 1e9, 3),
                     "name": (x.get("shortName") or "")[:40]})
    if len(keep) < 100:
        raise SystemExit(f"只取得 {len(keep)} 檔，明顯不完整，這次不更新（保留 repo 裡的舊檔）。")
    return keep


def main():
    print("向 Yahoo 篩選器要名單…")
    keep = build()
    caps = [x["cap"] for x in keep]
    payload = json.dumps({"built_at": dt.datetime.now().isoformat(timespec="seconds"),
                          "min_cap_b": MIN_CAP_B, "min_52w": MIN_52W, "rows": keep},
                         ensure_ascii=False, indent=1)
    print(f"完成：{len(keep)} 檔 "
          f"（>10B {sum(1 for c in caps if c >= 10)}／2-10B {sum(1 for c in caps if 2 <= c < 10)}"
          f"／1-2B {sum(1 for c in caps if c < 2)}）")

    if DRY:
        print(f"[dry-run] 不推送。內容 {len(payload.encode()) / 1024:.0f} KB")
        return

    if LOCAL:
        out = pathlib.Path(__file__).resolve().parent / PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        # 名單沒變就不覆寫（built_at 也不動），讓 git 看到「沒有變更」→ 不產生空 commit
        if out.exists():
            try:
                if json.loads(out.read_text(encoding="utf-8")).get("rows") == keep:
                    print("名單與現有檔案相同 → 不覆寫，git 不會有變更。")
                    return
            except Exception:
                pass
        out.write_text(payload, encoding="utf-8")
        print(f"✅ 已寫入 {out}（{len(payload.encode()) / 1024:.0f} KB），等 git commit。")
        return

    token = get_token()
    url = f"https://api.github.com/repos/{REPO}/contents/{PATH}"
    sha = None
    try:
        cur = api("GET", f"{url}?ref={BRANCH}", token)
        sha = cur.get("sha")
        old = base64.b64decode(cur.get("content", "")).decode("utf-8", "ignore")
        # built_at 每次都不同，只比對名單本身，沒變就不推（避免無意義的重新部署）
        try:
            if json.loads(old).get("rows") == keep:
                print("名單與 repo 上的完全相同 → 略過推送，Community Cloud 不會重新部署。")
                return
        except Exception:
            pass
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        print("repo 上還沒有這個檔案 → 建立新檔。")

    body = {"message": f"universe {dt.date.today()} ({len(keep)} 檔)",
            "content": base64.b64encode(payload.encode()).decode(),
            "branch": BRANCH}
    if sha:
        body["sha"] = sha
    res = api("PUT", url, token, body)
    print(f"✅ 已推送：{res['commit']['sha'][:7]} → {REPO}@{BRANCH}:{PATH}")
    print("Community Cloud 會自動重新部署，約一分鐘後生效。")


if __name__ == "__main__":
    main()
