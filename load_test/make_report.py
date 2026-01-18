import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}. Available: {list(df.columns)}")


def safe_float(v, default=0.0) -> float:
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-prefix", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--repo-id", type=int, default=0)
    ap.add_argument("--vus", type=int, default=0)
    ap.add_argument("--duration", default="")
    args = ap.parse_args()

    prefix = Path(args.csv_prefix)
    out_dir = Path(args.out_dir) if args.out_dir else prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = pd.read_csv(str(prefix) + "_stats.csv")
    hist = pd.read_csv(str(prefix) + "_stats_history.csv")

    row = stats[(stats["Name"] == "/chat") & (stats["Type"] == "POST")]
    if row.empty:
        row = stats[(stats["Name"] == "/chat")]
    if row.empty:
        raise ValueError("Could not find /chat row in stats csv")
    row = row.iloc[0]

    total = int(row.get("Request Count", 0) or 0)
    fails = int(row.get("Failure Count", 0) or 0)
    fail_rate = (fails / total) if total else 0.0

    p50 = safe_float(row.get("50%", row.get("Median Response Time", 0)))
    p95 = safe_float(row.get("95%", 0))
    p99 = safe_float(row.get("99%", 0))
    avg = safe_float(row.get("Average Response Time", 0))
    rps_avg = safe_float(row.get("Requests/s", 0))

    ts_col = pick_col(hist, ["Timestamp"])
    rps_col = pick_col(hist, ["Requests/s"])
    failps_col = pick_col(hist, ["Failures/s"])
    lat_col = pick_col(hist, ["Total Average Response Time", "Total Median Response Time"])

    x = pd.to_datetime(hist[ts_col], unit="s", errors="coerce")

    plt.figure()
    plt.plot(x, hist[rps_col])
    plt.xlabel("time")
    plt.ylabel("RPS")
    plt.title("Throughput (Requests/s)")
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_rps.png")
    plt.close()

    plt.figure()
    plt.plot(x, hist[failps_col])
    plt.xlabel("time")
    plt.ylabel("Failures/s")
    plt.title("Errors per second")
    plt.tight_layout()
    plt.savefig(out_dir / "errors_per_sec.png")
    plt.close()

    plt.figure()
    plt.plot(x, hist[lat_col])
    plt.xlabel("time")
    plt.ylabel("ms")
    plt.title(f"Latency over time ({lat_col})")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_avg_ms.png")
    plt.close()

    md = f"""## Load testing: /chat (steady-state)

### Setup
- Base URL: `{args.base_url}`
- repo_id: `{args.repo_id}`
- VUs: **{args.vus}**
- Duration: **{args.duration}**
- Workload: closed-loop (Locust users with think-time)

### Results (overall, /chat)
- Total requests: **{total}**
- RPS avg: **{round(rps_avg, 3)}**
- Errors: **{fails}** (**{round(fail_rate * 100, 2)}%**)
- Latency (ms): p50 **{round(p50, 2)}**, p95 **{round(p95, 2)}**, p99 **{round(p99, 2)}**
- Avg latency (ms): **{round(avg, 2)}**

### Charts
![Throughput](throughput_rps.png)
![Errors](errors_per_sec.png)
![Latency](latency_avg_ms.png)
"""
    (out_dir / "design_doc_load_test_section.md").write_text(md, encoding="utf-8")
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
