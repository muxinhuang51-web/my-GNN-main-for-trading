import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_CONFIG: Dict[str, Any] = {
    "train_window": 20,
    "top_neighbor_count": 0,
    "cluster_count": 20,
    "seed_value": 42,
    "min_cluster_valid_count": 5,
    "min_portfolio_valid_stocks": 50,
    "target_portfolio_valid_stocks": 100,
    "min_market_valid_stocks": 1000,
    "predictor_epochs": 3,
    "kmeans_n_init": 1,
}


DEFAULT_CANDIDATES: List[Dict[str, Any]] = [
    {"name": "c20_t100_tw20_e3_seed42"},
    {"name": "c15_t100_tw20_e3_seed42", "cluster_count": 15},
    {"name": "c20_t200_tw20_e3_seed42", "target_portfolio_valid_stocks": 200},
    {"name": "c10_t100_tw20_e3_seed42", "cluster_count": 10},
    {"name": "c25_t100_tw20_e3_seed42", "cluster_count": 25},
    {"name": "c20_t100_tw20_e3_seed7", "seed_value": 7},
    {"name": "c20_t150_tw20_e3_seed42", "target_portfolio_valid_stocks": 150},
    {"name": "c30_t100_tw20_e3_seed42", "cluster_count": 30},
]


CORR_EDGE_CANDIDATES: List[Dict[str, Any]] = [
    {"name": "c20_t100_corr5_tw20_e3_seed42", "top_neighbor_count": 5},
    {"name": "c20_t100_corr10_tw20_e3_seed42", "top_neighbor_count": 10},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded parameter search for cluster backtest.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="best_model.pt")
    parser.add_argument("--out-root", default="outputs/cluster_backtest_experiments")
    parser.add_argument("--summary-name", default="summary_bounded.json")
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--max-minutes", type=float, default=120.0)
    parser.add_argument("--trial-timeout-minutes", type=float, default=20.0)
    parser.add_argument("--min-days", type=int, default=20)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-corr-edges", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def tail_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-limit:]


def score_metrics(metrics: Dict[str, Any], min_days: int) -> Optional[float]:
    days = int(metrics.get("days") or 0)
    if days < min_days:
        return None

    sharpe = finite_float(metrics.get("sharpe"))
    annualized_return = finite_float(metrics.get("annualized_return"))
    max_drawdown = finite_float(metrics.get("max_drawdown"))
    mean_cluster_ic = finite_float(metrics.get("mean_cluster_ic"))

    # Sharpe is the primary signal. Annual return is squashed so short samples
    # cannot dominate, drawdown is negative and therefore penalizes the score.
    return sharpe + 0.25 * math.tanh(annualized_return) + 2.0 * max_drawdown + 0.5 * mean_cluster_ic


def build_candidates(allow_corr_edges: bool) -> List[Dict[str, Any]]:
    candidates = list(DEFAULT_CANDIDATES)
    if allow_corr_edges:
        candidates.extend(CORR_EDGE_CANDIDATES)
    return candidates


def build_trial_config(
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    config = dict(BASE_CONFIG)
    config.update({key: value for key, value in candidate.items() if key != "name"})
    config.update(
        {
            "data_dir": args.data_dir,
            "model_path": args.model_path,
            "out_dir": str(out_dir),
        }
    )
    if args.start_date:
        config["start_date"] = args.start_date
    if args.end_date:
        config["end_date"] = args.end_date
    return config


def run_trial_subprocess(
    trial_config: Dict[str, Any],
    device_name: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"config": trial_config, "device": device_name}, ensure_ascii=False)
    code = """
import json
import torch
import backtest_cluster as b

payload = json.loads(__PAYLOAD__)
config = payload["config"]
device_name = payload["device"]
if device_name == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_name)
config["device"] = device
_, metrics = b.run_backtest(**config)
print(json.dumps(metrics, ensure_ascii=False))
""".replace("__PAYLOAD__", repr(payload))
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def read_metrics(out_dir: Path) -> Optional[Dict[str, Any]]:
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_summary(path: Path, records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    completed = [record for record in records if record.get("status") == "completed"]
    best = None
    if completed:
        best = max(completed, key=lambda record: record.get("score", float("-inf")))

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "limits": {
            "max_trials": args.max_trials,
            "max_minutes": args.max_minutes,
            "trial_timeout_minutes": args.trial_timeout_minutes,
            "min_days": args.min_days,
            "allow_corr_edges": args.allow_corr_edges,
        },
        "best": best,
        "records": records,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main() -> int:
    args = parse_args()
    if args.max_trials <= 0:
        raise ValueError("--max-trials must be positive")
    if args.max_minutes <= 0:
        raise ValueError("--max-minutes must be positive")
    if args.trial_timeout_minutes <= 0:
        raise ValueError("--trial-timeout-minutes must be positive")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / args.summary_name
    candidates = build_candidates(args.allow_corr_edges)
    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    attempted_trials = 0

    for candidate in candidates:
        if attempted_trials >= args.max_trials:
            break
        elapsed_minutes = (time.perf_counter() - started) / 60.0
        if elapsed_minutes >= args.max_minutes:
            records.append(
                {
                    "name": candidate["name"],
                    "status": "skipped_budget_exhausted",
                    "elapsed_minutes": elapsed_minutes,
                }
            )
            break

        out_dir = out_root / candidate["name"]
        trial_config = build_trial_config(candidate, args, out_dir)
        record: Dict[str, Any] = {
            "name": candidate["name"],
            "config": trial_config,
            "out_dir": str(out_dir),
            "status": "planned",
        }

        if args.dry_run:
            record["status"] = "dry_run"
            records.append(record)
            attempted_trials += 1
            continue

        print(f"[优化] 开始 trial={candidate['name']} out_dir={out_dir}")
        attempted_trials += 1
        trial_started = time.perf_counter()
        timeout_seconds = args.trial_timeout_minutes * 60.0
        try:
            completed = run_trial_subprocess(trial_config, args.device, timeout_seconds)
            record["returncode"] = completed.returncode
            record["stdout_tail"] = tail_text(completed.stdout)
            record["stderr_tail"] = tail_text(completed.stderr)
            metrics = read_metrics(out_dir)
            if completed.returncode == 0 and metrics is not None:
                score = score_metrics(metrics, args.min_days)
                record["status"] = "completed" if score is not None else "completed_too_few_days"
                record["metrics"] = metrics
                record["score"] = score
            else:
                record["status"] = "failed"
                record["metrics"] = metrics
        except subprocess.TimeoutExpired as error:
            record["status"] = "timeout"
            record["stdout_tail"] = tail_text(error.stdout)
            record["stderr_tail"] = tail_text(error.stderr)

        record["duration_seconds"] = round(time.perf_counter() - trial_started, 3)
        records.append(record)
        write_summary(summary_path, records, args)
        print(f"[优化] 结束 trial={candidate['name']} status={record['status']} score={record.get('score')}")

    write_summary(summary_path, records, args)
    print(f"[优化] 汇总已保存: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
