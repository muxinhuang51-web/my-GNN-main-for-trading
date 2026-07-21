"""全链自驱动流水线：拉取 → 面板 → 校验 → T1(csi300+csi500) → T2(全市场) → 总汇总。

每个阶段幂等（断点续传/已有结果跳过），任一阶段失败即停链并保留日志。
用法：python -u -m exp.pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOG_DIR = ROOT / "outputs_v2" / "pipeline_logs"


def run_stage(name: str, args: list) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    print(f"\n===== [{time.strftime('%H:%M:%S')}] 阶段开始: {name} =====", flush=True)
    with log_path.open("a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} =====\n")
        log.flush()
        proc = subprocess.run([PY, "-u", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[流水线] 阶段 {name} 失败 (exit {proc.returncode})，日志: {log_path}", flush=True)
        raise SystemExit(proc.returncode)
    print(f"===== [{time.strftime('%H:%M:%S')}] 阶段完成: {name} =====", flush=True)


def main() -> int:
    t0 = time.time()
    run_stage("fetch", ["-m", "data_layer", "fetch"])
    run_stage("panel", ["-m", "data_layer", "panel"])
    run_stage("validate", ["-m", "data_layer", "validate"])

    for scope in ("csi300", "csi500"):
        run_stage(f"{scope}_encoders", ["-m", "exp.run_t1", "--scope", scope, "--stage", "encoders"])
        run_stage(f"{scope}_sweep", ["-m", "exp.run_t1", "--scope", scope, "--stage", "sweep"])
        run_stage(f"{scope}_baselines", ["-m", "exp.run_t1", "--scope", scope, "--stage", "baselines"])
        run_stage(f"{scope}_collect", ["-m", "exp.run_t1", "--scope", scope, "--stage", "collect"])

    run_stage("all_encoders", ["-m", "exp.run_t1", "--scope", "all", "--stage", "encoders", "--coarse"])
    run_stage("all_sweep", ["-m", "exp.run_t1", "--scope", "all", "--stage", "sweep", "--coarse"])
    run_stage("all_baselines", ["-m", "exp.run_t1", "--scope", "all", "--stage", "baselines", "--coarse"])
    run_stage("all_collect", ["-m", "exp.run_t1", "--scope", "all", "--stage", "collect", "--coarse"])

    hours = (time.time() - t0) / 3600
    print(f"\n[流水线] 全部完成，总耗时 {hours:.1f} 小时", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
