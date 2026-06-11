"""Dispatch NOCS LOSO runs without DDP."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_loso_protocols(split_json: Path) -> list[str]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    protocols = [
        split["protocol"]
        for split in data.get("splits", [])
        if str(split.get("protocol", "")).startswith("Y16_LOSO_")
    ]
    if not protocols:
        raise RuntimeError("No Y16_LOSO_* protocols found in %s" % split_json)
    return protocols


def parse_gpus(gpus: str) -> list[str]:
    return [item.strip() for item in gpus.split(",") if item.strip()]


def build_command(args: argparse.Namespace, protocol: str, seed: int, device: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.adagtcn_aligned.nocs.train_nocs",
        "--sequence-jsonl",
        str(args.sequence_jsonl),
        "--split-json",
        str(args.split_json),
        "--protocol",
        protocol,
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-dir",
        str(args.output_dir),
        "--ablation",
        args.ablation,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--max-len",
        str(args.max_len),
        "--stat-head",
        args.stat_head,
        "--residual-beta",
        str(args.residual_beta),
        "--lambda-residual-norm",
        str(args.lambda_residual_norm),
        "--lambda-gate",
        str(args.lambda_gate),
    ]
    if args.cache_records:
        cmd.append("--cache-records")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NOCS LOSO jobs, one process per GPU.")
    parser.add_argument("--sequence-jsonl", type=Path, default=Path("data/adagtcn_aligned/zuco_word_band_gaze_sequences.jsonl"))
    parser.add_argument("--split-json", type=Path, default=Path("reports/adagtcn_aligned/subject_splits.json"))
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path("outputs/nocs"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--gpus", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--stat-head", "--stat_head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--residual-beta", "--residual_beta", type=float, default=0.3)
    parser.add_argument("--lambda-residual-norm", "--lambda_residual_norm", type=float, default=0.01)
    parser.add_argument("--lambda-gate", "--lambda_gate", type=float, default=0.001)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protocols", nargs="*", default=None)
    parser.add_argument("--cache-records", action="store_true")
    args = parser.parse_args()

    protocols = args.protocols if args.protocols else load_loso_protocols(args.split_json)
    gpus = parse_gpus(args.gpus)
    slots = gpus if gpus else ["cpu"]
    tasks = [(protocol, seed) for seed in args.seeds for protocol in protocols]
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for idx, (protocol, seed) in enumerate(tasks):
            gpu = slots[idx % len(slots)]
            device = args.device if gpu != "cpu" else "cpu"
            cmd = build_command(args, protocol, seed, device)
            prefix = "CUDA_VISIBLE_DEVICES=%s " % gpu if gpu != "cpu" else ""
            print(prefix + " ".join(cmd))
        return

    failed = []
    for start in range(0, len(tasks), len(slots)):
        chunk = tasks[start : start + len(slots)]
        running: list[dict[str, object]] = []
        for slot_idx, (protocol, seed) in enumerate(chunk):
            gpu = slots[slot_idx]
            device = args.device if gpu != "cpu" else "cpu"
            cmd = build_command(args, protocol, seed, device)
            env = None
            if gpu != "cpu":
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = logs_dir / ("%s_seed%d_%s.log" % (protocol, seed, args.ablation))
            log_f = log_path.open("w", encoding="utf-8")
            print("Starting protocol=%s seed=%d gpu=%s log=%s" % (protocol, seed, gpu, log_path), flush=True)
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            running.append({"proc": proc, "log": log_f, "protocol": protocol, "seed": seed, "gpu": gpu, "log_path": str(log_path)})
        for item in running:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.wait()
            log_f = item["log"]
            log_f.close()
            if code != 0:
                failed_item = {k: v for k, v in item.items() if k not in {"proc", "log"}}
                failed_item["returncode"] = code
                failed.append(failed_item)
                print("FAILED protocol=%s seed=%s gpu=%s" % (item["protocol"], item["seed"], item["gpu"]), flush=True)
            else:
                print("Done protocol=%s seed=%s gpu=%s" % (item["protocol"], item["seed"], item["gpu"]), flush=True)

    failed_path = args.output_dir / "failed_jobs.json"
    failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit("Failed jobs written to %s" % failed_path)
    print("All NOCS jobs completed.")


if __name__ == "__main__":
    main()
