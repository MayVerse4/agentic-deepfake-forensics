"""Saves a benchmark summary permanently: a JSON file per benchmark plus an append-only
log results/benchmarks/BENCHMARKS.md (date, device, every metric, per-group table)."""
import datetime
import json
import os


def save(summary, out_dir, name):
    import torch
    summary["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    summary["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open(os.path.join(out_dir, name), "w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "BENCHMARKS.md"), "a", encoding="utf8") as fh:
        fh.write(f"\n## {summary['benchmark']}\n_{summary['date']} on {summary['device']}_\n\n")
        for k, v in summary.items():
            if k not in ("benchmark", "date", "device", "per_group"):
                fh.write(f"- **{k}**: {v}\n")
        if summary.get("per_group"):
            fh.write("\n| group | n | detected % |\n|---|---|---|\n")
            for g, (n, d) in sorted(summary["per_group"].items(), key=lambda kv: -kv[1][0]):
                fh.write(f"| {g} | {n} | {d} |\n")
    print("saved ->", os.path.join(out_dir, name), "and", os.path.join(out_dir, "BENCHMARKS.md"))
