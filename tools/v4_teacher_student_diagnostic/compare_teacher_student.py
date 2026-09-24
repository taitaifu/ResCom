#!/usr/bin/env python3
"""Run identical V4 test inference for Teacher and Student and compare metrics.

This is an offline diagnostic wrapper. It does not train or modify checkpoints.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _flatten(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def run_infer(args: argparse.Namespace, role: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_script = Path(__file__).resolve().parents[2] / "infer" / "infer_graph_model_v4.py"
    checkpoint = args.teacher_checkpoint if role == "teacher" else args.student_checkpoint
    cmd = [sys.executable, str(infer_script), "--checkpoint", str(checkpoint),
           "--model_role", role, "--split", args.split,
           "--infer_mode", "merged_csv",
           "--feature_dir", args.feature_dir, "--merged_csv", args.merged_csv,
           "--history_len", str(args.history_len), "--batch_size", str(args.batch_size),
           "--num_workers", str(args.num_workers), "--device", args.device,
           "--output_dir", str(output_dir)]
    if role == "teacher":
        cmd += ["--teacher_future_len", str(args.teacher_future_len)]
    subprocess.run(cmd, check=True)
    metrics_path = output_dir / "metrics_v4.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing inference metrics: {metrics_path}")
    return json.loads(metrics_path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher_checkpoint", required=True)
    p.add_argument("--student_checkpoint", required=True)
    p.add_argument("--feature_dir", default="Feature_Selection/DataSet")
    p.add_argument("--merged_csv", default="Feature_Selection/DataSet/merged_error_dataset.csv")
    default_output = Path(__file__).resolve().parent / "results"
    p.add_argument("--output_dir", default=str(default_output))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--history_len", type=int, default=9)
    p.add_argument("--teacher_future_len", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    root = Path(args.output_dir)
    teacher = run_infer(args, "teacher", root / "teacher")
    student = run_infer(args, "student", root / "student")
    tf, sf = _flatten(teacher), _flatten(student)
    rows = []
    for key in sorted(set(tf) | set(sf)):
        if key in tf or key in sf:
            tv, sv = tf.get(key), sf.get(key)
            rows.append({"metric": key, "teacher": tv, "student": sv,
                         "student_minus_teacher": None if tv is None or sv is None else sv - tv})
    (root / "comparison.json").write_text(json.dumps({"teacher": teacher, "student": student, "comparison": rows}, indent=2))

    def find(patterns: tuple[str, ...]) -> dict[str, float]:
        return {k: v for k, v in sf.items() if any(x in k.lower() for x in patterns)}

    force = find(("force", "fx", "fy", "fz"))
    print("[Teacher/Student Diagnostic]")
    print(f"teacher metrics: {root / 'teacher' / 'metrics_v4.json'}")
    print(f"student metrics: {root / 'student' / 'metrics_v4.json'}")
    print(f"comparison: {root / 'comparison.json'}")
    print("Student is close to Teacher only if the same force RMSE/Fx/Fy/Fz metrics are within ~5%.")
    if force:
        print("Student force-related metrics:")
        for key, value in sorted(force.items()):
            print(f"  {key} = {value:.6g}")
    print("Note: this wrapper compares reported metrics; z_force cosine requires a separate feature dump if not present in metrics_v4.json.")


if __name__ == "__main__":
    main()
