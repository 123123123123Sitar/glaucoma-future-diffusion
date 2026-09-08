#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.audit import audit_dataset
from glaucoma_forecast.data.schema import DatasetSetupError
from glaucoma_forecast.utils.logging import configure_logging


def infer_dataset(manifest: str | None, root: str | None) -> str:
    candidates = " ".join(x.lower() for x in [manifest or "", root or ""])
    if "grape" in candidates:
        return "grape"
    if "sigf" in candidates:
        return "sigf"
    if manifest and Path(manifest).exists():
        try:
            df = pd.read_csv(manifest, nrows=20)
            if "source_dataset" in df.columns:
                values = set(df["source_dataset"].dropna().astype(str).str.lower())
                if any("grape" in value for value in values):
                    return "grape"
                if any("sigf" in value for value in values):
                    return "sigf"
        except Exception:
            pass
    raise DatasetSetupError("Could not infer dataset. Pass --dataset sigf or --dataset grape.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SIGF or GRAPE longitudinal fundus data.")
    parser.add_argument("--dataset", choices=["sigf", "grape"])
    parser.add_argument("--root")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", default="outputs/audit")
    parser.add_argument("--json-logs", action="store_true")
    args = parser.parse_args()
    configure_logging(json_logs=args.json_logs)
    try:
        dataset = args.dataset or infer_dataset(args.manifest, args.root)
        root = args.root or str(Path(args.manifest).resolve().parent if args.manifest else Path("data") / dataset)
        summary = audit_dataset(dataset, root, args.output_dir, args.manifest)
    except DatasetSetupError as exc:
        print(f"Dataset setup error: {exc}", file=sys.stderr)
        return 2
    print(f"Audit complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
