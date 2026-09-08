#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or validate image quality model.")
    parser.add_argument("--manifest", required=True)
    parser.parse_args()
    raise RuntimeError("Quality model training requires labeled quality data. Use audit quality heuristics until labels are provided.")


if __name__ == "__main__":
    raise SystemExit(main())
