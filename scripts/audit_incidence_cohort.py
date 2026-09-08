#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.risk_trainer import validate_risk_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a candidate incident-glaucoma cohort.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    validate_risk_manifest(frame)
    eligible = (
        pd.to_numeric(frame["incidence_eligible"], errors="coerce").fillna(1).astype(int)
        if "incidence_eligible" in frame
        else pd.Series(1, index=frame.index)
    )
    incidence = frame[eligible == 1]
    events = int(pd.to_numeric(incidence["event_observed"]).sum())
    controls = int((pd.to_numeric(incidence["event_observed"]) == 0).sum())
    followup = pd.to_numeric(incidence["event_or_censor_time_years"])
    current_classes = (
        sorted(pd.to_numeric(frame["current_glaucoma_label"], errors="coerce").dropna().unique().tolist())
        if "current_glaucoma_label" in frame
        else []
    )
    report = {
        "eyes": len(incidence),
        "patients": int(incidence["patient_id"].astype(str).nunique()),
        "incident_events": events,
        "nonconverting_eyes": controls,
        "max_followup_years": float(followup.max()),
        "eyes_observable_at_10y": int((followup >= 10).sum()),
        "sites": int(frame["site_id"].nunique()) if "site_id" in frame else 0,
        "cameras": int(frame["camera_device"].nunique()) if "camera_device" in frame else 0,
        "current_glaucoma_classes": current_classes,
        "release_gates": {
            "minimum_500_events": events >= 500,
            "minimum_2000_controls": controls >= 2000,
            "has_10y_observations": bool((followup >= 10).any()),
            "current_head_has_both_classes": set(current_classes) >= {0.0, 1.0},
            "multiple_sites": (
                bool(frame["site_id"].nunique() >= 2) if "site_id" in frame else False
            ),
        },
    }
    report["ready_for_full_training"] = all(report["release_gates"].values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
