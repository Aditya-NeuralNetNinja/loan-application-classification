#!/usr/bin/env python3
"""Copy latest processed artifacts into app/assets for lightweight deployments."""

from __future__ import annotations

import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
ASSETS = BASE / "app" / "assets"

MAPPING = {
    PROCESSED / "model_results.json": ASSETS / "model_results.sample.json",
    PROCESSED / "model_leaderboard.csv": ASSETS / "model_leaderboard.sample.csv",
    PROCESSED / "optimal_threshold.json": ASSETS / "optimal_threshold.sample.json",
}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    copied = 0

    for src, dst in MAPPING.items():
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
            copied += 1
        else:
            print(f"Missing source, skipped: {src}")

    if copied == 0:
        raise SystemExit("No artifacts copied. Run notebooks 4a/4b first.")

    print(f"Done. Copied {copied} artifact file(s).")


if __name__ == "__main__":
    main()
