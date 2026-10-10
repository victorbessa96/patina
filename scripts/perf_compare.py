#!/usr/bin/env python3
"""perf_compare.py — the §12 nightly regression gate.

Compares the fresh harness JSONs against perf/baseline.json:
a >10% regression on any tracked metric fails (exit 1). Missing
metrics report as U (unmeasured) and don't fail — the protocol's
own honest-gap rule. A missing baseline file = first-run mode:
the fresh numbers BECOME the baseline candidate (printed, exit 0
— the commit of baseline.json is the human's call).

Usage: perf_compare.py <ref-scene.json> <i2p.json> <baseline.json>
"""

import json
import sys
from pathlib import Path

REGRESSION = 1.10  # >10% over baseline = fail


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    txt = p.read_text().strip()
    # the harnesses print the JSON as one stdout line; files may carry
    # surrounding log noise — take the last { ... } object on the line
    # that parses.
    for line in reversed(txt.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def metrics(ref: dict, i2p: dict) -> dict:
    """Flatten the two harness reports into the tracked metric set."""
    out = {}
    if "p95_ms" in ref:
        out["viewport_p95_ms"] = ref["p95_ms"]
    if "mean_ms" in ref:
        out["viewport_mean_ms"] = ref["mean_ms"]
    if "p95_ms" in i2p:
        out["input_photon_p95_ms"] = i2p["p95_ms"]
    if "p50_ms" in i2p:
        out["input_photon_p50_ms"] = i2p["p50_ms"]
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2

    fresh = metrics(load(sys.argv[1]), load(sys.argv[2]))
    baseline_path = Path(sys.argv[3])
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    if not baseline:
        print(f"first-run mode: no baseline at {baseline_path}")
        print("fresh metrics (become the baseline candidate):")
        print(json.dumps(fresh, indent=2))
        return 0

    if not fresh:
        print("FAIL: no fresh metrics parsed — harness output missing/unparsable")
        return 1

    failures = []
    for key, value in sorted(fresh.items()):
        base = baseline.get(key)
        if base is None:
            print(f"U  {key}: {value} (no baseline entry — unmeasured before)")
            continue
        ratio = value / base if base else float("inf")
        verdict = "OK" if ratio <= REGRESSION else "REGRESSION"
        print(f"{verdict:<12} {key}: {value} vs baseline {base} (x{ratio:.3f})")
        if ratio > REGRESSION:
            failures.append(key)

    if failures:
        print(f"FAIL: {len(failures)} metric(s) regressed >10%: {', '.join(failures)}")
        return 1
    print("PASS: no metric regressed >10% against the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
