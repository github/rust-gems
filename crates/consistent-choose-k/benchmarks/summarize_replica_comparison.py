#!/usr/bin/env python3
"""Convert Criterion's batch estimates to ns/query and ns/replica CSV.

Usage: python3 crates/consistent-choose-k/benchmarks/summarize_replica_comparison.py \
    target/criterion > comparison.csv
"""

import csv
import json
from pathlib import Path
import re
import sys


def summarize(root):
    rows = []
    for metadata in root.glob("**/new/benchmark.json"):
        benchmark = json.loads(metadata.read_text())
        group = benchmark["group_id"]
        if not group.startswith("sentinel_replicas/"):
            continue
        mode = group.removeprefix("sentinel_replicas/")
        algorithm = benchmark["function_id"]
        value = benchmark.get("value_str") or ""
        match = re.fullmatch(r"n(\d+)_k(\d+)", value)
        n, k = map(int, match.groups()) if match else (int(value or 0), 0)
        estimates = json.loads((metadata.parent / "estimates.json").read_text())
        mean = estimates["mean"]
        interval = mean["confidence_interval"]
        divisor = benchmark["throughput"]["Elements"]
        rows.append(
            (
                mode,
                algorithm,
                n,
                k,
                mean["point_estimate"] / divisor,
                interval["lower_bound"] / divisor,
                interval["upper_bound"] / divisor,
                estimates["std_dev"]["point_estimate"] / divisor,
                mean["point_estimate"] / divisor / (k if k and mode != "rank_replay" else 1),
            )
        )
    if not rows:
        raise SystemExit(f"No replica_comparison results found in {root}")
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["mode", "algorithm", "n", "k", "ns_query", "ci95_low", "ci95_high",
         "stddev_ns", "ns_replica"]
    )
    for row in sorted(rows):
        writer.writerow([*row[:4], *(f"{value:.3f}" for value in row[4:])])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    summarize(Path(sys.argv[1]))
