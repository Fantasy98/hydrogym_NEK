"""Summarize completed collective benchmarks; retain per-pass results for drift."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
groups = defaultdict(list)
for path in sorted(root.glob("*-timing-*.out")):
    for line in path.read_text().splitlines():
        if not line.startswith("RESULT,"):
            continue
        _, profile, repeat, operation, nbytes, ranks, batch, iterations, seconds = line.split(",")
        groups[(profile, int(repeat), operation, int(nbytes), int(ranks))].append(float(seconds))
if not groups:
    raise SystemExit("No completed timing rows found")
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["profile", "pass", "operation", "bytes", "solver_ranks", "batches",
                     "median_us", "min_us", "max_us"])
    for key, values in sorted(groups.items()):
        writer.writerow([*key, len(values), statistics.median(values) * 1e6,
                         min(values) * 1e6, max(values) * 1e6])
        if key[3] in (0, 8, 5132288):
            print(f"{key[0]:12} pass={key[1]} {key[2]:13} bytes={key[3]:7} "
                  f"median={statistics.median(values)*1e6:.2f} us ({len(values)} batches)")
print(f"Saved {root / 'summary.csv'}")
