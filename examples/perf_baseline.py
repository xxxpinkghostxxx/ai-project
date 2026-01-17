"""Quick baseline harness for the vectorized simulation engine."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project.system.vector_engine import run_micro_benchmark  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("perf_baseline")


def main() -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    results = {}
    for dev in devices:
        logger.info("Running micro benchmark on %s", dev)
        results[dev] = run_micro_benchmark(device=dev, n=20_000)

    print(json.dumps(results, indent=2))
    logger.info("Baseline complete: %s", results)


if __name__ == "__main__":
    main()
