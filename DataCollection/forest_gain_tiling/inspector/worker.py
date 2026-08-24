"""Run inspector work in a process configured for the requested PERIOD."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import ee

from config import settings
from gee.auth import get_ee_credentials
from gee_datasets.registry import Datasets
from inspector.export import export_inspector_tile
from inspector.service import fetch_tile_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fetch", "export"))
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    tile = payload["tile"]
    if tile.get("period") != settings.period:
        raise ValueError("Worker PERIOD does not match the requested tile period")

    ee.Initialize(get_ee_credentials(), project=settings.gee_project)
    datasets = Datasets()
    if args.action == "fetch":
        print(json.dumps(fetch_tile_metrics(tile, datasets)))
        return

    logger = logging.getLogger("gee.inspector")
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.INFO)
    output_dir = export_inspector_tile(
        tile, datasets, logger, Path(payload["output_root"])
    )
    print(json.dumps({"output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
