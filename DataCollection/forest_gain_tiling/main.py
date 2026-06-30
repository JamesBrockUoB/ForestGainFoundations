"""
Forest-gain tile export pipeline.

Commands
--------
  python main.py plan                              # build tile registry, print summary
  python main.py filter                            # raster-batch filter all AOIs with pending tiles
  python main.py filter --limit 5                  # filter only the first N AOIs (testing)
  python main.py run                               # process all valid tiles
  python main.py run --limit 500                   # next N valid tiles
  python main.py run --biome "Boreal Forests"      # filter by biome (substring match)
  python main.py run --region Neotropic            # filter by region
  python main.py run --aoi-id aoi_-73.25_-52.75   # single AOI (debug)
  python main.py run --status failed               # retry failed tiles
  python main.py status                            # print registry summary
  python main.py audit                             # report AOIs with no tiles

Filter flags
------------
  --limit      N            max number of AOIs to process (for testing)

Run flags
---------
  --aoi-id     AOI_ID       filter to a single AOI
  --biome      SUBSTRING    filter by biome (case-insensitive substring)
  --region     SUBSTRING    filter by region (case-insensitive substring)
  --limit      N            max tiles to process
  --status     STATUS       valid (default) | failed | rejected
  --stratify   KEY          biome | region
  --stratify-mode  MODE     prop (default) | equal
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import ee
from config import settings
from datasets.registry import Datasets
from enums import TileStatus
from export.tasks import run_hpc, run_local
from filtering.tasks import run_filter_hpc, run_filter_local
from registry.store import (
    audit_summary,
    build_aoi_audit,
    registry_summary,
    save_aoi_audit,
)
from tiling.grid import build_grid
from tiling.selection import (
    filter_candidates,
    log_strata_counts,
    stratified_sample,
)


def setup_logging(command: str) -> logging.Logger:
    settings.log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = settings.log_dir / f"gee_{command}_{ts}.log"

    logger = logging.getLogger("gee")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Log: {logfile}")
    return logger


def init_gee() -> Datasets:
    ee.Initialize(
        ee.ServiceAccountCredentials(None, settings.gee_credentials),
        project=settings.gee_project,
    )
    return Datasets()


def cmd_plan(args: argparse.Namespace) -> None:
    """Plan phase: generate tiles and add new ones to registry."""
    logger = setup_logging("plan")

    logger.info(f"Loading valid AOIs from {settings.valid_aois_path}…")
    with open(settings.valid_aois_path) as f:
        valid_aois = json.load(f)
    logger.info(f"  {len(valid_aois):,} valid AOIs")

    from registry.store import _get_db

    db_tile_count = _get_db().count_tiles()

    if db_tile_count > 0:
        logger.info(
            f"Database already has {db_tile_count:,} tiles. Skipping grid generation."
        )
    else:
        logger.info("Database is empty. Generating tile grid…")

        logger.info("Streaming tiles to database in batches…")
        from collections import Counter

        from registry.store import save_tiles_batch

        batch = []
        batch_size = 1000000
        new_count = 0
        biome_counts = Counter()
        region_counts = Counter()
        total = 0

        for t in build_grid(valid_aois, logger):
            batch.append(t)

            biome_counts[t["biome"]] += 1
            region_counts[t["region"]] += 1
            total += 1

            if len(batch) >= batch_size:
                new_count += save_tiles_batch(batch, batch_size=batch_size)
                batch = []

        if batch:
            new_count += save_tiles_batch(batch, batch_size=batch_size)

        settings.registry_db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Registry: {new_count:,} new tiles added → {settings.registry_db_path}"
        )

        sz = settings.tile_size_m
        lines = [
            "",
            "═" * 60,
            "  TILE PLAN SUMMARY",
            "═" * 60,
            f"  Total tiles : {total:>10,}",
            f"  Grid size   : {sz:.0f} m x {sz:.0f} m  ({settings.tile_pixels}x{settings.tile_pixels} px @ {settings.scale} m/px)",
            f"  CRS         : {settings.crs}",
            f"  Min overlap : {settings.min_aoi_overlap_frac*100:.0f}% of tile inside a single AOI",
            "",
            "  By biome:",
        ]
        for b, n in biome_counts.most_common():
            lines.append(f"    {b:<45} {n:>8,}  ({100*n/max(total,1):5.1f}%)")
        lines += ["", "  By region:"]
        for r, n in region_counts.most_common():
            lines.append(f"    {r:<30} {n:>8,}  ({100*n/max(total,1):5.1f}%)")
        lines += ["═" * 60, ""]
        print("\n".join(lines))


def cmd_status(args: argparse.Namespace) -> None:
    """Print registry status summary."""
    print(registry_summary())


def cmd_audit(args: argparse.Namespace) -> None:
    """Regenerate AOI coverage audit."""
    logger = setup_logging("audit")
    logger.info(f"Loading valid AOIs from {settings.valid_aois_path}…")
    with open(settings.valid_aois_path) as f:
        valid_aois = json.load(f)
    audit = build_aoi_audit(valid_aois)
    save_aoi_audit(audit)
    print(audit_summary(audit))


def cmd_filter(args: argparse.Namespace) -> None:
    """
    Filter phase: raster-batch viability/coverage checks on pending tiles.
    One aggregated GEE raster fetch per AOI covers all of that AOI's
    pending tiles. No exports happen here — tiles are marked valid or
    rejected.
    """
    logger = setup_logging("filter")

    init_gee()

    limit_aois = args.limit

    if settings.use_hpc:
        logger.info(f"Mode: HPC | workers={settings.num_workers}")
        run_filter_hpc(logger, limit_aois=limit_aois)
    else:
        logger.info("Mode: local sequential")
        run_filter_local(logger, limit_aois=limit_aois)

    print(registry_summary())


def cmd_run(args: argparse.Namespace) -> None:
    """
    Run phase: process valid tiles (pseudo-labels + export).
    Resumes from saved state - only processes tiles not yet complete/rejected.
    """
    logger = setup_logging("run")

    target_status = args.status or str(TileStatus.VALID)
    if target_status == str(TileStatus.REJECTED):
        logger.warning(
            "Targeting rejected tiles — these failed the filter checks "
            "and will likely be rejected again unless thresholds have changed."
        )

    logger.info("Loading candidates from registry (streaming)…")
    candidates = filter_candidates(
        status=target_status,
        aoi_id=args.aoi_id,
        biome=args.biome,
        region=args.region,
        logger=logger,
    )

    if args.stratify and args.limit:
        candidates = stratified_sample(
            candidates, args.stratify, args.limit, args.stratify_mode
        )
        log_strata_counts(candidates, args.stratify, logger, args.stratify_mode)
    elif args.limit:
        candidates = candidates[: args.limit]
        logger.info(f"Limited to {args.limit} tiles")

    if not candidates:
        logger.info("No tiles match the given filters.")
        return

    logger.info(f"Processing {len(candidates):,} tiles")

    ds = init_gee()

    if settings.use_hpc:
        logger.info(f"Mode: HPC | workers={settings.num_workers}")
        run_hpc(candidates, ds, logger)
    else:
        logger.info("Mode: local sequential")
        run_local(candidates, ds, logger)

    print(registry_summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forest-gain tile export pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="Build tile registry and AOI audit (no GEE calls)")
    sub.add_parser("status", help="Print current registry progress")
    sub.add_parser("audit", help="Report AOIs with no complete tiles")

    filter_p = sub.add_parser(
        "filter",
        help="Run raster-batch viability/coverage filters (no exports)",
    )
    filter_p.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Max number of AOIs to process (for testing on a subset)",
    )

    run_p = sub.add_parser("run", help="Submit and monitor export tasks")
    run_p.add_argument("--aoi-id", default=None)
    run_p.add_argument("--biome", default=None)
    run_p.add_argument("--region", default=None)
    run_p.add_argument("--limit", default=None, type=int)
    run_p.add_argument(
        "--status",
        default=str(TileStatus.VALID),
        choices=[
            str(s) for s in (TileStatus.VALID, TileStatus.FAILED, TileStatus.REJECTED)
        ],
    )
    run_p.add_argument("--stratify", default=None, choices=["biome", "region"])
    run_p.add_argument(
        "--stratify-mode",
        default="prop",
        choices=["prop", "equal"],
        dest="stratify_mode",
    )

    return parser


if __name__ == "__main__":
    settings.registry_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.aoi_audit_path.parent.mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "plan": cmd_plan,
        "status": cmd_status,
        "audit": cmd_audit,
        "filter": cmd_filter,
        "run": cmd_run,
    }
    dispatch[args.command](args)
