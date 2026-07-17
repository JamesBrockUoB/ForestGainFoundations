"""
Forest-gain tile export pipeline.

Period is controlled by the PERIOD environment variable (same convention as
generate_aois.py):
  PERIOD=p1 (default) — 2017 → 2020
  PERIOD=p2           — 2020 → 2024
Every command below operates on tiles tagged with the active period only.

Commands
--------
  PERIOD=p1 python main.py plan                     # build tile registry, print summary
  PERIOD=p1 python main.py filter --stage cheap      # cheap gain/NDVI filter
  PERIOD=p1 python main.py filter --stage imagery    # per-year S1+S2 availability filter
  python main.py filter --stage cheap --limit 5      # filter only the first N batches (testing)
  python main.py run                                # process all valid tiles
  python main.py run --limit 500                     # next N valid tiles
  python main.py run --biome "Boreal Forests"        # filter by biome (substring match)
  python main.py run --region Neotropic              # filter by region
  python main.py run --aoi-id aoi_-73.25_-52.75       # single AOI (debug)
  python main.py run --status failed                 # retry failed tiles
  python main.py status                              # print registry summary (active period)
  python main.py reset --status failed                # retry only failed tiles, keep error text
  python main.py reset --status rejected --yes        # re-filter previously rejected tiles, no prompt
  python main.py reset --clear-history                # nuke everything back to blank pending

Filter flags
------------
  --stage      cheap | imagery   which filter stage to run
                 cheap:   PENDING -> CHEAP_VALID | REJECTED (gain/NDVI thresholds;
                          canopy_mean is still computed and reported per tile but
                          no longer gates pass/fail — canopy height data is only
                          available for 2020, so it can't gate viability across
                          a full period like p1's 2017 endpoint)
                 imagery: CHEAP_VALID -> VALID | REJECTED (per-year S1+S2 availability
                          over every year in the active period)
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

All commands are implicitly scoped to the active PERIOD; reset in
particular always filters on period, so `reset --status failed` for
PERIOD=p1 will never touch p2 tiles.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import ee
from config import settings
from enums import TileStatus
from export.tasks import run_hpc, run_local
from filtering.tasks import run_filter_hpc, run_filter_local
from gee.auth import get_ee_credentials
from gee_datasets.registry import Datasets
from registry.store import registry_summary
from tiling.grid import build_grid
from tiling.selection import (
    filter_candidates,
    log_strata_counts,
    stratified_sample,
)


def setup_logging(command: str) -> logging.Logger:
    settings.log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = settings.log_dir / f"gee_{command}_{settings.period}_{ts}.log"

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
    logger.info(f"PERIOD={settings.period} ({settings.year_start}→{settings.year_end})")
    return logger


def init_ee() -> None:
    ee.Initialize(get_ee_credentials(), project=settings.gee_project)


def cmd_plan(args: argparse.Namespace) -> None:
    """Plan phase: generate tiles for the active period and add new ones to registry."""
    logger = setup_logging("plan")

    logger.info(f"Loading valid AOIs from {settings.valid_aois_path}…")
    with open(settings.valid_aois_path) as f:
        valid_aois = json.load(f)
    logger.info(f"  {len(valid_aois):,} valid AOIs (period={settings.period})")

    from registry.store import _get_db

    # Scoped to the active period: running `plan` for p2 after p1 has
    # already been planned must not skip generation just because the
    # (shared) database already has p1 rows in it.
    db_tile_count = _get_db().count_tiles(period=settings.period)

    if db_tile_count > 0:
        logger.info(
            f"Database already has {db_tile_count:,} tiles for period="
            f"{settings.period}. Skipping grid generation."
        )
    else:
        logger.info("No tiles for this period yet. Generating tile grid…")

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
            f"Registry: {new_count:,} new tiles added (period={settings.period}) "
            f"→ {settings.registry_db_path}"
        )

        sz = settings.tile_size_m
        lines = [
            "",
            "═" * 60,
            f"  TILE PLAN SUMMARY  (period={settings.period})",
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
    """Print registry status summary for the active period."""
    print(registry_summary(period=settings.period))


def cmd_filter(args: argparse.Namespace) -> None:
    logger = setup_logging(f"filter_{args.stage}")
    init_ee()

    if settings.use_hpc:
        logger.info(
            f"Mode: HPC | workers={settings.num_workers} | stage={args.stage} "
            f"| period={settings.period} | batch_size={args.batch_size} | Filtering tiles"
        )
        run_filter_hpc(
            logger,
            stage=args.stage,
            batch_size=args.batch_size,
            limit_batches=args.limit,
        )
    else:
        logger.info(
            f"Mode: local sequential | stage={args.stage} | period={settings.period} "
            f"| batch_size={args.batch_size} | Filtering tiles"
        )
        run_filter_local(
            logger,
            stage=args.stage,
            batch_size=args.batch_size,
            limit_batches=args.limit,
        )

    print(registry_summary(period=settings.period))


def cmd_run(args: argparse.Namespace) -> None:
    """
    Run phase: process valid tiles (pseudo-labels + export) for the active period.
    Resumes from saved state - only processes tiles not yet complete/rejected.
    """
    logger = setup_logging("run")

    target_status = args.status or str(TileStatus.VALID)
    if target_status == str(TileStatus.REJECTED):
        logger.warning(
            "Targeting rejected tiles — these failed the filter checks "
            "and will likely be rejected again unless thresholds have changed."
        )

    logger.info("Loading candidates from registry for export preparation")
    candidates = filter_candidates(
        status=target_status,
        aoi_id=args.aoi_id,
        biome=args.biome,
        region=args.region,
        period=settings.period,
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

    logger.info(f"Processing {len(candidates):,} tiles (period={settings.period})")

    init_ee()
    ds = Datasets()

    if settings.use_hpc:
        logger.info(f"Mode: HPC | workers={settings.num_workers}")
        run_hpc(candidates, logger, local_output=args.local_output)
    else:
        logger.info("Mode: local sequential")
        run_local(candidates, ds, logger, local_output=args.local_output)

    print(registry_summary(period=settings.period))


def cmd_reset(args: argparse.Namespace) -> None:
    """Reset tile statuses for the active period."""
    logger = setup_logging("reset")

    from registry.store import reset_tiles

    if args.to_status == args.status:
        logger.error(
            f"--to-status ({args.to_status}) is the same as --status "
            f"({args.status}) — nothing would change."
        )
        return

    label = args.status or "ALL"
    history_note = " and clear processing history" if args.clear_history else ""

    if not args.yes:
        confirm = input(
            f"This will reset {label} tiles in period={settings.period} to "
            f"'{args.to_status}'{history_note}. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            logger.info("Aborted.")
            return

    n = reset_tiles(
        status=args.status,
        period=settings.period,
        clear_history=args.clear_history,
        to_status=args.to_status,
    )
    logger.info(f"Reset {n:,} tiles (period={settings.period}) to '{args.to_status}'.")
    print(registry_summary(period=settings.period))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forest-gain tile export pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="Build tile registry from valid AOIs (no GEE calls)")
    sub.add_parser("status", help="Print current registry progress")

    filter_p = sub.add_parser(
        "filter",
        help="Run one stage of the tile-batch filter (no exports)",
    )
    filter_p.add_argument(
        "--stage",
        required=True,
        choices=["cheap", "imagery"],
        help="cheap: gain/NDVI thresholds (PENDING -> CHEAP_VALID); canopy_mean "
        "is reported but no longer gates. "
        "imagery: per-year Sentinel-1/2 availability over the active period "
        "(CHEAP_VALID -> VALID).",
    )
    filter_p.add_argument(
        "--batch-size",
        default=settings.filter_batch_size,
        type=int,
        help=f"Tiles per raster fetch (default: {settings.filter_batch_size})",
    )
    filter_p.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Max number of batches to process (for testing on a subset)",
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

    run_p.add_argument(
        "--local-output",
        action="store_true",
        help="Write exports and embeddings to DataCollection/data/test_tiles instead of HPC.",
    )

    _RESETTABLE_STATUSES = [
        str(s)
        for s in (
            TileStatus.PENDING,
            TileStatus.CHEAP_VALID,
            TileStatus.VALID,
            TileStatus.FAILED,
            TileStatus.REJECTED,
            TileStatus.SUBMITTED,
            TileStatus.COMPLETE,
        )
    ]

    reset_p = sub.add_parser(
        "reset", help="Reset tile statuses (scoped to active period)"
    )
    reset_p.add_argument(
        "--status",
        default=None,
        choices=_RESETTABLE_STATUSES,
        help="Only reset tiles currently in this status (default: reset all "
        "tiles not already at --to-status)",
    )
    reset_p.add_argument(
        "--to-status",
        default=str(TileStatus.PENDING),
        choices=_RESETTABLE_STATUSES,
        help="Status to reset tiles into (default: pending). E.g. "
        "'cheap_valid' to resume the imagery stage without redoing the cheap stage.",
    )
    reset_p.add_argument(
        "--clear-history",
        action="store_true",
        help="Also clear gee_task_id, submitted_at, completed_at, rejection_reason, error",
    )
    reset_p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    return parser


if __name__ == "__main__":
    settings.registry_db_path.parent.mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "plan": cmd_plan,
        "status": cmd_status,
        "filter": cmd_filter,
        "run": cmd_run,
        "reset": cmd_reset,
    }
    dispatch[args.command](args)
