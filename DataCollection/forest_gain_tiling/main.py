"""
Forest-gain tile export pipeline.

Commands
--------
  python main.py plan                              # build tile registry, print summary
  python main.py run                               # process all pending tiles
  python main.py run --limit 500                   # next N pending tiles
  python main.py run --biome "Boreal Forests"      # filter by biome (substring match)
  python main.py run --region Neotropic            # filter by region
  python main.py run --aoi-id aoi_-73.25_-52.75   # single AOI (debug)
  python main.py run --status failed               # retry failed tiles
  python main.py status                            # print registry summary
  python main.py audit                             # report AOIs with no tiles

Run flags
---------
  --aoi-id     AOI_ID       filter to a single AOI
  --biome      SUBSTRING    filter by biome (case-insensitive substring)
  --region     SUBSTRING    filter by region (case-insensitive substring)
  --limit      N            max tiles to process
  --status     STATUS       pending (default) | failed | rejected
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
from registry.store import (
    audit_summary,
    build_aoi_audit,
    load_registry,
    registry_summary,
    save_aoi_audit,
    save_registry,
)
from tiling.grid import build_global_grid, plan_summary
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
    logger = setup_logging("plan")

    logger.info(f"Loading valid AOIs from {settings.valid_aois_path}…")
    with open(settings.valid_aois_path) as f:
        valid_aois = json.load(f)
    logger.info(f"  {len(valid_aois):,} valid AOIs")

    tiles = build_global_grid(valid_aois, logger)
    print(plan_summary(tiles))

    registry = load_registry()
    new_count = 0
    for t in tiles:
        if t["tile_id"] not in registry:
            registry[t["tile_id"]] = t
            new_count += 1

    settings.registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_registry(registry)
    logger.info(
        f"Registry: {new_count:,} new tiles added, "
        f"{len(tiles)-new_count:,} already existed → {settings.registry_path}"
    )

    audit = build_aoi_audit(registry, valid_aois)
    settings.aoi_audit_path.parent.mkdir(parents=True, exist_ok=True)
    save_aoi_audit(audit)
    logger.info(f"AOI audit written → {settings.aoi_audit_path}")
    print(audit_summary(audit))


def cmd_status(args: argparse.Namespace) -> None:
    registry = load_registry()
    if not registry:
        print("Registry is empty. Run `python main.py plan` first.")
        return
    print(registry_summary(registry))


def cmd_audit(args: argparse.Namespace) -> None:
    registry = load_registry()
    if not registry:
        print("Registry is empty. Run `python main.py plan` first.")
        return
    logger = setup_logging("audit")
    logger.info(f"Loading valid AOIs from {settings.valid_aois_path}…")
    with open(settings.valid_aois_path) as f:
        valid_aois = json.load(f)
    audit = build_aoi_audit(registry, valid_aois)
    save_aoi_audit(audit)
    print(audit_summary(audit))
    uncovered = [aoi_id for aoi_id, v in audit.items() if not v["has_coverage"]]
    logger.info(
        f"AOIs with no complete tiles: {len(uncovered):,} — see {settings.aoi_audit_path}"
    )


def cmd_run(args: argparse.Namespace) -> None:
    logger = setup_logging("run")

    registry = load_registry()
    if not registry:
        logger.error("Registry is empty. Run `python main.py plan` first.")
        return

    target_status = args.status or str(TileStatus.PENDING)
    if target_status == str(TileStatus.REJECTED):
        logger.warning(
            "Targeting rejected tiles — these failed viability checks "
            "and will likely be rejected again unless thresholds have changed."
        )

    candidates = filter_candidates(
        registry,
        target_status,
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

    complete_count = sum(
        1 for e in registry.values() if e["status"] == str(TileStatus.COMPLETE)
    )
    logger.info(
        f"Processing {len(candidates):,} tiles  "
        f"(registry total: {len(registry):,}  complete: {complete_count:,})"
    )

    ds = init_gee()

    if settings.use_hpc:
        logger.info(f"Mode: HPC | workers={settings.num_workers}")
        run_hpc(candidates, registry, ds, logger)
    else:
        logger.info("Mode: local sequential")
        run_local(candidates, registry, ds, logger)

    print(registry_summary(registry))

    if settings.valid_aois_path.exists():
        with open(settings.valid_aois_path) as f:
            valid_aois = json.load(f)
        audit = build_aoi_audit(registry, valid_aois)
        save_aoi_audit(audit)
        print(audit_summary(audit))


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

    run_p = sub.add_parser("run", help="Submit and monitor export tasks")
    run_p.add_argument("--aoi-id", default=None)
    run_p.add_argument("--biome", default=None)
    run_p.add_argument("--region", default=None)
    run_p.add_argument("--limit", default=None, type=int)
    run_p.add_argument(
        "--status",
        default=str(TileStatus.PENDING),
        choices=[
            str(s) for s in (TileStatus.PENDING, TileStatus.FAILED, TileStatus.REJECTED)
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
    settings.registry_path.parent.mkdir(parents=True, exist_ok=True)
    settings.aoi_audit_path.parent.mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "plan": cmd_plan,
        "status": cmd_status,
        "audit": cmd_audit,
        "run": cmd_run,
    }
    dispatch[args.command](args)
