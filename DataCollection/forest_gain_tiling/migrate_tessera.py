"""
Replace existing TESSERA v1.0 embeddings with v1.1 (dclimate), tile by tile.

Run from the project root (next to main.py):

    python migrate_tessera_v11.py --root /path/to/tiles --dry-run     # report only
    python migrate_tessera_v11.py --root /path/to/tiles               # do it

Layout assumed: <root>/<tile_id>/embeddings/tessera_<year>.tif (what download_tessera writes).
The tile dict handed to the downloader is each tile's own <tile_id>/metadata.json with its "bounds" block
(crs, x_min_m..y_max_m, min_lon..max_lat) flattened to the top level, and tile_id set to the directory name.

A file counts as stale when it lacks the TESSERA_DATASET_VERSION/VARIANT tags for the pinned dataset (every
v1.0 file does). Each tile's <tile_id>/metadata.json also gets "tessera-version": "v1.1-dclimate", written only
once every year of the tile has been verified as v1.1 (a missing or different value means the tile is not
migrated yet). Re-running is safe and resumes where it stopped: finished tiles are skipped.
Stale files are deleted just before their replacement is fetched; a failed year leaves that year missing
(reported at the end and written to --failures), never a stale v1.0 file passed off as v1.1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event

from config import settings
from tessera.tessera import fetch_tessera_in_chunks, tile_is_current


def ensure_settings(years_per_request: int, year_timeout_s: int) -> None:
    """The download code reads these two from `settings`; supply them if your Settings does not define them."""
    for name, value in (
        ("tessera_years_per_request", years_per_request),
        ("tessera_year_timeout_s", year_timeout_s),
    ):
        if not hasattr(settings, name):
            try:
                setattr(settings, name, value)
            except Exception:  # frozen / validated settings object
                object.__setattr__(settings, name, value)
            print(
                f"settings.{name} not defined -> using {value} for this run", flush=True
            )
    missing = [
        n for n in ("years", "tile_pixels", "crs_wkt") if not hasattr(settings, n)
    ]
    if missing:
        sys.exit(
            f"settings is missing {', '.join(missing)}, which the TESSERA download needs"
        )


META, KEY, TARGET = "metadata.json", "tessera-version", "v1.1-dclimate"


def read_meta(tile_dir: Path) -> dict:
    """metadata.json as a dict ({} if absent). Raises ValueError if it exists but is not a JSON object."""
    p = tile_dir / META
    if not p.exists():
        return {}
    try:
        meta = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON ({exc})") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{p} is not a JSON object")
    return meta


def stamp(tile_dir: Path) -> None:
    """Add/overwrite "tessera-version" in metadata.json, keeping every other key; atomic write."""
    meta = read_meta(tile_dir)
    meta[KEY] = TARGET
    tmp = tile_dir / (META + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    tmp.replace(tile_dir / META)


REQUIRED = (
    "x_min_m",
    "y_min_m",
    "x_max_m",
    "y_max_m",
)  # geometry the download needs, from metadata.json


def classify(tile_dir: Path) -> dict[str, list[int]]:
    """Per-year state of a tile's embeddings: current / stale / missing; plus "unstamped" (metadata.json)."""
    out = {"current": [], "stale": [], "missing": [], "unstamped": []}
    for year in settings.years:
        f = tile_dir / "embeddings" / f"tessera_{year}.tif"
        out[
            (
                "missing"
                if not f.exists()
                else "current" if tile_is_current(f) else "stale"
            )
        ].append(year)
    if read_meta(tile_dir).get(KEY) != TARGET:
        out["unstamped"].append(1)
    return out


def migrate_tile(tile: dict, tile_dir: Path, log: logging.Logger) -> bool:
    """Fetch what is stale/missing, then stamp metadata.json only if every year is verified v1.1."""
    state = classify(tile_dir)
    if state["stale"] or state["missing"]:
        # per-tile event: fetch_tessera_in_chunks sets it on failure, which must not cancel other tiles
        if not fetch_tessera_in_chunks(tile, tile_dir, log, Event()):
            return False
        state = classify(tile_dir)
        if state["stale"] or state["missing"]:
            return False
    stamp(tile_dir)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="directory holding one sub-directory per tile",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="tiles in parallel (each runs up to 2 year downloads)",
    )
    ap.add_argument(
        "--years-per-request",
        type=int,
        default=2,
        help="years fetched per request (if not in settings)",
    )
    ap.add_argument(
        "--year-timeout",
        type=int,
        default=600,
        help="seconds allowed per year download (if not in settings)",
    )
    ap.add_argument("--limit", type=int, help="only the first N tiles that need work")
    ap.add_argument(
        "--failures", type=Path, default=Path("tessera_migration_failures.csv")
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    log = logging.getLogger("migrate")
    ensure_settings(args.years_per_request, args.year_timeout)

    tiles, todo, done, bad = {}, [], 0, []
    for d in sorted(p for p in args.root.iterdir() if (p / "embeddings").is_dir()):
        try:
            state = classify(d)
            meta = read_meta(d)
            tiles[d.name] = {
                **meta,
                **(meta.get("bounds") or {}),
                "tile_id": d.name,
            }  # flatten bounds
        except ValueError as exc:  # unreadable metadata.json: report, never overwrite
            bad.append((d.name, str(exc)))
            continue
        lacking = [k for k in REQUIRED if k not in tiles[d.name]]
        if lacking and (
            state["stale"] or state["missing"]
        ):  # cannot fetch without the geometry
            bad.append((d.name, f"metadata.json bounds lack {', '.join(lacking)}"))
            continue
        if state["stale"] or state["missing"] or state["unstamped"]:
            todo.append((d.name, d, state))
        else:
            done += 1

    n_stale = sum(len(s["stale"]) for *_, s in todo)
    n_missing = sum(len(s["missing"]) for *_, s in todo)
    n_stamp = sum(1 for *_, s in todo if not (s["stale"] or s["missing"]))
    print(
        f"{done} tiles already v1.1 | {len(todo)} tiles need work "
        f"({n_stale} stale v1.0 files, {n_missing} missing files, {n_stamp} tiles only need the metadata stamp)"
        + (
            f" | {len(bad)} tiles skipped (e.g. {bad[0][0]}: {bad[0][1]})"
            if bad
            else ""
        ),
        flush=True,
    )
    if args.limit:
        todo = todo[: args.limit]
    if args.dry_run or not todo:
        return 0

    t0, ok, failed = time.monotonic(), 0, list(bad)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(migrate_tile, tiles[tid], d, log): tid for tid, d, _ in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            tid = futs[fut]
            try:
                success = fut.result()
            except Exception as exc:  # no data / permanent errors for this tile
                success, why = False, f"{type(exc).__name__}: {exc}"
            else:
                why = "download failed" if not success else ""
            ok += success
            if not success:
                failed.append((tid, why))
            rate = (time.monotonic() - t0) / i
            print(
                f"[{i}/{len(todo)}] {tid} {'OK' if success else 'FAILED ' + why} "
                f"| {rate:.0f}s/tile, ~{rate * (len(todo) - i) / 60:.0f} min left",
                flush=True,
            )

    if failed:
        args.failures.write_text(
            "tile_id,reason\n" + "\n".join(f'{t},"{w}"' for t, w in failed) + "\n"
        )
    print(
        f"done: {ok} migrated, {len(failed)} failed"
        + (f" (see {args.failures})" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
