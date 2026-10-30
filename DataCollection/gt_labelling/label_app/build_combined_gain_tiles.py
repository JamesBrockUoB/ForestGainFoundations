#!/usr/bin/env python3
"""
Raw data  ->  one combined tile grid (gain_tiles_10m_combined.geojson)

Starts from just two raw sources:
  1. Copernicus HRL "Tree Cover Presence Change" (downloaded from CDSE by stage 1)
  2. your England woodland-creation polygons (a folder of .shp/.gpkg/.geojson, or one file)

Stages (run all by default, or pick some with --stages; every stage is resumable / skips finished work):
  download  CDSE catalogue -> HRL change + confidence tiles for 2018-21 and 2021-24      (needs CDSE_S3_ACCESS_KEY/SECRET_KEY)
  rasters   per-tile 20 m gain rasters (gain_period, gain_conf) -> one merged GeoTIFF
  ingest    upload merged GeoTIFF to a temp GCS bucket, ingest as ONE Earth Engine asset, delete bucket
                                                                                  (needs gcloud + earthengine authenticated)
  count     count HRL gain at 10 m per 2560 m EPSG:3035 tile with Earth Engine -> gain_tile_counts_gee10.csv
  combine   exact woodland area per tile, merged with the gain counts -> gain_tiles_10m_combined.geojson

Examples
  python build_combined_gain_tiles.py --bbox -6.5 49.8 2.0 55.9          # everything, England-only download
  python build_combined_gain_tiles.py --stages count combine              # asset already exists
  python build_combined_gain_tiles.py --stages combine                    # just redo the merge with the woodland data

Install:  pip install rasterio numpy pandas geopandas shapely affine earthengine-api boto3
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================== CONFIG
ROOT = Path("copernicus_tree_change")          # raw HRL downloads + per-tile gain rasters
RASTER_DIR = ROOT / "gee_gain_20m"
MERGED = RASTER_DIR / "hrl_gain_20m_merged.tif"
PERIOD_A, PERIOD_B = "2018-2021", "2021-2024"

# Earth Engine / GCS
PROJECT = "symbolic-base-346316"
BUCKET_NAME = "symbolic-base-346316-hrl-gain-tmp"
BUCKET_LOCATION = "EU"
ASSET = "projects/symbolic-base-346316/assets/hrl_gain_20m_eu"

# tile grid: tile (xi, yi) spans [xi*T, (xi+1)*T] x [yi*T, (yi+1)*T] in EPSG:3035
T, PX = 2560.0, 10.0
PX_TILE = (T / PX) ** 2
MIN_PCT = 1.0                                   # keep a tile if gain_pct >= MIN_PCT OR woodland_pct >= MIN_PCT

# woodland data
WOODLAND = "Woodland Creation 16-17 England"    # folder (searched recursively) or single file

# outputs of the count / combine stages
BLOCKS_CSV = "gain_blocks_nonempty.csv"
BLOCKS_DONE = "gain_blocks_done.json"
COUNTS_CSV = "gain_tile_counts_gee10.csv"       # tiles with gain > 0 only
TILES_DONE = "gain_tile_chunks_done10.json"
GAIN_ONLY_GEOJSON = "gain_tiles_10m.geojson"    # the original HRL-only grid, still written for reference
OUT = "gain_tiles_10m_combined.geojson"

# ---- download
CATALOGUE_URL = ("https://s3.waw3-1.cloudferro.com/swift/v1/CatalogueCSV/landcover_landuse/"
                 "tree_cover_density/{ds}/{ds}_cog.csv")
DATASETS = {
    "change": "clms_vlcc_tree-cover-presence-change_europe_20m_3yearly_v1",
    "confidence": "clms_vlcc_tree-cover-presence-change-confidence-layer_europe_20m_3yearly_v1",
}
S3_ENDPOINT, S3_BUCKET = "https://eodata.dataspace.copernicus.eu", "eodata"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
PERIOD_RE = re.compile(r"C(\d{4}-\d{4})")

# ---- raster building
NO_TREE_CODE, GAIN_CODE, STABLE_CODE = 0, 1, 10
CHANGE_VALUES = (0, 1, 2, 10)
CONF_BACKGROUND = (253, 255)
TILE_RE = re.compile(r"E\d{2,3}N\d{2,3}")
PRODUCT_RE = re.compile(r"HRLVLCC_([A-Z0-9]+)_C\d{4}-\d{4}")
CONF_SUFFIX = "CL"
STRIP = 2048
BAND_NAMES = ["gain_period", "gain_conf"]

# ---- EE counting
BLOCK, BLOCK_CHUNK, CHUNK, WORKERS = 4, 200, 250, 16


# =============================================================== STAGE 1: download
def fetch_catalogue(dataset, refresh=False):
    cache = ROOT / f"{dataset}_cog.csv"
    if refresh or not cache.exists():
        url = CATALOGUE_URL.format(ds=dataset)
        print(f"Fetching catalogue: {url}")
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                cache.write_bytes(r.read())
        except urllib.error.URLError as e:
            print(f"  WARNING: could not fetch catalogue for {dataset}: {e}")
            return []
    return list(csv.DictReader(io.StringIO(cache.read_text(encoding="utf-8")), delimiter=";"))


def _wkt_bbox(wkt):
    n = [float(x) for x in NUM.findall(wkt)]
    return min(n[0::2]), min(n[1::2]), max(n[0::2]), max(n[1::2])


def _row_period(row):
    m = PERIOD_RE.search(row["name"])
    return m.group(1) if m else f"{row['content_date_start'][:4]}-{row['content_date_end'][:4]}"


def _select_rows(rows, bbox):
    out = []
    for r in rows:
        if _row_period(r) not in (PERIOD_A, PERIOD_B) or "_03035_" not in r["name"]:
            continue                                  # skip other periods and overseas (non-3035) tiles
        if bbox:
            x0, y0, x1, y1 = _wkt_bbox(r["bbox"])
            if x1 < bbox[0] or x0 > bbox[2] or y1 < bbox[1] or y0 > bbox[3]:
                continue
        out.append(r)
    return out


def _list_objects(s3, prefix):
    objs = []
    for pfx in (prefix.rstrip("/") + "/", prefix):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=pfx):
            for o in page.get("Contents", []):
                k = o["Key"]
                if k == prefix or k.startswith((prefix + "/", prefix + ".")):
                    objs.append((k, o["Size"]))
        if objs:
            break
    return objs


def _download_tile(s3, row):
    prefix = row["s3_path"].replace(f"s3://{S3_BUCKET}/", "", 1).rstrip("/")
    dest_dir = ROOT / _row_period(row) / row["name"]
    objs = _list_objects(s3, prefix)
    if not objs:
        return row["name"], "no objects found under S3 path"
    for key, size in objs:
        if not key.lower().endswith((".tif", ".tiff")):
            continue
        dest = dest_dir / (key[len(prefix):].lstrip("/") or Path(key).name)
        if dest.exists() and dest.stat().st_size == size:
            continue                                  # resumable
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        s3.download_file(S3_BUCKET, key, str(tmp))
        tmp.replace(dest)
    return row["name"], None


def stage_download(bbox, workers=6, refresh=False):
    import boto3
    from botocore.config import Config

    ak, sk = os.environ.get("CDSE_S3_ACCESS_KEY"), os.environ.get("CDSE_S3_SECRET_KEY")
    if not (ak and sk):
        sys.exit("Set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY (free keys: "
                 "https://eodata-s3keysmanager.dataspace.copernicus.eu/)")
    ROOT.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=ak, aws_secret_access_key=sk,
                      region_name="default",
                      config=Config(max_pool_connections=max(10, workers * 2), retries={"max_attempts": 5}))
    failed = []
    for label in ("change", "confidence"):             # the confidence layer feeds gain_conf, so both are needed
        sel = _select_rows(fetch_catalogue(DATASETS[label], refresh), bbox)
        mb = sum(int(r["content_length"]) for r in sel) / 1e6
        print(f"\n[{label}] {len(sel):,} tiles selected (~{mb:,.0f} MB)")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_download_tile, s3, r): r for r in sel}
            for fut in as_completed(futs):
                try:
                    name, err = fut.result()
                except Exception as e:
                    name, err = futs[fut]["name"], f"{type(e).__name__}: {e}"
                if err:
                    failed.append((name, err))
                done += 1
                if done % 25 == 0 or done == len(sel):
                    print(f"  {done:,}/{len(sel):,} ({len(failed)} failed)", flush=True)
    if failed:
        (ROOT / "failed.txt").write_text("\n".join(f"{n}\t{e}" for n, e in failed), encoding="utf-8")
        print(f"{len(failed)} tile(s) failed - see {ROOT / 'failed.txt'}. Re-run to retry (finished files are skipped).")


# =============================================================== STAGE 2: per-tile 20 m rasters + merge
def _index(period):
    out = {}
    for tif in (ROOT / period).rglob("*.tif"):
        m = TILE_RE.search(tif.name)
        if m:
            out.setdefault(m.group(), []).append(tif)
    return out


def _product(path):
    m = PRODUCT_RE.search(path.name)
    return m.group(1) if m else None


def _is_conf_name(path):
    p = _product(path)
    return (p is not None and p.endswith(CONF_SUFFIX)) or "conf" in str(path.relative_to(ROOT)).lower()


def _looks_like_change(path, step=20):
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as s:
        a = s.read(1, out_shape=(max(s.height // step, 1), max(s.width // step, 1)), resampling=Resampling.nearest)
    valid = a[(a != 255) & (a != 253)]
    return None if valid.size == 0 else bool(np.isin(valid, CHANGE_VALUES).mean() > 0.99)


def _pick_change(files):
    cands = sorted(f for f in files if not _is_conf_name(f))
    if not cands:
        return None
    good = [f for f in cands if _looks_like_change(f) is True]
    if good:
        return good[0]
    if len(cands) == 1:
        if _looks_like_change(cands[0]) is False:
            print(f"  WARNING: {cands[0].name} is named like a change raster but its values are not {CHANGE_VALUES}")
        return cands[0]
    return None


def _pick_conf(files, change_file):
    confs = sorted(f for f in files if f != change_file and _is_conf_name(f))
    if not confs:
        return None
    want = (_product(change_file) or "") + CONF_SUFFIX
    return ([f for f in confs if _product(f) == want] or confs)[0]


def _in_bbox(path, bbox):
    import rasterio
    from rasterio.warp import transform_bounds
    if bbox is None:
        return True
    with rasterio.open(path) as s:
        l, b, r, t = transform_bounds("EPSG:4326", s.crs, *bbox)
        return not (r < s.bounds.left or l > s.bounds.right or t < s.bounds.bottom or b > s.bounds.top)


def _build_tile(key, ca, cb, qa, qb, force):
    import rasterio
    from rasterio.windows import Window
    dst_path = RASTER_DIR / f"{key}_gain_20m.tif"
    if dst_path.exists() and not force:
        return dst_path, None
    srcs = {"A": rasterio.open(ca), "B": rasterio.open(cb)}
    if qa is not None:
        srcs["QA"] = rasterio.open(qa)
    if qb is not None:
        srcs["QB"] = rasterio.open(qb)
    ha = 0.0
    try:
        A = srcs["A"]
        for name, s in srcs.items():
            if s.transform != A.transform or s.shape != A.shape:
                raise ValueError(f"{key}: raster '{name}' is not on the same grid as the period-A change raster")
        H, Wd = A.height, A.width
        prof = dict(driver="GTiff", dtype="uint8", count=len(BAND_NAMES), width=Wd, height=H, crs=A.crs,
                    transform=A.transform, nodata=0, tiled=True, blockxsize=512, blockysize=512,
                    compress="deflate", predictor=2, sparse_ok=True, BIGTIFF="IF_SAFER")
        with rasterio.open(dst_path, "w", **prof) as dst:
            for r0 in range(0, H, STRIP):
                win = Window(0, r0, Wd, min(STRIP, H - r0))
                a, b = srcs["A"].read(1, window=win), srcs["B"].read(1, window=win)
                # early gain: new tree cover in 2018-21 that is still there in 2021-24; late gain: none -> new in 2021-24
                early = (a == GAIN_CODE) & (b == STABLE_CODE)
                late = (a == NO_TREE_CODE) & (b == GAIN_CODE)
                if not (early.any() or late.any()):
                    continue
                period = np.where(early, 1, np.where(late, 2, 0)).astype("uint8")
                conf = np.zeros(a.shape, "uint8")
                for mask, q in ((early, "QA"), (late, "QB")):
                    if not mask.any():
                        continue
                    ha += float(mask.sum()) * 400.0 / 1e4      # 20 m pixel = 400 m2
                    if q in srcs:
                        conf = np.where(mask, srcs[q].read(1, window=win), conf).astype("uint8")
                dst.write(np.stack([period, conf]), window=win)
            for i, n in enumerate(BAND_NAMES, 1):
                dst.set_band_description(i, n)
    finally:
        for s in srcs.values():
            s.close()
    if ha == 0:
        dst_path.unlink()
        return None, 0.0
    return dst_path, ha


def _merge_tiles(paths, dst_path, res=20.0):
    import rasterio
    from affine import Affine
    from rasterio.windows import Window
    infos = []
    for p in paths:
        with rasterio.open(p) as s:
            infos.append((p, s.bounds, s.width, s.height, s.crs))
    crs = infos[0][4]
    if any(i[4] != crs for i in infos):
        raise ValueError("tiles are not all in the same CRS")
    minx, maxy = min(i[1].left for i in infos), max(i[1].top for i in infos)
    maxx, miny = max(i[1].right for i in infos), min(i[1].bottom for i in infos)
    wd, ht = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
    prof = dict(driver="GTiff", dtype="uint8", count=len(BAND_NAMES), width=wd, height=ht, crs=crs,
                transform=Affine(res, 0, minx, 0, -res, maxy), nodata=0, tiled=True, blockxsize=512,
                blockysize=512, compress="deflate", predictor=2, sparse_ok=True, BIGTIFF="IF_SAFER")
    with rasterio.open(dst_path, "w", **prof) as dst:
        for p, b, w, h, _ in infos:
            fx, fy = (b.left - minx) / res, (maxy - b.top) / res
            if abs(fx - round(fx)) > 1e-6 or abs(fy - round(fy)) > 1e-6:
                raise ValueError(f"{p.name} is not aligned to the common 20 m grid")
            xoff, yoff = int(round(fx)), int(round(fy))
            with rasterio.open(p) as s:
                for r0 in range(0, h, STRIP):
                    win = Window(0, r0, w, min(STRIP, h - r0))
                    data = s.read(window=win)
                    if data.any():
                        dst.write(data, window=Window(xoff, yoff + r0, w, win.height))
        for i, n in enumerate(BAND_NAMES, 1):
            dst.set_band_description(i, n)
    return wd, ht


def stage_rasters(bbox=None, force=False):
    idx_a, idx_b = _index(PERIOD_A), _index(PERIOD_B)
    keys = [k for k in sorted(set(idx_a) & set(idx_b)) if _in_bbox(idx_a[k][0], bbox)]
    pairs, skipped, no_conf = {}, [], []
    for k in keys:
        ca, cb = _pick_change(idx_a[k]), _pick_change(idx_b[k])
        if ca is None or cb is None:
            skipped.append(k)
            continue
        qa, qb = _pick_conf(idx_a[k], ca), _pick_conf(idx_b[k], cb)
        if qa is None or qb is None:
            no_conf.append(k)
        pairs[k] = (ca, cb, qa, qb)
    print(f"{len(keys)} tiles in both periods; {len(pairs)} with change rasters in both")
    if skipped:
        print(f"  skipped (no change-class raster): {skipped}")
    if no_conf:
        print(f"  WARNING: confidence raster missing for {no_conf}; gain_conf is 0 there")
    if not pairs:
        sys.exit(f"No tile pairs found under {ROOT}. Run the download stage first.")

    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    files, new_ha = [], 0.0
    for i, (key, args) in enumerate(pairs.items(), 1):
        path, ha = _build_tile(key, *args, force)
        if path is not None:
            files.append(path)
        if ha is not None:
            new_ha += ha
        print(f"[{i}/{len(pairs)}] {key}: " + ("cached" if ha is None else f"{ha:,.1f} ha" if path else "no candidates"))
    if not files:
        sys.exit("No gain pixels in any tile; check the class codes.")
    wd, ht = _merge_tiles(sorted(files), MERGED)           # always re-merge so it can't be stale
    print(f"\nMerged {len(files)} tiles -> {MERGED} ({wd} x {ht} px, {MERGED.stat().st_size / 1e6:,.0f} MB); "
          f"newly built this run: {new_ha:,.0f} ha")
    if MERGED.stat().st_size > 10e9:
        print("WARNING: over 10 GB; GEE's per-file upload limit would be exceeded")


# =============================================================== STAGE 3: ingest into Earth Engine
def _sh(cmd, check=True):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    for out in (r.stdout, r.stderr):
        if out.strip():
            print(out.strip())
    if check and r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def _asset_exists(asset):
    import ee
    try:
        ee.data.getAsset(asset)
        return True
    except ee.EEException:
        return False


def stage_ingest(keep_bucket=False, timeout_h=8.0):
    import ee
    ee.Initialize(project=PROJECT)
    if _asset_exists(ASSET):
        print(f"{ASSET} already exists - skipping ingest (delete the asset first to re-ingest).")
        return
    if not MERGED.exists():
        sys.exit(f"{MERGED} not found; run the rasters stage first.")
    tif = MERGED.name
    if _sh(f"gcloud storage buckets describe gs://{BUCKET_NAME}", check=False).returncode != 0:
        _sh(f"gcloud storage buckets create gs://{BUCKET_NAME} --project={PROJECT} --location={BUCKET_LOCATION} "
            f"--uniform-bucket-level-access --soft-delete-duration=0")
    _sh(f'gcloud storage cp "{MERGED}" gs://{BUCKET_NAME}/{tif}')
    manifest = {
        "name": ASSET,
        "tilesets": [{"sources": [{"uris": [f"gs://{BUCKET_NAME}/{tif}"]}]}],
        "bands": [{"id": "gain_period", "tilesetBandIndex": 0}, {"id": "gain_conf", "tilesetBandIndex": 1}],
        "missingData": {"values": [0]},
        "pyramidingPolicy": "MAX",
        "properties": {"resolution_m": 20, "gain_period_1": PERIOD_A, "gain_period_2": PERIOD_B},
    }
    Path("manifest.json").write_text(json.dumps(manifest, indent=2))
    _sh("earthengine upload image --manifest manifest.json")

    print("Waiting for ingestion (the asset appears when it finishes)...")
    t0 = time.time()
    while not _asset_exists(ASSET):
        if time.time() - t0 > timeout_h * 3600:
            sys.exit("Timed out waiting for ingestion. Check `earthengine task list`, then re-run with --stages count combine.")
        time.sleep(60)
        print(f"  {int((time.time() - t0) / 60)} min ... (`earthengine task list` shows status / errors)", flush=True)
    print("Ingestion complete.")
    if not keep_bucket:
        _sh(f"gcloud storage rm -r gs://{BUCKET_NAME}", check=False)


# =============================================================== STAGE 4: count gain per tile (Earth Engine, 10 m)
def _load_done(path):
    return set(json.loads(Path(path).read_text())) if Path(path).exists() else set()


def _parallel(fn, jobs, on_result, label, log_path, done):
    failed, completed = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fn, job): index for index, job in jobs}
        for fut in as_completed(futures):
            index = futures[fut]
            try:
                on_result(fut.result())
            except Exception as exc:
                failed.append(index)
                print(f"\n{label} chunk {index} failed: {str(exc)[:200]}")
                continue
            done.add(index)
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                Path(log_path).write_text(json.dumps(sorted(done)))
                print(f"{label}: {completed:,}/{len(futures):,}", end="\r")
    Path(log_path).write_text(json.dumps(sorted(done)))
    if failed:
        print(f"\n{label}: {len(failed)} chunks failed. Re-run this stage (without --fresh) to retry them.")
    print()


def stage_count(fresh=False):
    import ee
    ee.Initialize(project=PROJECT)
    if fresh:
        for f in (BLOCKS_CSV, BLOCKS_DONE, COUNTS_CSV, TILES_DONE, GAIN_ONLY_GEOJSON):
            Path(f).unlink(missing_ok=True)
        print("Cleared count-stage outputs; starting from scratch.")

    img = ee.Image(ASSET)
    period, conf = img.select("gain_period"), img.select("gain_conf")

    def m10(mask):
        return mask.unmask(0).toFloat().resample("bilinear").reproject("EPSG:3035", None, PX).gt(0.5)

    stack = ee.Image.cat([
        m10(period.gt(0)).rename("gain_px"),
        m10(period.eq(1)).rename("early_px"),
        m10(period.eq(2)).rename("late_px"),
        conf.updateMask(period.gt(0)).unmask(0).toFloat().rename("conf_sum"),
        period.gt(0).unmask(0).toFloat().rename("n20"),
    ])
    presence = period.gt(0).rename("g")

    def getinfo(fc, retries=3):
        for attempt in range(retries):
            try:
                return fc.getInfo()["features"]
            except Exception:
                if attempt == retries - 1:
                    raise

    def rect(xi, yi, w=1):
        return ee.Geometry.Rectangle([xi * T, yi * T, (xi + w) * T, (yi + w) * T], "EPSG:3035", False)

    # ---- Stage A: which BLOCK x BLOCK-tile blocks contain any gain
    grid = ee.data.getAsset(ASSET)["bands"][0]["grid"]
    aff = grid["affineTransform"]
    x0, y1 = aff["translateX"], aff["translateY"]
    x1 = x0 + grid["dimensions"]["width"] * abs(aff["scaleX"])
    y0 = y1 - grid["dimensions"]["height"] * abs(aff["scaleY"])
    tx0, ty0, tx1, ty1 = (int(np.floor(x0 / T)), int(np.floor(y0 / T)), int(np.ceil(x1 / T)), int(np.ceil(y1 / T)))
    blocks = [(bx, by) for bx in range(tx0 // BLOCK, -(-tx1 // BLOCK)) for by in range(ty0 // BLOCK, -(-ty1 // BLOCK))]
    chunks = [blocks[i:i + BLOCK_CHUNK] for i in range(0, len(blocks), BLOCK_CHUNK)]
    done = _load_done(BLOCKS_DONE)
    todo = [(i, chunks[i]) for i in range(len(chunks)) if i not in done]
    print(f"Stage A: {len(blocks):,} blocks, {len(chunks):,} chunks; {len(done):,} already complete.")

    def fn_a(chunk):
        feats = ee.FeatureCollection([ee.Feature(rect(bx * BLOCK, by * BLOCK, BLOCK), {"bx": bx, "by": by})
                                      for bx, by in chunk])
        red = presence.reduceRegions(collection=feats, reducer=ee.Reducer.count(), scale=20,
                                     crs="EPSG:3035", tileScale=4)
        return [{"bx": f["properties"]["bx"], "by": f["properties"]["by"]}
                for f in getinfo(red) if (f["properties"].get("count") or 0) > 0]

    first = [not Path(BLOCKS_CSV).exists()]

    def save_a(rows):
        if rows:
            pd.DataFrame(rows).to_csv(BLOCKS_CSV, mode="a", header=first[0], index=False)
            first[0] = False

    _parallel(fn_a, todo, save_a, "Stage A", BLOCKS_DONE, done)
    nonempty = (pd.read_csv(BLOCKS_CSV).drop_duplicates(["bx", "by"])
                if Path(BLOCKS_CSV).exists() else pd.DataFrame(columns=["bx", "by"]))

    # ---- Stage B: count gain per tile at 10 m inside those blocks
    if nonempty.empty:
        print("Stage B: no blocks contain gain.")
    else:
        counted = set()
        if Path(COUNTS_CSV).exists():
            c = pd.read_csv(COUNTS_CSV, usecols=["xi", "yi"])
            counted = set(zip(c.xi, c.yi))
        cands = sorted((bx * BLOCK + i, by * BLOCK + j) for bx, by in zip(nonempty.bx, nonempty.by)
                       for i in range(BLOCK) for j in range(BLOCK))
        cands = [t for t in cands if t not in counted]
        chunks = [cands[i:i + CHUNK] for i in range(0, len(cands), CHUNK)]
        done = _load_done(TILES_DONE)
        todo = [(i, chunks[i]) for i in range(len(chunks)) if i not in done]
        print(f"Stage B: {len(cands):,} tiles to count in {len(chunks):,} chunks; {len(done):,} already complete.")

        def fn_b(tiles):
            feats = ee.FeatureCollection([ee.Feature(rect(xi, yi), {"xi": xi, "yi": yi}) for xi, yi in tiles])
            red = stack.reduceRegions(collection=feats, reducer=ee.Reducer.sum(), scale=PX,
                                      crs="EPSG:3035", tileScale=4)
            rows = []
            for f in getinfo(red):
                p = f["properties"]
                if (p.get("gain_px") or 0) > 0:               # only tiles containing gain are written
                    n20, cs = p.get("n20") or 0, p.get("conf_sum") or 0
                    rows.append({"xi": p["xi"], "yi": p["yi"], "gain_px": p["gain_px"],
                                 "early_px": p.get("early_px") or 0, "late_px": p.get("late_px") or 0,
                                 "conf_mean": cs / n20 if n20 else np.nan})
            return rows

        first = [not Path(COUNTS_CSV).exists()]

        def save_b(rows):
            if rows:
                pd.DataFrame(rows).to_csv(COUNTS_CSV, mode="a", header=first[0], index=False)
                first[0] = False

        _parallel(fn_b, todo, save_b, "Stage B", TILES_DONE, done)

    if not Path(COUNTS_CSV).exists():
        sys.exit("No tile contains HRL gain; nothing to combine.")
    print(f"Counts in {COUNTS_CSV}")


# =============================================================== STAGE 5: combine with woodland polygons
def _read_shapes(path):
    import geopandas as gpd
    import shapely
    p = Path(path)
    if not p.exists():
        sys.exit(f"Woodland data not found: {p}")
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.suffix.lower() in {".shp", ".gpkg", ".geojson", ".json"}
        and not any(part.startswith(".") for part in f.parts))
    if not files:
        sys.exit(f"No spatial files in {p}")
    frames = []
    for f in files:
        g = gpd.read_file(f)
        if g.empty or g.crs is None:
            print(f"  skipping {f} (empty or no CRS)")
            continue
        frames.append(g.to_crs("EPSG:3035")[["geometry"]])
    g = pd.concat(frames, ignore_index=True)
    g["geometry"] = shapely.make_valid(g.geometry.values)
    print(f"Read {len(g):,} woodland polygons from {len(frames)} file(s)")
    return gpd.GeoDataFrame(g, geometry="geometry", crs="EPSG:3035")


def _woodland_per_tile(shapes):
    """Exact woodland area inside every lattice tile that touches a polygon."""
    import shapely
    from shapely import STRtree
    # dissolve first so overlapping polygons (e.g. the same site in several quarters) are not double counted
    parts = np.asarray(shapely.get_parts(shapely.union_all(shapes.geometry.values)))
    parts = parts[~shapely.is_empty(parts)]
    tree = STRtree(parts)
    cand = set()
    for bx0, by0, bx1, by1 in shapely.bounds(parts):
        for xi in range(int(np.floor(bx0 / T)), int(np.floor(bx1 / T)) + 1):
            for yi in range(int(np.floor(by0 / T)), int(np.floor(by1 / T)) + 1):
                cand.add((xi, yi))
    cand = np.array(sorted(cand))
    boxes = shapely.box(cand[:, 0] * T, cand[:, 1] * T, (cand[:, 0] + 1) * T, (cand[:, 1] + 1) * T)
    bi, pi = tree.query(boxes, predicate="intersects")
    area = shapely.area(shapely.intersection(boxes[bi], parts[pi]))
    out = pd.DataFrame({"xi": cand[:, 0], "yi": cand[:, 1],
                        "woodland_m2": np.bincount(bi, weights=area, minlength=len(boxes))})
    return out[out.woodland_m2 > 0]


def stage_combine(woodland=WOODLAND, counts_csv=COUNTS_CSV, out=OUT, min_pct=MIN_PCT):
    import geopandas as gpd
    import shapely
    shapes = _read_shapes(woodland)
    wood = _woodland_per_tile(shapes)
    gain = pd.read_csv(counts_csv).drop_duplicates(["xi", "yi"])

    df = gain.merge(wood, on=["xi", "yi"], how="outer")
    for c in ("gain_px", "early_px", "late_px", "woodland_m2"):
        df[c] = df[c].fillna(0.0)        # absent from the counts => exactly zero gain (stage 4 omits zero-gain tiles)
    for c in ("gain", "early", "late"):
        df[f"{c}_pct"] = 100 * df[f"{c}_px"] / PX_TILE
    df["woodland_pct"] = 100 * df.woodland_m2 / (T * T)

    # woodland_pct is only meaningful inside the woodland data's extent (England); elsewhere it is unknown, not 0
    x0, y0, x1, y1 = shapes.total_bounds
    inside = ((df.xi * T + T > x0) & (df.xi * T < x1) & (df.yi * T + T > y0) & (df.yi * T < y1))
    df.loc[~inside, ["woodland_pct", "woodland_m2"]] = np.nan

    has_gain = df.gain_pct >= min_pct
    has_wood = df.woodland_pct.fillna(0) >= min_pct
    df = df[has_gain | has_wood].copy()
    df["source"] = np.select([has_gain[df.index] & has_wood[df.index], has_wood[df.index]], ["both", "woodland"], "hrl")

    df["x_min_m"], df["y_min_m"] = df.xi * T, df.yi * T
    df["x_max_m"], df["y_max_m"] = df.x_min_m + T, df.y_min_m + T
    df["tile_id"] = "tile_" + df.xi.astype(int).astype(str) + "_" + df.yi.astype(int).astype(str)
    df = df.reset_index(drop=True)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:3035",
                           geometry=shapely.box(df.x_min_m, df.y_min_m, df.x_max_m, df.y_max_m))
    gdf.to_file(out, driver="GeoJSON")
    print(gdf.source.value_counts().to_string(), f"\n-> {out}  ({len(gdf):,} tiles)")

    # keep the original HRL-only grid too, for reference
    only = gdf[gdf.gain_pct >= min_pct]
    only.to_file(GAIN_ONLY_GEOJSON, driver="GeoJSON")
    print(f"-> {GAIN_ONLY_GEOJSON}  ({len(only):,} tiles with >= {min_pct:g}% gain)")
    return gdf


# =============================================================== main
STAGES = ["download", "rasters", "ingest", "count", "combine"]


def main():
    global WOODLAND, MIN_PCT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                    help="only HRL tiles intersecting this lon/lat box (default: all of Europe, a very large download)")
    ap.add_argument("--woodland", default=WOODLAND, help="folder or file with the woodland polygons")
    ap.add_argument("--min-pct", type=float, default=MIN_PCT, help="keep tiles with gain %% or woodland %% >= this")
    ap.add_argument("--fresh", action="store_true", help="count stage: discard checkpoints and recount")
    ap.add_argument("--force-rasters", action="store_true", help="rasters stage: rebuild cached per-tile rasters")
    ap.add_argument("--keep-bucket", action="store_true", help="ingest stage: do not delete the temp bucket afterwards")
    ap.add_argument("--workers", type=int, default=6, help="download threads")
    args = ap.parse_args()
    WOODLAND, MIN_PCT = args.woodland, args.min_pct

    for s in STAGES:
        if s not in args.stages:
            continue
        print(f"\n{'=' * 20} {s} {'=' * 20}")
        if s == "download":
            stage_download(args.bbox, args.workers)
        elif s == "rasters":
            stage_rasters(args.bbox, args.force_rasters)
        elif s == "ingest":
            stage_ingest(args.keep_bucket)
        elif s == "count":
            stage_count(args.fresh)
        elif s == "combine":
            stage_combine(WOODLAND, COUNTS_CSV, OUT, MIN_PCT)


if __name__ == "__main__":
    main()
