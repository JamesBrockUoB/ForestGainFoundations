# deadtrees.earth — Europe Composite Maps v5 (EPSG:3035)

This folder contains annual 10 m GeoTIFF map products for Europe in **EPSG:3035 (ETRS89 / LAEA Europe)** produced by the **deadtrees.earth** project, for the years **2017–2025**. These maps are a large-scale prototype of this approach and are meant to be interpreted with caution!

This is **version 5**. It has the same grid, encoding and file layout as the v1 maps (`v1_europe_composite_epsg_3035`), so v5 files can replace v1 files one to one. What changed is listed under "Changes since v1".

For questions and feedback (explicitly welcome!): clemens.mosig@uni-leipzig.de

## Citation

Mosig, C., Kattenborn, T., Montero Loaiza, D., Vanja-Jehle, J., Brandt, J., Jacobs, N., Khanal, S., Xing, E., Schwartz, M., Muller-Landau, H. C., Beloiu, M., Bozzini, A., Cheng, Y., Ganz, K., Grüning, B., Hartmann, H., Hempel, J., Horion, S., Junttila, S., Korznikov, K., Kraemer, G., Mönks, M., Nardi, D., Neumeier, P., Schmid, J., Soltani, S., Therese-Schmehl, M., Veitch-Michaelis, J., & Mahecha, M. (2026). Sub-pixel mapping of disturbance and tree mortality dynamics from Sentinel-2 time series around the globe. EarthArXiv. https://doi.org/10.31223/X5B18W

## Products

Each year is provided as two raster layers:

1) Tree cover (`tree_cover_<YEAR>.tif`): Fractional **tree canopy cover** per 10 m pixel.

2) Standing deadwood cover (`standing_deadwood_cover_<YEAR>.tif`): Fractional
**standing deadwood cover** per 10 m pixel. “Standing deadwood” refers to
standing dead trees.

**Fractional cover** is defined as the share of the pixel area covered by (dead)
tree crowns.

**Standing deadwood cover is a subset of and upper bounded by tree cover in its
definition (standing deadwood cover <= tree cover).** However, the model prediction
is not bounded and the maps may contain pixels where standing deadwood cover
exceeds tree cover.

Pixel values are encoded as integers in the range **0–255**, corresponding to **0–100%**
(band 1, `Byte`). Band 2 is a validity mask: `255` = covered by a processing block,
`0` = no data. Band 1 is `0` where there is no data; it has no nodata tag, so use band 2
to tell "no data" from "0 % cover".

For standing deadwood cover, accuracy decreases with small fractional cover
values. For conservative estimates consider filtering for >50% cover. For
tree cover, accuracy is generally very high. When comparing year-over-year
cover values for change detection, consider applying a minimum threshold of
20%. Also, the years 2017 and 2018 can be less stable.

In addition, one layer applies to all years:

3) Inference day of year (`inference_day_of_year.tif`): The day of year (1-366)
that the cover values at this location were predicted for. Within a processing
block the date is the same day of year every year. It is the middle of the
growing season at the block centre, taken from a MODIS/VIIRS phenology look-up;
the model sees the Sentinel-2 composites **before** that date. Across the
product the day ranges from **4 to 359** (median 196, mid July; 90 % of blocks
between day 116 and day 222), so consult this layer when the season matters,
e.g. when comparing regions or dating a disturbance within a year.

`UInt16`, `0` = no data, **100 m pixels** - coarser than the cover maps because
the value is constant over each ~30 km processing block; the grid is aligned
with the 10 m cover maps. Where two blocks overlap, the cell holds the mean of
the two days.

## Changes since v1

- **New model** (v5001 instead of v1004): retrained on a larger and corrected set of drone labels.
- **The negative bias in 2017/2018 has been reduced.**
- **Year-to-year tree cover is more consistent**, enabling better
  intercomparison of tree cover at one site.
- **Tree cover >= standing deadwood cover** often did not hold true in many larger mortality scenarios. A new constraint in the loss function now largely eliminates this issue, and where it is still violated, the difference is on the order of single percentage points.
- **Fixed cloud artifacts**: v1 showed many sites with cloud-like artifacts
  where tree cover and standing deadwood cover were ~20% off from what they should be. This
  has now been addressed and the cloud artifacts should be gone.
- **Inference dates are now taken at the right place.** v1 sampled the phenology look-up at the wrong
  longitude (see the v1 README). v5 samples it at the block centre, and uses a newer phenology
  version (v9). The dates therefore differ from v1: by a median of 20 days, and by more than 30 days
  for 36 % of the area, especially in southern Europe.
- **The date has a slightly different meaning.** In v1 the day of year was the centre of the 7-day
  composite that was predicted. In v5 it is the date the prediction is made for, and the model uses
  the composites strictly before it.
- **No clipping of values <4% to zero.**

## Known issues

- **Winter prediction dates around the Mediterranean.** For 116 blocks the phenology look-up places
  the middle of the growing season between December and February (111 blocks before day 60, 5 after
  day 300). They lie in southern Portugal and Spain, the Balearic Islands, Sardinia, Sicily, southern
  Italy, Greece and Crete. This follows the winter-green herbaceous vegetation there, not the tree
  canopy. Deciduous trees, orchards and vineyards are leafless at that time and may be mapped as
  standing deadwood. Use `inference_day_of_year.tif` to find these areas. We expect to change the
  date selection in a later version.
- **36 blocks have no 2025 value.** Their prediction date (late August or later) lies after the last
  Sentinel-2 composite available for 2025, so no 2025 prediction exists. They are no data in
  `tree_cover_2025.tif` and `standing_deadwood_cover_2025.tif`. Most are in northern Scotland, Orkney and
  Shetland; the others are scattered (western Ireland, Norway, Denmark, Poland, Sweden, Spain, Portugal, Italy):
    `block_32629_450000_5880000`, `block_32629_600000_4410000`, `block_32629_630000_4440000`,
    `block_32630_360000_6480000`, `block_32630_390000_6540000`, `block_32630_420000_6480000`,
    `block_32630_420000_6540000`, `block_32630_480000_6390000`, `block_32630_480000_6480000`,
    `block_32630_480000_6510000`, `block_32630_480000_6540000`, `block_32630_480000_6570000`,
    `block_32630_510000_6540000`, `block_32630_510000_6570000`, `block_32630_540000_6660000`,
    `block_32630_540000_6690000`, `block_32630_570000_6570000`, `block_32630_570000_6600000`,
    `block_32630_570000_6630000`, `block_32630_570000_6660000`, `block_32630_570000_6690000`,
    `block_32630_570000_6720000`, `block_32630_600000_6630000`, `block_32630_600000_6660000`,
    `block_32630_600000_6690000`, `block_32630_600000_6720000`, `block_32630_690000_4230000`,
    `block_32631_360000_4530000`, `block_32631_360000_4560000`, `block_32631_600000_6900000`,
    `block_32632_450000_6120000`, `block_32632_570000_6390000`, `block_32632_630000_6870000`,
    `block_32633_570000_4560000`, `block_32633_660000_5970000`, `block_32634_390000_6450000`

## Download (example)

- using wget: `wget
  https://data.rsc4earth.de/download/deadtrees.earth_maps/v5_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif`
- using rasterio:
  `rasterio.open("https://data.rsc4earth.de/download/deadtrees.earth_maps/v5_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif")`

### Add a raster in QGIS via HTTPS (manual)

1) In QGIS, open **Layer → Add Layer → Add Raster Layer...**
2) Choose **Protocol: HTTP(S)** and paste the file URL, e.g.
   `https://data.rsc4earth.de/download/deadtrees.earth_maps/v5_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif`
3) Click **Add**, then **Close**. The raster will stream over HTTPS.


### Subsetting over HTTP with a cutline (GDAL)

Use `gdalwarp` with `/vsicurl/` to stream a GeoTIFF via HTTPS, clip it to your AOI polygon, and save only that subset locally. The `-cutline ... -crop_to_cutline` options apply the GeoJSON geometry as the clip boundary, and `-dstalpha` masks pixels outside the AOI. `--config GDAL_PAM_ENABLED NO` avoids GDAL trying to write sidecar metadata for the remote file.

Example:
```bash
gdalwarp -of GTiff -b 1 \
  -cutline myshapefile.geojson -crop_to_cutline \
  -co COMPRESS=DEFLATE -co PREDICTOR=2 -co OVERVIEWS=AUTO \
  --config GDAL_PAM_ENABLED NO \
  "/vsicurl/https://data.rsc4earth.de/download/deadtrees.earth_maps/v5_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif" \
  standing_deadwood_cover_2024_subset.tif
```
