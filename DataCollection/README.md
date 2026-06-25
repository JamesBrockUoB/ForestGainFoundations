# deadtrees.earth — Europe Composite Maps (EPSG:3035)

This folder contains annual 10 m GeoTIFF map products for Europe in **EPSG:3035 (ETRS89 / LAEA Europe)** produced by the **deadtrees.earth** project. This maps are a first large-scale prototype of this approach and meant to be interpreted with caution!

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

**Standing deadwood cover is a subset of and upper bounded by tree cover in it's
definition (standing deadwood cover <= tree cover).** However, model prediction
is not bounded and maps may contain pixels where standing deadwood cover
exceeds tree cover.

Pixel values are encoded as integers in the range **0–255**, corresponding to **0–100%**.

For standing deadwood cover, accuracy decreases with small fractional cover
values. For conservative estimates consider filtering for >50% cover. For
forest cover, accuracy is generally very high. When comparing year over year
cover values for change detection, consider applying a minimum threshold of
20%. Also, the years 2017 and 2018 can be less stable. 

## Download (example)

- using wget: `wget
  https://data.rsc4earth.de/download/deadtrees.earth_maps/v1_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif`
- using rasterio:
  `rasterio.open("https://data.rsc4earth.de/download/deadtrees.earth_maps/v1_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif")`

### Add a raster in QGIS via HTTPS (manual)

1) In QGIS, open **Layer → Add Layer → Add Raster Layer...**
2) Choose **Protocol: HTTP(S)** and paste the file URL, e.g.
   `https://data.rsc4earth.de/download/deadtrees.earth_maps/v1_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif`
3) Click **Add**, then **Close**. The raster will stream over HTTPS.


### Subsetting over HTTP with a cutline (GDAL)

Use `gdalwarp` with `/vsicurl/` to stream a GeoTIFF via HTTPS, clip it to your AOI polygon, and save only that subset locally. The `-cutline ... -crop_to_cutline` options apply the GeoJSON geometry as the clip boundary, and `-dstalpha` masks pixels outside the AOI. `--config GDAL_PAM_ENABLED NO` avoids GDAL trying to write sidecar metadata for the remote file.

Example:
```bash
gdalwarp -of GTiff -b 1 \
  -cutline myshapefile.geojson -crop_to_cutline \
  -co COMPRESS=DEFLATE -co PREDICTOR=2 -co OVERVIEWS=AUTO \
  --config GDAL_PAM_ENABLED NO \
  "/vsicurl/https://data.rsc4earth.de/download/deadtrees.earth_maps/v1_europe_composite_epsg_3035/standing_deadwood_cover_2024.tif" \
  standing_deadwood_cover_2024_subset.tif
```
