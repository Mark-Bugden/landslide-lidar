# Provenance check of inputs 1–4 (gate G1)

**Outcome: G1 PASS.** [E] Run of 2026-11-26, 19:48–19:52 UTC, code at commit `3f73522`, config hash
`e9969c41…`. Every check agreed; no source was switched and nothing was worked around.

Re-run with `uv run python scripts/provenance_check.py` (add `--refetch` to download the sample again). The full
record is `results/provenance.json`. Sheet-grid coordinates are left out of it because they derive from the
surveyed-sheet layer, whose licence is not stated; they are kept in the gitignored
`data/cache/provenance/provenance_full.json`. Decisions: `docs/decisions/P1-03.md`.

## Results

| Check | Expected | Found [E] | Agree |
| --- | --- | --- | --- |
| Register features (row 2), `returnCountOnly=true` | 21,356 (brief), 21,356 (package file) | 21,356 | yes |
| Surveyed sheets (row 3), `returnCountOnly=true` | 860 (brief), 860 (package file) | 860 | yes |
| Geology features (row 4), `subregion IN (5 Carpathian subregions)` | 597 (package file) | 597 | yes |
| Package fields present in the live layers | all | all (rows 2–4) | yes |
| DTM grid: recipe vs `fetch_log.jsonl` | 860 equal | 860 of 860 | yes |
| DTM grid: recipe vs each raster's transform and size | 860 equal | 860 of 860 | yes |
| DTM grid: `fetch_log.jsonl` vs raster | 860 equal | 860 of 860 | yes |
| Package rasters: NaN fraction, zmin, zmax vs `fetch_log.jsonl` | 860 equal | 860 of 860 | yes |
| DTM re-download, georeferencing (size, transform, tiepoint, pixel scale, GeoKeys) | equal | 12 of 12 | yes |
| DTM re-download, values where the package is not NaN | bitwise equal | 81,849,089 of 81,849,089 px; 0 differ | yes |
| DTM re-download, package NaN cells explained | all | 0 package NaN cells in the sample | yes (vacuous) |
| DTM re-download, NoData or out-of-range download cells that the package kept | 0 | 0 | yes |
| sha256 of `data/raw_data/` against `MANIFEST.csv` | 863 match | 863 of 863 (11.41 GB) | yes |

**Sample.** 12 sheets drawn with seed 2611, stream `provenance_sample` (id 1): 24-44-15, 34-21-20, 12-12-23,
02-24-23, 25-31-05, 15-34-25, 25-14-20, 02-23-18, 25-21-14, 25-24-18, 24-14-18, 04-33-10. Two of them are border
sheets: 25-24-18 (5,083,803 cells of 0 m) and 04-33-10 (607,827). Their 0 m cells match bitwise too.

**Layer metadata** (read before any query) [E]:
- DMR 5G ImageServer: 2 × 2 m, F32, 1 band, SR wkid 102067 (latest 5514), vertical Baltic 1957 (Bpv), max image
  15,000 × 4,100. The largest recipe request is 2,902 × 2,496.
- The three ČGS layers: `esriGeometryPolygon`, native SR wkid 102067 (latest 5514). maxRecordCount is 1,000 for the
  register, 1,000 for the sheets and 2,000 for the geology.

## Things to know

- **No package raster has a NaN.** [E] A scan of all 860 package rasters finds 0 NaN cells and 0 negative cells, and
  the fetch log's `invalid_frac` is 0 everywhere. So the manifest's stated change ("NoData and values outside
  [−100, 2000] m set to NaN") changed no pixel, and the "every package NaN explained" clause holds trivially. 96 sheets
  contain 0 m cells (outside the border), matching the spike's 96 of 860 (PLAN.md §5; P1-08 owns that check).
- **The downloaded TIFF's CRS is encoded differently from the package's.** [E] Both declare
  `ProjectedCSTypeGeoKey = 5514`, PixelIsArea and identical tiepoint and pixel-scale tags. The service also writes an
  ESRI PE string (`X_Scale = −1`, `XY_Plane_Rotation = 90`), and GDAL 3.12 therefore reads the download's CRS as a
  `LOCAL_CS` with authority EPSG:5514 and no projection parameters. The package file carries the full EPSG:5514
  definition. The grid is the same; only the CRS encoding differs. Georeferencing was therefore compared at the tag
  level (decision D-new-4). Downstream code reads only the package rasters.
- **The reprojection depends on PROJ's per-area choice.** [E] With pyproj 3.8.0 / PROJ 9.8.1, the default
  EPSG:4326 → 5514 operation uses three Helmert variants across the sheets (824, 493 and 3 of 1,320 vertices). Any
  single fixed variant reproduces only 47–600 of the 860 grids. The tightest sheet, 24-33-01, has a buffered edge
  0.08 mm from a lattice line, and 5 sheets are under 1 mm. Later tasks should reuse `landslide.sheets.load_sheets` and
  read each sheet's grid from its raster.
- **Two files are not in the manifest:** `MANIFEST.csv` itself and `dmr5g/fetch_log.jsonl` (sha256 `a3869c7e…`).
  The fetch log's content is checked independently: its grids and statistics agree with the rasters for all 860
  sheets.
- **Rows 2–4 were only counted, not downloaded,** as the brief asks. The paging code of their fetch scripts is
  unit-tested but has not run against the live services.
- **A full DTM fetch** would be 860 requests and about 23 GB of uncompressed F32 responses (28.6 MB for a typical
  sheet; about 4 s per sheet in this run with the comparison included, so roughly 1 h). The brief's "about 11 GB" matches the deflate-compressed package (11.41 GB).
