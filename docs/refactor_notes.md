# Refactor Notes

The public repository is based on the Python parts of the original coursework only.

## Python DSM generation

The original DSM script generated TIN and IDW rasters for first, last, and all returns with:

- resolution: 1.5 m
- IDW weight: 2.0
- IDW radius: 2.5 m

The original visualization converted elevation arrays to `uint8` before plotting.
That conversion can wrap or truncate real elevation values and is not appropriate
for quantitative DSM visualization.

The refactored implementation keeps the raster values in their original numeric
type and computes min/max elevation directly from masked raster data.

## Python classification

The public classification pipeline is based on the later Decision Tree / Random
Forest experiment rather than the older SVM script.

Corrections and improvements:

- removed hard-coded Windows paths;
- replaced ambiguous image-axis heuristics with rasterio band-first reading;
- excluded ROI background class `0` from model training/evaluation;
- made the train/test split reproducible and stratified by class;
- reused exactly the same train/test indices for the with-DSM and without-DSM experiments;
- removed unnecessary full-image min-max normalization for tree-based models;
- fixed the confusion-matrix label order to classes `1..5`;
- added chunked full-scene prediction for large rasters;
- added GeoTIFF classification export and machine-readable metrics;
- added tests for sampling, model execution, and documented accuracy improvements.

## Evaluation limitation

The original RGB/texture/DSM stacks and ROI raster were not included during
repository cleanup. Therefore the exact classification run cannot be rerun from
this repository alone.

The real experiment results are preserved as documented coursework outputs.
The code is independently tested on deterministic synthetic arrays, but those
synthetic results are not presented as replacements for the real experiment.
