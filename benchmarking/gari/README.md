# GARI benchmarks


## Contents

- `submit.sh`: Slurm submission script for the benchmark sweep.
- `aggregated_results.jsonl`: Aggregated benchmark results used by the plots.
- `plot.py`: Plotting and summary-analysis script.
- `plots/`: PDF plots generated from `aggregated_results.jsonl`.

## steps before running jobs

```bash
bazel build --jobs=1 src:tesseract
```

For two-stage decoding, generate only the required physical-logical mode-N DEM
and compact mapping JSON. The input can be one `.stim` file or a directory,
which is scanned recursively:

```bash
bazel run --jobs=1 //src/py/_tesseract_py_util:gari_dem_utils -- \
  "testdata/surfacecodes/" --two-stage-only
```

This writes or updates only `<circuit-directory>/gari/<stem>_ogL_modeN.dem` and
`<circuit-directory>/gari/<stem>_two_stage_mapping.json`; existing generated
artifacts are left untouched. The compact JSON omits the legacy custom detector
orders; two-stage top orders are generated in memory by Tesseract.

A fixed beam with one top order performs one top-and-bottom pass:

```bash
./bazel-bin/src/tesseract \
  --gari-two-stage --circuit "<circuit>.stim" \
  --dem "<gari>/<stem>_ogL_modeN.dem" \
  --det-mapping-file "<gari>/<stem>_two_stage_mapping.json" \
  --sample-num-shots 10 --sample-seed 0 --threads 1 \
  --beam 20 --num-det-orders 1 --det-order-index \
  --gari-bottom-beam 2 --pqlimit 1000000 \
  --stats-out "<fixed-stats>.json"
```

Add `--beam-climbing` to test beams 0 through 20. Set
`--num-det-orders 21` to cycle through 21 randomized top index orders during
that outer schedule. Set `--gari-top-candidates K` to retain up to `K`
completed candidates from each top beam/order search before physical-cost
reranking; its default is 1.

Add `--gari-split-top` to decode the independent `D_X` and `D_Z` top blocks
with separate Tesseract instances. The mapping JSON records each block's
half-open detector-row, barred-column, and debt-row ranges as `offset` and
`count`. Existing top orders are stably filtered into both blocks, so their
method and seed are unchanged. With the split enabled, `--gari-top-candidates K`
requests up to `K` candidates from each block and tests their within-trial
Cartesian product (up to `K^2`) through the existing bottom debt cache.
Mappings generated before `top_components` was added remain valid unless this
flag is requested.

Set `--gari-bottom-num-det-orders 2` to complete each new debt using both the
natural and reversed bottom detector-index orders at the fixed
`--gari-bottom-beam`, retaining the lower original physical cost. The default
of 1 preserves the natural-order path. Detector-index ordering has only these
two distinct directions, so larger values are rejected.

Use `--gari-bottom-decoder pymatching` to complete each new debt with the
native PyMatching backend instead. Tesseract remains the default. The bottom
model is converted once per worker; one-target columns become boundary edges,
two-target columns become ordinary edges, and PyMatching's normal DEM handling
is used for parallel edges. The debt cache and bottom-cost reranking remain
active; with this backend, that cost is PyMatching's normalized, discretized
MWPM weight. PyMatching does not use the Tesseract-only bottom beam or detector
orders, so their defaults (`2` and `1`) must be retained. It returns logical
observables and cost but not physical error indices, making this backend
incompatible with `--dem-out`.

With `--stats-out`, the `gari_two_stage` JSON block reports
`bottom_decode_time_seconds` and `bottom_decode_time_fraction_of_total`. These
measure bottom solver calls on cache misses; the fraction uses the existing
aggregate `total_time_seconds` value. `bottom_decoder` identifies the selected
backend. The existing `bottom_beam` and `bottom_num_det_orders` keys are kept;
for PyMatching they record the required default values `2` and `1`.

`total_time_seconds` contains only accumulated shot-decoding time. Decoder
construction and the one-time GARI partition are reported separately as
`decoder_setup_time_seconds`. Timing results produced before this separation
included setup in the total and should be rerun before direct comparison.

Generate all GARI DEMs with the prior policies and detector orders for the
targeted circuits. From the repository root:

```bash
# Run DEM Generation under Bazel:
bazel run --jobs=1 //src/py/_tesseract_py_util:gari_dem_utils -- "testdata/bivariatebicyclecodes/"
bazel run --jobs=1 //src/py/_tesseract_py_util:gari_dem_utils -- "testdata/colorcodes/"
bazel run --jobs=1 //src/py/_tesseract_py_util:gari_dem_utils -- "testdata/surfacecodes/"

# Run test simulation under Bazel:
bazel run --jobs=1 //src/py:gari_simulation_test
```

The gari_dem_utils and simulation scripts require `numpy`, `scipy`, `matplotlib`, and `stim`, which are managed by Bazel when run via `bazel run`.

## Re-running jobs

From the repository root:

```bash
benchmarking/gari/submit.sh
```

The script assumes the Tesseract binary is available at
`./bazel-bin/src/tesseract`, reads circuits from `testdata/`, gari dems for the corresponding circuits, and submits jobs
with `sbatch`. The Slurm partition, memory, CPU count, and walltime are tuned
for the cluster used for the original PR benchmark and may need adjustment
before reuse.

Per-job stats are written under `out/`. Those raw per-job JSON files are not
included here; `aggregated_results.jsonl` is the aggregated dataset used for the
published plots.

## Re-making plots

From this directory:

```bash
python3 plot.py
```

The plotting script expects `aggregated_results.jsonl` in the current working
directory and writes outputs into `plots/`. It requires `matplotlib`, `numpy`,
`scipy`, and `stim`.

To try interactive html plot:
```bash
python3 interactive_plot.py
```
