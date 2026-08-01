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
`<circuit-directory>/gari/<stem>_mapping.json`; existing generated artifacts are
left untouched. The compact JSON omits the legacy custom detector orders;
two-stage top orders are generated in memory by Tesseract.

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
