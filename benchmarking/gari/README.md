# GARI benchmarks


## Contents

- `submit.sh`: Slurm submission script for the benchmark sweep.
- `aggregated_results.jsonl`: Aggregated benchmark results used by the plots.
- `plot.py`: Plotting and summary-analysis script.
- `plots/`: PDF plots generated from `aggregated_results.jsonl`.

## steps before running jobs

```bash
bazel build src:tesseract
```

Generate all the gari dems with all prior modes and detector ordering for the targeted circuits.
From the repository root:
```bash
python3 src/py/gari_dem_utils testdata/bivariatebicyclecodes/
python3 src/py/gari_dem_utils testdata/colorcodes/
python3 src/py/gari_dem_utils testdata/surfacecodes/
```

The gari_dem_utils requires `numpy`,`scipy`, and `stim`.

## Re-running jobs

From the repository root:

```bash
benchmarking/sparsify_errors/submit.sh
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
