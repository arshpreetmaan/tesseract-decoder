# Tesseract Trellis: GARI Integration Analysis

This comprehensive report provides a deep dive into the `tesseract_trellis` decoder's CLI parameters, its matrix-based DEM processing, and the architectural changes required to natively support GARI decoding.

## 1. CLI Parsing Logic & Tradeoffs

The `tesseract_trellis` decoder CLI is defined in `src/tesseract_trellis_main.cc`. Key arguments dictate data inputs, resource bounds, and heuristic tradeoffs between state tracking speed and logical accuracy.

### Core CLI Arguments
- `--circuit` / `--dem`: Input files for the Stim circuit or Detector Error Model (DEM).
- `--in` / `--in-format`: Input file and format (e.g., `b8`, `01`) containing the physical syndrome shots to decode.
- `--sample-num-shots`: Autogenerates test shots directly from the circuit without external file IO.
- `--threads`: Enables parallel decoding over the specified number of threads.
- `--ranking-mode`: Determines how branches are scored. Options are `mass`, `future-detcost`, or `future-active-detcost`.
- `--future-detcost-scale`: Multiplier for future detector-cost ranking penalties (default 2.0).

### Key Tradeoffs: `--beam` and `--beam-eps`
- **`--beam` (default: 1024)**: Sets the absolute maximum capacity for the beam width (number of states tracked simultaneously in the frontier).
  - **Tradeoff**: A higher `--beam` increases logical accuracy by tracking more potential error correction combinations, preventing early truncation of the true path. However, decoding time scales roughly linearly with the beam size because every state must be projected and expanded per fault.
- **`--beam-eps` (default: 0.0)**: A mass-threshold cutoff that dynamically truncates the beam early if the cumulative probability mass of the top tracked states reaches `1 - beam_eps`. 
  - **Tradeoff**: Using values $>0$ (e.g., $10^{-6}$) acts as a dynamic pruning mechanism. It accelerates decoding speed by discarding low-probability tails before hitting the strict `--beam` capacity limit. However, it risks logical inaccuracy if a "low probability" state locally is actually part of the globally optimal correction path.

## 2. Matrix-Based DEM Processing in Trellis

While traditional decoders might use A* search over a graph, Trellis natively treats the DEM as a parity check matrix $H$, where rows are detectors and columns are errors (faults).

1. **Parity Check Matrix parsing**: `parse_faults()` interprets the DEM. Each error mechanism becomes a `Fault` with a subset of active detectors (rows in $H$), an observable mask, and likelihood costs.
2. **Column-by-Column Sweep**: The decoder iterates over the columns (faults) of $H$ strictly sequentially.
3. **Active Frontier Tracking**: Rather than tracking all $M$ detectors, Trellis only tracks an active "frontier". As it iterates from fault $j=0$ to $E$, a detector enters the active frontier when it first appears in a fault, and retires from the frontier once the final fault containing it has been processed (`build_wide_layer_templates`).
4. **State Projection**: States are encoded in bitsets (`FixedWideStateWords<Words>`). For each fault, the decoder branches states (assuming the fault occurred vs. did not occur), XORs the fault's detector bitmask into the frontier, and accumulates probabilities. 
5. **Column Scaling / Early Retirement**: Because the state space size grows exponentially with the frontier width, detectors are greedily retired once they will no longer be referenced. This ensures the column-by-column tracking minimizes the maximum bit-width (`max_frontier_width`) needed for projection operations (`project_compiled_wide_state`).

## 3. GARI Integration Context

Based on the `gari_decoder_integration_report.md` and `pr_split_plan.md`, the GARI integration for the standard Tesseract/Simplex decoders introduced several key concepts:

- **DEM Sparsification (Y-Error Splits)**: GARI splits multi-connected errors (Y-type errors) into independent X and Z components coupled via newly introduced *virtual detectors*.
- **Detector Expansion ($M > N$)**: Physical syndrome shots contain $N$ detectors, but the GARI DEM contains $M$ detectors ($M > N$). A mapping JSON file (`--det-mapping-file`) maps physical indices to GARI indices on-the-fly. Unmapped virtual detectors are initialized with an even parity (0).
- **Search Topology (`--custom-order`)**: Standard Tesseract uses A* search and branches based on active detectors. GARI introduced JSON-supplied custom orderings (e.g., chronological sorting) to prioritize which active detectors are resolved first, preventing search tree explosions.
- **Heuristic Weighting**: Various prior weight modes derived from Linear Programming (LP) optimizations are used to ensure mathematically admissible costs for Tesseract.

## 4. Architectural Changes Required for `tesseract_trellis.cc`

To port GARI capabilities to `tesseract_trellis`, several exact C++ structural changes are required.

### A. Graph Parsing and JSON Mapping
1. **CLI Updates**: `Args` in `tesseract_trellis_main.cc` must be updated to accept `--det-mapping-file`.
2. **Shot Ingestion Loop**: The `extract()` method must be modified to apply `det_mapping[d]` when simulating or reading shots, directly converting the physical $N$ bits to GARI $M$ bits. 
3. **Parity Safety**: Trellis reads standard sparse shots. Any virtual GARI detector not present in the physical mapping must be implicitly assumed to have `0` (even) parity in the shot target masks (`actual_detector_words`). Explicit out-of-bounds safety checks must be added during mapping array lookups.

### B. Safe Virtual Tracking without Frontier Overflow
- In `tesseract_trellis.cc`, the state is compiled into fixed word sizes (e.g., `kMaxCompiledWideStateWords = 4`, supporting a max frontier of 256 active detectors). 
- **The Overflow Risk**: GARI introduces a massive number of virtual routing detectors. If the DEM columns (faults) are processed in random or purely component-blocked order (e.g., all X faults, then all Z faults), the active frontier will simultaneously hold detectors across all time steps, overflowing the 256-bit compilation limit.
- **Solution**: The `build_wide_layer_templates` function natively infers the tracker boundaries from the sequence of faults in the DEM. To safely process GARI without overflowing, the DEM must be pre-sorted chronologically prior to Trellis initialization, ensuring virtual detectors retire locally within their time slice.

### C. Matching `det_ordering`
- **Architectural Shift**: Instead of relying on a runtime `--custom-order` array, Trellis requires the DEM itself to be pre-permuted. A python utility (e.g., `gari_dem_utils.py`) should output a topologically/chronologically sorted GARI DEM. By reordering the columns of $H$ (the faults in the `.dem` file), Trellis will natively achieve the optimal scaling boundary without any C++ `det_order` scanning loops. 

***

## 5. Understanding Trellis Decoding: Frontier Width and Beam Width

When decoding error-correcting codes using a Trellis, we treat the decoding process as a sequence of steps. We process the Parity Check Matrix ($H$) one column (error) at a time. 

To understand `frontier_width` and `beam_width`, we must first understand the matrix structure, row/column weights, and the concept of a detector's "lifespan."

### A. The Toy Parity Check Matrix ($H$)

Let's define a small, sparse $4 \times 5$ Parity Check Matrix ($H$). 
* **Rows (D0 - D3)** represent **Detectors** (or parity checks/syndromes).
* **Columns (e0 - e4)** represent potential **Errors** (or bits/edges).

A `1` in the matrix means that a specific error triggers a specific detector.

| | `e0` | `e1` | `e2` | `e3` | `e4` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **D0** | **1** | 0 | **1** | 0 | 0 |
| **D1** | **1** | **1** | 0 | 0 | **1** |
| **D2** | 0 | **1** | 0 | **1** | 0 |
| **D3** | 0 | 0 | **1** | **1** | **1** |

### B. Row Weight and Column Weight

In the context of this matrix:

* **Column Weight**: The number of `1`s in a specific column. This tells us **how many detectors are triggered by a single error**.
  * *Example*: Error `e0` triggers `D0` and `D1`. Its column weight is **2**.
  * *Example*: Error `e1` triggers `D1` and `D2`. Its column weight is **2**.
* **Row Weight**: The number of `1`s in a specific row. This tells us **how many different errors can trigger a specific detector**.
  * *Example*: Detector `D0` is triggered by `e0` and `e2`. Its row weight is **2**.
  * *Example*: Detector `D1` is triggered by `e0`, `e1`, and `e4`. Its row weight is **3**.

### C. The Lifespan of a Detector (The Frontier)

The **Frontier** is the set of detectors that are currently "active" or "in memory" at a given step in the Trellis algorithm. 

A detector's lifespan dictates when it enters and leaves the frontier:
1. **Birth**: A detector is added to the frontier at the *first column* where it has a `1`.
2. **Retirement**: A detector is removed from the frontier immediately after processing the *last column* where it has a `1`.

Based on our matrix, the lifespans are:
* **D0**: Born at `e0`, Retires after `e2`.
* **D1**: Born at `e0`, Retires after `e4`.
* **D2**: Born at `e1`, Retires after `e3`.
* **D3**: Born at `e2`, Retires after `e4`.

### D. Column-by-Column Walkthrough and `frontier_width`

The `frontier_width` is the exact number of detectors active in the frontier *while processing a specific column*. It dictates the complexity of the Trellis at that step.

Let's walk through the algorithm column-by-column:

#### Step 0: Process Error `e0`
* **Triggered Detectors (Col Weight)**: `D0`, `D1`
* **Added to Frontier**: `D0`, `D1` (First appearances)
* **Active Frontier**: `{D0, D1}`
* 🧮 **`frontier_width`**: **2**
* **Retirements**: None (Both `D0` and `D1` appear again in later columns).

#### Step 1: Process Error `e1`
* **Triggered Detectors (Col Weight)**: `D1`, `D2`
* **Added to Frontier**: `D2` (First appearance)
* **Active Frontier**: `{D0, D1, D2}`  *(Note: `D0` is still tracked even though `e1` doesn't trigger it, because it's still alive waiting for `e2`)*
* 🧮 **`frontier_width`**: **3**
* **Retirements**: None.

#### Step 2: Process Error `e2`
* **Triggered Detectors (Col Weight)**: `D0`, `D3`
* **Added to Frontier**: `D3` (First appearance)
* **Active Frontier**: `{D0, D1, D2, D3}`
* 🧮 **`frontier_width`**: **4**
* **Retirements**: `D0`. (This is the last column where `D0` has a `1`. Its state is resolved, and we drop it from memory).
* *Remaining for next step: `{D1, D2, D3}`*

#### Step 3: Process Error `e3`
* **Triggered Detectors (Col Weight)**: `D2`, `D3`
* **Added to Frontier**: None
* **Active Frontier**: `{D1, D2, D3}`
* 🧮 **`frontier_width`**: **3**
* **Retirements**: `D2`. (Last appearance of `D2`).
* *Remaining for next step: `{D1, D3}`*

#### Step 4: Process Error `e4`
* **Triggered Detectors (Col Weight)**: `D1`, `D3`
* **Added to Frontier**: None
* **Active Frontier**: `{D1, D3}`
* 🧮 **`frontier_width`**: **2**
* **Retirements**: `D1`, `D3`. (Last appearances for both).
* *Remaining for next step: `{}` (Decoding complete)*

### E. `beam_width` and the State Space

At every step (column), the Trellis algorithm must track all possible partial combinations of errors to match the syndromes. Because a detector can theoretically be in $2$ states (parity satisfied or unsatisfied), the maximum number of paths (states) we must track at any given step is $2^{\text{frontier\_width}}$.

In our toy example at **Step 2**, `frontier_width = 4`. The Trellis evaluates $2^4 = 16$ possible states.

#### The Problem: State Space Explosion
In a real-world sparse matrix, `frontier_width` might grow to 50 or 100. Tracking $2^{50}$ states is computationally impossible. 

#### The Solution: `beam_width` (Beam Size)
`beam_width` is a hard numeric cap on the maximum number of states the algorithm is allowed to keep alive transitioning from one column to the next.

If we set `beam_width = 8` for our toy matrix:
* At Steps 0, 1, 3, and 4, the state spaces are $2^2=4$, $2^3=8$, $2^3=8$, and $2^2=4$. Because these are $\le 8$, we keep all states (exact Viterbi decoding).
* At **Step 2**, the state space expands to $16$. The algorithm will score all $16$ paths, **sort them by their likelihood (weight)**, and explicitly prune the $8$ worst paths. 
* Only the top `8` paths survive to Step 3.

**Summary**: 
* `frontier_width` is an *intrinsic property of the matrix ordering*, defining the raw size of the state space at a given column. 
* `beam_width` is an *algorithmic constraint*, forcibly pruning the state space to keep computation tractable at the cost of potentially dropping the optimal path.

### F. The Impact of Long Dependencies vs. Short Dependencies

To truly understand how Trellis scales constraints, consider what happens if we change a single connection deep in the matrix. If we change the entry at `(D1, e4)` from a `1` to a `0`, the matrix becomes:

| | `e0` | `e1` | `e2` | `e3` | `e4` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **D0** | **1** | 0 | **1** | 0 | 0 |
| **D1** | **1** | **1** | 0 | 0 | **0** |
| **D2** | 0 | **1** | 0 | **1** | 0 |
| **D3** | 0 | 0 | **1** | **1** | **1** |

**New Lifespans:**
* **D0**: Born at `e0`, Retires after `e2`.
* **D1**: Born at `e0`, **Retires after `e1`**. (Instead of `e4`)
* **D2**: Born at `e1`, Retires after `e3`.
* **D3**: Born at `e2`, Retires after `e4`.

#### Walkthrough (Modified 4-Detector Matrix):
* **Step e0:** Frontier = `{D0, D1}` (Width = 2)
* **Step e1:** Frontier = `{D0, D1, D2}` (Width = 3) ➡️ *D1 Retires Early!*
* **Step e2:** Frontier = `{D0, D2, D3}` (Width = **3**) ➡️ *D0 Retires*
* **Step e3:** Frontier = `{D2, D3}` (Width = **2**) ➡️ *D2 Retires*
* **Step e4:** Frontier = `{D3}` (Width = **1**) ➡️ *D3 Retires*

#### Comparison 1: Modified 4-Detector vs. Original 4-Detector
In the original matrix, `D1` lived all the way to `e4`. The old Step e2 had `frontier_width = 4` (evaluating $2^4 = 16$ states).
By dropping that single connection at `(D1, e4)`, **the peak frontier width for the entire algorithm dropped from 4 down to 3**. That single zero heavily optimizes the bottleneck step, requiring Trellis to track only 8 states maximum instead of 16.

#### Comparison 2: Modified 4-Detector vs. Original 2-Detector (D0 & D1 only)
If we ran Trellis on a matrix with *only* D0 and the original D1 (where `D1, e4 = 1`), the width at Step e4 would be 1 (`{D1}`).
Even though the Modified 4-Detector matrix has **twice as many rows**, its frontier width at Step e4 is exactly the same (Width = 1). 
Furthermore, in the 2-detector matrix, `D1` was dead weight—it sat in the frontier from `e2` to `e3` doing nothing but consuming bits to resolve a parity check at `e4`. 

In the 4-detector modified matrix, memory is handed off efficiently like a relay race (`D1` finishes, hands its active slot to `D2`, which hands off to `D3`).

**Conclusion for GARI Scaling:** 
The 256-bit structural limit in Trellis is heavily dependent on **dependency length**. If a massive number of virtual detectors have tightly clumped `1`s, they will efficiently pop in and out of the frontier array like a relay race, keeping the max width securely under the cap. If they have stretched out `1`s spanning the whole block, they overlap and crash the frontier limit.

orignial tresllis d=3 cc_z
num_shots = 100000 num_low_confidence = 0 num_errors = 557 states_expanded = 190618 states_merged = 190618 max_beam = 1024 frontier_width = 12 total_time_seconds = 1104.294651000006
kept_states min=1 median=512 mean=532.4525139664804 max=1024
branch_masses obs0=0.9999976549480732 obs1=2.345051926692678e-06
phase_times_seconds expand=0.006857999999999994 collapse=0.0003380000000000008 truncate=0.003090000000000003 reconstruct=0
num_shots = 100000 num_low_confidence = 0 num_errors = 557 total_time_seconds = 1104.294651000006


gari-dem-rev
num_shots = 100000 num_low_confidence = 0 num_errors = 594 states_expanded = 437429 states_merged = 437429 max_beam = 1024 frontier_width = 99 total_time_seconds = 2569.933325000002
kept_states min=1 median=1024 mean=959.2741228070175 max=1024
branch_masses obs0=1 obs1=0
phase_times_seconds expand=0.011924 collapse=0.00167000000000001 truncate=0.01044200000000002 reconstruct=0
num_shots = 100000 num_low_confidence = 0 num_errors = 594 total_time_seconds = 2569.933325000002


d=5
num_shots = 1000 num_low_confidence = 0 num_errors = 2 states_expanded = 2781024 states_merged = 2781024 max_beam = 1024 frontier_width = 29 total_time_seconds = 193.2362500000001
kept_states min=1 median=1024 mean=997.4978479196557 max=1024
branch_masses obs0=0.9999999996616125 obs1=3.383876137828544e-10
phase_times_seconds expand=0.09308500000000013 collapse=0.008905000000000206 truncate=0.08161600000000063 reconstruct=0
num_shots = 1000 num_low_confidence = 0 num_errors = 2 total_time_seconds = 193.2362500000001