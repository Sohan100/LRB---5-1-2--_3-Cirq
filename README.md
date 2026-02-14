# Qutrit RB: Physical and Logical Benchmarking for the Folded [[5,1,2]]_3 Code

This repository implements an end-to-end randomized benchmarking (RB) workflow
for qutrit systems, with an emphasis on comparing:

- Physical RB on an unencoded single qutrit baseline.
- Logical RB (LRB) on the folded [[5,1,2]]_3 qutrit code.
- Multiple logical noise models under matched physical depolarizing rates.

The codebase is designed for long-running parameter sweeps, resumable
checkpointing, and post-processing with dedicated plotting utilities.

## Why This Project Exists

In encoded quantum computing experiments, physical gate quality does not
directly tell you logical gate quality. Error correction structure, syndrome
selection, and effective noise channels can change the decay behavior
substantially. This project gives you a controlled simulation framework to
answer:

- How does logical RB decay compare to physical RB at the same `p`?
- When does logical infidelity cross above physical infidelity?
- How sensitive are results to the logical noise model and readout policy?

The repository is organized so simulation and visualization are separate:

- Sim scripts generate raw checkpoints and fit parameters.
- Plot scripts consume saved artifacts and produce publication-style figures.

## Repository Highlights

Core simulation entry points:

- `qutrit_physical_rb_sim.py`
  - Physical single-qutrit RB baseline.
- `qutrit_logical_rb_logical_noise_sim.py`
  - Logical RB with sampled logical depolarizing noise.
- `qutrit_logical_rb_terminal_check_sim.py`
  - Logical RB with terminal-only stabilizer checks and local physical noise.

Sequence generators:

- `rb_seq.py`
  - Physical RB sequence builder.
- `logical_rb_seq.py`
  - Logical RB sequence builder.

Plotting:

- `qutrit_rb_plotting.py`
  - Fidelity/survival/success plots and logical-vs-physical overlays.

Support modules:

- `gates.py`, `noise.py`, `pauli.py`
- `qutrit_clifford.py`, `qutrit_logical_clifford.py`,
  `qutrit_logical_pauli.py`
- `qutrit_folded_logical_clifford.py`,
  `qutrit_folded_logical_plus_state.py`
- `rb_checkpoint.py`

Notebook:

- `Plotting LRB and RB Notebook.ipynb`

Cluster scripts:

- `run_logical_rb_logical_noise3.sh`
- `run_logical_rb_logical_noise4.sh`
- `run_logical_rb_terminal_check.sh`

## Installation and Environment

This project is pure Python and is easiest to run in a virtual environment.

Suggested packages:

- `cirq`
- `numpy`
- `scipy`
- `matplotlib`
- `networkx`
- `sympy`
- `jupyter` (for notebook workflows)

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install cirq numpy scipy matplotlib networkx sympy jupyter
```

Windows PowerShell example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install cirq numpy scipy matplotlib networkx sympy jupyter
```

## End-to-End Workflow

Typical analysis loop:

1. Run physical RB sweep to generate baseline file.
2. Run one logical RB variant across the same `p` values.
3. Use `qutrit_rb_plotting.py` (or the notebook) to overlay LRB vs RB.
4. Inspect fitted fidelities and threshold behavior.

Artifacts are saved in your chosen checkpoint directory, then reused by
plotting without rerunning simulation.

## Running Simulations Locally

### 1) Physical RB Baseline

Run full default sweep:

```bash
python qutrit_physical_rb_sim.py \
  --checkpoint-dir qutrit_rb_results_logical_depolarizing_noise_noise \
  --seed 24 \
  --repetitions 5000
```

Run a single error rate:

```bash
python qutrit_physical_rb_sim.py \
  --checkpoint-dir qutrit_rb_results_logical_depolarizing_noise_noise \
  --error-rate 0.0311537409
```

Primary outputs:

- `physicalRB_SimulationResults.npy`
- `physical_rb_infidelities.npy`

### 2) Logical RB: Sampled Logical Depolarizing Noise

Run full default sweep:

```bash
python qutrit_logical_rb_logical_noise_sim.py \
  --checkpoint-dir qutrit_rb_results_logical_depolarizing_noise_noise \
  --reps 10000 \
  --seed 24
```

Run a single error rate:

```bash
python qutrit_logical_rb_logical_noise_sim.py \
  --checkpoint-dir qutrit_rb_results_logical_depolarizing_noise_noise \
  --error-rate 0.0311537409 \
  --reps 10000
```

Primary outputs:

- `logicalRB_p{p}.pkl` per physical `p`
- `final_logical_error_rate.pkl`

### 3) Logical RB: Terminal Check + Local Noise After Each Logical Gate

Run full default sweep:

```bash
python qutrit_logical_rb_terminal_check_sim.py \
  --checkpoint-dir qutrit_rb_results_terminal_check_local_noise \
  --reps 10000 \
  --seed 24
```

Run a single error rate:

```bash
python qutrit_logical_rb_terminal_check_sim.py \
  --checkpoint-dir qutrit_rb_results_terminal_check_local_noise \
  --error-rate 0.0311537409 \
  --reps 10000
```

Primary outputs:

- `logicalRB_p{p}.pkl` per physical `p`
- `final_logical_error_rate.pkl`

## Plotting and Overlays

All plotting is centralized in `qutrit_rb_plotting.py` using
`LogicalRbPlotter`.

### Minimal Overlay Example

```python
from qutrit_rb_plotting import LogicalRbPlotter

plotter = LogicalRbPlotter(
    logical_checkpoint_dir=(
        "qutrit_rb_results_logical_depolarizing_noise_noise"
    ),
    experiment_name="logical_noise",
)

fidelity_summary = plotter.overlay_with_physical_rb(
    physical_results_path=(
        "qutrit_rb_results_logical_depolarizing_noise_noise/"
        "physicalRB_SimulationResults.npy"
    ),
    error_rates=None,         # infer from logicalRB_p*.pkl
    show_plot=True,           # force on-screen rendering
    fit_curves=True,          # enable exponential fits + fidelity text
    y_range=(0.0, 1.2),       # optional fixed y-limits
)

print(fidelity_summary)
```

The overlay figure title for logical-noise mode is:

`Logical RB vs Physical RB under Logical Depolarizing Channel`

Per panel, fitted values are annotated at high precision:

- `F_LRB=...`
- `F_RB=...`

When `y_range=None`, the overlay uses adaptive limits equivalent to:

- lower bound `-0.1`
- upper bound `max(y_values) + 0.45`

This avoids compressed traces while preserving full depth-domain visibility.

### Terminal-Check Overlay Without Fits

For terminal-check LRB, you can explicitly disable fit overlays:

```python
from qutrit_rb_plotting import LogicalRbPlotter

plotter = LogicalRbPlotter(
    logical_checkpoint_dir="qutrit_rb_results_terminal_check_local_noise",
    experiment_name="terminal_check",
)

plotter.overlay_terminal_check_with_physical_rb_no_fit(
    physical_results_path=(
        "qutrit_rb_results_terminal_check_local_noise/"
        "physicalRB_SimulationResults.npy"
    ),
    error_rates=None,
    show_plot=True,
    y_range=None,
)
```

Equivalent low-level call:

`overlay_with_physical_rb(..., fit_curves=False)`

### Other Plot APIs

`LogicalRbPlotter` also supports:

- `plot_fidelity_curves(...)`
- `plot_survival_ratios(...)`
- `plot_logical_success(...)`
- `plot_logical_vs_physical_infidelity()`
- `overlay_terminal_check_with_physical_rb_no_fit(...)`

Generated files are saved under:

- `Fidelity Curves/`
- `Survival Probability Curves/`
- `Logical Success Curves/`

inside your selected output directory.

## Notebook Usage

Use `Plotting LRB and RB Notebook.ipynb` for interactive plotting.

Recommended notebook setup:

1. Select your experiment key (`logical_noise` or `terminal_check`).
2. Point `checkpoint_dir` and `physical_results_path` to matching runs.
3. For logical-noise overlays, call:
   `overlay_with_physical_rb(..., show_plot=True, fit_curves=True)`.
4. For terminal-check overlays, call:
   `overlay_terminal_check_with_physical_rb_no_fit(...)`.
5. Keep a markdown header before each plotting code cell describing the plot.

If your backend does not render figures automatically, add:

```python
%matplotlib inline
```

in the first notebook cell.

## How Circuit Depth Is Defined

Default RB depth list is defined in sequence generators:

- `rb_seq.py` and `logical_rb_seq.py` default to:
  - `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- Default circuits per depth:
  - `30`

Depth labels are carried explicitly with each generated circuit using
`depth_keys`.

### Why There Used to Be `2 * (...)`

Older code inferred depth by index using:

`2 * (circuit_index // circuits_per_sequence)`

That works only when depth spacing is exactly 2. This repository now stores
explicit per-circuit depth keys and resolves them robustly. Legacy fallback is
kept for compatibility with old checkpoints.

### How to Change Depths Safely

To customize depth schedule:

1. Pass `sequenceLengths=[...]` and optionally `circuitsPerSequence=...` into
   `RBSeq`/`LogicalRBSeq` construction sites.
2. Keep depth labels from `get_depth_keys()` as the source of truth.
3. Avoid deriving depth from circuit index arithmetic.

If you want CLI-level control, add parser flags in the simulation scripts and
forward them to sequence constructors.

## Checkpoint and Data Schema

### Logical RB per-rate file: `logicalRB_p{p}.pkl`

Common keys (snake_case with legacy camelCase compatibility):

- `expectation_values` / `expectationValues`
- `survival_ratios` / `survivalRatios`
- `logical_successes` / `logicalSuccesses`
- sequence payload:
  - logical-noise mode: `rb_circuits`, `rb_string_sequences`
  - terminal-check mode: `rb_sequences`, `rb_string_sequences`
- depth metadata:
  - `sequence_lengths`
  - `depth_keys`
- `index` (resume cursor)

### Logical sweep summary: `final_logical_error_rate.pkl`

- Logical-noise mode stores paired physical/logical rate arrays.
- Terminal-check mode stores a `p -> logical_error_rate` mapping.

### Physical RB payload: `physicalRB_SimulationResults.npy`

Dictionary keyed by physical `p`, each value includes:

- `x_data` (depths)
- `y_data` (mean expectation values)
- `fit_params`
- `infidelity`
- `expectation_values` (raw repeated samples by depth)
- `average` (depth -> mean)

## Running on SLURM

Provided scripts use one common pattern:

- One SLURM job requests `N` tasks.
- The script launches one backgrounded `srun --exclusive` per `p` value.
- Each `srun` writes an individual per-probability log file.
- The script `wait`s until all simulations complete.

Scripts:

- `run_logical_rb_logical_noise3.sh`
- `run_logical_rb_logical_noise4.sh`
- `run_logical_rb_terminal_check.sh`

Edit these fields before submission:

- checkpoint directory
- script path (`SIM_SCRIPT`)
- scripts root (`SCRIPTS_DIR`)
- repetitions (`REPS`)
- queue/account constraints
- error-rate list
- thread count (`OMP_NUM_THREADS`)

Then submit with:

```bash
sbatch run_logical_rb_terminal_check.sh
```

and similarly for other scripts.

## Testing and Sanity Checks

Lightweight manual test scripts:

- `test_gates.py`
- `test_noise.py`

They are smoke-test style scripts, not a strict `pytest` unit suite.

Run:

```bash
python test_gates.py
python test_noise.py
```

## Practical Notes and Troubleshooting

- Resume behavior:
  - If a `logicalRB_p{p}.pkl` file exists, the simulation resumes from the
    saved `index`.
- Plot save collisions on Windows:
  - If a PDF is open in a viewer, plotting may save to a `_new` fallback name.
- Mismatched rates between logical and physical files:
  - Overlay code tries float-tolerant lookup, but matching exact `p` lists is
    still best.
- Legacy script flags:
  - Some older SLURM scripts may reference options removed from current CLIs.
    Use the command examples in this README as the source of truth.

## Suggested Reproducible Campaign

1. Run physical RB once into a stable directory.
2. Run logical-noise LRB sweep into that same directory.
3. Generate overlay and save fidelity summary.
4. Run terminal-check LRB sweep into a second directory.
5. Generate overlay for that directory with the same physical baseline.
6. Compare fitted `F_LRB` vs `F_RB` across applicable models.

This gives you a clean side-by-side study of noise-model sensitivity while
keeping physical baseline constant.

## License and Attribution

This repository includes a license file at `LICENSE`.
