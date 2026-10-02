# Multi-Agent Learning of Reactive IPD Strategies

Code for J. Wolfe and J. B. Plotkin, *Multi-Agent Learning of Reactive IPD Strategies*
(in preparation): every simulation, figure and numerical check in the paper.

## Setup

- **Python 3.12:** `uv sync --locked` with [uv](https://docs.astral.sh/uv/), or
  `pip install -r requirements.txt`.
- **LaTeX** (for figure text): a full TeX Live or MacTeX install, including `dvipng`.
- **C compiler**, only for the long run (Fig. S6): `xcode-select --install` on macOS,
  `apt install build-essential` on Debian/Ubuntu. Set `CC` if the compiler isn't `cc`.

## Usage

Download simulation data from
[doi:10.5281/zenodo.23044556](https://doi.org/10.5281/zenodo.23044556) and unzip it into
`data/`. Then:

```bash
make figures   # all figures, as PDFs in figures/ (about 1 min)
make fig2      # one figure (fig1-fig3, figS1-figS11)
make checks    # run notebooks/checks.ipynb, which checks the claims in the paper
make data      # regenerate any missing data (all of it: about 9 h on an 8-core Apple M3)
```

`make help` lists every target. To rerun one experiment, delete its output in
`data/generated/` and run `make data`.

## Layout

```text
src/reactive_learning/   model, analysis and plotting code
scripts/generate/        one script per experiment, writing data/generated/
scripts/figures/         one script per figure, writing figures/
notebooks/               the same figures (main_figs, sup_figs) and the checks
data/README.md           contents of every data file
```

Each figure script is a copy of a notebook cell; after editing a cell, run
`make figure-scripts`.

## Figures and run times

Run times are for regenerating the data on an Apple M3 (8 cores), as logged in
`data/timings.csv`.

| Figure | Script (`scripts/figures/`) | Data (`scripts/generate/`) | Run time |
| --- | --- | --- | --- |
| 1 | `fig01_conceptual_comparison.py` | none | |
| 2, 3, S1, S5 | `fig02_…`, `fig03_…`, `figS01_…`, `figS05_…` | `agent_main_data.py` | 48 min |
| S2 | `figS02_lattice_dynamics.py` | `lattice_data.py` | 23 min |
| S3 | `figS03_axis_entry_distribution.py` | `axis_entry_data.py` | < 1 min |
| S4 | `figS04_certificate_phase_diagrams.py` | `convergence_phase_diagrams_data.py` | < 1 min |
| S6 | `figS06_long_run.py` | `agent_variable_long_run_data.py` | 3.6 h |
| S7 | `figS07_gtft_q_shock.py` | `gtft_q_shock_data.py` | < 1 min |
| S8 | `figS08_gtft_disk_displacement.py` | `gtft_disk_displacement_data.py` | 1 min |
| S9 | `figS09_basin_heatmaps.py` | `basin_data.py` (all cores) | 2.9 h |
| S10 | `figS10_noisy_population.py` | `agent_noise_data.py` | 5 min |
| S11 | `figS11_noisy_pair.py` | `agent_pair_data.py` | 1 min |
| checks | `notebooks/checks.ipynb` | `self_interactions_data.py`, `axis_entry_rate_sensitivity_data.py` | 56 min |

Runs are deterministic and seeded, so they reproduce the archived data exactly on the same
hardware and software. Across machines, results can differ in the last digits.

## Citation

J. Wolfe and J. B. Plotkin, *Multi-Agent Learning of Reactive IPD Strategies*
(in preparation). Code: MIT license. Data: CC BY 4.0.
