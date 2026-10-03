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

### Fetching data

Download simulation data from
[doi:10.5281/zenodo.23044556](https://doi.org/10.5281/zenodo.23044556) and unzip it into
`data/`. Alternatively, generate it yourself by running `make data`
(total run time: ~9 hours, see [Run times](#run-times) for breakdown).

The generators depend on each other's output, so we recommend running them through `make data`, 
which handles the order.

### Reproducing results

From the terminal, run:

```bash
make figures   # all figures, as PDFs in figures/ (generates missing data first; ~1 min if data already present)
make fig2      # one figure (fig1-fig3, figS1-figS11)
make checks    # run notebooks/checks.ipynb, which checks the claims in the paper
make data      # generate any missing data
```

`make help` lists every target. To rerun one experiment, delete its output in
`data/generated/` and run `make data`.

Alternatively, use the notebooks in `notebooks/` to generate the figures and checks
interactively (data must be present in `data/generated/`).

## Layout

```text
src/reactive_learning/   model, analysis and plotting code
scripts/generate/        one script per experiment, writing data/generated/
scripts/figures/         one script per figure, copied from the notebooks (edit the notebooks)
notebooks/               the same figures (main_figs, sup_figs) and the checks
data/README.md           contents of every data file
```

## Run times

Run times are for regenerating the data on an Apple M3 (8 cores), as logged in
`data/timings.csv`.

| Figure | Script (`scripts/figures/`) | Data (`scripts/generate/`) | Run time |
| --- | --- | --- | --- |
| 1 | `fig01_conceptual_comparison.py` | none | — |
| 2, 3, S1, S5 | `fig02_…`, `fig03_…`, `figS01_…`, `figS05_…` | `agent_main_data.py` | 48&nbsp;min |
| S2 | `figS02_lattice_dynamics.py` | `lattice_data.py` | 23&nbsp;min |
| S3 | `figS03_axis_entry_distribution.py` | `axis_entry_data.py` | <&nbsp;1&nbsp;min |
| S4 | `figS04_certificate_phase_diagrams.py` | `convergence_phase_diagrams_data.py` | <&nbsp;1&nbsp;min |
| S6 | `figS06_long_run.py` | `agent_variable_long_run_data.py` | 3.6&nbsp;h |
| S7 | `figS07_gtft_q_shock.py` | `gtft_q_shock_data.py` | <&nbsp;1&nbsp;min |
| S8 | `figS08_gtft_disk_displacement.py` | `gtft_disk_displacement_data.py` | 1&nbsp;min |
| S9 | `figS09_basin_heatmaps.py` | `basin_data.py` (all cores) | 2.9&nbsp;h |
| S10 | `figS10_noisy_population.py` | `agent_noise_data.py` | 5&nbsp;min |
| S11 | `figS11_noisy_pair.py` | `agent_pair_data.py` | 1&nbsp;min |
| checks | `notebooks/checks.ipynb` | `self_interactions_data.py`, `axis_entry_rate_sensitivity_data.py` | 56&nbsp;min |

Runs are seeded, so they reproduce the archived data exactly on the same
hardware and software. Across machines, results can differ slightly.

## Citation

J. Wolfe and J. B. Plotkin, *Multi-Agent Learning of Reactive IPD Strategies*
(in preparation). Code: MIT license. Data: CC BY 4.0.
