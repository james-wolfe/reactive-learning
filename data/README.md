# Data for "Multi-Agent Learning of Reactive IPD Strategies"

Every data file is produced by a script in `scripts/generate/` from the repository
[github.com/james-wolfe/reactive-learning](https://github.com/james-wolfe/reactive-learning); there is no external or
hand-edited input. The data files are in `generated/` (about 200 MB total).
They are not stored in the git repository;
download them from [doi:10.5281/zenodo.23044556](https://doi.org/10.5281/zenodo.23044556).
`timings.csv` records the run time of each generator (see the main README of the repo).

All runs use $b = 1$, $c = 1/2$ and $\varepsilon = 10^{-3}$ unless the table says otherwise.
Archives are NumPy `.npz` files of plain numeric and string arrays; none needs `allow_pickle`.
Most also store their parameters (`b`, `c`, `epsilon`, `seed`, ...) as scalar entries.

## Files used by the paper

| File in `generated/` | Generator | Used by | Contents |
| --- | --- | --- | --- |
| `agent_variable.npz`, `agent_uniform.npz` | `agent_main_data.py` | Figs. 2, 3, S1, S5; checks | `steps`; `agents` (saved step, agent, p/q); `eta`. N = 2500, seed 29; variable rates Uniform[0.01, 0.39] for 25,000 steps, uniform rate 0.2 for 15,000 |
| `lattice_variable.npz`, `lattice_uniform.npz` | `lattice_data.py` | Fig. S2 | `steps`; `freq` (p point, q point, learning rate, saved step) on a 50 x 50 grid over [ε, 1-ε]² for each of 100 learning rates, starting from equal mass at every point; `eta` |
| `axis_entry.npz` | `axis_entry_data.py` | Fig. S3; checks | first step at which every agent has q = ε (`entry_steps`) and the population then (`agents`); the Theorem 1 example distributions (`fraction_alld`, `p_max`, `beta_shapes`) and their survival curves |
| `convergence_phase_diagrams.npz` | `convergence_phase_diagrams_data.py` | Fig. S4 | certificate margins -U (`*_alld`) and P - Q (`*_tft`) on 1001 x 1001 grids for ALLD atom + uniform and for Beta distributions |
| `agent_variable_long_run_n125_rate100.npz` | `agent_variable_long_run_data.py` | Fig. S6; checks | 125 weighted agents sampled from `agent_variable.npz` at step 25,000 (`source_indices`, `weights`), rates x 100, 10^10 updates on a log-spaced schedule; `elapsed_seconds` |
| `gtft_q_shock.npz` | `gtft_q_shock_data.py` | Fig. S7 | `agents` (regime, step, agent, p/q) and `eta` (regime, agent) for 250 agents after q -> q + 10^-6 at GTFT; regimes `variable`, `uniform` |
| `gtft_disk_displacement.npz` | `gtft_disk_displacement_data.py` | Fig. S8 | `mean_p`, `mean_q`, `cooperation` (regime, replicate, saved step) and `final_agents` for 500 replicates per regime of 250 agents |
| `basin.npz` | `basin_data.py` | Fig. S9 | `cooperation_fraction` (N, alpha, b/c) for N in {10, 100, 1000}, 25 populations per cell |
| `agent_noise.npz` | `agent_noise_data.py` | Fig. S10 | `low_noise_agents`, `high_noise_agents` (saved step, agent, p/q): N = 2500, rate 0.2, noise SD 0.01 and 0.05, 2,000 steps |
| `agent_pairs.npz` | `agent_pair_data.py` | Fig. S11 | `average_cooperation` (pair, saved step) for 1000 isolated pairs, rate 0.02, noise SD 0.001, no self-play |
| `self_interactions/{uniform,variable}_eps0.001_self0.npz` | `self_interactions_data.py` | checks (Sec. II B) | the main runs repeated without self-play, stopped once the outcome is established |
| `axis_entry_rate_sensitivity/*.npz` | `axis_entry_rate_sensitivity_data.py` | checks (Sec. III B) | the main runs restarted from their first entry into q = ε with the learning rates swapped; one archive per case, described in the generator |

`agent_main_data.py` must run before `axis_entry_data.py`,
`agent_variable_long_run_data.py` and `axis_entry_rate_sensitivity_data.py`;
the Makefile enforces this order. The long run needs a C compiler.

## License and citation

The data are released under CC BY 4.0. If you use them, please cite J. Wolfe and
J. B. Plotkin, *Multi-Agent Learning of Reactive IPD Strategies* (in preparation), and
this dataset ([doi:10.5281/zenodo.23044556](https://doi.org/10.5281/zenodo.23044556)).
