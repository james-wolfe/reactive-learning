# Data, figures and numerical checks for "Multi-Agent Learning of Reactive IPD Strategies".
# `make help` lists the targets.
#
# A data file is generated only when it is missing. Data-on-data prerequisites are order-only
# (after `|`), so file times never trigger a rerun: data unpacked from Zenodo is used as is,
# whatever times the unzip left, and editing a generator does not rerun it (several take
# hours). To rerun an experiment, delete its output and everything computed from it (see
# below), then run `make data`. Every generator runs through scripts/timed.py, which appends
# its wall-clock time to data/timings.csv.

UV ?= uv
PYTHON := $(UV) run --locked python
TIMED := $(PYTHON) scripts/timed.py
EXECUTE := env -u MPLBACKEND $(UV) run --locked jupyter nbconvert --to notebook --execute --inplace \
	--ExecutePreprocessor.timeout=-1
export MPLBACKEND := Agg
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib

D := data/generated
F := figures
S := scripts/figures
PACKAGE := $(wildcard src/reactive_learning/*.py)

.DEFAULT_GOAL := help
.PHONY: help all sync data figures notebooks checks figure-scripts check-figure-scripts \
	timings requirements \
	lint format clean clean-data \
	fig1 fig2 fig3 figS1 figS2 figS3 figS4 figS5 figS6 figS7 figS8 figS9 figS10 figS11

help:
	@echo "make sync            install the locked Python environment (uv)"
	@echo "make figures         every paper figure (PDFs in figures/), generating missing data"
	@echo "make fig2, figS3 ... one figure"
	@echo "make checks          execute notebooks/checks.ipynb (fails if any check fails)"
	@echo "make all             figures and checks"
	@echo "make data            every data file, without figures"
	@echo "make notebooks       execute main_figs.ipynb and sup_figs.ipynb in place"
	@echo "make figure-scripts  copy the notebook figure cells to scripts/figures/"
	@echo "make timings         print data/timings.csv (run times of the generators)"
	@echo "make requirements    regenerate requirements.txt from uv.lock"
	@echo "make clean           delete the paper figures;  make clean-data  delete all data"

all: figures checks

sync:
	$(UV) sync --locked

# ---------------------------------------------------------------------------------------
# Data. Generators that write several files are keyed on their first output.

MAIN := $(D)/agent_variable.npz $(D)/agent_uniform.npz
LATTICE := $(D)/lattice_variable.npz $(D)/lattice_uniform.npz
LONG_RUN := $(D)/agent_variable_long_run_n125_rate100.npz
SELF_PLAY := $(D)/self_interactions/variable_eps0.001_self0.npz \
	$(D)/self_interactions/uniform_eps0.001_self0.npz
RATE_SWAP := $(addprefix $(D)/axis_entry_rate_sensitivity/, small_rate_long.npz \
	uniform_to_variable.npz switch_to_uniform_0.2.npz keep_variable_rates.npz)
FIGURE_DATA := $(MAIN) $(LATTICE) $(LONG_RUN) $(addprefix $(D)/, agent_noise.npz \
	agent_pairs.npz basin.npz axis_entry.npz convergence_phase_diagrams.npz gtft_q_shock.npz \
	gtft_disk_displacement.npz)
CHECK_DATA := $(MAIN) $(LONG_RUN) $(SELF_PLAY) $(RATE_SWAP) $(D)/axis_entry.npz

data: $(sort $(FIGURE_DATA) $(CHECK_DATA))

$(word 1,$(MAIN)):
	$(TIMED) scripts/generate/agent_main_data.py
$(word 2,$(MAIN)): | $(word 1,$(MAIN))

$(word 1,$(LATTICE)):
	$(TIMED) scripts/generate/lattice_data.py
$(word 2,$(LATTICE)): | $(word 1,$(LATTICE))

$(D)/agent_noise.npz:
	$(TIMED) scripts/generate/agent_noise_data.py
$(D)/agent_pairs.npz:
	$(TIMED) scripts/generate/agent_pair_data.py
$(D)/basin.npz:
	$(TIMED) scripts/generate/basin_data.py
$(D)/convergence_phase_diagrams.npz:
	$(TIMED) scripts/generate/convergence_phase_diagrams_data.py
$(D)/gtft_q_shock.npz:
	$(TIMED) scripts/generate/gtft_q_shock_data.py
$(D)/gtft_disk_displacement.npz:
	$(TIMED) scripts/generate/gtft_disk_displacement_data.py

# Computed from the main runs: delete these too when rerunning those.
$(D)/axis_entry.npz: | $(MAIN)
	$(TIMED) scripts/generate/axis_entry_data.py
$(word 1,$(RATE_SWAP)): | $(MAIN)
	$(TIMED) scripts/generate/axis_entry_rate_sensitivity_data.py
$(wordlist 2,4,$(RATE_SWAP)): | $(word 1,$(RATE_SWAP))

$(word 1,$(SELF_PLAY)):
	$(TIMED) scripts/generate/self_interactions_data.py
$(word 2,$(SELF_PLAY)): | $(word 1,$(SELF_PLAY))

# Also computed from the main runs. Compiled C kernel (see README). The long run refuses to
# overwrite an existing archive; resume an interrupted run with
# `make figS6 LONG_RUN_ARGS=--resume`.
$(LONG_RUN): | $(word 1,$(MAIN))
	$(TIMED) scripts/generate/agent_variable_long_run_data.py $(LONG_RUN_ARGS)

# ---------------------------------------------------------------------------------------
# Figures. Each script is extracted verbatim from a cell of main_figs.ipynb / sup_figs.ipynb.

FIGURES := $(addprefix $(F)/, cartoon.pdf agent_based_dist.pdf steps_with_streams.pdf \
	average_population_welfare.pdf continuous_over_time.pdf axis_entry_p_distribution.pdf \
	convergence_phase_diagrams.pdf steps_with_streams_GTFT.pdf long_run_traj.pdf \
	gtft_perturbation_trajectories_q_infinitesimal.pdf \
	gtft_perturbation_disk_displacement_r0.01_lines.pdf heatmaps.pdf \
	agent_based_dist_noisy_runs.pdf avg_coop_two_agents_noisy.pdf)

figures: check-figure-scripts $(FIGURES)

check-figure-scripts:
	@$(PYTHON) scripts/sync_figure_scripts.py --check

fig1: $(word 1,$(FIGURES))
fig2: $(word 2,$(FIGURES))
fig3: $(word 3,$(FIGURES))
figS1: $(word 4,$(FIGURES))
figS2: $(word 5,$(FIGURES))
figS3: $(word 6,$(FIGURES))
figS4: $(word 7,$(FIGURES))
figS5: $(word 8,$(FIGURES))
figS6: $(word 9,$(FIGURES))
figS7: $(word 10,$(FIGURES))
figS8: $(word 11,$(FIGURES))
figS9: $(word 12,$(FIGURES))
figS10: $(word 13,$(FIGURES))
figS11: $(word 14,$(FIGURES))

$(F)/cartoon.pdf: $(S)/fig01_conceptual_comparison.py $(PACKAGE)
	$(PYTHON) $<
$(F)/agent_based_dist.pdf: $(S)/fig02_agent_trajectories.py $(PACKAGE) $(MAIN)
	$(PYTHON) $<
$(F)/steps_with_streams.pdf: $(S)/fig03_stages_and_gradient_fields.py $(PACKAGE) $(MAIN)
	$(PYTHON) $<
$(F)/average_population_welfare.pdf: $(S)/figS01_population_payoffs.py $(PACKAGE) $(MAIN)
	$(PYTHON) $<
$(F)/continuous_over_time.pdf: $(S)/figS02_lattice_dynamics.py $(PACKAGE) $(LATTICE)
	$(PYTHON) $<
$(F)/axis_entry_p_distribution.pdf: $(S)/figS03_axis_entry_distribution.py $(PACKAGE) \
		$(D)/axis_entry.npz
	$(PYTHON) $<
$(F)/convergence_phase_diagrams.pdf: $(S)/figS04_certificate_phase_diagrams.py $(PACKAGE) \
		$(D)/convergence_phase_diagrams.npz
	$(PYTHON) $<
$(F)/steps_with_streams_GTFT.pdf: $(S)/figS05_gtft_transition.py $(PACKAGE) $(MAIN)
	$(PYTHON) $<
$(F)/long_run_traj.pdf: $(S)/figS06_long_run.py $(PACKAGE) $(LONG_RUN)
	$(PYTHON) $<
$(F)/gtft_perturbation_trajectories_q_infinitesimal.pdf: $(S)/figS07_gtft_q_shock.py \
		$(PACKAGE) $(D)/gtft_q_shock.npz
	$(PYTHON) $<
$(F)/gtft_perturbation_disk_displacement_r0.01_lines.pdf: $(S)/figS08_gtft_disk_displacement.py \
		$(PACKAGE) $(D)/gtft_disk_displacement.npz
	$(PYTHON) $<
$(F)/heatmaps.pdf: $(S)/figS09_basin_heatmaps.py $(PACKAGE) $(D)/basin.npz
	$(PYTHON) $<
$(F)/agent_based_dist_noisy_runs.pdf: $(S)/figS10_noisy_population.py $(PACKAGE) \
		$(D)/agent_noise.npz
	$(PYTHON) $<
$(F)/avg_coop_two_agents_noisy.pdf: $(S)/figS11_noisy_pair.py $(PACKAGE) $(D)/agent_pairs.npz
	$(PYTHON) $<

# ---------------------------------------------------------------------------------------
# Notebooks, checks and housekeeping.

notebooks: $(FIGURE_DATA)
	$(EXECUTE) notebooks/main_figs.ipynb notebooks/sup_figs.ipynb

checks: $(CHECK_DATA)
	$(EXECUTE) notebooks/checks.ipynb

figure-scripts:
	$(PYTHON) scripts/sync_figure_scripts.py

timings:
	@column -s, -t data/timings.csv

requirements:
	$(UV) export --locked --no-dev --no-hashes -o requirements.txt

lint:
	$(UV) run --locked ruff check .

format:
	$(UV) run --locked ruff format src scripts

clean:
	rm -f $(FIGURES)

clean-data:
	rm -f $(sort $(FIGURE_DATA) $(CHECK_DATA))
