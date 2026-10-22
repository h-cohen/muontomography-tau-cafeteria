# Reproduce everything into a fresh root:       make reproduce REPRO=runs/repro (REPRODUCE.md)
# Baseline chain into runs/ and results/:        make all
# Cheap analysis/viewer smoke run (CI):          make all FAST=1
#                                               (writes only runs/fast/)
CONFIG  := configs/cafeteria.yaml
CLI     := uv run cafetomo $(if $(FAST),--fast,)
RUNS    := $(if $(FAST),runs/fast,runs)
RESULTS := $(if $(FAST),runs/fast/results,results)
CACHE   := $(RUNS)/.cache
# Conservative invalidation: any package or dependency change rebuilds the
# scientific chain. Grouped targets regenerate missing companion JSON files.
CODE    := $(wildcard src/cafetomo/*.py) pyproject.toml uv.lock
INGEST  := $(RUNS)/ingest/meta.json $(RESULTS)/data.json
RECON   := $(RUNS)/voxels/meta.json $(RESULTS)/reconstruction.json
ANALYSIS := $(RESULTS)/beamdepth.json $(RESULTS)/beams.json $(RESULTS)/autofocus.json
NUMERIC = $(RESULTS)/data.json $(POSE) $(RESULTS)/reconstruction.json $(ANALYSIS) \
           $(RESULTS)/validation.json $(RESULTS)/uncertainty.json $(RESULTS)/inputs.json \
           $(RESULTS)/depthdiagnostics.json $(RESULTS)/arraydepth.json
# main.tex reads generated/numbers.tex and generated/figures/ relative to paper-draft/.
# GEN is that `generated` directory: paper-draft/generated, or runs/fast/generated
# under FAST. latexmk runs in paper-draft/ with TEXINPUTS led by GEN's parent, so
# `generated/...` resolves to GEN before paper-draft/ is searched, and -outdir keeps
# every build product in GEN; BIBINPUTS lets bibtex (run from GEN) find refs.bib.
GEN     := $(if $(FAST),runs/fast/generated,paper-draft/generated)
TEXROOT := $(abspath $(dir $(GEN)))
POSE    := $(RESULTS)/pose.json
FIGS    := setup opacity height depth
FIGPDF  := $(FIGS:%=$(GEN)/figures/%.pdf)
AUTHORFIG := $(wildcard paper-draft/figures/voxels.png paper-draft/figures/room_voxels.pdf paper-draft/figures/room_voxels.png)

.PHONY: all ingest ingest-raw beams2d channelcal rawcheck raw selfcal opacity reconstruct analysis depthcheck arraydepth validation uncertainty export figures numbers paper viewer test clean arxiv
.DELETE_ON_ERROR:

all: viewer
ifeq ($(strip $(FAST)),)
	$(MAKE) paper
endif

$(RESULTS)/inputs.json: configs/paper-inputs.json
	mkdir -p $(RESULTS)
	cp $< $@

ingest: $(INGEST)
$(INGEST) &: $(CONFIG) $(wildcard data/*.root) $(CODE)
	$(CLI) ingest --config $(CONFIG) --out $(RUNS)/ingest --results $(RESULTS)

# Raw-cluster pos0 (see README "Raw event data"); never part of `all`, never read by the paper.
RAWRUNS := runs/raw
ingest-raw: $(RAWRUNS)/ingest/meta.json
$(RAWRUNS)/ingest/meta.json: $(CONFIG) $(wildcard data/*.root) $(CODE)
	$(CLI) ingest-raw --config $(CONFIG) --out $(RAWRUNS)/ingest --cache $(RAWRUNS)/cache \
	  --jobs $(or $(JOBS),4)

beams2d: $(RAWRUNS)/beams2d/meta.json
$(RAWRUNS)/beams2d/meta.json: $(RAWRUNS)/ingest/meta.json $(CODE)
	$(CLI) beams2d --config $(CONFIG) --ingest $(RAWRUNS)/ingest --out $(RAWRUNS)/beams2d

channelcal: $(RAWRUNS)/channelcal/meta.json
$(RAWRUNS)/channelcal/meta.json: $(CONFIG) $(CODE)
	$(CLI) channelcal --config $(CONFIG) --out $(RAWRUNS)/channelcal --cache $(RAWRUNS)/cache \
	  --jobs $(or $(JOBS),4)

# rawcheck reads the reviewed ROOT ingest and pose as fixed inputs: they are deliberately not
# prerequisites, so a raw check can never rebuild the baseline.
rawcheck: $(RAWRUNS)/results/rawdata.json
$(RAWRUNS)/results/rawdata.json: $(RAWRUNS)/ingest/meta.json $(RAWRUNS)/channelcal/meta.json $(CODE)
	$(CLI) rawcheck --config $(CONFIG) --pose $(POSE) --ingest $(RAWRUNS)/ingest \
	  --root-ingest $(RUNS)/ingest --channelcal $(RAWRUNS)/channelcal --cache $(RAWRUNS)/cache \
	  --out $@

raw: beams2d channelcal rawcheck

selfcal: $(POSE)
$(POSE): $(INGEST) $(CODE)
	$(CLI) selfcal --config $(CONFIG) --ingest $(RUNS)/ingest --out $@

opacity: $(RUNS)/opacity/meta.json
$(RUNS)/opacity/meta.json: $(POSE) $(CODE)
	$(CLI) opacity --config $(CONFIG) --pose $(POSE) --ingest $(RUNS)/ingest --out $(RUNS)/opacity

reconstruct: $(RECON)
$(RECON) &: $(RUNS)/opacity/meta.json $(CODE)
	$(CLI) reconstruct --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity \
	  --out $(RUNS)/voxels --results $(RESULTS) --cache $(CACHE)

analysis: $(ANALYSIS)
$(ANALYSIS) &: $(RECON) $(CODE)
	$(CLI) analyze --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity \
	  --voxels $(RUNS)/voxels --results $(RESULTS) --cache $(CACHE)

depthcheck: $(RESULTS)/depthdiagnostics.json
$(RESULTS)/depthdiagnostics.json: $(ANALYSIS) $(POSE) $(INGEST) $(RUNS)/opacity/meta.json $(CODE)
	$(CLI) depthcheck --config $(CONFIG) --pose $(POSE) --ingest $(RUNS)/ingest \
	  --opacity $(RUNS)/opacity --results $(RESULTS)

validation: $(RESULTS)/validation.json
$(RESULTS)/validation.json: $(ANALYSIS) $(RUNS)/opacity/meta.json $(POSE) $(CODE)
	$(CLI) validate --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity \
	  --results $(RESULTS) --cache $(CACHE)

uncertainty: $(RESULTS)/uncertainty.json
$(RESULTS)/uncertainty.json: $(ANALYSIS) $(POSE) $(INGEST) \
                            $(RUNS)/opacity/meta.json $(RUNS)/voxels/meta.json $(CODE)
	$(CLI) uncertainty --config $(CONFIG) --pose $(POSE) --ingest $(RUNS)/ingest \
	  --opacity $(RUNS)/opacity --voxels $(RUNS)/voxels --results $(RESULTS) \
	  --out $(RUNS)/bootstrap --cache $(CACHE)

export: $(RUNS)/export/meta.json
$(RUNS)/export/meta.json: $(RESULTS)/uncertainty.json $(RESULTS)/arraydepth.json $(CODE)
	$(CLI) export --config $(CONFIG) --pose $(POSE) --voxels $(RUNS)/voxels \
	  --bootstrap $(RUNS)/bootstrap --results $(RESULTS) --out $(RUNS)/export

viewer: $(RUNS)/viewer.html
$(RUNS)/viewer.html: $(RUNS)/export/meta.json $(wildcard viewer/src/*.mjs) viewer/shell.html $(CODE)
	$(CLI) viewer --export $(RUNS)/export --out $@

figures: $(FIGPDF)
$(GEN)/figures/%.pdf: paper-draft/figures/make_%.py paper-draft/figures/style.py \
                      $(RESULTS)/uncertainty.json $(RESULTS)/validation.json \
                      $(RESULTS)/beams.json $(RESULTS)/beamdepth.json $(POSE) \
                      $(RESULTS)/data.json $(RESULTS)/reconstruction.json \
                      $(RESULTS)/autofocus.json $(RUNS)/voxels/meta.json $(RUNS)/opacity/meta.json
	uv run python paper-draft/figures/make_$*.py --config $(CONFIG) --pose $(POSE) \
	  --results $(RESULTS) --runs $(RUNS) --out $@

$(GEN)/figures/depth.pdf: $(RESULTS)/inputs.json

arraydepth: $(RESULTS)/arraydepth.json
$(RESULTS)/arraydepth.json: $(RESULTS)/beams.json $(RUNS)/opacity/meta.json $(POSE) $(CODE)
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m cafetomo.arraydepth $(if $(FAST),--fast,) \
	  --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity --ingest $(RUNS)/ingest \
	  --beams $(RESULTS)/beams.json --out $@

numbers: $(GEN)/numbers.tex
$(GEN)/numbers.tex: $(NUMERIC) $(CODE)
	$(CLI) numbers --results $(RESULTS) --out $@

ifneq ($(strip $(FAST)),)
paper:
	@echo "FAST=1 only supports analysis/viewer smoke; reviewed publication requires a full run" >&2; exit 1

# FAST has no reviewed internal diagnostics or promoted scalars. A precompiled
# FAST PDF may still be packaged by arxiv (the archive regression fixture).
$(GEN)/paper.pdf:
	@echo "FAST=1 only supports analysis/viewer smoke; reviewed publication requires a full run" >&2; exit 1

arxiv: $(GEN)/paper.pdf
else
paper: internal-paper
arxiv: paper
endif

test:
	uv run ruff check . && uv run ruff format --check .
	uv run vulture
	uv run pytest -m "not slow"
	node --test viewer/test/*.test.mjs
	@if test -f $(PRODUCTION)/generated/MANIFEST.json; then $(MAKE) --no-print-directory production-check; fi

arxiv:
	tar -czf $(GEN)/arxiv.tar.gz -C paper-draft main.tex refs.bib sections \
	  -C $(TEXROOT) generated/numbers.tex generated/figures \
	  -C $(abspath $(GEN)) main.bbl

clean:
	rm -rf runs paper-draft/generated

# Explicit isolated campaign; choose a fresh IVOUT for each attempt.
IVOUT ?= runs/internal-validation/pilot-001
IVCACHE ?= runs/internal-validation/scientific-cache
IVSETTINGS ?= configs/internal-validation.yaml
.PHONY: internal-validate
internal-validate:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(CLI) internal-validate \
	  --config $(CONFIG) --ingest $(RUNS)/ingest --settings $(IVSETTINGS) \
	  --out $(IVOUT) --cache $(IVCACHE)

# Reviewed pilot publication: uses archived baseline numbers/figures, never refits them.
IVPAPERARCHIVE ?= runs/internal-validation/pilot-2026-10-13-a
IVCORRECTION ?= runs/internal-validation/weight-correction-2026-10-14
IVNOMINAL ?= runs/response-aware-2026-10-07/voxels
IVFIGOUT ?= runs/figure-review/selected-paper-publication-2026-10-18-fix3
IVTRACE ?= runs/internal-validation/recovery-trace-2026-10-14
IVSUMMARY ?= results/internalvalidation.json
TOOLS ?= runs/tools
TECTONIC ?= $(TOOLS)/tectonic
IVSELECTED := $(IVFIGOUT)/internal-main.pdf $(IVFIGOUT)/internal-overlay.pdf \
              $(IVFIGOUT)/internal-support-stability.pdf \
              $(IVFIGOUT)/internal-prediction-recovery.pdf $(IVFIGOUT)/internal-numbers.tex
.PHONY: internal-selected internal-preflight internal-paper arxiv-internal
$(IVPAPERARCHIVE)/manifest.json:
	@test -f $@ || { echo "missing $@: run make internal-validate IVOUT=$(IVPAPERARCHIVE)" >&2; exit 1; }
$(IVCORRECTION)/comparison.json:
	@test -f $@ || { echo "missing $@: run make weight-correction" >&2; exit 1; }
internal-selected: $(IVSELECTED)

internal-preflight:
	@if test -f $(IVSUMMARY); then \
	  uv run python paper-draft/figures/make_internal_paper.py --verify-selection \
	    --archive $(IVPAPERARCHIVE) --correction $(IVCORRECTION) \
	    --main-volume $(IVNOMINAL)/volume_full.npz --main-coverage $(IVNOMINAL)/views.npy \
	    --overlay paper-draft/figures/voxels.png --summary $(IVSUMMARY) \
	    --baseline-results $(RESULTS); \
	fi

$(IVSELECTED) &: paper-draft/figures/make_internal_paper.py paper-draft/figures/make_support.py \
                paper-draft/figures/internal_figure_utils.py paper-draft/figures/style.py \
                $(IVPAPERARCHIVE)/manifest.json $(IVCORRECTION)/comparison.json \
                | internal-preflight
	uv run python paper-draft/figures/make_internal_paper.py --archive $(IVPAPERARCHIVE) \
	  --correction $(IVCORRECTION) --main-volume $(IVNOMINAL)/volume_full.npz \
	  --main-coverage $(IVNOMINAL)/views.npy --overlay paper-draft/figures/voxels.png \
	  --out $(IVFIGOUT) --summary $(IVSUMMARY) --baseline-results $(RESULTS)

internal-paper: internal-selected
	uv run python paper-draft/figures/make_internal_paper.py --verify-selection \
	  --archive $(IVPAPERARCHIVE) --correction $(IVCORRECTION) \
	  --main-volume $(IVNOMINAL)/volume_full.npz --main-coverage $(IVNOMINAL)/views.npy \
	  --overlay paper-draft/figures/voxels.png --summary $(IVSUMMARY) \
	  --baseline-results $(RESULTS)
	mkdir -p $(GEN)/figures
	uv run python paper-draft/figures/make_height.py --config $(CONFIG) --pose $(POSE) \
	  --results $(RESULTS) --runs $(dir $(IVNOMINAL)) --out $(GEN)/figures/height.pdf
	cp $(IVFIGOUT)/internal-*.pdf $(IVFIGOUT)/internal-numbers.tex $(GEN)/figures/
	cp $(IVFIGOUT)/display-settings.json $(GEN)/figures/internal-display-settings.json
	uv run python -c "from pathlib import Path; from cafetomo.numbers import write_numbers; write_numbers(Path('$(RESULTS)'), Path('$(GEN)/numbers.tex'))"
	cd paper-draft && $(abspath $(TECTONIC)) --keep-logs --keep-intermediates -o $(abspath $(GEN)) main.tex
	mv $(GEN)/main.pdf $(GEN)/paper.pdf

arxiv-internal: arxiv
	cp $(GEN)/arxiv.tar.gz $(GEN)/arxiv-internal.tar.gz

# Production paper (Overleaf submodule). Stage from the reviewed baseline,
# sync into paper-production/generated, check, and push only on request.
PRODUCTION ?= paper-production
PRODSTAGE  := runs/production-stage
PRODRUNS   ?= runs/response-aware-2026-10-07
PRODUCTION_FIGURES := parallax depth opacity setup slices zprofile arrayfit pose coverage autofocus
# Nominal reconstruction rerun with every iterate's chi2 kept (volume bit-identical to PRODRUNS).
CONVRUNS   ?= runs/convergence-2026-10-21
PRODUCTION_RAW_FIGURES := maps2d channelcal
PRODUCTION_LINT ?= 3d.tex,2d.tex,rawdata.tex,sky.tex
.PHONY: production-stage production-sync production-check production-push
production-stage: internal-paper
	test -d $(PRODRUNS)/opacity -a -d $(PRODRUNS)/bootstrap -a -d $(PRODRUNS)/ingest
	rm -rf $(PRODSTAGE) && mkdir -p $(PRODSTAGE)/figures
	$(CLI) numbers --results $(RESULTS) --out $(PRODSTAGE)/numbers.tex
	$(CLI) numbers --results $(RESULTS)/production --out $(PRODSTAGE)/production-numbers.tex
	$(CLI) config-numbers --config $(CONFIG) --volume $(PRODRUNS)/voxels/volume_full.npz \
	  --out $(PRODSTAGE)/config-numbers.tex
	$(CLI) raw-numbers --rawdata $(RAWRUNS)/results/rawdata.json --config $(CONFIG) \
	  --out $(PRODSTAGE)/raw-numbers.tex
	cp $(IVFIGOUT)/internal-numbers.tex $(IVFIGOUT)/internal-main.pdf $(PRODSTAGE)/
	mv $(PRODSTAGE)/internal-main.pdf $(PRODSTAGE)/figures/
	for f in $(PRODUCTION_FIGURES); do \
	  uv run python paper-draft/figures/make_$$f.py --config $(CONFIG) --pose $(POSE) \
	    --results $(RESULTS) --runs $(PRODRUNS) --frame detector --out $(PRODSTAGE)/figures/$$f.pdf || exit 1; \
	done
	uv run python paper-draft/figures/make_convergence.py --config $(CONFIG) --pose $(POSE) \
	  --results $(RESULTS) --runs $(CONVRUNS) --frame detector --out $(PRODSTAGE)/figures/convergence.pdf
	for f in $(PRODUCTION_RAW_FIGURES); do \
	  uv run python paper-draft/figures/make_$$f.py --config $(CONFIG) --pose $(POSE) \
	    --results $(RESULTS) --runs $(RAWRUNS) --frame detector --out $(PRODSTAGE)/figures/$$f.pdf || exit 1; \
	done
production-sync: production-stage
	$(CLI) production-sync --production $(PRODUCTION) --staged $(PRODSTAGE) --results $(RESULTS)
production-check:
	$(CLI) production-check --production $(PRODUCTION) --results $(RESULTS) --lint $(PRODUCTION_LINT)
production-push: production-check
	git -C $(PRODUCTION) push origin HEAD:main

# Registered studies on the internal-validation pilot (scripts tracked in studies/).
.PHONY: weight-correction recovery-trace autofocus-study tools reproduce
weight-correction:
	test ! -e $(IVCORRECTION)
	uv run python studies/weight-correction/compare.py --pilot $(IVPAPERARCHIVE) --out $(IVCORRECTION)
recovery-trace:
	test ! -e $(IVTRACE)
	for step in register run compare; do \
	  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python studies/recovery-trace/trace.py $$step --pilot $(IVPAPERARCHIVE) \
	    --out $(IVTRACE) --ingest $(RUNS)/ingest || exit 1; \
	done
	for step in register run; do \
	  uv run python studies/recovery-trace/support_supplement.py $$step \
	    --pilot $(IVPAPERARCHIVE) --out $(IVTRACE)/support-supplement || exit 1; \
	done

# Autofocus with the corrected layer; promoted into results/production/ (production only).
AFRUNS ?= runs/autofocus-study-2026-10-21
autofocus-study:
	test ! -e $(AFRUNS)
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(CLI) autofocus-study --config $(CONFIG) --pose $(POSE) \
	  --ingest $(RUNS)/ingest --opacity $(RUNS)/opacity --results $(RESULTS) \
	  --out $(AFRUNS)/autofocuslayer.json --cache $(AFRUNS)/cache
	mkdir -p $(RESULTS)/production
	cp $(AFRUNS)/autofocuslayer.json $(RESULTS)/production/autofocuslayer.json

# Everything from the inputs into one fresh root REPRO (never the tracked results/ or the
# reviewed archives). Compare afterwards with `cafetomo compare-results`. See REPRODUCE.md.
REPRO_VARS = RUNS=$(REPRO)/baseline RESULTS=$(REPRO)/results GEN=$(REPRO)/draft \
  IVOUT=$(REPRO)/internal-validation/pilot IVCACHE=$(REPRO)/internal-validation/cache \
  IVPAPERARCHIVE=$(REPRO)/internal-validation/pilot \
  IVCORRECTION=$(REPRO)/internal-validation/weight-correction \
  IVTRACE=$(REPRO)/internal-validation/recovery-trace IVNOMINAL=$(REPRO)/baseline/voxels \
  IVFIGOUT=$(REPRO)/figure-review/selection IVSUMMARY=$(REPRO)/results/internalvalidation.json \
  RAWRUNS=$(REPRO)/raw PRODRUNS=$(REPRO)/baseline CONVRUNS=$(REPRO)/baseline \
  AFRUNS=$(REPRO)/autofocus-study PRODSTAGE=$(REPRO)/production-stage
reproduce:
	test -n "$(REPRO)"
	test ! -e $(REPRO)
	$(MAKE) $(REPRO_VARS) viewer validation depthcheck arraydepth $(REPRO)/results/inputs.json
	$(MAKE) $(REPRO_VARS) internal-validate
	$(MAKE) $(REPRO_VARS) weight-correction
	$(MAKE) $(REPRO_VARS) recovery-trace
	$(MAKE) $(REPRO_VARS) raw
	$(MAKE) $(REPRO_VARS) autofocus-study
	$(MAKE) $(REPRO_VARS) production-stage


# Pinned Tectonic (LaTeX engine for both papers), verified by checksum.
TECTONIC_URL ?= https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz
TECTONIC_SHA256 := 8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7
tools: $(TOOLS)/tectonic
$(TOOLS)/tectonic:
	mkdir -p $(TOOLS)
	curl -fsSL -o $(TOOLS)/tectonic.tar.gz "$(TECTONIC_URL)"
	echo "$(TECTONIC_SHA256)  $(TOOLS)/tectonic.tar.gz" | sha256sum -c -
	tar -xzf $(TOOLS)/tectonic.tar.gz -C $(TOOLS) tectonic
