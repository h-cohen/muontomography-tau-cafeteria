# Reproduce every result, figure and the paper:  make all
# Cheap smoke run (CI):                         make all FAST=1   (writes runs/fast/, never results/
#                                               or paper/generated/)
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
           $(RESULTS)/depthdiagnostics.json
# main.tex reads generated/numbers.tex and generated/figures/ relative to paper/.
# GEN is that `generated` directory: paper/generated, or runs/fast/generated
# under FAST. latexmk runs in paper/ with TEXINPUTS led by GEN's parent, so
# `generated/...` resolves to GEN before paper/ is searched, and -outdir keeps
# every build product in GEN; BIBINPUTS lets bibtex (run from GEN) find refs.bib.
GEN     := $(if $(FAST),runs/fast/generated,paper/generated)
TEXROOT := $(abspath $(dir $(GEN)))
POSE    := $(RESULTS)/pose.json
FIGS    := setup opacity height depth
FIGPDF  := $(FIGS:%=$(GEN)/figures/%.pdf)
AUTHORFIG := $(wildcard paper/figures/room_voxels.pdf paper/figures/room_voxels.png)

.PHONY: all ingest selfcal opacity reconstruct analysis validation uncertainty export figures paper viewer test clean arxiv
.DELETE_ON_ERROR:

all: paper viewer

$(RESULTS)/inputs.json: configs/paper-inputs.json
	mkdir -p $(RESULTS)
	cp $< $@

ingest: $(INGEST)
$(INGEST) &: $(CONFIG) $(wildcard data/*.root) $(CODE)
	$(CLI) ingest --config $(CONFIG) --out $(RUNS)/ingest --results $(RESULTS)

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
$(RUNS)/export/meta.json: $(RESULTS)/uncertainty.json $(CODE)
	$(CLI) export --config $(CONFIG) --pose $(POSE) --voxels $(RUNS)/voxels \
	  --bootstrap $(RUNS)/bootstrap --results $(RESULTS) --out $(RUNS)/export

viewer: $(RUNS)/viewer.html
$(RUNS)/viewer.html: $(RUNS)/export/meta.json $(wildcard viewer/src/*.mjs) viewer/shell.html $(CODE)
	$(CLI) viewer --export $(RUNS)/export --out $@

figures: $(FIGPDF)
$(GEN)/figures/%.pdf: paper/figures/make_%.py paper/figures/style.py \
                      $(RESULTS)/uncertainty.json $(RESULTS)/validation.json \
                      $(RESULTS)/beams.json $(RESULTS)/beamdepth.json $(POSE) \
                      $(RESULTS)/data.json $(RESULTS)/reconstruction.json \
                      $(RESULTS)/autofocus.json $(RUNS)/voxels/meta.json $(RUNS)/opacity/meta.json
	uv run python paper/figures/make_$*.py --config $(CONFIG) --pose $(POSE) \
	  --results $(RESULTS) --runs $(RUNS) --out $@

$(GEN)/figures/depth.pdf: $(RESULTS)/inputs.json

$(GEN)/numbers.tex: $(NUMERIC) $(CODE)
	$(CLI) numbers --results $(RESULTS) --out $@

paper: $(GEN)/paper.pdf
$(GEN)/paper.pdf: paper/main.tex $(wildcard paper/sections/*.tex) paper/refs.bib \
                  $(GEN)/numbers.tex $(FIGPDF) $(AUTHORFIG)
	cd paper && TEXINPUTS=$(TEXROOT): BIBINPUTS=$(abspath paper): latexmk -pdf -interaction=nonstopmode -halt-on-error \
	  -outdir=$(abspath $(GEN)) main.tex
	mv $(GEN)/main.pdf $@

test:
	uv run ruff check . && uv run ruff format --check .
	uv run vulture
	uv run pytest -m "not slow"
	node --test viewer/test/*.test.mjs

arxiv: paper
	tar -czf $(GEN)/arxiv.tar.gz -C paper main.tex sections refs.bib $(AUTHORFIG:paper/%=%) \
	  -C $(abspath $(GEN)) main.bbl -C $(TEXROOT) \
	  generated/numbers.tex $(FIGS:%=generated/figures/%.pdf)

clean:
	rm -rf runs paper/generated
