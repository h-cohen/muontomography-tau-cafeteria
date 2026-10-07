# Reproduce every result, figure and the paper:  make all
# Cheap smoke run (CI):                         make all FAST=1   (writes runs/fast/, never results/)
CONFIG  := configs/cafeteria.yaml
CLI     := uv run cafetomo $(if $(FAST),--fast,)
RUNS    := $(if $(FAST),runs/fast,runs)
RESULTS := $(if $(FAST),runs/fast/results,results)
CACHE   := $(RUNS)/.cache
GEN     := paper/generated
POSE    := $(RESULTS)/pose.json
FIGS    := setup opacity backprojection autofocus triangulation depth volume uncertainty
FIGPDF  := $(FIGS:%=$(GEN)/figures/%.pdf)

.PHONY: all ingest selfcal opacity reconstruct analysis validation uncertainty export figures paper viewer test clean arxiv
.DELETE_ON_ERROR:

all: paper viewer

ingest: $(RUNS)/ingest/meta.json
$(RUNS)/ingest/meta.json: $(CONFIG) $(wildcard data/*.root)
	$(CLI) ingest --config $(CONFIG) --out $(RUNS)/ingest --results $(RESULTS)

selfcal: $(POSE)
$(POSE): $(RUNS)/ingest/meta.json
	$(CLI) selfcal --config $(CONFIG) --ingest $(RUNS)/ingest --out $@

opacity: $(RUNS)/opacity/meta.json
$(RUNS)/opacity/meta.json: $(POSE)
	$(CLI) opacity --config $(CONFIG) --pose $(POSE) --ingest $(RUNS)/ingest --out $(RUNS)/opacity

reconstruct: $(RUNS)/voxels/meta.json
$(RUNS)/voxels/meta.json: $(RUNS)/opacity/meta.json
	$(CLI) reconstruct --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity \
	  --out $(RUNS)/voxels --results $(RESULTS) --cache $(CACHE)

analysis: $(RESULTS)/beamdepth.json
$(RESULTS)/beamdepth.json: $(RUNS)/voxels/meta.json
	$(CLI) analyze --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity \
	  --voxels $(RUNS)/voxels --results $(RESULTS) --cache $(CACHE)

validation: $(RESULTS)/validation.json
$(RESULTS)/validation.json: $(RESULTS)/beamdepth.json
	$(CLI) validate --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity \
	  --results $(RESULTS) --cache $(CACHE)

uncertainty: $(RESULTS)/uncertainty.json
$(RESULTS)/uncertainty.json: $(RESULTS)/beamdepth.json
	$(CLI) uncertainty --config $(CONFIG) --pose $(POSE) --ingest $(RUNS)/ingest \
	  --opacity $(RUNS)/opacity --voxels $(RUNS)/voxels --results $(RESULTS) \
	  --out $(RUNS)/bootstrap --cache $(CACHE)

export: $(RUNS)/export/meta.json
$(RUNS)/export/meta.json: $(RESULTS)/uncertainty.json
	$(CLI) export --config $(CONFIG) --pose $(POSE) --voxels $(RUNS)/voxels \
	  --bootstrap $(RUNS)/bootstrap --results $(RESULTS) --out $(RUNS)/export

viewer: $(RUNS)/viewer.html
$(RUNS)/viewer.html: $(RUNS)/export/meta.json $(wildcard viewer/src/*.mjs) viewer/shell.html
	$(CLI) viewer --export $(RUNS)/export --out $@

figures: $(FIGPDF)
$(GEN)/figures/%.pdf: paper/figures/make_%.py paper/figures/style.py \
                      $(RESULTS)/uncertainty.json $(RESULTS)/validation.json
	uv run python paper/figures/make_$*.py --config $(CONFIG) --pose $(POSE) \
	  --results $(RESULTS) --runs $(RUNS) --out $@

$(GEN)/numbers.tex: $(RESULTS)/uncertainty.json $(RESULTS)/validation.json
	$(CLI) numbers --results $(RESULTS) --out $@

paper: $(GEN)/paper.pdf
$(GEN)/paper.pdf: paper/main.tex $(wildcard paper/sections/*.tex) paper/refs.bib \
                  $(GEN)/numbers.tex $(FIGPDF)
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=generated main.tex
	mv $(GEN)/main.pdf $@

test:
	uv run ruff check .
	uv run vulture
	uv run pytest -m "not slow"
	node --test viewer/test/*.test.mjs

arxiv: paper
	tar -czf $(GEN)/arxiv.tar.gz -C paper main.tex sections refs.bib \
	  generated/numbers.tex generated/main.bbl $(FIGS:%=generated/figures/%.pdf)

clean:
	rm -rf runs paper/generated
