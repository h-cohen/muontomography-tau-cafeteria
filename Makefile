# Reproduce every result, figure and the paper:  make all
# Cheap smoke run (CI):                         make all FAST=1   (writes runs/fast/, never results/
#                                               or paper/generated/)
CONFIG  := configs/cafeteria.yaml
CLI     := uv run cafetomo $(if $(FAST),--fast,)
RUNS    := $(if $(FAST),runs/fast,runs)
RESULTS := $(if $(FAST),runs/fast/results,results)
CACHE   := $(RUNS)/.cache
# main.tex reads generated/numbers.tex and generated/figures/ relative to paper/.
# GEN is that `generated` directory: paper/generated, or runs/fast/generated
# under FAST. latexmk runs in paper/ with TEXINPUTS led by GEN's parent, so
# `generated/...` resolves to GEN before paper/ is searched, and -outdir keeps
# every build product in GEN; BIBINPUTS lets bibtex (run from GEN) find refs.bib.
GEN     := $(if $(FAST),runs/fast/generated,paper/generated)
TEXROOT := $(abspath $(dir $(GEN)))
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
$(RESULTS)/validation.json: $(RESULTS)/beamdepth.json $(RUNS)/opacity/meta.json $(POSE)
	$(CLI) validate --config $(CONFIG) --pose $(POSE) --opacity $(RUNS)/opacity \
	  --results $(RESULTS) --cache $(CACHE)

uncertainty: $(RESULTS)/uncertainty.json
$(RESULTS)/uncertainty.json: $(RESULTS)/beamdepth.json $(POSE) $(RUNS)/ingest/meta.json \
                            $(RUNS)/opacity/meta.json $(RUNS)/voxels/meta.json
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

# Written alongside a stage's main target; listed so rules that read them can
# depend on them.
$(RESULTS)/data.json: $(RUNS)/ingest/meta.json ;
$(RESULTS)/reconstruction.json: $(RUNS)/voxels/meta.json ;
$(RESULTS)/beams.json $(RESULTS)/autofocus.json: $(RESULTS)/beamdepth.json ;

figures: $(FIGPDF)
$(GEN)/figures/%.pdf: paper/figures/make_%.py paper/figures/style.py \
                      $(RESULTS)/uncertainty.json $(RESULTS)/validation.json \
                      $(RESULTS)/beams.json $(RESULTS)/beamdepth.json $(POSE) \
                      $(RESULTS)/data.json $(RESULTS)/reconstruction.json \
                      $(RESULTS)/autofocus.json $(RUNS)/voxels/meta.json $(RUNS)/opacity/meta.json
	uv run python paper/figures/make_$*.py --config $(CONFIG) --pose $(POSE) \
	  --results $(RESULTS) --runs $(RUNS) --out $@

$(GEN)/numbers.tex: $(RESULTS)/uncertainty.json $(RESULTS)/validation.json
	$(CLI) numbers --results $(RESULTS) --out $@

paper: $(GEN)/paper.pdf
$(GEN)/paper.pdf: paper/main.tex $(wildcard paper/sections/*.tex) paper/refs.bib \
                  $(GEN)/numbers.tex $(FIGPDF)
	cd paper && TEXINPUTS=$(TEXROOT): BIBINPUTS=$(abspath paper): latexmk -pdf -interaction=nonstopmode -halt-on-error \
	  -outdir=$(abspath $(GEN)) main.tex
	mv $(GEN)/main.pdf $@

test:
	uv run ruff check . && uv run ruff format --check .
	uv run vulture
	uv run pytest -m "not slow"
	node --test viewer/test/*.test.mjs

arxiv: paper
	tar -czf $(GEN)/arxiv.tar.gz -C paper main.tex sections refs.bib -C $(TEXROOT) \
	  generated/numbers.tex generated/main.bbl $(FIGS:%=generated/figures/%.pdf)

clean:
	rm -rf runs paper/generated
