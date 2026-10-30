# Reproducing every result

Everything in `results/`, both papers' numbers and figures, and the viewer is
regenerated from this repository and its input data with `make`. Nothing is
taken from a local archive.

## Inputs

| Input | Where | Used by |
|---|---|---|
| ROOT angular histograms (pos0, pos1, open sky) | `data/` (in git) | baseline chain |
| Raw DAQ event files (~40 GB) | outside the repo, `../../01_data/processed/tau_cafeteria_raw_filtered/{pos0,pos1,roof}` (`raw:` in `configs/cafeteria.yaml`) | raw chain only |
| Viewer render for the draft's box illustration | `paper-draft/figures/voxels.png` (in git, author-supplied) | draft paper |
| Analysis settings | `configs/cafeteria.yaml`, `configs/internal-validation.yaml`, `configs/paper-inputs.json` | all |

## Environment

```bash
git clone https://github.com/h-cohen/muontomography-tau-cafeteria.git
cd muontomography-tau-cafeteria
uv sync                     # Python environment from uv.lock
make tools                  # Tectonic 0.17.0 into runs/tools/, checksum-verified
git submodule update --init paper-production && git -C paper-production checkout main   # production paper (Overleaf; needs a git token)
```

Node ≥ 24 is needed only for `make test` (viewer tests).

## One command

```bash
make reproduce REPRO=runs/repro JOBS=4
```

`REPRO` must not exist. Every stage writes under it — never into the tracked
`results/` or the reviewed archives — in this order:

| Stage | Make target(s) | Output under `REPRO` | Time (this laptop) |
|---|---|---|---|
| Baseline: ingest → pose fit → opacity → voxel reconstruction → height and depth analyses → validation phantoms → bootstrap and systematics → continued-array depth → viewer | `viewer validation depthcheck arraydepth` | `baseline/`, `results/*.json` | ~6 h |
| Internal-validation pilot (held-out prediction, support, persistence, synthetic recovery) | `internal-validate` | `internal-validation/pilot/` | ~2 h |
| Corrected-weight comparison | `weight-correction` (`studies/weight-correction/`) | `internal-validation/weight-correction/` | ~3 min |
| Recovery trace replay and support supplement | `recovery-trace` (`studies/recovery-trace/`) | `internal-validation/recovery-trace/` | ~1 h |
| Raw chain: raw ingest, 2D maps, channel calibration, raw-versus-ROOT checks | `raw` | `raw/` | ~15 min |
| Production 3D on the joint open-sky reference, with the corrected autofocus | `joint3d` (after `sky-joint`) | `joint3d/`, `results/production/3d/` | ~17 h |
| Internal-validation pilot, weight correction and recovery trace on the production 3D | `joint3d-internal` | `internal-validation/*-joint3d/`, `figure-review/joint3d/`, `results/production/3d/internalvalidation.json` | ~6 h |
| Registered 3D systematics, time stability and depth controls | `systematics3d` | `systematics3d-2026-10-30/`, `results/production/3d/systematics.json` | ~3 h |
| Draft panels and scalars; draft paper | `internal-selected`, `internal-paper` (inside `production-stage`) | `figure-review/selection/`, `results/internalvalidation.json`, `draft/paper.pdf` | minutes |
| Production paper macros and figures | `production-stage` | `production-stage/` | minutes |

Each stage can also be run alone with the same variables (see `REPRO_VARS` in
the Makefile); stages refuse to overwrite an existing output directory. To
continue an interrupted reproduction, `make reproduce-stages REPRO=runs/repro`
(finished stages are skipped; remove a partially written study directory first).

## Compare with the committed results

```bash
uv run cafetomo compare-results --reference results --candidate runs/repro/results --out runs/repro/compare-results.json
uv run cafetomo compare-results --reference paper-production/generated --candidate runs/repro/production-stage --out runs/repro/compare-generated.json
```

Each file is reported as identical, equal as data (NaN equals NaN), different
(every differing number with its delta, scaled by the quoted uncertainty when
one exists) or missing.

## Why numbers can differ

The committed baseline (`results/` up to `uncertainty.json`) was produced on
2026-10-07 from commit `a9d066f` with uncommitted changes, and the code has
changed since (opacity zero-spread weights, pose fit, bootstrap streams, full
solver history, the autofocus layer fix). A reproduction with today's code is
therefore expected to differ slightly; differences are reported, never tuned
away. The corrected-weight comparison reproduces its archive exactly when run on
the archived pilot.

## Verified on 2026-10-22

`make reproduce` in a fresh clone of commit `ccafaa6` (fixes up to `006aec9`
applied while it ran), compared with the committed files:

- Identical: every baseline result (`data`, `pose`, `reconstruction`, `beams`,
  `beamdepth`, `arraydepth`, `depthdiagnostics`, `inputs`), the corrected
  autofocus `results/production/autofocuslayer.json`, and the production
  paper's `config-numbers.tex`, `raw-numbers.tex`, `production-numbers.tex`.
- Identical scalars in `results/internalvalidation.json` (held-out deviances,
  folds, recovery, replicas, corrected weight objectives); its input list
  differs only in paths and file sizes.
- Expected differences, both from bugs fixed in the code: the original
  autofocus values in `autofocus.json`, `uncertainty.json` (`autofocus_z_*`) and
  `validation.json` (`focus_*`) — the reproduction gives exactly the corrected
  autofocus — and `original_weight_objective` (archived 5.04e22, reproduced
  0.424): the archived value is the numerical failure of the pre-fix opacity
  weights, which the current code no longer produces. The draft no longer quotes it.
- Runtime: baseline ~3 h, pilot ~1.3 h, autofocus study ~2 h, the rest minutes.

## Updating the papers from a reproduction

Results become paper numbers only by explicit promotion: review the comparison,
copy the accepted files into `results/` (production-only ones into
`results/production/`), render a fresh reviewed draft selection
(`IVFIGOUT=runs/figure-review/<new>`, see README "Reviewed internal pilot in the
paper"), then `make production-sync` and, when agreed, `make production-push`.
