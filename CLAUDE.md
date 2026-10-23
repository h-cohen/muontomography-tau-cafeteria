# CLAUDE.md — cafetomo

Paper-companion repo: two-position muon tomography of the TAU cafeteria ceiling (arXiv preprint,
*Seeing a Ceiling with Cosmic Muons: Three-Dimensional Imaging from Two Views*). Also read `AGENTS.md`
(scientific interpretation rules) and `README.md` (commands, status, evidence).
Handoff / current state: `docs/dev/handoff-2026-10-14.md` (gitignored; read first). Completion report and
review records: `docs/dev/reports/`. Specs/plans: `docs/dev/{specs,plans}/` (old checkboxes ≠ open work).

## Status (2026-10-23)
- Internal-validation pilot (7 tasks), weight correction, trace replay and the short draft paper
  (`paper-draft/`) are COMPLETE and reviewed. Do not rerun or redispatch them. Scientific validation is
  NOT complete (depth, calibrated uncertainty, null/misspecification controls open — handoff "Next
  scientific work").
- Production paper (`paper-production/`, Overleaf, pushed): user owns `3d.tex`; we also wrote `2d.tex`,
  `rawdata.tex`, `sky.tex` and edited coauthors' `setup.tex` (8.5 m = roof-slab top above floor; live times;
  open-sky role without simulation). Coauthors edit in Overleaf. The 3D nominal voxel-field figure is
  a placeholder: the user will supply the image.
- Done in production: corrected autofocus (layer bug fixed; `results/production/autofocuslayer.json`,
  7.21 ± 0.35 m; draft keeps 7.18 m), solver-convergence and coverage/resolution figures, raw-chain 2D maps
  and Landau⊗Gauss MPV (PRELIMINARY macros from `runs/raw/results/rawdata.json`, not promoted).
- Done (2D, 2026-10-23): roof split into its two orientations (`raw.subsets` SKY_A runs 37-57,
  SKY_B runs 58-77 turned 90 deg cw, user-confirmed); `ingest-raw --sky`; preregistered joint
  open-sky reference (`make sky-joint`, `studies/sky-orientation/`, `results/production/skyjoint.json`):
  sky azimuthal harmonics fitted from the A/B split, chi2/dof 1.06, no gain change. Assumption: roof A
  shares the room's detector axes (not recorded; 180 deg flip = smooth shift ≤ 0.09 in λ, a systematic).
  Production `sky.tex` (subsection "Two roof orientations", Fig. 5 = roof rates `skyroof`) and `2d.tex`
  (λ maps of pos0/pos1 vs joint reference, `skyjoint`) are written and pushed.
- Open next (user's list): 3D analysis rerun with the joint sky reference (the 3D still uses the DAQ
  sky histogram); then fix the reproduction (`runs/repro` failed at raw stage, `rawdata.json`
  AttributeError) and run a full `make production-sync` (last syncs staged by hand, MANIFEST credits the
  whole `generated/` to the latest commit); continued-array depth systematics and null/misspecification controls (preregister first); review and
  promote `rawdata.json`; coauthor items: beam spacing 1.2 m (clear gap?) vs pitch 1.59 m, 0.6 m sideways
  pose offset, empty `\label{}`s in setup.tex.

## Resume on another machine
- Nothing is copied: `REPRODUCE.md` (clone, `uv sync`, `make tools`, submodule; then `make reproduce
  REPRO=<fresh>` regenerates everything, `cafetomo compare-results` compares with `results/`). Raw DAQ
  files (~40 GB) live outside the repo at `../../01_data/processed/tau_cafeteria_raw_filtered` (`raw:`).
  Overleaf git token in a credential helper (username `git`); Overleaf is intermittently unreachable.
- Studies once run from gitignored dirs are tracked in `studies/` (weight correction, recovery trace).
- Check: `make test`, `make -s internal-preflight`, `make production-check`.

## Paper framing (user priorities)
- 3D voxel room/ceiling reconstruction is the primary result. Height, pitch, depth, held-out prediction,
  support and persistence are supporting evidence.
- Terms: "ceiling height"/"beam-bottom height" (vertical position, detector-referenced); depth/thickness
  h = z_top − z_bottom; width across the beams; analysis frame: beams along y (production paper: along X).
  Never "beam height".
- Effective voxel opacity is 1/m, not mass density. Voxel-profile FWHM (≈2.3 m) is an image diagnostic,
  not physical depth. Flexible matched-beam depth is bound/unresolved (NaN) — never a measured thickness.
- Continued-array depth/width/bottom spreads are conditional counting spreads (fixed pose, relative centres,
  array completeness), not total errors.
- On-site tape values (bottom 7.3 m, depth 1.2 m) and rough 2.42 m separation are external comparisons,
  NEVER fit targets or constraints. Tracker-to-floor origin is not established. User would like depth
  near 1.2 m — matching it is not a validation goal.
- Retrospective pilot reused data: descriptive, no p-values/significance; frequencies ≠ posterior probs.
- Keep `AUTHORS` placeholder and user-supplied `paper-draft/figures/voxels.png` (viewer render) unchanged. Double-column A4.

## Measurement
- One 4-plane scintillator tracker; active width 35.375 cm, layer gap 38.9 cm. Positions pos0 (origin)
  and pos1 (fitted pose, free baseline, floor-offset prior only); open-sky roof run = response reference.
- λ = −ln[n_pos / ((t_pos/t_sky)·n_sky)] per world sky-tangent bin; live times from `dT` → absolute gauge.
- Voxel inversion: λ = A q, aperture-averaged path lengths, non-negative iterative solver, horizontal-only TV.
- Concrete ρ pinning (`muonphysics.py`) is an optional comparison, not the primary depth model.

## Pipeline
- `make all` (→ viewer; `FAST=1` → `runs/fast/` smoke only, FAST paper unsupported). Bootstrap dominates runtime.
- Paper: `make paper TECTONIC=runs/tools/tectonic`, `make arxiv TECTONIC=runs/tools/tectonic` — guarded
  archived publication (source/hash preflight), no baseline refit. Panels/scalars from
  `paper-draft/figures/make_internal_paper.py` + `results/internalvalidation.json`.
- Internal validation: `make internal-validate IVOUT=runs/internal-validation/<new>` (fresh root; renderer
  refuses existing outputs; changed inputs need fresh `IVFIGOUT`/`IVSUMMARY`, review, explicit promotion).
- Numerical campaigns: register before outcomes, single-threaded BLAS, frozen source/config, disjoint outputs.
- Production paper: Overleaf submodule `paper-production/` (coauthors edit prose there). `make
  production-sync` stages macros (`numbers.tex`, `internal-numbers.tex`, `config-numbers.tex` from config +
  code constants + archive, `raw-numbers.tex` PRELIMINARY, `production-numbers.tex` from
  `results/production/`) and figures (`--frame detector`) into its `generated/` (+MANIFEST), commits in the
  submodule, never pushes. Push order: `make production-push`, then commit the submodule pointer, then push
  GitHub — only on explicit user request. `paper-draft/` = short inner draft (guarded).
- Axis naming: analysis code/results use beams along y; the production paper uses the detector naming
  (beams along X, pos1 displaced along Y). Figures swap via `paper-draft/figures/frame.py`; text symbols are
  swapped by hand. Never edit `paper-draft/figures/style.py` or add files to top-level `results/`: both are
  inputs of the reviewed draft selection (fix3); production-only results go in `results/production/`.
- NEVER `make clean` while evidence archives are needed: it deletes all `runs/` and `paper-draft/generated/`
  (Tectonic, caches, pilot archives). Evidence map in handoff.

## Rules
- Every paper number is a macro (results/*.json, results/production/, config/constants/archive,
  preliminary raw chain); every figure has a script. Never type numbers into paper sections
  (`make production-check` lints `PRODUCTION_LINT`).
- Never tune `configs/cafeteria.yaml` to hit expected values; report what the pipeline yields.
- Uncertainties: bootstrap = statistical; systematics are shifts, combined only in the error budget.
- NaN = not measured; errors fail loudly; no partial outputs; keep failures/bounds/censored outcomes.
  Lengths in metres (detector block in cm).
- Known limits: depth weakly constrained by two views; y of pos1 weakly constrained along beams;
  azimuth < 1° invisible to selfcal; coarse lattices alias beam pitch (use twin half-voxel lattices).
- Code: flat package `src/cafetomo/`, TDD, `make test` (ruff check+format, vulture, pytest -m "not slow",
  node ≥24 viewer tests). No predecessor-project names, no provenance notes, no TODOs/dead code.
- Git: main only; author Hadar Cohen <hal.nls@gmail.com>, Conventional Commits, no trailers. Never commit
  `runs/`, `paper-draft/generated/`, `docs/dev/`, `.superpowers/`. Push/merge only on explicit user request.
- User prefers concise communication and autonomous progress.
