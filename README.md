# Muon tomography of the TAU cafeteria ceiling

[Code and data repository](https://github.com/h-cohen/muontomography-tau-cafeteria).

Code, data and manuscript for *Seeing the Ceiling with Cosmic Muons: Two-View Tomography of a Real Room*.
Two positions of one four-plane tracker, normalized by an open-sky run, provide a
three-dimensional room reconstruction, beam-height and pitch estimates, and
model-dependent beam-depth estimates.

The detector separation is self-calibrated; its scale remains conditional on the
floor-offset prior and fitting objective. On-site bottom/depth measurements
(7.3 m / 1.2 m) are comparison inputs, not fitting targets. The continued-array
depth estimate fixes relative beam centres and array completeness; its count
spread excludes uncertainty in those assumptions.

## Setup

Run commands from the repository root. Requirements:

- Python >= 3.12, managed by [uv](https://docs.astral.sh/uv/).
- GNU make >= 4.3.
- Node >= 24 for viewer unit tests.
- For `make paper`/`make all`: `latexmk`, `pdflatex`, BibTeX and the LaTeX packages
  `siunitx`, `natbib`, `hyperref`, `booktabs` and `geometry`.

```bash
uv sync
uv run playwright install chromium
```

The Playwright installation is needed for browser tests. On Linux, its system
libraries can also be installed with `uv run playwright install --with-deps chromium`.
The ROOT input histograms are under `data/`; settings are in `configs/cafeteria.yaml`.

## Full execution

Build the analysis, paper and self-contained viewer:

```bash
make all
```

Run the analysis and generate all figures/numbers without a TeX installation:

```bash
make validation uncertainty arraydepth viewer figures numbers
```

Make resolves upstream dependencies automatically. The full run includes the
50-replica reconstruction bootstrap and a separate 50-replica conditional depth
study. Up-to-date stages are reused; changes to analysis code, dependencies or
configuration rebuild affected stages.

To force a complete rebuild, including the paper:

```bash
make -B all
```

## Individual stages

| Stage | Command | Output |
|---|---|---|
| Ingest ROOT counts and live times | `make ingest` | `runs/ingest/`, `results/data.json` |
| Fit the second detector pose | `make selfcal` | `results/pose.json` |
| Opacity maps and weights | `make opacity` | `runs/opacity/` |
| Full and single-position voxel reconstructions | `make reconstruct` | `runs/voxels/`, `results/reconstruction.json` |
| Autofocus, beam triangulation and flexible depth fit | `make analysis` | `results/{autofocus,beams,beamdepth}.json` |
| Alternative count-weight depth diagnostic | `make depthcheck` | `results/depthdiagnostics.json` |
| Conditional continued-array depth study | `make arraydepth` | `results/arraydepth.json` |
| Phantom validation | `make validation` | `results/validation.json` |
| Count bootstrap and sensitivity budget | `make uncertainty` | `runs/bootstrap/`, `results/uncertainty.json` |
| Viewer data export | `make export` | `runs/export/` |
| Self-contained interactive viewer | `make viewer` | `runs/viewer.html` |
| Scripted paper figures | `make figures` | `paper/generated/figures/` |
| Result macros for the paper | `make numbers` | `paper/generated/numbers.tex` |
| Compile paper | `make paper` | `paper/generated/paper.pdf` |
| Package arXiv sources | `make arxiv` | `paper/generated/arxiv.tar.gz` |

## Build and open the viewer

```bash
make viewer
xdg-open runs/viewer.html
```

The HTML embeds its data and can be opened directly in a WebGL-capable browser.
Alternatively, serve the generated directory:

```bash
uv run python -m http.server 8000 --bind 127.0.0.1 --directory runs
```

Open [http://localhost:8000/viewer.html](http://localhost:8000/viewer.html).
Stop the server with Ctrl+C. This serves the file; it does not rerun the analysis.

To rebuild only the HTML from an existing export, bypassing upstream Make targets:

```bash
uv run cafetomo viewer --export runs/export --out runs/viewer.html
```

The viewer displays the reconstructed room and toggleable beam boxes. A bound
box fit is labelled as a conditional layer with unresolved physical depth.
The continued-array estimate is reported separately in `results/arraydepth.json`.

## FAST smoke execution

```bash
make all FAST=1
```

Without TeX:

```bash
make validation uncertainty arraydepth viewer figures numbers FAST=1
```

FAST uses reduced solver/replica settings and writes into `runs/fast/`, including
`runs/fast/results/`, `runs/fast/generated/` and `runs/fast/viewer.html`. It does
not overwrite the full-run `results/` or `paper/generated/` and is not used for
paper measurements. Add `FAST=1` to the analysis, viewer, figure, paper or archive
commands above.

Open the smoke viewer:

```bash
xdg-open runs/fast/viewer.html
```

Or use the same server above and open
[http://localhost:8000/fast/viewer.html](http://localhost:8000/fast/viewer.html).

## Paper and submission archive

```bash
make figures numbers
make paper
make arxiv
```

Every paper quantity is generated from `results/*.json`. Add the author-provided
3D room figure as `paper/figures/room_voxels.pdf` or `paper/figures/room_voxels.png`,
then run `make paper` again. Without that file, the paper retains a labelled
placeholder. The archive includes the figure when present, generated figures,
number macros, manuscript sections and the formatted bibliography.

If the local Tectonic executable is available under `runs/tools/`, it can compile
already-generated paper inputs without `latexmk`:

```bash
runs/tools/tectonic --keep-logs --keep-intermediates --outdir paper/generated paper/main.tex
cp paper/generated/main.pdf paper/generated/paper.pdf
make arxiv -o paper/generated/paper.pdf
```

Generate `figures` and `numbers` first. The final command packages that existing
PDF's source inputs without invoking the Make PDF recipe.

## Direct analysis commands

Inspect the CLI options:

```bash
uv run cafetomo --help
uv run cafetomo reconstruct --help
uv run cafetomo --fast analyze --help
uv run python -m cafetomo.arraydepth --help
```

Run the conditional study against existing full-run products:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m cafetomo.arraydepth \
  --config configs/cafeteria.yaml \
  --pose results/pose.json \
  --opacity runs/opacity \
  --ingest runs/ingest \
  --beams results/beams.json \
  --out results/arraydepth.json \
  --replicas 50
```

When using direct commands, supply the fitted `--pose` after self-calibration.
The Make recipes already supply it.

## Verification

```bash
make test
uv run pytest -m slow
```

`make test` runs lint/format checks, vulture, fast Python and browser tests, and
Node viewer tests. The slow suite includes phantom recoveries and independent
noiseless ray-box checks for the conditional depth model.

Run selected checks:

```bash
uv run pytest -q tests/test_array_depth_study.py
uv run pytest -m browser
node --test viewer/test/*.test.mjs
```

## Outputs and cleanup

Tracked numerical results live in `results/`. Generated analysis products,
bootstrap arrays and viewer files live in `runs/`; paper build outputs live in
`paper/generated/`. These generated directories are not committed.

```bash
make clean
```

This removes `runs/` and `paper/generated/`, including caches and local diagnostic
runs; it leaves the tracked result JSON files in place.

## Citation and license

See `CITATION.cff`. MIT license; see `LICENSE`.
