# Muon tomography of the TAU cafeteria ceiling

[Code and data repository](https://github.com/h-cohen/muontomography-tau-cafeteria).

Code, data and manuscript for *Three-Dimensional Muon Imaging of a Building Ceiling from Two Detector Positions*.
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
  `siunitx`, `natbib`, `hyperref`, `booktabs`, `geometry`, `float` and `balance`.

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

The viewer displays the reconstructed room and toggleable fitted-beam bodies and outlines.
Use **Fitted beams** to show/hide the overlay, **beam opacity** to fade it
(0% hides both faces and edges), and **beam color** to choose its color. These controls are
independent of voxel opacity, thresholds and the fitted dimensions. The PNG
export includes the currently displayed overlays; uncheck **Fitted beams** or
set beam opacity to 0% for a voxel-only room image.

**Beam fit** selects the geometry shown. When available, the default is the
**Conditional continued array (z depth)**: its vertical extent is
`ztop - zbottom = h`, currently 1.45 m. The legend marks it conditional and labels
the 0.11 m spread as counting variation under fixed array/pose assumptions.
**Matched-beam baseline** remains selectable: its roughly 0.067 m vertical extent
is a bound optimizer result, labelled physically unresolved. Older exports with
only one fit still work.

Depth always means **z-axis thickness**, not the long y-axis span or transverse
x width. The earlier flat overlay displayed the unresolved baseline, not the
1.45 m conditional estimate. Switching fits changes only the displayed boxes;
it does not stretch the voxel field or rerun the inference. The color and opacity
controls affect filled faces and outlines in either model. For presentation, the
conditional overlay hides its leftmost and rightmost boxes, and displayed boxes
use 80% of the fitted y span, centred on the same midpoint. These display crops
do not alter the exported fits or their vertical z thickness.

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

Every paper quantity is generated from `results/*.json`. The author-provided 3D room figure is retained as
`paper/figures/voxels.jpeg` (copied from the supplied generated figure). Replace
that source image and run `make paper` to update the two-column manuscript. The archive includes the figure when present, generated figures,
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

## How beam depth is estimated

### Data and geometric information

Depth fitting uses the measured angular opacity maps, not the rendered voxel
volume. For each world-direction bin, opacity is `lambda = -ln(n_pos / n_expected)`,
where the expected open-sky counts include the live-time ratio. The fitted pose
places both exposures in the same metre-based coordinates. Usable rows within
the configured transverse-analysis band enter a weighted least-squares fit.

A beam has shared bottom height `z0`, width `w` and vertical depth `h`, extending
along y. A deeper beam changes shadow shape when viewed obliquely; schematically,
its transverse extent includes a contribution `h * abs(tan(theta_x))` in addition
to width. Two views sample different angles, but width, depth, opacity amplitude
and background can still compensate for one another.

For each ray, paths through the box are calculated from its intersections with
the bottom, top and side faces. The detector aperture is not a point: only
midpoint positions admitted by the coincidence geometry are averaged. In local
detector axes their widths are `W - D * abs(tan(theta))`, rotated into the world
frame by the fitted azimuth.

### Flexible matched-beam fit

`make analysis` produces `results/beamdepth.json`. The model fits bottom, width,
depth and each matched beam centre, with an independent nonnegative opacity
coefficient per beam and a polynomial background per position. Opacity and
background coefficients are projected out at each geometric step. The transverse
mean path is integrated analytically where the complete footprint lies inside
the finite beam extent; other crossings use quadrature.

Multiple starts explore a common parameter domain. The current nominal fit
reaches a shallow-depth bound. Its raw `h` and box outlines are optimizer
diagnostics; `h_measurement` and `zbottom_measurement` are NaN and
`depth_resolved` is false. `make depthcheck` repeats the fit with analytic
count weights, while fixed-depth nuisance refits expose the thin-layer plateau.
The bootstrap distribution of these raw parameters is not a physical-depth
confidence interval.

### Conditional continued-array fit

`make arraydepth` produces `results/arraydepth.json`. It continues the inferred
beam spacing with two additional centres at each end of the matched array.
Relative centres and detector pose are fixed; a common transverse shift, bottom,
width, depth and independent nonnegative normal-column opacities remain free.
Width starts at the rough 0.3 m estimate and is fitted, not fixed to that value.
Background coefficients are projected out under the configured polynomial model.

Here the aperture average is taken in transmitted flux before the logarithm:

```text
lambda_i = background_i - log(mean_A(exp(-sum_k((q_k / h) * path_ik(A)))))
```

`q_k` is a fitted normal-incidence opacity, not a pinned concrete density.
Background is assumed constant across the aperture for a given direction.
The current quadrature uses 32 transverse samples and 3 samples on the second
projected aperture coordinate. Two identical depth-start rules are used for
nominal, bootstrap and numerical recovery checks. The JSON records bounds,
quadrature dimensions, fitted centres, column opacities and boundary flags.

The current conditional result is depth **1.45 m**, width **0.486 m** and bottom
**6.178 m in tracker coordinates**. Their count spreads are **0.11 m**,
**0.015 m** and **0.091 m**, respectively. All 50 replicas resample both room
exposures and their shared roof reference, retaining nominal rows and weights.
Every replica is kept; failed fits stop the run. No current depth replica reaches
its bound. Independent 3D ray-box tests check noiseless numerical recovery.

These spreads hold pose, relative centres and array completeness fixed. They
exclude uncertainty from those assumptions, actual cross-section, background
misspecification and angular-bin modelling. The residual statistic is about
2.36 per nominal degree of freedom, so this is an exploratory conditional
estimate. Changing array/background/centre assumptions shifts depth much more
than aperture refinement or flux-versus-mean-path averaging. The on-site 1.2 m
is an external comparison, never a fitting target; comparing the bottom with
7.3 m also requires a registered tracker-to-floor height origin.

### Why the voxel beam looks thicker

The voxel volume solves a different inverse problem: a distributed, nonnegative
field of effective opacity. With only two positions, many vertical distributions
can explain similar projections. Finite angular/aperture resolution, the limited
view geometry and reconstruction choices can distribute beam-related signal
across several z layers. The room field also contains background structures;
voxel opacity and threshold controls alter which of those layers are visible.
Vertical TV smoothing is disabled in this campaign, but that does not remove
the geometric null space or the other reconstruction ambiguities.

The saved cross-check in `beamdepth.json` averages beam columns minus
between-beam columns over a y band, then measures the half-maximum crossings of
that contrast profile. Its current FWHM is **2.30 m**, from about **6.17 m** to
**8.47 m**. This is a reconstruction diagnostic, not the fitted physical depth.
The analytic feature-height resolution scale is about **0.545 m**; it is not a
measured point-spread kernel and cannot simply be subtracted from the FWHM to
recover beam thickness. Crossings outside the solve grid are reported as NaN.

Consequently, counting visible z voxels and multiplying by the 0.2 m spacing
measures a display-dependent extent. It does not replace the angular-data box
fit or establish that the beam is physically as thick as the visible volume.

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

Internal validation runs are isolated from the paper pipeline:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run cafetomo internal-validate \
  --config configs/cafeteria.yaml --ingest runs/ingest \
  --settings configs/internal-validation.yaml \
  --out runs/internal-validation/pilot-001 \
  --cache runs/internal-validation/scientific-cache
```

The settings explicitly select `pilot`, `full`, or `smoke`. Global `--fast`
forces smoke scope. Pilot settings are declared before fitting: three count
splits, five guarded angular folds, inner selection over nine TV/damping choices,
two direct controls and five count realizations for each of two synthetic scenes.
Inner candidates share one training-only nuisance fit; the selected model refits
pose on all outer training counts. Angular inner/outer folds retain the complete
roof calibration and its acquisition time. Count folds thin the shared roof once.
The first fold of each protocol has two conditional prediction replicas; other
folds report descriptive predictions. Five refitted-pose count replicas and
one-factor reconstruction sensitivities use the first count training partition.

All new artifacts live under the supplied output root. Completed JSON summaries
appear in `results/` only after all stages succeed; `manifest.json` records
incomplete runs and failures. Scientific completed-training cache keys include
counts, masks, exposure, configuration and package source, with payload checksums.
The local cache is trusted generated data, not an interchange format. Use a new
output root for each attempt and reuse the scientific cache to avoid repeated
completed training fits. Full mode requires a written `production_budget` in the
settings before outcomes. These checks remain retrospective and conditional;
persistence and geometric support do not measure real-world accuracy.
