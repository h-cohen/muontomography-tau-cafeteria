# Muon tomography of the TAU cafeteria ceiling

[Code and data repository](https://github.com/h-cohen/muontomography-tau-cafeteria).

Code, data and manuscript for *Seeing a Ceiling with Cosmic Muons:
Three-Dimensional Imaging from Two Views*.
Two positions of one four-plane tracker, normalized by an open-sky run, provide a
three-dimensional room reconstruction. Ceiling-height and pitch estimates,
conditional beam-depth fits, and internal validation support that main result.

The detector separation is self-calibrated; its scale remains conditional on the
floor-offset prior and fitting objective. On-site bottom/depth measurements
(7.3 m / 1.2 m) are comparison inputs, not fitting targets. The continued-array
depth estimate fixes relative beam centres and array completeness; its count
spread excludes uncertainty in those assumptions.

## Current status

The retrospective internal-validation pilot and its reviewed paper are complete.
The larger scientific validation program remains open: physical beam depth,
calibrated uncertainty, and voxel-solver convergence have not been established.

- Held-out count Poisson deviance is 59.71% below a uniform slab; guarded angular
  deviance is 5.41% below it overall, with two of five angular folds worse.
  These reused-data checks are descriptive, not independent significance tests.
- Synthetic recovery retains thin-scene boundary fits (four of five), extra
  peaks (seven of ten noisy cases), and substantial vertical centroid bias.
- A separately registered weight correction removes a constant-bootstrap
  roundoff defect. The original pilot is retained unchanged.
- Ten noisy recovery replays and two direct controls reproduce original outcomes
  within registered tolerances, with saved volumes and solver diagnostics.
  All 48 support selections reproduce with background diagnostics retained.
- Latest verification: 465 Python and 118 viewer tests passed; 22 slow tests
  were deselected from the default suite. Registered numerical campaigns ran
  separately. The six-page, double-column paper and source archive compiled
  with identical extracted text.

Local agent handoff: `docs/dev/handoff-2026-10-14.md` (intentionally untracked).
It maps the completed work, preserved reviews, numerical evidence, and next tasks.

## Setup

Run commands from the repository root. Requirements:

- Python >= 3.12, managed by [uv](https://docs.astral.sh/uv/).
- GNU make >= 4.3.
- Node >= 24 for viewer unit tests.
- For full `make paper`/`make arxiv`/`make all`: Tectonic at `runs/tools/tectonic`
  (or set `TECTONIC` to its executable path) and the reviewed pilot,
  correction, nominal volume, and baseline inputs described below.

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

Make resolves upstream dependencies automatically. The analysis commands above
include the 50-replica reconstruction bootstrap and a separate 50-replica
conditional depth study. Up-to-date stages are reused; changes to analysis code,
dependencies or configuration rebuild affected stages. `make all` builds the
viewer and then the guarded publication; internal validation is a separate
registered campaign and is not launched by `make all`.

To force a fresh full analysis without replacing the reviewed products, choose
an unused output directory:

```bash
make -B validation uncertainty arraydepth viewer figures numbers \
  RUNS=runs/rebuild-YYYY-MM-DD \
  RESULTS=runs/rebuild-YYYY-MM-DD/results \
  GEN=runs/rebuild-YYYY-MM-DD/generated
```

Replace `YYYY-MM-DD` with a unique run label. This does not promote new paper
measurements. Publication uses explicitly reviewed archives and baseline hashes;
changed baseline results require a fresh reviewed selection before compilation.

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

To rebuild the viewer of the published baseline (the nominal full-data volume in
the paper) without refitting anything, export the archived volume with the
canonical `results/` into a fresh directory, then build the HTML:

```bash
uv run cafetomo export --config configs/cafeteria.yaml --pose results/pose.json \
  --voxels runs/response-aware-2026-10-07/voxels \
  --bootstrap runs/response-aware-2026-10-07/bootstrap \
  --results results --out runs/viewer-baseline/export
uv run cafetomo viewer --export runs/viewer-baseline/export --out runs/viewer.html
xdg-open runs/viewer.html
```

Choose a new `--out` directory rather than writing into the archived
`runs/response-aware-2026-10-07/export/`.

The viewer opens with the clip box set by `volume.viewer_clip_m` in
`configs/cafeteria.yaml` (x, y, z in world metres; display-only, read by no
analysis): the ceiling region over both positions, clip fractions
x 0.32–0.68, y 0.31–0.67, z 0.54–0.90 on the baseline grid. Re-export to apply a
change. Without it, the x/y `viewer_crop_xy_m` box is used. Note that
`viewer_crop_xy_m` also sets the beam parallax-scan range, so it is not a pure
display setting.

The viewer displays the reconstructed room and toggleable fitted-beam bodies and outlines.
Overlays sit in cards under **Camera & view**. Use **Fitted beams** to show/hide
the overlay, its **opacity** slider to fade it (0% hides both faces and edges),
and its round color swatch to choose its color (defaults: gray, 50%). These controls are
independent of voxel opacity, thresholds and the fitted dimensions. The PNG
export includes the currently displayed overlays and the legend panel (colour
bar and every visible legend line); uncheck **Fitted beams** or set the beam
opacity to 0% for a voxel-only room image.

**smoothing** (Transfer section, **Voxel cubes** at cube size 1× only) is a
display-only interpolation of the solved voxels; it adds no information and
never changes the reconstruction (the 0.2 m grid is a grid, not a resolution:
the banner reports a vertical depth resolution of about 0.27 m).

- **Off** (default) draws each solved voxel as a block, exactly as before.
- **2×**, **4×** and **8×** sample the trilinearly interpolated density on a
  0.1 m, 0.05 m or 0.025 m sub-grid and draw the samples at or above
  **cube threshold** as cubes of that size, with the same face shading, rim and
  opacity as Off.
- **Isosurface** draws the smooth surface of the interpolated field at
  **cube threshold**, shaded by its gradient like the fog and coloured by the
  interpolated density just inside it.

Every sub-sample or surface point belongs to one solved parent voxel and is
hidden exactly when that voxel is: the sigma, coverage and SNR gates, the clip
box, the clip plane and the slice decide per solved voxel, and gate fields are
never interpolated. At cube size > 1 the control is disabled, because merged
cubes bake their gates into block means and would have no single solved
parent. Fast preview while interacting drops to Off and the idle render restores
the chosen mode. Hover still reports the solved voxel and its solved value. The
legend (and the PNG) adds a line such as `display: trilinear 4× (solved 0.2 m)`.

**Fields of view** (off by default) draws each detector position's geometric
acceptance as a translucent square frustum rising to the top of the volume.
A track is accepted when it crosses both outer planes of the active square
(width W, layer gap D), so the frustum starts as the W×W detector square and
widens by W/D per metre of height (|tan θ| ≤ W/D per detector axis, ≈ 0.91);
each frustum is rotated by its fitted azimuth. **show** selects **Separate views**
(every frustum) or **Intersection only** (the region all positions see,
outlined by horizontal contour rings up to the shared ceiling footprint; offered
only when the views overlap). **Shade by acceptance** (on by default) fills the
region with a glow whose brightness follows the tracker's relative acceptance
per solid angle for the direction from each detector centre, the same formula
as `Detector.acceptance`: (1 − |tx|/T)(1 − |ty|/T)(1 + tx² + ty²)^−1.5 with
T = W/D, normalised to 1 at the zenith. It is brightest straight above each
detector and falls to zero at the acceptance edge. Separate views add their
weights; the intersection takes their geometric mean, which vanishes wherever any
view is blind. Unchecked, separate views draw flat translucent walls and the
intersection a uniform fill. **opacity** defaults to 60%, which draws the
overlay as designed; lower values fade it (0% hides it) and 100% saturates it to
a solid fill. The color swatch recolors it. The
frusta come from `detector_geometry` in `meta.json`; exports made before that
field existed hide the control. Like the beams, this is display-only context.

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

FAST uses reduced solver/replica settings and writes into `runs/fast/`, including
`runs/fast/results/` and `runs/fast/viewer.html`. It can also render the baseline
figures and numbers under `runs/fast/generated/`, but it does not generate the
reviewed internal publication or its diagnostic scalars. `FAST=1 paper` fails
with a scope reminder; `FAST=1 arxiv` can package an already compiled FAST PDF
for the archive regression fixture. The normal FAST smoke target is
analysis/viewer only. No FAST target writes full-run `results/` or
`paper/generated/`.

Open the smoke viewer:

```bash
xdg-open runs/fast/viewer.html
```

Or use the same server above and open
[http://localhost:8000/fast/viewer.html](http://localhost:8000/fast/viewer.html).

## Paper and submission archive

```bash
make paper TECTONIC=runs/tools/tectonic
make arxiv TECTONIC=runs/tools/tectonic
```

The reviewed paper uses baseline numbers from `results/*.json` and separate
internal diagnostics from `results/internalvalidation.json`. Its source archive
contains the selected PDF panels, number macros, manuscript sections and the
formatted bibliography. The author-provided `paper/figures/voxels.jpeg` is a
selected, hashed input for the separately labelled conditional illustration.

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

## Internal validation execution

Internal validation runs are isolated from the paper pipeline:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run cafetomo internal-validate \
  --config configs/cafeteria.yaml --ingest runs/ingest \
  --settings configs/internal-validation.yaml \
  --out runs/internal-validation/pilot-001 \
  --cache runs/internal-validation/scientific-cache
```

Or, after generating `runs/ingest/`, use the Make wrapper with a fresh output root:

```bash
make internal-validate IVOUT=runs/internal-validation/pilot-NEW \
  IVSETTINGS=configs/internal-validation.yaml
```

The wrapper uses `runs/ingest/`. The completed reviewed campaign instead used
`runs/response-aware-2026-10-07/ingest/`; preserve that distinction when
reproducing its evidence. Inspect options with `uv run cafetomo internal-validate --help`.

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
counts, masks, exposure, configuration, package source, and count-source order
(which controls Poisson draw order), with payload checksums.
The local cache is trusted generated data, not an interchange format. Use a new
output root for each attempt and reuse the scientific cache to avoid repeated
completed training fits. Full mode requires a written `production_budget` in the
settings before outcomes. These checks remain retrospective and conditional;
persistence and geometric support do not measure real-world accuracy.

Angular point scores condition on the observed complete roof calibration. Their
representative Poisson uncertainty replicas include full-reference counting
variation: one complete roof-count draw is shared across all room views in each
replica, while the full roof acquisition time stays unchanged. Fold and replica
manifests record this scope explicitly. These intervals therefore differ from
uncertainty with a fixed observed roof. Every fold's input counts, assignments,
masks, live times and reference/nuisance scope are registered before any fold
fitting, including for failed or interrupted attempts. Excluded training bins are
sanitized in archives and accompanied by masks defining their missingness.

## Reviewed internal pilot in the paper

The selected publication panels use the completed original pilot and the
separately registered clipped-bootstrap weight correction. Their quantities
are promoted only to `results/internalvalidation.json`; baseline result JSON
files retain their original values. The nominal main image is explicitly
`runs/response-aware-2026-10-07/voxels/volume_full.npz`, with both-position
coverage from its `views.npy`. Training-fold support and persistence have their
own fitted grids and scope. The original `paper/figures/voxels.jpeg` remains an
unchanged, separately labelled conditional box illustration.

```bash
make internal-selected
make internal-paper TECTONIC=runs/tools/tectonic
make arxiv TECTONIC=runs/tools/tectonic
```

Normal `make paper` and `make arxiv` use this same reviewed publication path.
They require the completed original/corrective archives and the archived baseline
figures; a fresh checkout must restore or generate those reviewed inputs first.
These publication targets never run baseline analysis again. The selected summary
binds all 16 consumed source hashes and their selected roles, baseline JSON
hashes, and nominal reconstruction metadata. Every normal paper/archive build
checks those identities before copying panels or compiling, including when its
PDFs already exist. Changed or missing inputs require a new reviewed promotion.
`make all` completes the analysis/viewer before this guarded build. The default reviewed selection lives in
`runs/figure-review/selected-paper-publication-2026-10-14-fix1/`. To create a new selection,
set `IVFIGOUT` to a new directory and `IVSUMMARY` to a new JSON path outside
`results/`. Set `IVPAPERARCHIVE`, `IVCORRECTION`, and `IVNOMINAL` to the intended
source bundles, then run `make internal-selected` with those variables. Review
the new panels, source hashes and scalars before copying that JSON to
`results/internalvalidation.json`; then build with the same `IVFIGOUT` and source
arguments. Existing output directories are refused by the renderer. Selected
paths are checkout-relative where possible, so identical relocated archives
retain their identity; changed bytes still fail the guard.
The generated paper is `paper/generated/paper.pdf`; its self-contained LaTeX
source archive is `paper/generated/arxiv.tar.gz` (`arxiv-internal` is an alias).

The full original diagnostic set is under
`runs/figure-review/pilot-2026-10-13-a-2026-10-14/`: 16 per-view prediction
panels, all 48 declared support settings and eight fold-volume images, outcome
and recovery plots, and separately labelled persistence groups. The original
Poisson-weight sensitivity is retained as a numerical failure, while corrected
weight evidence is identified as a separate registered comparison. Neither
bundle supplies external geometry, calibrated physical depth, or posterior
confidence. Count and angular summaries remain separate descriptive checks
using repeated data; the pilot's negative vertical and detection findings are
reported in the manuscript.

The original pilot did not retain its ten synthetic voxel fields or solver
histories. The [registered recovery trace](runs/internal-validation/recovery-trace-2026-10-14/registration.json)
replays the same ten noisy cases and two known-geometry controls as a separate
supplement. Its [comparison](runs/internal-validation/recovery-trace-2026-10-14/comparison-adjudicated.json)
checks every original outcome field except elapsed time: all 12 agree within
the registered tolerances. The raw comparison retains 32 matching NaN pairs.
Each noisy case has simulated counts, input metadata, a fitted volume, row
weights, and solver diagnostics; each direct control has its fit inputs and
volume. These are fixed-budget 150-iteration reconstructions;
the solver history does not establish convergence. The original findings and
29 promoted paper scalars are unchanged.

The [support diagnostic supplement](runs/internal-validation/recovery-trace-2026-10-14/support-supplement/registration.json)
records polynomial coefficients, powers, excluded rows, and eligible rows for
all eight folds and six settings. Its corrected comparison reproduces all 48
original masks exactly. The first rebuild, which omitted angular observation
masks, is retained in `attempt-1/` with its correction note and amendment record.

## Citation and license

See `CITATION.cff`. MIT license; see `LICENSE`.
