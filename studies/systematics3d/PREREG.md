# 3D systematics, stability and depth controls (registered 2026-10-30, before any variant ran)

Production 3D baseline: `runs/joint3d-2026-10-29` (joint open-sky reference m3, gain-corrected
raw counts, pedestal-based hits), pose `results/production/3d/pose.json`, config
`runs/joint3d-2026-10-29/config.yaml`. Source frozen at the commit that adds this file.
Single-threaded BLAS; outputs in a fresh `runs/systematics3d-<date>/`; nothing is tuned.

## Estimators (unchanged code)

`measure()` with the volume: autofocus height `autofocus_z`, triangulated `beams_z`, `beams_zx`,
`beams_pitch`, voxel profile `zprofile_{fwhm,bottom,top}`; then the continued-array fit
(`arraydepth.fit_array_depth`, triangulated centres of the same variant continued by their median
spacing, height seed its `beams_z`): `h`, `w`, `bottom`, `chisq_per_dof`. Pose fixed at the nominal
fit throughout (pose is its own systematic). Every variant changes one input. The nominal is
recomputed by the same script and must reproduce `results/production/3d` (autofocus, beams,
arraydepth nominal) to 1e-6 m, else the study stops.

Shift of a variant = variant − recomputed nominal. A family's systematic = the largest |shift| over
its variants (an envelope, not a standard deviation), entered in quadrature in the budget.

## Item 6 — open-sky reference (family `sky`)

The four alternatives of the production sky table, from the existing `sky-joint` ingests
(`runs/gain1/skyswap/{joint/ingest_m3_phi180, joint/ingest_m3_phiB, joint/ingest_m2_phi0,
ingest_SKY}`): roof A half-turned; roof B sharing the room axes; harmonics m ≤ 2; plain sum of both
roof parts. Opacity rebuilt from each ingest at the nominal pose; rows and weights nominal.

## Item 7 — time stability of the room runs (family `stability`)

Halves: pos0 day files 1–5 vs 6–10; pos1 (one file, 5 clock segments) split at the segment
boundary closest to equal live time. Each half is counted with the whole exposure's channel gains
(only time differs); pedestals for the hit selection are fitted per half, as each production
source fits its own; the reference is the nominal joint sky. Two tests:

1. Shape: early vs late angular shape scatter per position on 0.05-tan bins (`rawcheck.epoch_scatter`)
   against its counting expectation.
2. Results: every estimator on the early-half and late-half datasets (each against the full sky).
   The difference Δ is compared with σ_Δ = 2 σ_stat (each half has half the counts, so its
   statistical spread is √2 σ_stat; σ_stat from `uncertainty.json`/`arraydepth.json`).

Decision rule: if |Δ| > 2 σ_Δ for a quantity, |Δ|/2 enters its budget as "time stability";
otherwise the test is reported and no row is added.

## Item 8 — reconstruction choices (family `reconstruction`)

TV strength α × {1/2, 2} (0.015, 0.06); coverage damping μ × {1/2, 2} (0.025, 0.1); voxel spacing
{0.15, 0.25} m; iterate rule: budget k*/2 = 15 iterations (the best-χ² iterate of a shorter run,
i.e. stopping early). Reported: shifts of `autofocus_z` and of the voxel profile, and for the
same-grid variants the Pearson correlation of the volume with the nominal over voxels seen by both
positions. The triangulated heights, pitch and array depth do not use the volume (shift 0 by
construction; computed anyway as a check).

## Item 9 — goodness of fit

The continued-array statistical spreads are scaled by S = sqrt(χ²/ν) of the nominal fit
(PDG scale factor) when χ²/ν > 1. The triangulation already scales its covariance by the residual.

## Item 10 — depth systematics and controls (continued array)

Systematic families on h, w, bottom (in addition to `sky`, `stability`, item 9):

- `centres`: relative centres rescaled about their mean by the pitch ± its statistical spread.
- `completeness`: centres continued by 1 and by 3 beams per end (nominal 2).
- `pose`: opacity rebuilt at each pose coordinate ± its spread (6 refits; envelope).
- `background`: background degree 1 and 3 (nominal 2).

Controls, on simulated band opacities λ = beam model + nominal fitted background + Gaussian noise
of the per-row σ (seed `SeedSequence(0, spawn_key=(7,))`), 20 replicas each, nominal centres, width,
columns and centre height z_c = bottom + h/2 of the nominal fit:

- recovery: true h ∈ {0.6, 1.2, 1.8} m (bottom = z_c − h/2); report mean and spread of fitted h.
- null: no beams (all columns 0); report the fitted h, columns and how often the fit reaches a bound.
- misspecified slab: true h = 1.2 m plus a uniform slab term 0.1·sqrt(1+sx²+sy²) not in the model.
- misspecified array: true h = 1.2 m with the continued (outer) beams absent (columns 0).

No outcome is a pass/fail gate on the reported depth; controls are reported as found. The depth is
quoted as a measurement only with its full budget; if the recovery bias at the nominal h exceeds
the total uncertainty, the paper says so.
