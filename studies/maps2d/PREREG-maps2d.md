# Preregistration: stripe measurements on the 2D opacity maps

Registered 2026-10-28, before the estimator below has been run on any data. Not blind:
the smoothed maps of pos0 and pos1 against the joint reference (production Fig. "2d-maps",
flag-only hit selection) have been seen; the beams are stripes along tan θ_x, the
transverse member a stripe along tan θ_y near tan θ_x ≈ +0.4, and the beam-level analysis of
the 3D section (`results/beams.json`: period 0.232, parallax height 6.85 m) is known.
The band and window choices below were fixed from those maps, not from any
measured stripe position, width or contrast. Outcomes are reported whatever they are.

Axes are the detector's (production paper naming): beams along X, pos1 displaced mainly
along Y. Code works in the analysis frame (beams along y); the study converts.

## Question

What do the 2D angular maps measure directly, per position, before any 3D model, and what
does the pedestal-based neighbour selection (`raw.neighbour_sigmas` = 3) change in them
relative to the DAQ's hit flag alone?

## Inputs (frozen)

- Counts: the preregistered primary joint reference (m ≤ 3, φ = 0;
  `studies/sky-orientation`) as built by `make sky-joint`, i.e.
  `<SKYRUNS>/joint/ingest_m3_phi0/counts_{pos0,pos1,SKY}.npz`, for two hit selections:
  flag-only (`runs/repro2/skyswap`, commit 658cb32) and neighbour selection
  (`runs/sel3/skyswap`, commit 423af08 or later with unchanged `src/`).
- View: `configs/cafeteria.yaml` `view` (0.025-tan bins, count floors 150 / 100), opacity
  per bin and σ² = 1/n_pos + 1/N_ref as in `beamview.opacity_view`. No smoothing enters
  any measurement.

## Estimators

1. **Beam profile** λ_b(t_y): weighted mean (1/σ²) over |tan θ_x| ≤ 0.25 of each
   tan θ_y column (excludes the transverse member), kept where ≥ 5 bins are measured and
   |tan θ_y| ≤ 0.65.
2. **Transverse profile** λ_T(t_x): the same over |tan θ_y| ≤ 0.25, |tan θ_x| ≤ 0.65.
3. **Stripes:** local maxima of a profile with prominence ≥ 2 σ of the profile point.
   Each is fitted, by weighted least squares on the unsmoothed profile within ±0.10 tan
   of the maximum, with a Gaussian on a linear background:
   centre t_k, FWHM w_k, amplitude a_k (the contrast Δλ above the local background).
   A fit whose centre leaves its window or whose width hits 0.01 / 0.30 is kept and flagged.
4. **Matching across positions:** pos1's beam profile is shifted by s and correlated with
   pos0's (Pearson, on common points); the shift s* maximizing it in
   s ∈ [0.15, 0.40] (heights 4.7–12.6 m for a 1.9 m baseline: a range wide enough to
   contain every candidate, and excluding the one-period aliases) pairs stripes k of
   pos0 with the pos1 stripe nearest t_k − s*. Paired shifts Δt_k = t_k(pos0) − t_k(pos1).
5. **Summary:** per position the mean period (mean spacing of consecutive matched
   stripes), mean FWHM, mean contrast and contrast over its error; across positions the mean
   and scatter of Δt_k and the transverse-member shift Δt_T.

## Uncertainties

200 Poisson replicas of the position counts and of the reference counts (N_ref as a
Poisson mean), seed `SeedSequence(cfg.uncertainty.seed, spawn_key=(3, selection))`; each
quantity's σ is its replica standard deviation, the stripe list held at its nominal
matching. Statistical only. The two selections share their events, so their difference
is not given a significance: it is reported as the two values with their own errors.

## Reporting

`studies/maps2d/maps2d.py` writes `maps2d.json` per selection and a comparison; promotion
to `results/production/maps2d.json` (macros `Maps2d...`) only after review. Figure: beam
profiles of both positions (points with errors, fitted stripes), pos1 shifted by s*; a
panel with the transverse profiles.
