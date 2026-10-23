# Preregistration: joint open-sky reference from both roof orientations

Registered 2026-10-23, before any fit of this model. Not blind: the SKY_B − SKY_A dipole
(±0.025 in λ at |t| > 0.5, `skyswap.json`, `overview.py`) was already seen. Outcomes are
reported whatever they are; failures and censored results are kept.

## Model
Counts per fine detector bin d (500×500 ingest grid, |t| inside acceptance):
  E_A(d) = T_A · R(d) · S(w_A(d)),   E_B(d) = T_B · c · R(d) · S(w_B(d))
- R(d): detector response, free per bin (profiled out exactly).
- S(w) = exp s(w), sky azimuthal structure in the world frame;
  s(w) = Σ_{m=1..3} τ^m [a_m(τ) cos mψ + b_m(τ) sin mψ], a_m, b_m quadratic in τ
  (τ = |w| in tangent, ψ its azimuth): 18 parameters. m = 0 is degenerate with R; m = 4
  cancels under a 90° turn and is not fitted.
- c: free epoch rate ratio (pressure, ageing); expected ≈ 1.
- w_A = d (roof A taken as the indoor detector orientation); w_B = rotation of d by
  `turn_deg` = −90° (clockwise from above, confirmed by the user).
Profiling R gives n_A ~ Binomial(n_A + n_B, p), logit p = ln(T_A/(c T_B)) + s(w_A) − s(w_B).
Fit by maximum likelihood (exact binomial on fine bins).

## Reference for a room position p at world azimuth φ_p
  N_ref,p(d) = (n_A + n_B) · (T_A + T_B) S(w_p(d)) / (T_A S(w_A) + c T_B S(w_B)),
live time T_A + T_B. It reduces to the whole roof run when s = 0, c = 1.

## Fixed choices
Harmonics m = 1..3, quadratic radial coefficients, fine bins with n_A + n_B ≥ 1, the
same acceptance as `opacity`. No change after seeing the fit.

## Checks (pass criteria)
1. Goodness of fit: binomial deviance / dof on 0.025-tan view bins within 1 ± 3√(2/dof).
2. Detector-frame residual structure (gain-change test): pulls of n_A vs fit projected
   on tan θ_x and tan θ_y in view bins; any projection with |mean pull| > 3/√(bins per
   row) in more than 5% of rows fails.
3. c − 1 reported with its error; |c − 1| > 0.02 is flagged (rate drift between epochs).
4. The fitted s must reproduce the SKY_B − SKY_A sky-ratio dipole (residual map shown).

## Systematics (shifts, not combined with statistics)
- Orientation of roof A relative to the indoor detector: φ_p ∈ {0°, 180°} primary
  (aligned up to sign, the user's assumption); {+90°, −90°} reported as the alternative
  that roof B, not A, matched the indoor orientation. The spread of λ_p across these is
  the orientation shift.
- Model: the λ shift between m ≤ 2 and m ≤ 3 fits. m = 4 is not identifiable from a 90°
  turn and is bounded only by check 1.
- Reference choice: λ against SKY_A, SKY_B and the whole run, for comparison.

## Outputs
`runs/skyjoint/` (gitignored): fit json, synthetic ingest dirs per φ hypothesis with
`counts_SKY.npz` = N_ref, then `cafetomo beams2d` on each; summary figure. Script:
`studies/sky-orientation/joint_sky.py`.
