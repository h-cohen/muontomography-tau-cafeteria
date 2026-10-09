# AGENTS.md — cafetomo

Paper-companion repository for two-position muon tomography of the TAU cafeteria.
Read README.md for execution commands, model definitions and scientific limits.
For ongoing development, consult docs/dev/plans/2026-10-07-status-summary.md when
available; local plans and execution notes are gitignored.

## Scientific interpretation

- Detector pose and separation are fitted. The rough 2.42 m estimate is not an
  exact baseline constraint. On-site bottom/depth values are independent
  comparisons in configs/paper-inputs.json, never optimization targets.
- Depth is vertical thickness: h = ztop - zbottom. Width is along x; beams extend
  along y. Keep detector reference coordinates distinct from floor heights.
- The flexible matched-beam model fits independent nonnegative opacity
  coefficients. A bound fit remains an optimizer diagnostic, not a physical
  depth measurement. Concrete pinning is an optional comparison.
- The continued-array model fixes relative centres and array completeness while
  fitting a common shift, bottom, width, depth and column opacities. Its bootstrap
  spread is conditional counting variation, excluding those structural assumptions.
- Voxel-profile extent is a reconstruction diagnostic, not physical beam depth.
  Display cropping, color and opacity change presentation rather than inference.
- Use the detector-admitted footprint. Preserve the distinction between averaging
  paths and averaging transmitted flux before taking its logarithm.
- Keep all bootstrap outcomes; report boundary censoring and fail on invalid fits.
  NaN represents an unmeasured quantity. Lengths are metres, detector inputs cm.

## Development and artifacts

- Keep paper quantities generated from results/*.json and figures scripted.
- Verify relevant changes with make test and the applicable slow numerical checks.
  Use FAST=1 for isolated smoke outputs, not paper measurements.
- Source is in src/cafetomo/; viewer code is in viewer/. Use existing interfaces.
- Keep runs/, paper/generated/, docs/dev/ and .superpowers/ out of commits.
- Git author: Hadar Cohen <hal.nls@gmail.com>. Use Conventional Commits without
  attribution trailers. Integration target is main; preserve existing work.
