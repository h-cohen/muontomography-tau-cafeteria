# Muon tomography of the TAU cafeteria ceiling

Code, data and manuscript for *A Full-Room Voxel Model from Only Two Muon Detectors*.
Two exposures of one four-plane scintillator tracker under the cafeteria ceiling,
normalised by an open-sky run of the same detector, yield a full-room voxel model,
the height and pitch of the ceiling beams, and their vertical depth.

The second detector position is self-calibrated from the data; a single external
length sets the absolute scale. See the paper for the treatment and its uncertainty.

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), GNU make, a LaTeX distribution with
latexmk and bibtex (siunitx, natbib, hyperref), and Node >= 24 (viewer unit tests only).

    uv sync
    make all          # results/*.json, paper/generated/paper.pdf, runs/viewer.html

`make all FAST=1` runs the same chain with cheap settings into `runs/fast/` (never
touching `results/` or `paper/generated/`); the committed `results/` come from the full run.

| Stage | Command | Output |
|---|---|---|
| Ingest ROOT histograms | `make ingest` | `runs/ingest/` |
| Self-calibrate the second position | `make selfcal` | `results/pose.json` |
| Opacity and weights | `make opacity` | `runs/opacity/` |
| Voxel reconstruction | `make reconstruct` | `runs/voxels/`, `results/reconstruction.json` |
| Autofocus, beams, beam depth | `make analysis` | `results/{autofocus,beams,beamdepth}.json` |
| Phantom validation | `make validation` | `results/validation.json` |
| Bootstrap and systematics | `make uncertainty` | `results/uncertainty.json` |
| Paper figures | `make figures` | `paper/generated/figures/*.pdf` |
| Paper | `make paper` | `paper/generated/paper.pdf` |
| Interactive viewer | `make viewer` | `runs/viewer.html` (self-contained) |

## Supplementary viewer

`make viewer` builds `runs/viewer.html`, a single self-contained WebGL page with the data
embedded. It shows the reconstructed volume with the fitted ceiling beams drawn as
wireframe boxes (toggleable) and a legend giving the beam depth h with its uncertainty.

## Layout

`src/cafetomo/` analysis package, `configs/cafeteria.yaml` every setting,
`data/` the input histograms, `results/` every number the paper quotes,
`paper/` manuscript and figure scripts, `viewer/` WebGL viewer, `tests/`.

## Tests

    make test                       # lint, dead-code check, pytest (incl. browser), viewer unit tests
    uv run playwright install chromium   # once, for the browser tests
    uv run pytest -m slow           # phantom recoveries on the real geometry (minutes)

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
