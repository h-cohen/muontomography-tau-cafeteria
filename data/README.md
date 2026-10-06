# Data

Pre-binned DAQ histograms of one four-plane scintillator tracker (4 planes x 23 bars).
No event-level data is included or needed.

| File | Role | md5 |
|---|---|---|
| `HistsOutDataCafePos0.root` | cafeteria, position pos0 (frame origin) | dfc65118fd27977251c9a3ed29fb30b5 |
| `HistsOutDataCafePos1.root` | cafeteria, position pos1 | c7cb1a6bb4bb3eb17a90773cea0aa48a |
| `HistsOutSkyRoofRuns37-77.root` | open sky on the roof, same detector | 2e58fc1ecd8cc78d907299765701db90 |

Totals (sum of `txty`; live time = sum of `dT` bin centre x count, in seconds):

| File | tracks in `txty` | live time (s) |
|---|---|---|
| `HistsOutDataCafePos0.root` | 6.33e6 | 8.75e5 |
| `HistsOutDataCafePos1.root` | 2.41e6 | 3.34e5 |
| `HistsOutSkyRoofRuns37-77.root` | 2.85e7 | 3.58e6 |

Histograms used:
- `txty` — TH2, 800 x 800 bins over tan(theta_x), tan(theta_y) in [-2, 2]; axis 0 = tan(theta_x).
- `dT` — TH1 of time between consecutive tracks; the sum of bin centre x count is the live time, in seconds (the axis is unlabelled in the file; unit inferred from the ~7 Hz track rate in all three files: 7.2, 7.2, 8.0).

The analysis crops `txty` to |tan| <= 1.25 (500 x 500 bins of 0.005) and rebins by 10.
