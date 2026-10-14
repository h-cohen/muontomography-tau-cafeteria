// Detector fields of view: the region of the room each position can see.
// A track is accepted when it crosses both outer planes of the active square
// (width W, gap D), so per detector axis |tan| <= W/D, and the union of
// accepted lines through the square is a square frustum: half-width W/2 at
// the detector, growing by W/D per metre of height. Its footprint is rotated
// by the detector azimuth exactly as Pose.rotation does (world = R * local).
// Pure geometry only - app.mjs owns the GL buffers and the draw.

// Floats per vertex: world xyz, then an alpha weight in (0, 1] that the
// shader multiplies by the user's opacity. Weights shape the look: walls fade
// as they rise, edges stay crisp, the ceiling footprint stays readable.
export const FOV_STRIDE = 4;
const WALL_BASE = 0.28, WALL_TOP = 0.05;
const EDGE_BASE = 0.95, EDGE_TOP = 0.4, FOOTPRINT = 0.55;

function acceptanceTan(geometry) {
  const w = geometry && geometry.active_width_m, d = geometry && geometry.layer_gap_m;
  if (!(Number.isFinite(w) && Number.isFinite(d) && w > 0 && d > 0)) {
    throw new Error('detector_geometry needs a positive active_width_m and layer_gap_m');
  }
  return { half: w / 2, tan: w / d };
}

// fovGeometry(detectors, geometry, zTop) -> { faces, edges }
// detectors: meta.json `detectors` [{x, y, z, az_deg}]; geometry: meta.json
// `detector_geometry`; zTop: world height the frusta reach (the volume top).
// faces: 4 walls x 2 triangles per detector, for gl.TRIANGLES; edges: 4
// rising edges, the detector square and the footprint at zTop, for gl.LINES.
export function fovGeometry(detectors, geometry, zTop) {
  const { half, tan } = acceptanceTan(geometry);
  const faces = [], edges = [];
  detectors.forEach((det, k) => {
    const { x, y, z } = det, az = det.az_deg || 0;
    if (![x, y, z, az, zTop].every(Number.isFinite)) {
      throw new Error(`detector ${k}: pose and zTop must be finite`);
    }
    if (!(zTop > z)) throw new Error(`detector ${k}: zTop must lie above the detector`);
    const a = az * Math.PI / 180, co = Math.cos(a), si = Math.sin(a);
    const top = half + (zTop - z) * tan;
    // Corners counter-clockwise in the detector frame, rotated into the world.
    const ring = (h, zz, wgt) => [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([u, v]) =>
      [x + co * u * h - si * v * h, y + si * u * h + co * v * h, zz, wgt]);
    const lo = ring(half, z, WALL_BASE), hi = ring(top, zTop, WALL_TOP);
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      faces.push(...lo[i], ...lo[j], ...hi[j], ...lo[i], ...hi[j], ...hi[i]);
    }
    const loE = ring(half, z, EDGE_BASE), riseTop = ring(top, zTop, EDGE_TOP);
    const footprint = ring(top, zTop, FOOTPRINT);
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      edges.push(...loE[i], ...riseTop[i]);
      edges.push(...loE[i], ...loE[j]);
      edges.push(...footprint[i], ...footprint[j]);
    }
  });
  return { faces: new Float32Array(faces), edges: new Float32Array(edges) };
}

// The legend line: the acceptance edge the frusta are drawn at.
export function fovLegendText(geometry) {
  const { tan } = acceptanceTan(geometry);
  return `field of view: |tan θ| ≤ ${tan.toFixed(2)} per axis (W/D)`;
}
