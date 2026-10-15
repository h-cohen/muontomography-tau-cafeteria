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

// fovGain(opacity) -> alpha gain for the opacity slider. Weights draw as
// alpha = 1 - (1 - w)^gain and the acceptance field's extinction scales by
// gain, so the 60% default renders the weights as designed, 0% hides the
// overlay and 100% saturates it to a solid fill.
const DEFAULT_OPACITY = 0.6, MAX_GAIN = 1e4;
export function fovGain(opacity) {
  if (!(opacity > 0)) return 0;
  if (opacity >= 1) return MAX_GAIN;
  const odds = (o) => o / (1 - o);
  return Math.min(MAX_GAIN, odds(opacity) / odds(DEFAULT_OPACITY));
}

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

// acceptanceWeight(tx, ty, geometry) -> [0, 1]
// cafetomo.detector.Detector.acceptance, (W-|tx|D)(W-|ty|D)(1+tx^2+ty^2)^-1.5,
// divided by its zenith value W^2: the relative rate per solid angle of the
// direction (tx, ty) in the detector frame. The GLSL twin in app.mjs
// (FOV_FIELD_FRAGMENT_SRC) must stay in step.
export function acceptanceWeight(tx, ty, geometry) {
  const { tan } = acceptanceTan(geometry);
  const sx = Math.max(0, 1 - Math.abs(tx) / tan), sy = Math.max(0, 1 - Math.abs(ty) / tan);
  return sx * sy * (1 + tx * tx + ty * ty) ** -1.5;
}

// The world-aligned box enclosing every frustum: where the volumetric
// field pass marches.
export function fovFieldBounds(detectors, geometry, zTop) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, zTop];
  const { edges } = fovGeometry(detectors, geometry, zTop);
  for (let i = 0; i < edges.length; i += FOV_STRIDE) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], edges[i + k]);
      max[k] = Math.max(max[k], edges[i + k]);
    }
  }
  return { min, max };
}

// The frustum cross-section at height z: its square, counter-clockwise, in world xy.
function section(det, half, tan, z) {
  const a = (det.az_deg || 0) * Math.PI / 180, co = Math.cos(a), si = Math.sin(a);
  const h = half + (z - det.z) * tan;
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([u, v]) =>
    [det.x + co * u * h - si * v * h, det.y + si * u * h + co * v * h]);
}

// Sutherland-Hodgman: convex polygon clipped by a convex counter-clockwise one.
function clipConvex(subject, clip) {
  let out = subject;
  for (let i = 0; i < clip.length && out.length; i++) {
    const [ax, ay] = clip[i], [bx, by] = clip[(i + 1) % clip.length];
    const side = ([px, py]) => (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    const input = out;
    out = [];
    for (let j = 0; j < input.length; j++) {
      const p = input[j], q = input[(j + 1) % input.length];
      const sp = side(p), sq = side(q);
      if (sp >= 0) out.push(p);
      if ((sp >= 0) !== (sq >= 0)) {
        const t = sp / (sp - sq);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
  }
  return out;
}

function area(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    s += x0 * y1 - x1 * y0;
  }
  return Math.abs(s) / 2;
}

// fovIntersectionRings(detectors, geometry, zTop) -> Float32Array line list
// Outline of the region every position sees, as its horizontal cross-section
// at RING_LEVELS heights from where the views first overlap up to zTop (the
// top ring is the shared ceiling footprint). Empty when the views never meet
// below zTop.
const RING_LEVELS = 7, RING_LOW = 0.25;
export function fovIntersectionRings(detectors, geometry, zTop) {
  if (detectors.length < 2) throw new Error('an intersection needs at least two detector positions');
  const { half, tan } = acceptanceTan(geometry);
  fovGeometry(detectors, geometry, zTop); // validates every pose against zTop
  const common = (z) => detectors.slice(1).reduce(
    (poly, det) => clipConvex(poly, section(det, half, tan, z)), section(detectors[0], half, tan, z));
  const overlaps = (z) => area(common(z)) > 1e-9;
  if (!overlaps(zTop)) return new Float32Array(0);
  // Cross-sections only grow with height, so the first overlap is a bisection.
  let lo = Math.max(...detectors.map((d) => d.z)), hi = zTop;
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2;
    if (overlaps(mid)) hi = mid; else lo = mid;
  }
  const z0 = hi + 0.02 * (zTop - hi);
  const out = [];
  for (let k = 0; k < RING_LEVELS; k++) {
    const f = k / (RING_LEVELS - 1), z = z0 + f * (zTop - z0);
    const w = k === RING_LEVELS - 1 ? FOOTPRINT : RING_LOW + f * (EDGE_TOP - RING_LOW);
    const poly = common(z);
    poly.forEach((p, i) => {
      const q = poly[(i + 1) % poly.length];
      out.push(p[0], p[1], z, w, q[0], q[1], z, w);
    });
  }
  return new Float32Array(out);
}

// The legend line: the acceptance edge the frusta are drawn at, and how.
export function fovLegendText(geometry, { mode = 'separate', weighted = false } = {}) {
  const { tan } = acceptanceTan(geometry);
  const what = mode === 'intersection' ? 'shared field of view' : 'field of view';
  const shade = weighted ? ', shaded by acceptance' : '';
  return `${what}: |tan θ| ≤ ${tan.toFixed(2)} per axis (W/D)${shade}`;
}
