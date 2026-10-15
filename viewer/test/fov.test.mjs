import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  fovGeometry, fovLegendText, FOV_STRIDE, acceptanceWeight, fovGain, fovFieldBounds, fovIntersectionRings,
} from '../src/fov.mjs';
import { sortedBeamFaces } from '../src/beams.mjs';

const GEOMETRY = { active_width_m: 0.35, layer_gap_m: 0.4 };
const DET = { id: 'pos0', x: 1, y: -2, z: 0, az_deg: 0 };

function vertices(v) {
  const out = [];
  for (let i = 0; i < v.length; i += FOV_STRIDE) out.push([...v.slice(i, i + FOV_STRIDE)]);
  return out;
}

test('one detector: 4 side quads as 8 triangles and 12 edges, xyz + alpha weight', () => {
  const { faces, edges } = fovGeometry([DET], GEOMETRY, 4);
  assert.equal(FOV_STRIDE, 4);
  assert.equal(faces.length, 8 * 3 * FOV_STRIDE);
  assert.equal(edges.length, 12 * 2 * FOV_STRIDE);
  const two = fovGeometry([DET, { ...DET, x: 3 }], GEOMETRY, 4);
  assert.equal(two.faces.length, 2 * faces.length);
  assert.equal(fovGeometry([], GEOMETRY, 4).faces.length, 0);
});

test('frustum spans the detector square at its z and the acceptance edge W/D at zTop', () => {
  const pts = vertices(fovGeometry([DET], GEOMETRY, 4).faces);
  const base = pts.filter((p) => p[2] === 0);
  const top = pts.filter((p) => p[2] === 4);
  assert.equal(base.length + top.length, pts.length);
  const half = (ps, i, c) => Math.max(...ps.map((p) => Math.abs(p[i] - c)));
  assert.ok(Math.abs(half(base, 0, 1) - 0.175) < 1e-6);
  assert.ok(Math.abs(half(base, 1, -2) - 0.175) < 1e-6);
  const topHalf = 0.175 + 4 * (0.35 / 0.4);
  assert.ok(Math.abs(half(top, 0, 1) - topHalf) < 1e-5);
  assert.ok(Math.abs(half(top, 1, -2) - topHalf) < 1e-5);
});

test('detector azimuth rotates the footprint as Pose.rotation does', () => {
  const az = 30;
  const pts = vertices(fovGeometry([{ ...DET, x: 0, y: 0, az_deg: az }], GEOMETRY, 4).edges);
  const t = 0.175 + 4 * (0.35 / 0.4);
  const a = az * Math.PI / 180;
  const want = [Math.cos(a) * t - Math.sin(a) * t, Math.sin(a) * t + Math.cos(a) * t];
  assert.ok(pts.some((p) => p[2] === 4 && Math.hypot(p[0] - want[0], p[1] - want[1]) < 1e-5));
});

test('walls fade upward; edges stay brighter than the walls they bound', () => {
  const { faces, edges } = fovGeometry([DET], GEOMETRY, 4);
  const weight = (ps, z) => Math.max(...ps.filter((p) => p[2] === z).map((p) => p[3]));
  const f = vertices(faces), e = vertices(edges);
  assert.ok(weight(f, 0) > weight(f, 4));
  assert.ok(weight(e, 0) > weight(f, 0));
  assert.ok(weight(e, 4) > weight(f, 4));
  for (const p of [...f, ...e]) assert.ok(p[3] > 0 && p[3] <= 1);
});

test('missing or degenerate geometry throws instead of drawing a wrong cone', () => {
  assert.throws(() => fovGeometry([DET], null, 4), /detector_geometry/);
  assert.throws(() => fovGeometry([DET], { active_width_m: 0.35, layer_gap_m: 0 }, 4), /detector_geometry/);
  assert.throws(() => fovGeometry([DET], GEOMETRY, 0), /above/);
  assert.throws(() => fovGeometry([{ ...DET, x: NaN }], GEOMETRY, 4), /finite/);
});

test('legend states the per-axis acceptance edge', () => {
  assert.equal(fovLegendText({ active_width_m: 0.36, layer_gap_m: 0.4 }),
    'field of view: |tan θ| ≤ 0.90 per axis (W/D)');
});

test('face sorting honours a wider vertex stride', () => {
  const near = [0, 0, 0, 0.5, 1, 0, 0, 0.5, 0, 1, 0, 0.5];
  const far = near.map((v, i) => (i % 4 === 2 ? 5 : v));
  // matrix maps z to depth directly
  const m = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
  const out = sortedBeamFaces(new Float32Array([...near, ...far]), m, 4);
  assert.deepEqual([...out.slice(0, 12)], far);
  assert.deepEqual([...out.slice(12)], near);
});

test('acceptance weight is the pipeline Detector.acceptance normalised at the zenith', () => {
  // Detector.acceptance: (W-|tx|D)(W-|ty|D)(1+tx^2+ty^2)^-1.5, here / W^2.
  const g = { active_width_m: 0.36, layer_gap_m: 0.4 };
  assert.equal(acceptanceWeight(0, 0, g), 1);
  const want = (1 - 0.3 / 0.9) * (1 - 0.6 / 0.9) * (1 + 0.09 + 0.36) ** -1.5;
  assert.ok(Math.abs(acceptanceWeight(0.3, -0.6, g) - want) < 1e-12);
  assert.equal(acceptanceWeight(0.9, 0, g), 0);
  assert.equal(acceptanceWeight(0, -1.2, g), 0);
});

test('field bounds enclose every frustum from the lowest detector to zTop', () => {
  const dets = [DET, { ...DET, x: 3, az_deg: 20 }];
  const { min, max } = fovFieldBounds(dets, GEOMETRY, 4);
  assert.equal(min[2], 0);
  assert.equal(max[2], 4);
  for (const p of vertices(fovGeometry(dets, GEOMETRY, 4).edges)) {
    for (let i = 0; i < 3; i++) assert.ok(p[i] >= min[i] - 1e-9 && p[i] <= max[i] + 1e-9);
  }
});

test('intersection rings lie inside every frustum and start where the views first overlap', () => {
  const dets = [{ ...DET, x: 0, y: 0 }, { ...DET, x: 2, y: 0, az_deg: 10 }];
  const pts = vertices(fovIntersectionRings(dets, GEOMETRY, 4));
  assert.ok(pts.length > 0);
  const tan = 0.35 / 0.4;
  for (const p of pts) {
    for (const d of dets) {
      const a = d.az_deg * Math.PI / 180, dx = p[0] - d.x, dy = p[1] - d.y;
      const u = Math.cos(a) * dx + Math.sin(a) * dy, v = -Math.sin(a) * dx + Math.cos(a) * dy;
      const half = 0.175 + (p[2] - d.z) * tan;
      assert.ok(Math.abs(u) <= half + 1e-5 && Math.abs(v) <= half + 1e-5);
    }
  }
  const zs = pts.map((p) => p[2]);
  assert.equal(Math.max(...zs), 4);
  // Bases 2 m apart overlap once 2 * (0.175 + z * tan) >= 2 (rotation aside).
  assert.ok(Math.min(...zs) > 0.5 && Math.min(...zs) < 1.5);
});

test('disjoint views have no intersection rings; one view cannot intersect', () => {
  const far = [DET, { ...DET, x: 100 }];
  assert.equal(fovIntersectionRings(far, GEOMETRY, 4).length, 0);
  assert.throws(() => fovIntersectionRings([DET], GEOMETRY, 4), /two/);
});

test('opacity gain: 0 hides, the 60% default is unit gain, 100% is solid', () => {
  assert.equal(fovGain(0), 0);
  assert.ok(Math.abs(fovGain(0.6) - 1) < 1e-12);
  assert.ok(fovGain(0.3) < fovGain(0.6) && fovGain(0.6) < fovGain(0.9));
  // At 100% a faint wall weight already renders opaque: 1 - (1 - w)^gain.
  assert.ok(1 - (1 - 0.05) ** fovGain(1) > 0.999);
});
