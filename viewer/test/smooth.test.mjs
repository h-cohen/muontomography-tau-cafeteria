import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  effectiveSmoothing, subCellCentres, parentVoxel, trilinear, neighbourhoodMax,
  smoothCubeMarch, isoMarch, smoothingLegendText,
} from '../src/smooth.mjs';
import { solvedVoxelGate, voxelGates } from '../src/gates.mjs';

// numpy C-order (z fastest), as the viewer holds layers.
function grid(shape, f) {
  const [nx, ny, nz] = shape;
  const out = new Float32Array(nx * ny * nz);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    out[i * ny * nz + j * nz + k] = f(i, j, k);
  }
  return out;
}
const META = { shape: [5, 3, 4], origin_m: [0, 0, 1], spacing_m: 0.2 };
const KEEP_ALL = () => true;

test('smoothing applies only to per-voxel cubes and drops to Off while previewing', () => {
  const base = { renderMode: 'cubes', cubeBlock: 1, smoothing: '4', adaptiveQuality: true, interacting: false };
  assert.equal(effectiveSmoothing(base), '4');
  assert.equal(effectiveSmoothing({ ...base, renderMode: 'fog' }), 'off');
  assert.equal(effectiveSmoothing({ ...base, cubeBlock: 2 }), 'off');
  assert.equal(effectiveSmoothing({ ...base, interacting: true }), 'off');
  assert.equal(effectiveSmoothing({ ...base, interacting: true, adaptiveQuality: false }), '4');
  assert.equal(effectiveSmoothing({ ...base, smoothing: 'iso' }), 'iso');
  assert.equal(effectiveSmoothing({ ...base, smoothing: '8' }), '8');
});

test('sub-cell centres tile the parent voxel and map back to it', () => {
  const c2 = subCellCentres(3, 1, 2, 2);
  assert.equal(c2.length, 8);
  assert.deepEqual(c2[0], [3.25, 1.25, 2.25]);
  assert.deepEqual(c2[7], [3.75, 1.75, 2.75]);
  const c4 = subCellCentres(0, 0, 0, 4);
  assert.equal(c4.length, 64);
  assert.deepEqual(c4[0], [0.125, 0.125, 0.125]);
  for (const p of [...c2, ...subCellCentres(3, 1, 2, 4)]) assert.deepEqual(parentVoxel(p), [3, 1, 2]);
});

test('trilinear matches GL texel-centre convention: exact at centres, linear between, clamped', () => {
  const shape = [5, 3, 4];
  const data = grid(shape, (i, j, k) => i + 10 * j + 100 * k);
  assert.equal(trilinear(data, shape, [2.5, 1.5, 3.5]), 2 + 10 + 300);
  assert.ok(Math.abs(trilinear(data, shape, [2.75, 1.5, 0.5]) - 12.25) < 1e-6);
  assert.ok(Math.abs(trilinear(data, shape, [1.0, 0.5, 0.5]) - 0.5) < 1e-6);
  assert.equal(trilinear(data, shape, [0.1, 0.5, 0.5]), 0);       // clamp below the first centre
  assert.equal(trilinear(data, shape, [4.9, 0.5, 0.5]), 4);       // and above the last
});

test('neighbourhood max bounds every trilinear sample inside a parent voxel', () => {
  const shape = [5, 3, 4];
  const data = grid(shape, (i, j, k) => Math.sin(i * 1.3 + j * 0.7 - k));
  const nmax = neighbourhoodMax(data, shape);
  for (let i = 0; i < 5; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 4; k++) {
    for (const p of subCellCentres(i, j, k, 4)) {
      assert.ok(trilinear(data, shape, p) <= nmax[i * 12 + j * 4 + k] + 1e-6);
    }
  }
  const single = grid(shape, (i, j, k) => (i === 2 && j === 1 && k === 2 ? 1 : 0));
  const m = neighbourhoodMax(single, shape);
  assert.equal(m[1 * 12 + 0 * 4 + 1], 1);   // diagonal neighbour sees it
  assert.equal(m[0 * 12 + 0 * 4 + 0], 0);   // two voxels away does not
});

test('solved-voxel gate: a sub-sample is hidden exactly when its parent voxel is', () => {
  const shape = META.shape;
  const snr = grid(shape, (i) => (i === 3 ? 0.5 : 10));
  const settings = {
    sigmaGateEnabled: false, sigmaGateValue: 1, coverageGateEnabled: false, minRays: 2,
    snrGateEnabled: true, minSnr: 3,
    clipMin: [0, 0, 0], clipMax: [1, 1, 0.5], clipPlaneEnabled: false, clipPlaneNormal: [0, 0, 1], clipPlaneD: 0,
  };
  const keep = solvedVoxelGate(settings, voxelGates(settings, { snr }), META);
  assert.equal(keep(1, 1, 1), true);
  assert.equal(keep(3, 1, 1), false);           // SNR gate on the parent
  assert.equal(keep(1, 1, 2), false);           // clip box judged at the voxel centre (0.625 > 0.5)
  assert.equal(keep(1, 1, 1.9 | 0), true);
  assert.equal(keep(-1, 0, 0), false);          // outside the grid
  assert.equal(keep(5, 0, 0), false);
});

// A straight-down ray onto a bright column at x = 2 (all y, all z).
function column() {
  const data = grid(META.shape, (i) => (i === 2 ? 1 : 0));
  const down = (x) => [[x, 0.3, 5], [0, 0, -1]];
  return { data, down };
}

test('2x / 4x cubes hit only sub-cells at or above threshold, reporting the parent', () => {
  const { data, down } = column();
  // Voxel 2 spans x 0.4..0.6 m; its centre is x 0.5. Sub-cells inside it
  // interpolate toward 0 at its sides: 2x centres (x 0.45, 0.55) read 0.75.
  for (const f of [2, 4, 8]) {
    const hit = smoothCubeMarch(...down(0.5), META, data, f, 0.5, KEEP_ALL);
    assert.deepEqual(hit.parent, [2, 1, 3]);
    assert.equal(hit.axis, 2);
  }
  // Just inside voxel 1, the outermost 4x sub-cell reads 0.375: below 0.5.
  assert.equal(smoothCubeMarch(...down(0.39), META, data, 4, 0.5, KEEP_ALL), null);
  // ... while a lower threshold catches it, still reporting parent 1.
  assert.deepEqual(smoothCubeMarch(...down(0.39), META, data, 4, 0.3, KEEP_ALL).parent, [1, 1, 3]);
});

test('isosurface sits where the interpolated field crosses the threshold', () => {
  const { data } = column();
  const g = grid(META.shape, (i, j, k) => k);   // field rising with z: 0..3
  // Upward ray: f = (z_voxel - 0.5) crosses 1.75 at voxel z 2.25 -> world 1 + 0.45 m.
  const up = isoMarch([0.5, 0.3, 0], [0, 0, 1], META, g, 1.75, KEEP_ALL);
  assert.ok(Math.abs(up.point[2] - 1.45) < 1e-3, up.point);
  assert.deepEqual(up.parent, [2, 1, 2]);
  // Downward ray starts inside (f = 3 at the top): the surface is the box face.
  const down = isoMarch([0.5, 0.3, 5], [0, 0, -1], META, g, 1.75, KEEP_ALL);
  assert.ok(Math.abs(down.point[2] - 1.8) < 1e-3, down.point);
  assert.deepEqual(down.parent, [2, 1, 3]);
  const side = isoMarch([-1, 0.3, 1.5], [1, 0, 0], META, data, 0.5, KEEP_ALL);
  // Entering the column from x-: f = 0.5 exactly at the face x = 0.4 m.
  assert.ok(Math.abs(side.point[0] - 0.4) < 1e-3, side.point);
  assert.deepEqual(side.parent, [2, 1, 2]);
});

test('gated parents stay hidden in every mode, including the surface', () => {
  const { data, down } = column();
  const noColumn = (i) => i !== 2;
  for (const f of [2, 4, 8]) assert.equal(smoothCubeMarch(...down(0.5), META, data, f, 0.5, noColumn), null);
  assert.equal(isoMarch(...down(0.5), META, data, 0.5, noColumn), null);
  assert.equal(isoMarch([-1, 0.3, 1.5], [1, 0, 0], META, data, 0.5, noColumn), null);
});

test('legend names the display interpolation and the solved spacing', () => {
  assert.equal(smoothingLegendText('off', 0.2, 0.01), '');
  assert.equal(smoothingLegendText('2', 0.2, 0.01), 'display: trilinear 2× (solved 0.2 m)');
  assert.equal(smoothingLegendText('4', 0.2, 0.01), 'display: trilinear 4× (solved 0.2 m)');
  assert.equal(smoothingLegendText('8', 0.2, 0.01), 'display: trilinear 8× (solved 0.2 m)');
  assert.equal(smoothingLegendText('iso', 0.2, 0.0125),
    'display: trilinear isosurface at 0.0125 1/m (solved 0.2 m)');
  assert.equal(smoothingLegendText('iso', 0.2, 0.5, ''),
    'display: trilinear isosurface at 0.5 (solved 0.2 m)');
});
