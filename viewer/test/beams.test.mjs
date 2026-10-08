import { test } from 'node:test';
import assert from 'node:assert/strict';
import { beamBoxVertices, beamLegendText } from '../src/beams.mjs';

const BOX = { x: 1.0, w: 0.4, zbottom: 6.0, ztop: 7.25, y_extent: [-5, 5] };

function points(v) {
  const out = [];
  for (let i = 0; i < v.length; i += 3) out.push([v[i], v[i + 1], v[i + 2]]);
  return out;
}

test('one box is 12 edges x 2 endpoints x 3 coords', () => {
  assert.equal(beamBoxVertices({ boxes: [BOX] }).length, 72);
  assert.equal(beamBoxVertices({ boxes: [BOX, { ...BOX, x: 2.7 }] }).length, 144);
  assert.equal(beamBoxVertices({ boxes: [] }).length, 0);
});

test('vertices are the box corners in world metres, no voxel conversion', () => {
  const pts = points(beamBoxVertices({ boxes: [BOX] }));
  const xs = new Set(pts.map((p) => p[0].toFixed(4)));
  const ys = new Set(pts.map((p) => p[1].toFixed(4)));
  const zs = new Set(pts.map((p) => p[2].toFixed(4)));
  assert.deepEqual([...xs].sort(), ['0.8000', '1.2000']);
  assert.deepEqual([...ys].sort(), ['-5.0000', '5.0000']);
  assert.deepEqual([...zs].sort(), ['6.0000', '7.2500']);
});

test('every edge is axis-aligned and runs the full box side', () => {
  const pts = points(beamBoxVertices({ boxes: [BOX] }));
  const lengths = [];
  for (let e = 0; e < pts.length; e += 2) {
    const d = pts[e].map((c, i) => Math.abs(pts[e + 1][i] - c));
    assert.equal(d.filter((c) => c > 1e-6).length, 1, `edge ${e / 2} is not axis-aligned`);
    lengths.push(Math.max(...d));
  }
  const counts = (L) => lengths.filter((l) => Math.abs(l - L) < 1e-5).length;
  assert.equal(counts(0.4), 4);
  assert.equal(counts(10), 4);
  assert.equal(counts(1.25), 4);
});

test('a malformed or empty box throws instead of drawing', () => {
  assert.throws(() => beamBoxVertices({ boxes: [{ ...BOX, ztop: NaN }] }), /beam box 0/);
  assert.throws(() => beamBoxVertices({ boxes: [{ ...BOX, y_extent: [1] }] }), /y_extent/);
  assert.throws(() => beamBoxVertices({ boxes: [BOX, { ...BOX, ztop: 5 }] }), /beam box 1: empty/);
});

test('legend shows the total uncertainty when present, the bare value otherwise', () => {
  assert.equal(beamLegendText({ h: 1.234, h_sigma: 0.087 }), 'beam depth h = 1.23 ± 0.09 m');
  assert.equal(beamLegendText({ h: 1.234, h_sigma: null }), 'beam depth h = 1.23 m');
  assert.throws(() => beamLegendText({ h: null }), /finite depth/);
});

test('unresolved physical depth is labelled as a conditional fit layer', () => {
  assert.equal(
    beamLegendText({ h: 0.067, h_sigma: 0.4, depth_resolved: false }),
    'box-fit layer; physical depth unresolved',
  );
});
