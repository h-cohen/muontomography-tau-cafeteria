import { test } from 'node:test';
import assert from 'node:assert/strict';
import { COLORMAP_NAMES, DEFAULT_COLORMAP, colormapStops } from '../src/colormap.mjs';

test('exposes the four presets', () => {
  for (const n of ['viridis', 'inferno', 'magma', 'grayscale']) {
    assert.ok(COLORMAP_NAMES.includes(n), n);
  }
});

test('stops span t=0..1 with a rising alpha ramp', () => {
  const s = colormapStops('viridis');
  assert.equal(s[0].t, 0);
  assert.equal(s[s.length - 1].t, 1);
  assert.ok(s[0].a <= s[s.length - 1].a);          // alpha rises with density
  for (const st of s) {
    for (const c of ['r', 'g', 'b', 'a']) {
      assert.ok(st[c] >= 0 && st[c] <= 255, `${c}=${st[c]}`);
    }
  }
});

test('grayscale is neutral (r==g==b at every stop)', () => {
  for (const st of colormapStops('grayscale')) {
    assert.equal(st.r, st.g);
    assert.equal(st.g, st.b);
  }
});

test('unknown name falls back to viridis', () => {
  assert.deepEqual(colormapStops('nope'), colormapStops('viridis'));
});

test('reversed presets: light at low density, dark at high, same alpha ramp', () => {
  for (const n of ['viridis', 'inferno', 'magma', 'grayscale']) {
    assert.ok(COLORMAP_NAMES.includes(`${n}_r`), `${n}_r`);
    const fwd = colormapStops(n);
    const rev = colormapStops(`${n}_r`);
    rev.forEach((st, i) => {
      const mirror = fwd[fwd.length - 1 - i];
      assert.deepEqual([st.r, st.g, st.b], [mirror.r, mirror.g, mirror.b]);
      assert.equal(st.a, fwd[i].a);
    });
  }
  const g = colormapStops('grayscale_r');
  assert.ok(g[0].r > g[g.length - 1].r);
});

test('the default is the reversed viridis', () => {
  assert.equal(DEFAULT_COLORMAP, 'viridis_r');
});
