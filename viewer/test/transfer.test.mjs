import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildTransferLUT } from '../src/transfer.mjs';

test('buildTransferLUT has correct byte length', () => {
  const lut = buildTransferLUT([{ t: 0, r: 0, g: 0, b: 0, a: 0 }, { t: 1, r: 255, g: 255, b: 255, a: 255 }], 256);
  assert.equal(lut.length, 256 * 4);
  assert.ok(lut instanceof Uint8Array);
});

test('buildTransferLUT interpolates linearly between two stops', () => {
  const stops = [
    { t: 0, r: 0, g: 0, b: 0, a: 0 },
    { t: 1, r: 255, g: 255, b: 255, a: 255 },
  ];
  const lut = buildTransferLUT(stops, 3); // samples at t = 0, 0.5, 1
  assert.deepEqual([lut[0], lut[1], lut[2], lut[3]], [0, 0, 0, 0]);
  assert.deepEqual([lut[8], lut[9], lut[10], lut[11]], [255, 255, 255, 255]);
  assert.ok(Math.abs(lut[4] - 127) <= 1);
});
