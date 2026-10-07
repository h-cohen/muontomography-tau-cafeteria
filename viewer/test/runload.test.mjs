import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readRun, embeddedFiles } from '../src/runload.mjs';

function npyBuffer(values, shape) {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  let h = header;
  while ((10 + h.length + 1) % 64 !== 0) h += ' ';
  h += '\n';
  const buf = new ArrayBuffer(10 + h.length + values.length * 4);
  const u8 = new Uint8Array(buf);
  u8.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(buf).setUint16(8, h.length, true);
  for (let i = 0; i < h.length; i++) u8[10 + i] = h.charCodeAt(i);
  new Float32Array(buf, 10 + h.length).set(values);
  return buf;
}
const jsonFile = (name, obj) => ({ name, text: async () => JSON.stringify(obj), arrayBuffer: async () => { throw new Error('json'); } });
const npyFile = (name, values, shape) => ({ name, text: async () => { throw new Error('npy'); }, arrayBuffer: async () => npyBuffer(values, shape) });

const META = { shape: [2, 2, 2], spacing_m: 0.5, origin_m: [0, 0, 0], layers: ['volume', 'sigma', 'profile2d'], suggested_iso: [0.3, 0.6], value_range: [0, 1] };
const vol = [0, 1, 2, 3, 4, 5, 6, 7];

test('missing meta.json throws the same error as before', async () => {
  await assert.rejects(readRun([npyFile('volume.npy', vol, [2, 2, 2])]), /no meta\.json/);
});

test('only full-grid layers load; sigmaMax and cubeThreshold derived', async () => {
  const r = await readRun([
    jsonFile('meta.json', META),
    npyFile('volume.npy', vol, [2, 2, 2]),
    npyFile('sigma.npy', [1, 9, 1, 1, 1, 1, 1, 1], [2, 2, 2]),
    npyFile('profile2d.npy', [1, 2, 3, 4], [2, 2]),       // 2D: skipped
  ]);
  assert.deepEqual([...r.layers.keys()], ['volume', 'sigma']);
  assert.equal(r.sigmaMax, 9);
  assert.equal(r.cubeThreshold, 0.3);
});

test('cubeThreshold falls back to mid value_range; sigmaMax 1 without sigma', async () => {
  const r = await readRun([jsonFile('meta.json', { ...META, layers: ['volume'], suggested_iso: null, value_range: [0, 2] }), npyFile('volume.npy', vol, [2, 2, 2])]);
  assert.equal(r.cubeThreshold, 1);
  assert.equal(r.sigmaMax, 1);
});

test('embeddedFiles: null without the tag, File objects with it', async () => {
  assert.equal(embeddedFiles({ getElementById: () => null }), null);
  const files = { 'meta.json': btoa(JSON.stringify(META)), 'note.json': btoa('[1,2]') };
  const doc = {
    getElementById: (id) => (id === 'embedded-run' ? { textContent: JSON.stringify({ files }) } : null),
  };
  const out = embeddedFiles(doc);
  assert.deepEqual(out.map((f) => f.name), ['meta.json', 'note.json']);
  assert.deepEqual(JSON.parse(await out[0].text()), META);
  assert.deepEqual(JSON.parse(await out[1].text()), [1, 2]);
});

test('embeddedFiles round-trips binary bytes through readRun', async () => {
  const b64 = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)));
  const files = {
    'meta.json': btoa(JSON.stringify({ ...META, layers: ['volume'] })),
    'volume.npy': b64(npyBuffer(vol, [2, 2, 2])),
  };
  const doc = { getElementById: () => ({ textContent: JSON.stringify({ files }) }) };
  const r = await readRun(embeddedFiles(doc));
  assert.deepEqual([...r.layers.get('volume')], vol);
});
