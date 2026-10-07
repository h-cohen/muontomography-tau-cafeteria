import { parseNpy } from './npy.mjs';

// Reads a run directory (File-like objects: name, text(), arrayBuffer()) into
// one dataset record. Pure apart from the file reads; app.mjs applies the
// record to GL, DOM and state. Optional layers are skipped when absent, never
// an error: a run without a bootstrap has no sigma or snr.
export async function readRun(files) {
  const byName = new Map();
  for (const f of files) byName.set(f.name, f);
  const metaFile = byName.get('meta.json');
  if (!metaFile) throw new Error('selected directory has no meta.json');
  const meta = JSON.parse(await metaFile.text());
  const npyData = async (f) => parseNpy(await f.arrayBuffer()).data;

  // Only layers with one value per voxel are raymarch-able; lower-dimensional
  // diagnostics that share meta.layers are skipped here, correctly, not
  // reported as errors.
  const voxelCount = meta.shape[0] * meta.shape[1] * meta.shape[2];
  const layers = new Map();
  for (const name of meta.layers) {
    const f = byName.get(`${name}.npy`);
    if (!f) continue;
    const data = await npyData(f);
    if (data.length === voxelCount) layers.set(name, data);
  }

  // Plain loop, not Math.max(...sig): a spread blows the call stack on a
  // campaign-sized sigma array.
  let sigmaMax = 1;
  if (layers.has('sigma')) {
    const sig = layers.get('sigma');
    sigmaMax = 0;
    for (let i = 0; i < sig.length; i++) if (sig[i] > sigmaMax) sigmaMax = sig[i];
  }

  const iso = Array.isArray(meta.suggested_iso) ? meta.suggested_iso[0] : null;
  const vr = Array.isArray(meta.value_range) ? meta.value_range : [0, 1];
  const cubeThreshold = Number.isFinite(iso) ? iso : 0.5 * (vr[0] + vr[1]);

  return { meta, layers, sigmaMax, cubeThreshold };
}

// A self-contained build carries its run inline, as a JSON script element
// with id "embedded-run" holding {"files": {"meta.json": "<base64>",
// "volume.npy": "<base64>", ...}} (written by cafetomo/viewerbuild.py).
// Turning it back into File objects lets readRun stay the single loader.
export function embeddedFiles(doc) {
  const el = doc.getElementById('embedded-run');
  if (!el) return null;
  const { files } = JSON.parse(el.textContent);
  return Object.entries(files).map(([name, b64]) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new File([bytes], name);
  });
}
