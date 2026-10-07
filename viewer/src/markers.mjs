// Detector position markers: a small 3-axis cross per detector, in world
// space (metres), so results can be referenced back to the instrument.
// Pure geometry only - no GL here; app.mjs owns the buffer upload and draw.

export const MARKER_HALF_M = 0.6;

// markerVertices(detectors) -> Float32Array
// detectors: [{x, y, z}, ...] (meta.json's `detectors` list; extra fields
// such as `id` are ignored).
// Returns a flat XYZ line-list: for each detector, 3 axes x 2 endpoints x 3
// coords = 18 floats, suitable for gl.LINES.
export function markerVertices(detectors) {
  const out = new Float32Array(detectors.length * 6 * 3);
  let o = 0;
  for (const d of detectors) {
    const { x, y, z } = d;
    // x-arm
    out[o++] = x - MARKER_HALF_M; out[o++] = y; out[o++] = z;
    out[o++] = x + MARKER_HALF_M; out[o++] = y; out[o++] = z;
    // y-arm
    out[o++] = x; out[o++] = y - MARKER_HALF_M; out[o++] = z;
    out[o++] = x; out[o++] = y + MARKER_HALF_M; out[o++] = z;
    // z-arm
    out[o++] = x; out[o++] = y; out[o++] = z - MARKER_HALF_M;
    out[o++] = x; out[o++] = y; out[o++] = z + MARKER_HALF_M;
  }
  return out;
}

// dedupeDetectors(detectors) -> [{ x, y, z, label }, ...]
// Groups meta.json's detectors by world position
// (rounded to guard against float noise) so a single position visited by
// several exposures gets one label with all ids joined by "/", in the order
// they first appear.
export function dedupeDetectors(detectors) {
  const POS_DECIMALS = 6;
  const groups = new Map();
  const order = [];
  for (const d of detectors || []) {
    const { x, y, z, id } = d;
    const key = [x, y, z].map((v) => Number(v).toFixed(POS_DECIMALS)).join(',');
    if (!groups.has(key)) {
      groups.set(key, { x, y, z, ids: [] });
      order.push(key);
    }
    if (id !== undefined) groups.get(key).ids.push(id);
  }
  return order.map((key) => {
    const g = groups.get(key);
    return { x: g.x, y: g.y, z: g.z, label: g.ids.join('/') };
  });
}

// projectToScreen(world, viewProj, width, height) -> { x, y } | null
// Projects a world-space point (metres) through the same column-major
// view-projection matrix the raymarch/hover-pick path uses (see
// cameraMatrices() in app.mjs) into CSS pixel coordinates relative to the
// canvas's top-left corner. Returns null when the point is behind the
// camera (clip w <= 0), since screen coordinates are meaningless there.
export function projectToScreen(world, viewProj, width, height) {
  const [x, y, z] = world;
  const m = viewProj;
  const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
  const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
  const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (cw <= 0) return null;
  const ndcX = cx / cw;
  const ndcY = cy / cw;
  return {
    x: (ndcX * 0.5 + 0.5) * width,
    y: (1 - (ndcY * 0.5 + 0.5)) * height,
  };
}
