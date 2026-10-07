// The fitted beam-depth model (meta.json `beams`, from the parametric box fit
// to the measured opacity) as wireframe boxes. The fit, the voxel grid and the
// detector markers share one world frame (metres, z up, origin at the first
// position), and the viewer draws every overlay in that frame with the same
// view-projection as the raymarch, so a box needs no voxel-index conversion:
// it lands where the fit put it, and it can be read against the volume
// directly. Pure geometry only - app.mjs owns the GL buffer and the draw.

// The 12 edges of a box as pairs of corner indices; corner c has x = hi when
// bit 0 is set, y = hi for bit 1, z = hi for bit 2.
const BOX_EDGES = [
  [0, 1], [2, 3], [4, 5], [6, 7],   // along x
  [0, 2], [1, 3], [4, 6], [5, 7],   // along y
  [0, 4], [1, 5], [2, 6], [3, 7],   // along z
];

// A malformed box would draw somewhere plausible-looking and wrong, so it
// throws instead of being skipped.
function boxBounds(box, k) {
  const vals = [box.x, box.w, box.zbottom, box.ztop, ...(box.y_extent || [])];
  if (vals.length !== 6 || !vals.every(Number.isFinite)) {
    throw new Error(`beam box ${k}: needs finite x, w, zbottom, ztop and a 2-element y_extent`);
  }
  if (!(box.w > 0) || !(box.ztop > box.zbottom) || !(box.y_extent[1] > box.y_extent[0])) {
    throw new Error(`beam box ${k}: empty extent`);
  }
  return {
    lo: [box.x - box.w / 2, box.y_extent[0], box.zbottom],
    hi: [box.x + box.w / 2, box.y_extent[1], box.ztop],
  };
}

// beamBoxVertices(beams) -> Float32Array
// beams: meta.json's `beams` block, {boxes: [{x, w, zbottom, ztop, y_extent}]}.
// Returns a flat world-space XYZ line list, 12 edges x 2 endpoints x 3 coords
// = 72 floats per box, for gl.LINES (the layout markerVertices uses).
export function beamBoxVertices(beams) {
  const boxes = beams.boxes;
  const out = new Float32Array(boxes.length * BOX_EDGES.length * 6);
  let o = 0;
  boxes.forEach((box, k) => {
    const { lo, hi } = boxBounds(box, k);
    const corner = (c) => [c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]];
    for (const [a, b] of BOX_EDGES) {
      out.set(corner(a), o);
      out.set(corner(b), o + 3);
      o += 6;
    }
  });
  return out;
}

// The legend line under the colour bar. The total uncertainty is shown when
// the error budget has run; without it the bare fit value is shown rather
// than a statistical-only error that would read as the full one.
export function beamLegendText(beams) {
  if (!Number.isFinite(beams.h)) throw new Error('beams.h must be a finite depth in metres');
  const sigma = beams.h_sigma;
  return Number.isFinite(sigma)
    ? `beam depth h = ${beams.h.toFixed(2)} ± ${sigma.toFixed(2)} m`
    : `beam depth h = ${beams.h.toFixed(2)} m`;
}
