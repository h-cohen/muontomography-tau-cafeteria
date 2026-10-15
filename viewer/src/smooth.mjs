import { rayBox, voxelMarch } from './grid.mjs';

// Display-only smoothing for the "Voxel cubes" render mode. The solved grid
// is never resampled or re-solved: 2x / 4x / 8x draw cubes on a finer sub-grid of
// the TRILINEARLY interpolated density, and 'iso' draws the interpolated
// field's surface at the cube threshold. Interpolation adds no information.
// Every sub-sample or surface point belongs to one PARENT solved voxel and is
// hidden exactly when that voxel is (gates.mjs solvedVoxelGate); gate fields
// are never interpolated.
//
// These are the CPU references (hover picking, tests). The GLSL twins in
// app.mjs (smoothCubeMarch / isoMarch in FRAGMENT_SRC) march the same cells
// and samples, skipping parents that are gated or whose 3x3x3 neighbourhood
// maximum is below the threshold (neighbourhoodMax), which cannot change
// the first hit. Coordinates here are voxel coordinates: voxel k spans
// [k, k+1) with its centre at k + 0.5, the GL texel convention.

export const SMOOTHING_MODES = ['off', '2', '4', '8', 'iso'];

// Smoothing applies to per-voxel cubes only: a merged cube grid (cube size
// > 1) bakes its gates into block means, so its samples would no longer have
// a single solved parent. Fast preview while interacting drops it to Off.
export function effectiveSmoothing(state) {
  if (state.renderMode !== 'cubes' || (state.cubeBlock | 0) > 1) return 'off';
  if (state.adaptiveQuality && state.interacting) return 'off';
  return SMOOTHING_MODES.includes(state.smoothing) ? state.smoothing : 'off';
}

// Centres of the factor^3 sub-cells of voxel (i, j, k).
export function subCellCentres(i, j, k, factor) {
  const out = [];
  for (let a = 0; a < factor; a++) for (let b = 0; b < factor; b++) for (let c = 0; c < factor; c++) {
    out.push([i + (a + 0.5) / factor, j + (b + 0.5) / factor, k + (c + 0.5) / factor]);
  }
  return out;
}

export function parentVoxel(p) {
  return [Math.floor(p[0]), Math.floor(p[1]), Math.floor(p[2])];
}

// Trilinear density at voxel coordinate p, clamped to the edge centres:
// the shader's manual trilinear (tex * size - 0.5) and hardware LINEAR.
export function trilinear(data, shape, p) {
  const [nx, ny, nz] = shape;
  const n = [nx, ny, nz];
  const lo = [0, 0, 0], hi = [0, 0, 0], f = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const q = p[a] - 0.5, fl = Math.floor(q);
    f[a] = q - fl;
    lo[a] = Math.min(n[a] - 1, Math.max(0, fl));
    hi[a] = Math.min(n[a] - 1, Math.max(0, fl + 1));
  }
  const at = (x, y, z) => data[x * ny * nz + y * nz + z];
  const lerp = (u, v, t) => u + (v - u) * t;
  const x0 = lerp(at(lo[0], lo[1], lo[2]), at(hi[0], lo[1], lo[2]), f[0]);
  const x1 = lerp(at(lo[0], hi[1], lo[2]), at(hi[0], hi[1], lo[2]), f[0]);
  const x2 = lerp(at(lo[0], lo[1], hi[2]), at(hi[0], lo[1], hi[2]), f[0]);
  const x3 = lerp(at(lo[0], hi[1], hi[2]), at(hi[0], hi[1], hi[2]), f[0]);
  return lerp(lerp(x0, x1, f[1]), lerp(x2, x3, f[1]), f[2]);
}

// Max of the finite values in each voxel's clamped 3x3x3 neighbourhood: an
// upper bound on every trilinear sample inside that voxel (its samples blend
// only the voxel and its neighbours). -3.4e38 where all are NaN.
export function neighbourhoodMax(data, shape) {
  const [nx, ny, nz] = shape;
  const out = new Float32Array(nx * ny * nz);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    let m = -3.4e38;
    for (let a = Math.max(0, i - 1); a <= Math.min(nx - 1, i + 1); a++) {
      for (let b = Math.max(0, j - 1); b <= Math.min(ny - 1, j + 1); b++) {
        for (let c = Math.max(0, k - 1); c <= Math.min(nz - 1, k + 1); c++) {
          const v = data[a * ny * nz + b * nz + c];
          if (v > m) m = v;
        }
      }
    }
    out[i * ny * nz + j * nz + k] = m;
  }
  return out;
}

// 2x / 4x / 8x: first sub-cell (on the factor-times finer grid) whose parent is
// kept and whose centre density is >= threshold. Returns
// { fine: [i, j, k], parent: [i, j, k], axis } or null; axis is the face the
// ray entered the sub-cell through, as voxelMarch reports it.
export function smoothCubeMarch(origin, dir, meta, data, factor, threshold, keep) {
  const fine = {
    shape: meta.shape.map((n) => n * factor),
    origin_m: meta.origin_m,
    spacing_m: meta.spacing_m / factor,
  };
  const hit = voxelMarch(origin, dir, fine, (fi, fj, fk) => {
    const p = [(fi + 0.5) / factor, (fj + 0.5) / factor, (fk + 0.5) / factor];
    const [pi, pj, pk] = parentVoxel(p);
    return keep(pi, pj, pk) && trilinear(data, meta.shape, p) >= threshold;
  });
  if (!hit) return null;
  const parent = [0, 1, 2].map((a) => Math.floor(hit[a] / factor));
  return { fine: hit.slice(0, 3), parent, axis: hit[3] };
}

// Isosurface: samples every 1/stepsPerVoxel voxel from the box entry; the
// first sample "inside" (parent kept AND density >= threshold) after one
// that is not is refined by bisection on the same predicate, so the point
// lands on the density crossing or, where a gate cuts the region, on the
// hidden voxel's face. Returns { point, parent, t } or null.
export const ISO_STEPS_PER_VOXEL = 4;
export function isoMarch(origin, dir, meta, data, threshold, keep, stepsPerVoxel = ISO_STEPS_PER_VOXEL) {
  const s = meta.spacing_m, o = meta.origin_m, n = meta.shape;
  const hit = rayBox(origin, dir, o, [0, 1, 2].map((a) => o[a] + n[a] * s));
  if (!hit) return null;
  const [tEnter, tExit] = hit;
  const voxelAt = (t) => [0, 1, 2].map((a) => (origin[a] + dir[a] * t - o[a]) / s);
  const parentAt = (v) => v.map((x, a) => Math.min(n[a] - 1, Math.max(0, Math.floor(x))));
  const inside = (t) => {
    const v = voxelAt(t);
    const [i, j, k] = parentAt(v);
    return keep(i, j, k) && trilinear(data, n, v) >= threshold;
  };
  const dt = s / stepsPerVoxel;
  let prevT = tEnter;
  for (let i = 0; ; i++) {
    const t = tEnter + (i + 0.5) * dt;
    if (t > tExit) return null;
    if (inside(t)) {
      let lo = prevT, hi = t;
      for (let b = 0; b < 20; b++) {
        const mid = (lo + hi) / 2;
        if (inside(mid)) hi = mid; else lo = mid;
      }
      const point = [0, 1, 2].map((a) => origin[a] + dir[a] * hi);
      return { point, parent: parentAt(voxelAt(hi)), t: hi };
    }
    prevT = t;
  }
}

// The legend line that labels a smoothed view as display-only.
// `unit` labels the threshold (the active layer's unit; '' for none).
export function smoothingLegendText(mode, spacing, threshold, unit = '1/m') {
  const solved = `(solved ${Number(spacing.toPrecision(3))} m)`;
  if (mode === '2' || mode === '4' || mode === '8') return `display: trilinear ${mode}× ${solved}`;
  if (mode === 'iso') {
    const level = `${Number(threshold.toPrecision(3))}${unit ? ` ${unit}` : ''}`;
    return `display: trilinear isosurface at ${level} ${solved}`;
  }
  return '';
}
