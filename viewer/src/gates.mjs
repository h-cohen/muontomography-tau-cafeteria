import { insideClipBox, insideClipPlane } from './clip.mjs';

// The voxel gate set: the ONE reference for which voxels the viewer hides.
// Every CPU consumer (hover picking, the cube-size merge, the auto window)
// asks this module, and render() uploads `uniforms` for the two GLSL copies
// (voxelGated and the fog march in app.mjs), which must follow the same rules.
// tests/viewer/test_gate_parity.py checks shader against this, gate by gate.
//
// NaN semantics, defined once:
//   sigma NaN -> kept   (no bootstrap sigma is not "too uncertain")
//   rays  NaN -> kept   (the shader's rule; rays.npy is counts, never NaN)
//   snr   NaN -> hidden (the bootstrap said nothing: not significant)
// A gate is inert when it is off OR its layer is not loaded.
export function voxelGates(settings, layers) {
  const sigma = settings.sigmaGateEnabled && layers.sigma ? layers.sigma : null;
  const rays = settings.coverageGateEnabled && layers.rays ? layers.rays : null;
  const snr = settings.snrGateEnabled && layers.snr ? layers.snr : null;
  const sigmaValue = settings.sigmaGateValue, minRays = settings.minRays, minSnr = settings.minSnr;
  // The auto window deliberately ignores the sigma gate: sigma is a
  // continuous slider, and re-windowing on every drag would shift the colours
  // under the user's hand. Coverage and SNR hide the noise shell whose
  // inflated values would otherwise set the colour scale.
  const keepForWindow = (n) => (!rays || !(rays[n] < minRays)) && (!snr || snr[n] >= minSnr);
  const keep = (n) => (!sigma || !(sigma[n] > sigmaValue)) && keepForWindow(n);
  return {
    keep,
    keepForWindow,
    windowActive: !!(rays || snr),
    uniforms: {
      sigmaEnabled: !!sigma, sigmaValue,
      coverageEnabled: !!rays, minRays,
      snrEnabled: !!snr, minSnr,
    },
    key: [!!sigma, sigmaValue, !!rays, minRays, !!snr, minSnr].join('|'),
  };
}

// The gate of one SOLVED voxel (i, j, k), as cube mode judges it: clip box,
// clip plane and slice at the voxel centre, then sigma / coverage / SNR on
// the voxel itself. Display smoothing (smooth.mjs) asks this for the parent
// of every sub-sample or surface point, so interpolation never reaches past
// a hidden voxel and gate fields are never interpolated. GLSL twin:
// parentGated in app.mjs (voxelGated at the voxel centre).
export function solvedVoxelGate(settings, voxel, meta) {
  const spatial = spatialGates(settings);
  const [nx, ny, nz] = meta.shape;
  return (i, j, k) => {
    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
    const tex = [(i + 0.5) / nx, (j + 0.5) / ny, (k + 0.5) / nz];
    return spatial.keep(tex) && voxel.keep(i * ny * nz + j * nz + k);
  };
}

// Geometric gates at a sample: clip box, then clip plane, the shader's
// order. `tex` is the [0,1]^3 volume coordinate.
export function spatialGates(settings) {
  return {
    keep(tex) {
      if (!insideClipBox(tex, settings.clipMin, settings.clipMax)) return false;
      if (settings.clipPlaneEnabled && !insideClipPlane(tex, settings.clipPlaneNormal, settings.clipPlaneD)) return false;
      return true;
    },
  };
}
