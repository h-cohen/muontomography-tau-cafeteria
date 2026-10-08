import { identity, multiply, perspective, lookAt, invert } from './mat4.mjs';
import { reorderForTexture, cubeGrid } from './grid.mjs';
import { buildTransferLUT } from './transfer.mjs';
import { orbitToEye, CAMERA_PRESETS } from './camera.mjs';
import { computeHistogram, robustWindow, windowToBandPx, bandPxToWindow } from './histogram.mjs';
import { availableLayers } from './layers.mjs';
import { initDock } from './dock.mjs';
import { COLORMAP_NAMES, colormapStops } from './colormap.mjs';
import { captureView, loadViews, saveViews } from './views.mjs';
import { SHORTCUTS, keyToAction } from './shortcuts.mjs';
import { markerVertices, dedupeDetectors, projectToScreen } from './markers.mjs';
import { beamBoxVertices, beamDisplayGeometry, beamFaceVertices, sortedBeamFaces, beamLegendText } from './beams.mjs';
import { voxelGates, spatialGates } from './gates.mjs';
import { pickVoxel } from './picker.mjs';
import { readRun, embeddedFiles } from './runload.mjs';
import { createState, effectsFor } from './model.mjs';

const VERTEX_SRC = `#version 300 es
out vec2 vUv;
void main() {
  vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = pos;
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;

uniform mat4 uInvViewProj;
uniform vec3 uCameraPos;
uniform sampler3D uVolume;
uniform sampler3D uSigmaTex;
uniform sampler2D uTransferLUT;
uniform vec2 uWindow;        // [lo, hi] density remap before LUT lookup
uniform vec3 uClipMin;
uniform vec3 uClipMax;
uniform vec3 uWorldMin;
uniform vec3 uWorldExtent;
uniform bool uClipPlaneEnabled;
uniform vec3 uClipPlaneNormal;
uniform float uClipPlaneD;
uniform bool uSigmaGateEnabled;
uniform float uSigmaGateValue;
uniform sampler3D uRaysTex;
uniform bool uCoverageGateEnabled;
uniform float uMinRays;
uniform sampler3D uSnrTex;
uniform bool uSnrGateEnabled;
uniform float uMinSnr;
uniform int uSteps;
uniform bool uManualTrilinear;
uniform bool uShading;
uniform vec3 uVolSize;    // float(nx, ny, nz)
uniform float uRefStep;   // path length the transfer-function alpha is defined per
uniform float uMinStepVoxels;  // minimum step length, in voxels (test-driven; default 0.5)
uniform int uRenderMode;       // 0 density fog, 1 voxel cubes
uniform float uCubeThreshold;  // cube mode: voxels >= this become opaque cubes
uniform sampler3D uCubeTex;    // cube-mode grid: the volume itself, or its b^3 block means
uniform vec3 uCubeSize;        // cube-grid shape
uniform float uCubeVoxel;      // cube-grid spacing (m) = volume spacing * block size
uniform bool uCubeBaked;       // true: sigma/coverage/SNR gates already applied in the merge
uniform float uOpacity;        // display-only opacity multiplier, 0..1 (1 = unchanged)

vec3 unproject(vec2 ndc, float z) {
  vec4 clip = vec4(ndc, z, 1.0);
  vec4 world = uInvViewProj * clip;
  return world.xyz / world.w;
}

float sampleDensity(vec3 tex) {
  if (!uManualTrilinear) return texture(uVolume, tex).r;
  vec3 p = tex * uVolSize - 0.5;
  vec3 f = fract(p);
  ivec3 mx = ivec3(uVolSize) - 1;
  ivec3 a = clamp(ivec3(floor(p)), ivec3(0), mx);
  ivec3 b = clamp(ivec3(floor(p)) + 1, ivec3(0), mx);
  float c000 = texelFetch(uVolume, ivec3(a.x, a.y, a.z), 0).r;
  float c100 = texelFetch(uVolume, ivec3(b.x, a.y, a.z), 0).r;
  float c010 = texelFetch(uVolume, ivec3(a.x, b.y, a.z), 0).r;
  float c110 = texelFetch(uVolume, ivec3(b.x, b.y, a.z), 0).r;
  float c001 = texelFetch(uVolume, ivec3(a.x, a.y, b.z), 0).r;
  float c101 = texelFetch(uVolume, ivec3(b.x, a.y, b.z), 0).r;
  float c011 = texelFetch(uVolume, ivec3(a.x, b.y, b.z), 0).r;
  float c111 = texelFetch(uVolume, ivec3(b.x, b.y, b.z), 0).r;
  return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
             mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}

// Cube mode: every gate evaluated once per voxel, at its centre. GLSL twin of
// gates.mjs (the reference): uniforms come from voxelGates().uniforms;
// tests/viewer/test_gate_parity.py checks the two agree.
bool voxelGated(vec3 tex) {
  bool clipped = any(lessThan(tex, uClipMin)) || any(greaterThan(tex, uClipMax));
  if (uClipPlaneEnabled) {
    float d = dot(tex - vec3(0.5), uClipPlaneNormal) - uClipPlaneD;
    clipped = clipped || d < 0.0;
  }
  if (!uCubeBaked) {
    if (uSigmaGateEnabled) clipped = clipped || texture(uSigmaTex, tex).r > uSigmaGateValue;
    if (uCoverageGateEnabled && !clipped) clipped = texture(uRaysTex, tex).r < uMinRays;
    if (uSnrGateEnabled && !clipped) clipped = !(texture(uSnrTex, tex).r >= uMinSnr);
  }
  return clipped;
}

// Voxel-to-voxel DDA (Amanatides & Woo), the GPU twin of voxelMarch in
// grid.mjs: first voxel >= uCubeThreshold that passes every gate becomes an
// opaque cube, shaded by the face the ray entered through, with a darker rim
// so neighbouring cubes read as separate blocks.
vec4 cubeMarch(vec3 o, vec3 dir, vec3 safeDir) {
  float vs = uCubeVoxel;
  ivec3 n = ivec3(uCubeSize);
  // The cube grid's own box (a merged grid can overhang the volume by a
  // partial block); voxelMarch in grid.mjs intersects the same box.
  vec3 invDir = 1.0 / safeDir;
  vec3 t0s = (uWorldMin - o) * invDir;
  vec3 t1s = (uWorldMin + uCubeSize * vs - o) * invDir;
  vec3 tsm = min(t0s, t1s), tbg = max(t0s, t1s);
  float tEnter = max(max(max(tsm.x, tsm.y), tsm.z), 0.0);
  float tExit = min(min(tbg.x, tbg.y), tbg.z);
  if (tExit <= tEnter) return vec4(0.0);
  int axis = -1;
  if (tEnter > 0.0) axis = (tsm.x >= tsm.y && tsm.x >= tsm.z) ? 0 : (tsm.y >= tsm.z ? 1 : 2);
  vec3 p = o + dir * tEnter;
  ivec3 idx = clamp(ivec3(floor((p - uWorldMin) / vs)), ivec3(0), n - 1);
  ivec3 stp = ivec3(sign(safeDir));
  vec3 tDelta = abs(vec3(vs) / safeDir);
  vec3 tMax = (uWorldMin + (vec3(idx) + vec3(greaterThan(safeDir, vec3(0.0)))) * vs - o) / safeDir;
  float tCur = tEnter;
  vec4 accum = vec4(0.0);
  for (int g = 0; g < 2048; g++) {
    vec3 centre = uWorldMin + (vec3(idx) + 0.5) * vs;
    vec3 tex = clamp((centre - uWorldMin) / uWorldExtent, 0.0, 1.0);
    if (!voxelGated(tex)) {
      float v = texelFetch(uCubeTex, idx, 0).r;
      if (v >= uCubeThreshold) {
        float t = clamp((v - uWindow.x) / max(uWindow.y - uWindow.x, 1e-6), 0.0, 1.0);
        vec3 rgb = texture(uTransferLUT, vec2(t, 0.5)).rgb;
        float shade = axis == 0 ? 0.78 : (axis == 1 ? 0.62 : 1.0);
        if (axis >= 0) {
          vec3 l = fract((o + dir * tCur - uWorldMin) / vs);
          vec3 e = min(l, 1.0 - l);
          e[axis] = 1.0;
          if (min(min(e.x, e.y), e.z) < 0.03) shade *= 0.55;   // rim: 3% of the face width
        }
        // uOpacity = 1: the first cube is opaque (accum.a = 1, loop ends) --
        // exactly the old first-hit return. Below 1, cubes behind show through.
        float a = uOpacity;
        accum += (1.0 - accum.a) * vec4(rgb * shade * a, a);
        if (accum.a > 0.98 || a <= 0.0) return accum;   // premultiplied, like the fog
      }
    }
    int a = 0;
    if (tMax.y < tMax[a]) a = 1;
    if (tMax.z < tMax[a]) a = 2;
    if (tMax[a] > tExit) break;
    idx[a] += stp[a];
    if (idx[a] < 0 || idx[a] >= n[a]) break;
    tCur = tMax[a];
    tMax[a] += tDelta[a];
    axis = a;
  }
  return accum;
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 nearP = unproject(ndc, -1.0);
  vec3 farP = unproject(ndc, 1.0);
  vec3 dir = normalize(farP - nearP);

  // March ONLY inside the volume box (slab intersection). A ray that misses
  // the box costs nothing, and all samples land in the volume at a fixed
  // fraction of a voxel -- the old loop spread uSteps over the whole
  // near..far frustum (~100 m), sampling a fine voxel grid only every
  // other voxel while most steps fell in empty space.
  // Guard the slab division: GLSL ES 3.00 leaves 1.0/0.0 undefined, and while
  // this stack's IEEE ±inf happens to resolve parallel rays correctly, a
  // sign-preserving epsilon keeps invDir finite everywhere without changing
  // the result (measure-zero exact-axis-aligned rays only).
  vec3 sgn = vec3(greaterThanEqual(dir, vec3(0.0))) * 2.0 - 1.0;
  vec3 safeDir = mix(dir, sgn * 1e-8, lessThan(abs(dir), vec3(1e-8)));
  if (uRenderMode == 1) { outColor = cubeMarch(nearP, dir, safeDir); return; }
  vec3 invDir = 1.0 / safeDir;
  vec3 t0s = (uWorldMin - nearP) * invDir;
  vec3 t1s = (uWorldMin + uWorldExtent - nearP) * invDir;
  vec3 tsm = min(t0s, t1s), tbg = max(t0s, t1s);
  float tEnter = max(max(max(tsm.x, tsm.y), tsm.z), 0.0);
  float tExit = min(min(tbg.x, tbg.y), tbg.z);
  if (tExit <= tEnter) { outColor = vec4(0.0); return; }

  float voxel = uWorldExtent.x / uVolSize.x;       // cubic voxels (single spacing)
  float stepLen = max(uMinStepVoxels * voxel, (tExit - tEnter) / float(uSteps));
  // Opacity correction: the transfer function's alpha is defined per
  // uRefStep of path (the legacy sampling length), so the calibrated look is
  // independent of how finely we now sample.
  float alphaExp = stepLen / uRefStep;
  vec4 accum = vec4(0.0);

  for (int i = 0; i < 1024; i++) {
    float tt = tEnter + (float(i) + 0.5) * stepLen;
    if (tt > tExit || accum.a > 0.98) break;
    vec3 pos = nearP + dir * tt;
    vec3 tex = clamp((pos - uWorldMin) / uWorldExtent, 0.0, 1.0);
    bool clipped = any(lessThan(tex, uClipMin)) || any(greaterThan(tex, uClipMax));
    if (uClipPlaneEnabled) {
      float d = dot(tex - vec3(0.5), uClipPlaneNormal) - uClipPlaneD;
      clipped = clipped || d < 0.0;
    }
    // Gates: GLSL twin of gates.mjs (voxelGates + spatialGates); keep in step.
    if (uSigmaGateEnabled) {
      float sigma = texture(uSigmaTex, tex).r;
      clipped = clipped || sigma > uSigmaGateValue;
    }
    if (uCoverageGateEnabled && !clipped) {
      clipped = texture(uRaysTex, tex).r < uMinRays;
    }
    if (uSnrGateEnabled && !clipped) {
      // !(>=) so a NaN SNR (bootstrap said nothing) is hidden, matching hover.
      clipped = !(texture(uSnrTex, tex).r >= uMinSnr);
    }
    if (!clipped) {
      float density = sampleDensity(tex);
      float t = clamp((density - uWindow.x) / max(uWindow.y - uWindow.x, 1e-6), 0.0, 1.0);
      vec4 c = texture(uTransferLUT, vec2(t, 0.5));
      if (uShading && c.a > 0.004) {   // gradient only where the sample shows
        vec3 h = 1.0 / uVolSize;   // one voxel per axis; voxels are cubic, so the
                                   // tex-space difference is proportional to the world gradient
        vec3 g = vec3(
          sampleDensity(tex + vec3(h.x, 0.0, 0.0)) - sampleDensity(tex - vec3(h.x, 0.0, 0.0)),
          sampleDensity(tex + vec3(0.0, h.y, 0.0)) - sampleDensity(tex - vec3(0.0, h.y, 0.0)),
          sampleDensity(tex + vec3(0.0, 0.0, h.z)) - sampleDensity(tex - vec3(0.0, 0.0, h.z)));
        float gm = length(g);
        if (gm > 1e-6) {
          float lambert = abs(dot(-g / gm, -dir));
          c.rgb *= 0.35 + 0.65 * lambert;
        }
      }
      c.a = 1.0 - pow(1.0 - clamp(c.a * uOpacity, 0.0, 0.999), alphaExp);
      c.rgb *= c.a;
      accum += (1.0 - accum.a) * c;
    }
  }

  outColor = accum;
}`;

// Detector markers and fitted beam boxes use a SEPARATE minimal GL program
// from the raymarch shader above: flat-colored world-space lines, drawn with
// gl.LINES after the raymarch fullscreen triangle. This keeps
// FRAGMENT_SRC/VERTEX_SRC (the raymarch shader) untouched.
const MARKER_VERTEX_SRC = `#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uMarkerViewProj;
void main() {
  gl_Position = uMarkerViewProj * vec4(aPos, 1.0);
}`;

const MARKER_FRAGMENT_SRC = `#version 300 es
precision highp float;
uniform vec4 uMarkerColor;
out vec4 outColor;
void main() {
  outColor = uMarkerColor;
}`;

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile error: ${log}`);
  }
  return sh;
}

function linkProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`program link error: ${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

function makeVolumeTexture(gl, shape, data) {
  const [nx, ny, nz] = shape;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  // NEAREST filtering avoids depending on OES_texture_float_linear, which
  // is not guaranteed on every WebGL2 implementation.
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, nx, ny, nz, 0, gl.RED, gl.FLOAT, reorderForTexture(data, shape));
  return tex;
}

function makeLutTexture(gl, lutBytes) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, lutBytes.length / 4, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, lutBytes);
  return tex;
}

export function initViewer(root) {
  const canvas = root.querySelector('#gl-canvas');
  const fileInput = root.querySelector('#load-run-input');
  const detectorLabelsEl = root.querySelector('#detector-labels');
  // preserveDrawingBuffer: true so toDataURL()/screenshot readback after a
  // draw call sees the frame just rendered, not a backbuffer the browser
  // has already cleared for the next composite.
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 is required');
  gl.getExtension('EXT_color_buffer_float');
  const floatLinear = !!gl.getExtension('OES_texture_float_linear');

  const program = linkProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
  const vao = gl.createVertexArray();

  const markerProgram = linkProgram(gl, MARKER_VERTEX_SRC, MARKER_FRAGMENT_SRC);
  const markerUniforms = {
    uMarkerViewProj: gl.getUniformLocation(markerProgram, 'uMarkerViewProj'),
    uMarkerColor: gl.getUniformLocation(markerProgram, 'uMarkerColor'),
  };
  const markerVao = gl.createVertexArray();
  const markerBuffer = gl.createBuffer();
  gl.bindVertexArray(markerVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, markerBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  // Fitted beam boxes: the SAME markerProgram as the detector crosses, a
  // second VAO/buffer because the geometry is unrelated.
  const beamVao = gl.createVertexArray();
  const beamBuffer = gl.createBuffer();
  gl.bindVertexArray(beamVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, beamBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  const beamFaceVao = gl.createVertexArray();
  const beamFaceBuffer = gl.createBuffer();
  gl.bindVertexArray(beamFaceVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, beamFaceBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  let beamFaces = new Float32Array(0);
  let beamModels = {};
  const CAMERA_NEAR = 0.05, CAMERA_FAR = 100;
  const RAY_STEPS = 200; // max samples per ray INSIDE the volume box
  const ADAPTIVE_STEPS = 64; // fast-preview step count while state.interacting
  // Magenta: distinct from the amber detector crosses and from every
  // colormap's ramp, so a box edge never reads as density.

  const uniforms = {};
  for (const name of [
    'uInvViewProj', 'uCameraPos', 'uVolume', 'uSigmaTex', 'uTransferLUT',
    'uWindow', 'uClipMin', 'uClipMax', 'uClipPlaneEnabled', 'uClipPlaneNormal',
    'uClipPlaneD', 'uSigmaGateEnabled', 'uSigmaGateValue', 'uSteps',
    'uWorldMin', 'uWorldExtent', 'uManualTrilinear', 'uShading', 'uVolSize', 'uRefStep',
    'uMinStepVoxels',
    'uRaysTex', 'uCoverageGateEnabled', 'uMinRays',
    'uSnrTex', 'uSnrGateEnabled', 'uMinSnr', 'uRenderMode', 'uCubeThreshold',
    'uCubeTex', 'uCubeSize', 'uCubeVoxel', 'uCubeBaked', 'uOpacity',
  ]) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }

  const dummyVolume = makeVolumeTexture(gl, [1, 1, 1], new Float32Array([0]));
  const state = Object.assign(createState(), {
    gl, program, uniforms, floatLinear,
    raysTex: dummyVolume, snrTex: dummyVolume, volumeTex: dummyVolume, sigmaTex: dummyVolume,
    lutTex: makeLutTexture(gl, buildTransferLUT(colormapStops('viridis'))),
  });

  function worldBounds() {
    if (!state.meta) return { min: [0, 0, 0], extent: [1, 1, 1] };
    const { shape, spacing_m, origin_m } = state.meta;
    const extent = shape.map((n) => n * spacing_m);
    return { min: origin_m, extent };
  }

  // Gate sets are rebuilt from state at each use (cheap, and tests mutate
  // state directly before calling render()/pick()).
  function stateVoxelGates() {
    return voxelGates(state, {
      sigma: state.layerData.get('sigma'),
      rays: state.hasRays ? state.layerData.get('rays') : null,
      snr: state.hasSnr ? state.layerData.get('snr') : null,
    });
  }
  function stateSpatialGates() {
    return spatialGates(state);
  }
  // World Z is the physical vertical (detectors at z=0, ceiling above). Near
  // straight-down/up (|pitch| > PITCH_GIMBAL_LIMIT) the z-up vector goes
  // parallel to the eye-to-target axis and lookAt degenerates, so we fall
  // back to y-up there.
  const PITCH_GIMBAL_LIMIT = 1.4;

  // Single source of truth for eye/view/proj/viewProj/invViewProj, shared by
  // render() and castHoverRay() so the two can never drift out of sync (see
  // the cross-ref comment that used to live on castHoverRay).
  function cameraMatrices() {
    const { yaw, pitch, distance, target } = state.camera;
    const eye = orbitToEye(target, yaw, pitch, distance);
    const up = Math.abs(pitch) > PITCH_GIMBAL_LIMIT ? [0, 1, 0] : [0, 0, 1];
    const view = lookAt(eye, target, up);
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, CAMERA_NEAR, CAMERA_FAR);
    const viewProj = multiply(proj, view);
    const invViewProj = invert(viewProj) || identity();
    return { eye, view, proj, viewProj, invViewProj };
  }

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  // Upgrades the live 3D textures' filter mode from the NEAREST that
  // makeVolumeTexture always creates them with. Hardware LINEAR is only used
  // when OES_texture_float_linear is present (R32F textures otherwise cannot
  // be linearly filtered on WebGL2); the shader's manual 8-tap trilinear
  // (uManualTrilinear) covers smooth sampling everywhere else. Must be
  // called after every makeVolumeTexture assignment, since a fresh texture
  // always starts NEAREST.
  function applyVolumeFilter() {
    const filter = (state.smoothSampling && state.floatLinear) ? gl.LINEAR : gl.NEAREST;
    // Only the density is smoothed. The sigma texture stays NEAREST: the sigma
    // gate is a per-voxel decision, and filtering it would make the gate edge
    // differ between GPUs with and without float-linear (the manual path
    // always gates per voxel) and disagree with nearest-voxel hover.
    // The rays texture is a per-voxel count gate, NEAREST for the same reason.
    const sets = [[state.volumeTex, filter], [state.sigmaTex, gl.NEAREST],
                  [state.raysTex, gl.NEAREST], [state.snrTex, gl.NEAREST]];
    for (const [tex, f] of sets) {
      if (!tex) continue;
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, f);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, f);
    }
  }
  state.applyVolumeFilter = applyVolumeFilter;

  // Adaptive quality: camera drags, zooms and slider drags call
  // beginInteraction() to mark state.interacting=true for a fast (low-step,
  // unshaded) preview render, then idle for INTERACTION_IDLE_MS with no
  // further beginInteraction() call before render() runs once more at full
  // quality. idleNow() is a test hook that fires that idle render
  // immediately, without waiting on the real timer, so headless tests need
  // not sleep to observe it.
  const INTERACTION_IDLE_MS = 150;
  let interactionTimer = null;
  function idleNow() {
    if (interactionTimer != null) {
      clearTimeout(interactionTimer);
      interactionTimer = null;
    }
    state.interacting = false;
    render();
  }
  function beginInteraction() {
    state.interacting = true;
    if (interactionTimer != null) clearTimeout(interactionTimer);
    interactionTimer = setTimeout(idleNow, INTERACTION_IDLE_MS);
  }
  state.beginInteraction = beginInteraction;
  state.idleNow = idleNow;

  // The one way a UI control changes display state: assign the patch, run the
  // follow-ups model.mjs declares for those fields (fixed order), and render
  // a fast preview when the change is part of a continuous drag.
  function commit(patch, { interactive = false } = {}) {
    const effects = effectsFor(patch);
    Object.assign(state, patch);
    for (const e of effects) {
      if (e === 'filter') applyVolumeFilter();
      else if (e === 'window') { if (state.applyWindowForLayer) state.applyWindowForLayer(state.activeLayer); }
      else if (e === 'histogram') drawHistogram();
      else if (e === 'lut') {
        gl.deleteTexture(state.lutTex);
        state.lutTex = makeLutTexture(gl, buildTransferLUT(state.transferStops));
        drawXferEditor();
      } else if (e === 'render') {
        if (interactive) beginInteraction();
        render();
      }
    }
  }

  function render() {
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(width * (window.devicePixelRatio || 1)));
    canvas.height = Math.max(1, Math.round(height * (window.devicePixelRatio || 1)));
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (currentTheme() === 'light') {
      gl.clearColor(0.90, 0.92, 0.94, 1);
    } else {
      gl.clearColor(0.04, 0.05, 0.06, 1);
    }
    gl.clear(gl.COLOR_BUFFER_BIT);

    const { eye, viewProj, invViewProj } = cameraMatrices();

    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniformMatrix4fv(uniforms.uInvViewProj, false, invViewProj);
    gl.uniform3fv(uniforms.uCameraPos, eye);

    const { min, extent } = worldBounds();
    gl.uniform3fv(uniforms.uWorldMin, min);
    gl.uniform3fv(uniforms.uWorldExtent, extent);
    gl.uniform2fv(uniforms.uWindow, state.window || [0, 1]);
    gl.uniform3fv(uniforms.uClipMin, state.clipMin);
    gl.uniform3fv(uniforms.uClipMax, state.clipMax);
    gl.uniform1i(uniforms.uClipPlaneEnabled, state.clipPlaneEnabled ? 1 : 0);
    gl.uniform3fv(uniforms.uClipPlaneNormal, state.clipPlaneNormal);
    gl.uniform1f(uniforms.uClipPlaneD, state.clipPlaneD);
    const vg = stateVoxelGates().uniforms;
    gl.uniform1i(uniforms.uSigmaGateEnabled, vg.sigmaEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uSigmaGateValue, vg.sigmaValue);
    gl.uniform1i(uniforms.uCoverageGateEnabled, vg.coverageEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uMinRays, vg.minRays);
    gl.uniform1i(uniforms.uSnrGateEnabled, vg.snrEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uMinSnr, vg.minSnr);
    gl.uniform1i(uniforms.uRenderMode, state.renderMode === 'cubes' ? 1 : 0);
    gl.uniform1f(uniforms.uCubeThreshold, state.cubeThreshold);
    gl.uniform1f(uniforms.uOpacity, state.opacity);
    const cg = state.renderMode === 'cubes' ? ensureCubeGrid() : null;
    const cShape = cg ? cg.shape : (state.meta ? state.meta.shape : [1, 1, 1]);
    const cVoxel = cg ? cg.spacing : 1;
    gl.uniform3fv(uniforms.uCubeSize, [cShape[0], cShape[1], cShape[2]]);
    gl.uniform1f(uniforms.uCubeVoxel, cVoxel);
    gl.uniform1i(uniforms.uCubeBaked, cg && cg.baked ? 1 : 0);
    // Fast preview while interacting (drag/zoom/slider): fewer march steps
    // and no shading for this frame only -- state.shading itself is left
    // untouched, so the checkbox/readout never lies about the setting, and
    // the very next idle full-quality render restores it.
    const previewing = !!(state.adaptiveQuality && state.interacting);
    const steps = previewing ? ADAPTIVE_STEPS : RAY_STEPS;
    gl.uniform1i(uniforms.uSteps, steps);
    // Legacy per-sample path: uSteps spread over the whole near..far span. The
    // LUT/window were tuned against it, so alpha stays defined per this length.
    // The old step was actually (FAR-NEAR)/(uSteps*cos(theta)) per pixel, theta
    // the off-axis angle from unprojecting a frustum instead of a box, so
    // corner pixels had a longer, less-dense step than the centre; this single
    // uniform reference matches the old CENTRE look exactly (edges are now
    // slightly denser than the legacy render, not less).
    gl.uniform1f(uniforms.uRefStep, (CAMERA_FAR - CAMERA_NEAR) / steps);
    gl.uniform1f(uniforms.uMinStepVoxels, state.minStepVoxels != null ? state.minStepVoxels : 0.5);
    gl.uniform1i(uniforms.uManualTrilinear, (state.smoothSampling && !state.floatLinear) ? 1 : 0);
    gl.uniform1i(uniforms.uShading, (previewing ? false : state.shading) ? 1 : 0);
    const volShape = state.meta ? state.meta.shape : [1, 1, 1];
    gl.uniform3fv(uniforms.uVolSize, [volShape[0], volShape[1], volShape[2]]);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, state.volumeTex);
    gl.uniform1i(uniforms.uVolume, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, state.sigmaTex);
    gl.uniform1i(uniforms.uSigmaTex, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, state.lutTex);
    gl.uniform1i(uniforms.uTransferLUT, 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_3D, state.raysTex);
    gl.uniform1i(uniforms.uRaysTex, 3);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_3D, state.snrTex);
    gl.uniform1i(uniforms.uSnrTex, 4);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_3D, cg && cg.tex ? cg.tex : state.volumeTex);
    gl.uniform1i(uniforms.uCubeTex, 5);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (state.showDetectors && state.meta && state.markerVertexCount > 0) {
      drawMarkers(viewProj);
    }
    if (state.showBeams && state.beamOpacity > 0 && state.meta && state.beamVertexCount > 0) {
      drawBeams(viewProj);
    }
    drawGizmo();
    updateDetectorLabels();
  }
  state.render = render;

  // Detector position markers: drawn with their own tiny GL program (see
  // MARKER_VERTEX_SRC/MARKER_FRAGMENT_SRC above), using the SAME viewProj
  // render() just computed so the crosses sit correctly in the scene.
  function drawMarkers(viewProj) {
    gl.useProgram(markerProgram);
    gl.bindVertexArray(markerVao);
    gl.uniformMatrix4fv(markerUniforms.uMarkerViewProj, false, viewProj);
    gl.uniform4fv(markerUniforms.uMarkerColor, [1.0, 0.75, 0.1, 1.0]);
    gl.drawArrays(gl.LINES, 0, state.markerVertexCount);
    gl.bindVertexArray(null);
  }

  function rebuildMarkerBuffer() {
    const detectors = state.detectors || [];
    const verts = markerVertices(detectors);
    state.markerVertexCount = verts.length / 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, markerBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    state.detectorLabels = dedupeDetectors(detectors);
    rebuildDetectorLabelEls();
  }

  // HTML overlay labels ("P0", "P1", ...) for detector positions -
  // crisp, theme-aware text the GL raymarch/marker programs can't give us
  // cheaply. One <div> per distinct world position (see dedupeDetectors in
  // markers.mjs); rebuilt only when the detector set changes, repositioned
  // every render() via the SAME viewProj the scene draws with.
  let detectorLabelEls = [];
  function rebuildDetectorLabelEls() {
    if (!detectorLabelsEl) return;
    detectorLabelsEl.innerHTML = '';
    detectorLabelEls = state.detectorLabels.map((d) => {
      const el = document.createElement('div');
      el.className = 'detector-label';
      el.textContent = d.label;
      el.hidden = true;
      detectorLabelsEl.appendChild(el);
      return el;
    });
  }

  function updateDetectorLabels() {
    if (!detectorLabelsEl) return;
    if (!state.showDetectors || !state.meta || detectorLabelEls.length === 0) {
      for (const el of detectorLabelEls) el.hidden = true;
      return;
    }
    const { viewProj } = cameraMatrices();
    const { width, height } = root.querySelector('#canvas-wrap').getBoundingClientRect();
    state.detectorLabels.forEach((d, idx) => {
      const el = detectorLabelEls[idx];
      const p = projectToScreen([d.x, d.y, d.z], viewProj, width, height);
      if (!p || p.x < 0 || p.x > width || p.y < 0 || p.y > height) {
        el.hidden = true;
        return;
      }
      el.hidden = false;
      el.style.left = `${p.x}px`;
      el.style.top = `${p.y}px`;
    });
  }

  // Fitted beam boxes: the SAME markerProgram and viewProj as the detector
  // crosses, so the boxes sit in the scene exactly where the fit put them.
  function drawBeams(viewProj) {
    gl.useProgram(markerProgram);
    gl.bindVertexArray(beamVao);
    gl.uniformMatrix4fv(markerUniforms.uMarkerViewProj, false, viewProj);
    const rgb = [1, 3, 5].map((i) => parseInt(state.beamColor.slice(i, i + 2), 16) / 255);
    gl.uniform4fv(markerUniforms.uMarkerColor, [...rgb, state.beamOpacity]);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(beamFaceVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, beamFaceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sortedBeamFaces(beamFaces, viewProj), gl.DYNAMIC_DRAW);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.drawArrays(gl.TRIANGLES, 0, beamFaces.length / 3);
    gl.disable(gl.CULL_FACE);
    gl.bindVertexArray(beamVao);
    gl.drawArrays(gl.LINES, 0, state.beamVertexCount);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  // A run without a fitted beam model hides the control and
  // the legend line instead of offering a toggle that draws nothing; one with
  // it shows the boxes by default, since they are the paper's headline.
  function applyBeams(beams) {
    const controlEl = root.querySelector('#beams-control');
    const toggleEl = root.querySelector('#toggle-beams');
    const legendEl = root.querySelector('#beam-legend');
    const displayed = beams ? beamDisplayGeometry(beams) : null;
    const verts = displayed ? beamBoxVertices(displayed) : new Float32Array(0);
    state.beamVertexCount = verts.length / 3;
    beamFaces = displayed ? beamFaceVertices(displayed) : new Float32Array(0);
    state.beamDisplayBoxes = displayed ? displayed.boxes : [];
    state.beamFaceVertexCount = beamFaces.length / 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, beamBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    state.showBeams = !!beams;
    controlEl.hidden = !beams;
    toggleEl.checked = !!beams;
    legendEl.hidden = !beams;
    legendEl.textContent = beams ? beamLegendText(beams) : '';
  }

  // Axis-orientation gizmo (bottom-left): projects the three world axes
  // into view space using the SAME yaw/pitch as render()'s camera, so the
  // triad always matches what's on screen. Distance/target don't affect
  // direction, so we reuse orbitToEye/lookAt with target=[0,0,0],
  // distance=1 purely to get the rotation basis.
  function drawGizmo() {
    const gizmoCanvas = root.querySelector('#gizmo-canvas');
    if (!gizmoCanvas) return;
    const ctx = gizmoCanvas.getContext('2d');
    const w = gizmoCanvas.width, h = gizmoCanvas.height;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.32;

    const { yaw, pitch } = state.camera;
    const eye = orbitToEye([0, 0, 0], yaw, pitch, 1);
    const up = Math.abs(pitch) > PITCH_GIMBAL_LIMIT ? [0, 1, 0] : [0, 0, 1];
    const view = lookAt(eye, [0, 0, 0], up);
    // view[0..2] = view-space coords of world X axis, view[4..6] of world
    // Y, view[8..10] of world Z (see mat4.mjs lookAt column layout).
    const axes = [
      { label: 'X', color: '#e5484d', dx: view[0], dy: view[1] },
      { label: 'Y', color: '#3fb950', dx: view[4], dy: view[5] },
      { label: 'Z', color: '#4d9de5', dx: view[8], dy: view[9] },
    ];
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const axis of axes) {
      // Canvas y grows downward; view-space y grows upward.
      const ex = cx + axis.dx * r, ey = cy - axis.dy * r;
      ctx.strokeStyle = axis.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      ctx.fillStyle = axis.color;
      ctx.fillText(axis.label, cx + axis.dx * (r + 10), cy - axis.dy * (r + 10));
    }
  }
  function drawHistogram() {
    const canvas = root.querySelector('#histogram-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const data = state.layerData.get(state.activeLayer);
    const readout = root.querySelector('#window-readout');
    if (!data || !state.window) return;
    const [lo, hi] = state.window;
    const hist = computeHistogram(data, lo, hi, 64);
    const max = Math.max(...hist, 1);
    const barW = canvas.width / hist.length;
    ctx.fillStyle = '#8cf';
    for (let i = 0; i < hist.length; i++) {
      const h = (hist[i] / max) * canvas.height;
      ctx.fillRect(i * barW, canvas.height - h, barW - 1, h);
    }

    // The window band IS the control: draw it as a translucent accent
    // rectangle with two edge handles over the bars, in the same pixel
    // space #histogram-canvas pointer events are converted into below.
    const { loPx, hiPx } = windowToBandPx(state.window, state.layerMax, canvas.width);
    ctx.fillStyle = 'rgba(140, 204, 255, 0.18)';
    ctx.fillRect(loPx, 0, hiPx - loPx, canvas.height);
    ctx.fillStyle = 'rgba(140, 204, 255, 0.9)';
    ctx.fillRect(loPx - 1.5, 0, 3, canvas.height);
    ctx.fillRect(hiPx - 1.5, 0, 3, canvas.height);

    if (readout) {
      readout.textContent = `${state.window[0].toFixed(3)} – ${state.window[1].toFixed(3)} 1/m`;
    }
    drawLegend();
  }

  // Colorbar legend (bottom-right): fills #legend-canvas with the current
  // transfer LUT and labels #legend-ticks with the active window's lo/mid/hi.
  // Called from drawHistogram()/drawXferEditor() so it stays in sync with
  // every place the LUT or window changes (colormap, window drag, layer
  // switch, initial load).
  function drawLegend() {
    const canvas = root.querySelector('#legend-canvas');
    const ticksEl = root.querySelector('#legend-ticks');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const lut = buildTransferLUT(state.transferStops, canvas.width);
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let x = 0; x < canvas.width; x++) {
      for (let y = 0; y < canvas.height; y++) {
        const idx = (y * canvas.width + x) * 4;
        img.data[idx] = lut[x * 4]; img.data[idx + 1] = lut[x * 4 + 1];
        img.data[idx + 2] = lut[x * 4 + 2]; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    if (ticksEl) {
      const w = state.window || [0, 1];
      const mid = (w[0] + w[1]) / 2;
      const unit = state.activeLayer === 'volume' ? ' 1/m' : '';
      const fmt = (v) => v.toFixed(3) + unit;
      ticksEl.textContent = `${fmt(w[0])}  ${fmt(mid)}  ${fmt(w[1])}`;
    }
  }
  state.drawLegend = drawLegend;

  function drawXferEditor() {
    const canvas = root.querySelector('#xfer-canvas');
    const ctx = canvas.getContext('2d');
    const lut = buildTransferLUT(state.transferStops, canvas.width);
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let x = 0; x < canvas.width; x++) {
      for (let y = 0; y < canvas.height; y++) {
        const idx = (y * canvas.width + x) * 4;
        img.data[idx] = lut[x * 4]; img.data[idx + 1] = lut[x * 4 + 1];
        img.data[idx + 2] = lut[x * 4 + 2]; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.fillStyle = '#fff';
    for (const s of state.transferStops) {
      ctx.fillRect(s.t * canvas.width - 2, 0, 4, canvas.height);
    }
    drawLegend();
  }
  state.drawHistogram = drawHistogram;
  state.drawXferEditor = drawXferEditor;

  // Frame the camera on the grid's world-space center, sized to the volume's
  // diagonal. Shared by loadRun (on every fresh run) and #frame-all-btn (to
  // recover the view after the user has panned/zoomed away), so the framing
  // math lives in exactly one place.
  function frameAll() {
    if (!state.meta) return;
    const { shape, spacing_m, origin_m } = state.meta;
    state.camera.target = [
      origin_m[0] + 0.5 * shape[0] * spacing_m,
      origin_m[1] + 0.5 * shape[1] * spacing_m,
      origin_m[2] + 0.5 * shape[2] * spacing_m,
    ];
    const diag = Math.hypot(shape[0] * spacing_m, shape[1] * spacing_m, shape[2] * spacing_m);
    const FRAME_MARGIN = 1.6;
    state.camera.distance = FRAME_MARGIN * diag;
  }

  async function loadRun(files) {
    state.ready = false;
    const onboardingEl = root.querySelector('#overlay-onboarding');
    if (onboardingEl) onboardingEl.hidden = true;

    const run = await readRun(files);
    const meta = run.meta;
    state.meta = meta;
    state.detectors = meta.detectors || [];
    rebuildMarkerBuffer();

    beamModels = meta.beam_models || (meta.beams ? { matched: meta.beams } : {});
    const selector = root.querySelector('#beam-model');
    selector.replaceChildren();
    for (const key of Object.keys(beamModels)) {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = key === 'conditional' ? 'Conditional continued array (z depth)' : 'Matched-beam baseline';
      selector.append(option);
    }
    selector.value = meta.beam_model_default || Object.keys(beamModels)[0] || '';
    applyBeams(beamModels[selector.value] || null);

    const banner = root.querySelector('#resolution-banner');
    const res = meta.resolution || {};
    banner.textContent = res.verdict || '';
    banner.classList.toggle('not-resolved', !res.depth_resolved);
    const runNameEl = root.querySelector('#run-name');
    if (runNameEl) runNameEl.textContent = meta.run || '';

    // readRun keeps only layers with one value per voxel (nx*ny*nz): those
    // are the raymarch-able ones offered as radios.
    state.layerData.clear();
    for (const [name, data] of run.layers) state.layerData.set(name, data);

    // Frame the camera on the grid's world-space center, sized to the
    // volume's diagonal, instead of the fixed target=[0,0,0]/distance=3
    // defaults, which orphan a (typically off-origin, many-metre) volume
    // off-screen or reduced to a speck.
    frameAll();

    applyViewerCrop(meta);
    state.loadSeq += 1;
    {
      const sel = root.querySelector('#cube-size');
      if (sel) {
        for (const opt of sel.options) {
          const b = parseInt(opt.value, 10);
          opt.textContent = `${b}\u00d7 (${(b * meta.spacing_m).toFixed(2)} m)`;
        }
      }
    }
    {
      state.cubeThreshold = run.cubeThreshold;
      const el = root.querySelector('#cube-threshold');
      if (el) el.value = String(state.cubeThreshold);
    }

    state.activeLayer = 'volume';
    state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('volume'));
    if (state.layerData.has('sigma')) {
      state.sigmaTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('sigma'));
    }
    state.sigmaMax = run.sigmaMax;
    state.hasRays = state.layerData.has('rays');
    state.raysTex = state.hasRays
      ? makeVolumeTexture(gl, meta.shape, state.layerData.get('rays'))
      : dummyVolume;
    for (const id of ['#coverage-gate-enabled', '#coverage-gate-value']) {
      const el = root.querySelector(id);
      if (el) el.disabled = !state.hasRays;
    }
    state.hasSnr = state.layerData.has('snr');
    state.snrTex = state.hasSnr
      ? makeVolumeTexture(gl, meta.shape, state.layerData.get('snr'))
      : dummyVolume;
    for (const id of ['#snr-gate-enabled', '#snr-gate-value']) {
      const el = root.querySelector(id);
      if (el) el.disabled = !state.hasSnr;
    }
    applyVolumeFilter();

    // Window each layer to ITS OWN robust range (see histogram.mjs
    // robustWindow), not meta.value_range: value_range[1] is typically a
    // single outlier voxel, so windowing to [min,max] renders near-black.
    function applyWindowForLayer(key) {
      const data = state.layerData.get(key);
      let max = 0;
      if (data) {
        for (let i = 0; i < data.length; i++) {
          const v = data[i];
          if (!Number.isNaN(v) && v > max) max = v;
        }
      }
      state.layerMax = max;
      state.window = data ? robustWindow(gatedForWindow(data)) : [0, 1];
      drawHistogram();
    }
    state.applyWindowForLayer = applyWindowForLayer;

    // The auto window comes from the voxels the window gates KEEP (see
    // keepForWindow in gates.mjs for why sigma is not one of them).
    function gatedForWindow(data) {
      const g = stateVoxelGates();
      if (!g.windowActive) return data;
      const out = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) out[i] = g.keepForWindow(i) ? data[i] : NaN;
      return out;
    }

    function setActiveLayer(key) {
      state.activeLayer = key;
      gl.deleteTexture(state.volumeTex);
      state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get(key));
      applyVolumeFilter();
      applyWindowForLayer(key);
      drawHistogram();
      render();
    }
    state.setActiveLayer = setActiveLayer;

    const panel = root.querySelector('#layer-panel');
    panel.innerHTML = '';
    for (const layer of availableLayers([...state.layerData.keys()])) {
      const id = `layer-${layer.key}`;
      const label = document.createElement('label');
      label.style.display = 'block';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'active-layer';
      radio.id = id;
      radio.checked = layer.key === 'volume';
      radio.addEventListener('change', () => setActiveLayer(layer.key));
      label.appendChild(radio);
      label.appendChild(document.createTextNode(' ' + layer.label));
      panel.appendChild(label);
    }

    applyWindowForLayer('volume');
    drawHistogram();
    drawXferEditor();
    render();
    state.ready = true;
  }

  // Load failures are reported in the console and on window.__viewerError (read
  // by the Playwright tests), never as a silently half-loaded view.
  function loadRunReportingErrors(files) {
    loadRun(files).catch((err) => {
      console.error(err);
      window.__viewerError = String(err);
    });
  }
  fileInput.addEventListener('change', (ev) => loadRunReportingErrors(Array.from(ev.target.files)));

  root.querySelector('#export-png-btn').addEventListener('click', () => {
    render();
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'cafetomo-volume-view.png';
      a.click();
      URL.revokeObjectURL(url);
    });
  });

  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', (ev) => {
    dragging = true; lastX = ev.clientX; lastY = ev.clientY;
  });
  window.addEventListener('pointerup', () => { dragging = false; });
  window.addEventListener('pointermove', (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
    lastX = ev.clientX; lastY = ev.clientY;
    const yaw = state.camera.yaw + dx * 0.01;
    const pitch = Math.max(-1.5, Math.min(1.5, state.camera.pitch + dy * 0.01));
    commit({ camera: { ...state.camera, yaw, pitch } }, { interactive: true });
  });
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const distance = Math.max(0.5, state.camera.distance * (1 + ev.deltaY * 0.001));
    commit({ camera: { ...state.camera, distance } }, { interactive: true });
  }, { passive: false });

  for (const key of Object.keys(CAMERA_PRESETS)) {
    const btn = root.querySelector(`#camera-preset-${key}`);
    btn.addEventListener('click', () => {
      const yaw = CAMERA_PRESETS[key].yaw;
      const pitch = CAMERA_PRESETS[key].pitch;
      commit({ camera: { ...state.camera, yaw, pitch } });
    });
  }

  root.querySelector('#frame-all-btn').addEventListener('click', () => {
    frameAll();
    render();
  });

  const toggleDetectorsEl = root.querySelector('#toggle-detectors');
  if (toggleDetectorsEl) {
    toggleDetectorsEl.addEventListener('change', (ev) => {
      commit({ showDetectors: ev.target.checked });
    });
  }

  const toggleSmoothEl = root.querySelector('#toggle-smooth');
  if (toggleSmoothEl) {
    toggleSmoothEl.addEventListener('change', (ev) => {
      commit({ smoothSampling: ev.target.checked });
    });
  }

  const toggleShadingEl = root.querySelector('#toggle-shading');
  if (toggleShadingEl) {
    toggleShadingEl.addEventListener('change', (ev) => {
      commit({ shading: ev.target.checked });
    });
  }

  const toggleAdaptiveEl = root.querySelector('#toggle-adaptive');
  if (toggleAdaptiveEl) {
    toggleAdaptiveEl.addEventListener('change', (ev) => {
      commit({ adaptiveQuality: ev.target.checked });
    });
  }

  root.querySelector('#toggle-beams').addEventListener('change', (ev) => {
    commit({ showBeams: ev.target.checked });
  });

  root.querySelector('#beam-model').addEventListener('change', ev => {
    const visible = state.showBeams;
    applyBeams(beamModels[ev.target.value]);
    state.showBeams = visible;
    root.querySelector('#toggle-beams').checked = visible;
    render();
  });
  root.querySelector('#beam-opacity').addEventListener('input', (ev) => {
    const value = Number(ev.target.value);
    root.querySelector('#beam-opacity-readout').textContent = `${Math.round(100 * value)}%`;
    commit({ beamOpacity: value });
  });
  root.querySelector('#beam-color').addEventListener('input', (ev) => {
    commit({ beamColor: ev.target.value });
  });

  function renderViewList() {
    const list = loadViews();
    const ul = root.querySelector('#saved-views');
    ul.innerHTML = '';
    list.forEach((view, i) => {
      const li = document.createElement('li');
      li.textContent = view.name + ' ';
      const applyBtn = document.createElement('button');
      applyBtn.textContent = 'Apply';
      applyBtn.addEventListener('click', () => {
        if (!state.meta) return;
        state.camera = {
          yaw: view.camera.yaw,
          pitch: view.camera.pitch,
          distance: view.camera.distance,
          target: [...view.camera.target],
        };
        // setActiveLayer rebuilds the layer texture AND resets state.window
        // to that layer's own robust default as a side effect, so it must
        // run BEFORE the saved window is applied, not after — otherwise the
        // saved window is silently clobbered by the robust default.
        if (view.activeLayer && state.setActiveLayer) {
          state.setActiveLayer(view.activeLayer);
          state.window = [view.window[0], view.window[1]];
          if (state.drawHistogram) state.drawHistogram();
          render();
        } else {
          state.window = [view.window[0], view.window[1]];
          render();
        }
      });
      const deleteBtn = document.createElement('button');
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', () => {
        const current = loadViews();
        current.splice(i, 1);
        saveViews(current);
        renderViewList();
      });
      li.appendChild(applyBtn);
      li.appendChild(deleteBtn);
      ul.appendChild(li);
    });
  }

  root.querySelector('#save-view-btn').addEventListener('click', () => {
    if (!state.meta || !state.setActiveLayer) return;
    const nameInput = root.querySelector('#view-name');
    const list = loadViews();
    list.push(captureView(state, nameInput.value || 'view ' + (list.length + 1)));
    saveViews(list);
    renderViewList();
  });

  renderViewList();

  const histCanvas = root.querySelector('#histogram-canvas');
  let draggingWindowEdge = null; // 'lo' | 'hi' | null, local to this control
  function histCanvasPx(ev) {
    const rect = histCanvas.getBoundingClientRect();
    return (ev.clientX - rect.left) * (histCanvas.width / rect.width);
  }
  histCanvas.addEventListener('pointerdown', (ev) => {
    const px = histCanvasPx(ev);
    const { loPx, hiPx } = windowToBandPx(state.window, state.layerMax, histCanvas.width);
    draggingWindowEdge = Math.abs(px - loPx) <= Math.abs(px - hiPx) ? 'lo' : 'hi';
  });
  window.addEventListener('pointerup', () => { draggingWindowEdge = null; });
  window.addEventListener('pointermove', (ev) => {
    if (!draggingWindowEdge) return;
    const px = histCanvasPx(ev);
    const value = bandPxToWindow(px, state.layerMax, histCanvas.width);
    const w = [...state.window];
    if (draggingWindowEdge === 'lo') {
      w[0] = Math.min(value, w[1]);
    } else {
      w[1] = Math.max(value, w[0]);
    }
    commit({ window: w }, { interactive: true }); // continuous drag: fast preview, full render on idle
  });

  const xferCanvas = root.querySelector('#xfer-canvas');
  let draggingStop = null;
  xferCanvas.addEventListener('pointerdown', (ev) => {
    const rect = xferCanvas.getBoundingClientRect();
    const t = (ev.clientX - rect.left) / rect.width;
    draggingStop = state.transferStops.reduce((best, s) =>
      Math.abs(s.t - t) < Math.abs(best.t - t) ? s : best);
  });
  window.addEventListener('pointerup', () => { draggingStop = null; });
  window.addEventListener('pointermove', (ev) => {
    if (!draggingStop) return;
    const rect = xferCanvas.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    draggingStop.t = t;
    commit({ transferStops: state.transferStops }, { interactive: true }); // continuous drag: fast preview, full render on idle
  });

  const axes = { x: 0, y: 1, z: 2 };
  // meta.viewer_crop_xy_m ([[x0,x1],[y0,y1]] metres) sets the initial clip box
  // in x/y. Display-only: the exporter writes it from the site config, the
  // solve box is unchanged. Without it the clip box is left as it is.
  function applyViewerCrop(meta) {
    const crop = meta.viewer_crop_xy_m;
    if (!Array.isArray(crop) || crop.length !== 2) return;
    for (const [idx, axis] of [[0, 'x'], [1, 'y']]) {
      const o = meta.origin_m[idx];
      const ext = meta.shape[idx] * meta.spacing_m;
      const lo = Math.min(1, Math.max(0, (crop[idx][0] - o) / ext));
      const hi = Math.min(1, Math.max(0, (crop[idx][1] - o) / ext));
      state.clipMin[idx] = lo;
      state.clipMax[idx] = hi;
      for (const [id, v] of [[`#clip-${axis}-min`, lo], [`#clip-${axis}-max`, hi]]) {
        const el = root.querySelector(id);
        if (el) el.value = String(v);
        const val = root.querySelector(`${id}-val`);
        if (val) val.textContent = v.toFixed(2);
      }
    }
  }
  for (const axis of Object.keys(axes)) {
    const minInput = root.querySelector(`#clip-${axis}-min`);
    const minVal = root.querySelector(`#clip-${axis}-min-val`);
    minVal.textContent = Number(minInput.value).toFixed(2);
    minInput.addEventListener('input', (ev) => {
      const next = [...state.clipMin];
      next[axes[axis]] = parseFloat(ev.target.value);
      minVal.textContent = Number(ev.target.value).toFixed(2);
      commit({ clipMin: next }, { interactive: true });
    });
    const maxInput = root.querySelector(`#clip-${axis}-max`);
    const maxVal = root.querySelector(`#clip-${axis}-max-val`);
    maxVal.textContent = Number(maxInput.value).toFixed(2);
    maxInput.addEventListener('input', (ev) => {
      const next = [...state.clipMax];
      next[axes[axis]] = parseFloat(ev.target.value);
      maxVal.textContent = Number(ev.target.value).toFixed(2);
      commit({ clipMax: next }, { interactive: true });
    });
  }

  root.querySelector('#slice-axis').addEventListener('change', () => updateSlice(false));
  root.querySelector('#slice-pos').addEventListener('input', () => updateSlice(true));
  function updateSlice(interactive) {
    const axis = root.querySelector('#slice-axis').value;
    const pos = parseFloat(root.querySelector('#slice-pos').value);
    let clipMin, clipMax;
    if (axis === 'none') {
      clipMin = [0, 0, 0];
      clipMax = [1, 1, 1];
    } else {
      const idx = axes[axis];
      const half = 0.02;
      clipMin = [0, 0, 0]; clipMax = [1, 1, 1];
      clipMin[idx] = Math.max(0, pos - half);
      clipMax[idx] = Math.min(1, pos + half);
    }
    commit({ clipMin, clipMax }, { interactive });
  }

  root.querySelector('#clip-plane-enabled').addEventListener('change', (ev) => {
    commit({ clipPlaneEnabled: ev.target.checked });
  });
  const clipPlaneDInput = root.querySelector('#clip-plane-d');
  const clipPlaneDVal = root.querySelector('#clip-plane-d-val');
  clipPlaneDVal.textContent = parseFloat(clipPlaneDInput.value).toFixed(2);
  clipPlaneDInput.addEventListener('input', (ev) => {
    const clipPlaneD = parseFloat(ev.target.value);
    clipPlaneDVal.textContent = clipPlaneD.toFixed(2);
    commit({ clipPlaneD }, { interactive: true });
  });

  root.querySelector('#sigma-gate-enabled').addEventListener('change', (ev) => {
    commit({ sigmaGateEnabled: ev.target.checked });
  });
  root.querySelector('#sigma-gate-value').addEventListener('input', (ev) => {
    const frac = parseFloat(ev.target.value);
    const max = state.sigmaMax || 1;
    commit({ sigmaGateValue: frac * max }, { interactive: true });
  });

  root.querySelector('#coverage-gate-enabled').addEventListener('change', (ev) => {
    commit({ coverageGateEnabled: ev.target.checked });
  });
  root.querySelector('#coverage-gate-value').addEventListener('input', (ev) => {
    const n = parseFloat(ev.target.value);
    if (!Number.isFinite(n)) return;
    commit({ minRays: n }, { interactive: true });
  });

  // The cube-mode grid: at block size 1 the active layer itself (gates run in
  // the shader); above 1 the b^3 block means of the voxels that pass the
  // sigma / coverage / SNR gates (blockAverage), gates then "baked".
  function ensureCubeGrid() {
    const meta = state.meta;
    if (!meta) return null;
    const b = Math.max(1, state.cubeBlock | 0);
    const vgates = stateVoxelGates();
    const key = [b, state.activeLayer, state.loadSeq, vgates.key].join('|');
    if (key === state.cubeGridKey && state.cubeGrid) return state.cubeGrid;
    if (state.cubeGrid && state.cubeGrid.tex) gl.deleteTexture(state.cubeGrid.tex);
    const grid = cubeGrid(state.layerData.get(state.activeLayer), meta, b, vgates.keep);
    grid.tex = grid.baked ? makeVolumeTexture(gl, grid.shape, grid.values) : null;
    state.cubeGrid = grid;
    state.cubeGridKey = key;
    return state.cubeGrid;
  }
  state.ensureCubeGrid = ensureCubeGrid;

  root.querySelector('#cube-size').addEventListener('change', (ev) => {
    commit({ cubeBlock: Math.max(1, parseInt(ev.target.value, 10) || 1) });
  });

  root.querySelector('#render-mode').addEventListener('change', (ev) => {
    commit({ renderMode: ev.target.value === 'cubes' ? 'cubes' : 'fog' });
  });
  const opacityReadout = root.querySelector('#opacity-readout');
  root.querySelector('#opacity').addEventListener('input', (ev) => {
    const v = parseFloat(ev.target.value);
    if (!Number.isFinite(v)) return;
    const opacity = Math.min(1, Math.max(0, v));
    if (opacityReadout) opacityReadout.textContent = `${Math.round(opacity * 100)}%`;
    commit({ opacity }, { interactive: true });
  });
  root.querySelector('#cube-threshold').addEventListener('input', (ev) => {
    const v = parseFloat(ev.target.value);
    if (!Number.isFinite(v)) return;
    commit({ cubeThreshold: v }, { interactive: true });
  });

  root.querySelector('#snr-gate-enabled').addEventListener('change', (ev) => {
    commit({ snrGateEnabled: ev.target.checked });
  });
  root.querySelector('#snr-gate-value').addEventListener('input', (ev) => {
    const n = parseFloat(ev.target.value);
    if (!Number.isFinite(n)) return;
    commit({ minSnr: n }, { interactive: true });
  });

  // Hover picking: client pixel -> NDC -> pickVoxel (picker.mjs), sharing
  // cameraMatrices() with render() so it always agrees with what was drawn.
  function castHoverRay(clientX, clientY) {
    if (!state.meta) return null;
    const rect = canvas.getBoundingClientRect();
    const ndc = [((clientX - rect.left) / rect.width) * 2 - 1, -(((clientY - rect.top) / rect.height) * 2 - 1)];
    const { invViewProj } = cameraMatrices();
    return pickVoxel(ndc, invViewProj, {
      meta: state.meta,
      data: state.layerData.get(state.activeLayer),
      window: state.window,
      renderMode: state.renderMode,
      raySteps: RAY_STEPS,
      minStepVoxels: state.minStepVoxels,
      voxel: stateVoxelGates(),
      spatial: stateSpatialGates(),
      cubeGrid: state.renderMode === 'cubes' ? ensureCubeGrid() : null,
      cubeThreshold: state.cubeThreshold,
    });
  }

  // Test hook: state.pick(x, y) -> {i,j,k,value} | null, exercised directly
  // by Playwright tests without simulating pointermove events.
  state.pick = (x, y) => castHoverRay(x, y);

  canvas.addEventListener('pointermove', (ev) => {
    const hit = castHoverRay(ev.clientX, ev.clientY);
    const el = root.querySelector('#hover-readout');
    const wrapRect = root.querySelector('#canvas-wrap').getBoundingClientRect();
    el.style.left = (ev.clientX - wrapRect.left) + 'px';
    el.style.top = (ev.clientY - wrapRect.top) + 'px';
    el.hidden = !hit;
    el.textContent = hit
      ? `voxel (${hit.i}, ${hit.j}, ${hit.k})  value ${hit.value.toFixed(4)}`
      : '';
  });

  // ---- theme toggle (dark default, persisted in localStorage) ----
  const THEME_KEY = 'cafetomo-viewer:theme';
  const themeToggleBtn = root.querySelector('#theme-toggle');
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme === 'light' ? 'light' : 'dark');
    if (themeToggleBtn) themeToggleBtn.textContent = theme === 'light' ? 'Light' : 'Dark';
  }
  let startTheme = 'dark';
  try { startTheme = localStorage.getItem(THEME_KEY) || 'dark'; } catch { /* ignore */ }
  applyTheme(startTheme);
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const next = currentTheme() === 'light' ? 'dark' : 'light';
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
      render();
    });
  }

  render();
  initDock(root);

  const cmapSel = root.querySelector('#colormap-select');
  for (const name of COLORMAP_NAMES) {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    cmapSel.appendChild(opt);
  }
  let lastCmap = 'viridis';
  try { lastCmap = localStorage.getItem('cafetomo-viewer:colormap') || 'viridis'; } catch { /* ignore */ }
  cmapSel.value = COLORMAP_NAMES.includes(lastCmap) ? lastCmap : 'viridis';
  function applyColormap(name) {
    try { localStorage.setItem('cafetomo-viewer:colormap', name); } catch { /* ignore */ }
    commit({ transferStops: colormapStops(name) });
  }
  cmapSel.addEventListener('change', (ev) => applyColormap(ev.target.value));
  applyColormap(cmapSel.value);

  // ---- shortcuts cheatsheet + onboarding card ----
  const shortcutsTable = root.querySelector('#shortcuts-table');
  if (shortcutsTable) {
    for (const s of SHORTCUTS) {
      const tr = document.createElement('tr');
      const keysTd = document.createElement('td');
      keysTd.className = 'keys';
      keysTd.textContent = s.keys;
      const labelTd = document.createElement('td');
      labelTd.textContent = s.label;
      tr.appendChild(keysTd);
      tr.appendChild(labelTd);
      shortcutsTable.appendChild(tr);
    }
  }

  const ONBOARDED_KEY = 'cafetomo-viewer:onboarded';
  const onboardingOverlay = root.querySelector('#overlay-onboarding');
  const shortcutsOverlay = root.querySelector('#overlay-shortcuts');
  let onboarded = false;
  try { onboarded = localStorage.getItem(ONBOARDED_KEY) === '1'; } catch { /* ignore */ }
  if (onboardingOverlay) onboardingOverlay.hidden = onboarded;

  const onboardingDismissBtn = root.querySelector('#onboarding-dismiss-btn');
  if (onboardingDismissBtn) {
    onboardingDismissBtn.addEventListener('click', () => {
      try { localStorage.setItem(ONBOARDED_KEY, '1'); } catch { /* ignore */ }
      if (onboardingOverlay) onboardingOverlay.hidden = true;
    });
  }

  window.addEventListener('keydown', (ev) => {
    const targetTag = ev.target && ev.target.tagName;
    if (targetTag === 'INPUT' || targetTag === 'SELECT' || targetTag === 'TEXTAREA') return;
    const action = keyToAction(ev);
    if (!action) return;
    if (action.startsWith('layer')) {
      const n = Number(action.slice('layer'.length));
      const radios = root.querySelectorAll('#layer-panel input[type="radio"]');
      const radio = radios[n - 1];
      if (radio) radio.click();
      return;
    }
    if (action.startsWith('preset-')) {
      const key = action.slice('preset-'.length);
      const btn = root.querySelector(`#camera-preset-${key}`);
      if (btn) btn.click();
      return;
    }
    if (action === 'frame-all') {
      frameAll();
      render();
      return;
    }
    if (action === 'toggle-help') {
      if (shortcutsOverlay) shortcutsOverlay.hidden = !shortcutsOverlay.hidden;
      return;
    }
    if (action === 'close-overlay') {
      if (shortcutsOverlay) shortcutsOverlay.hidden = true;
      if (onboardingOverlay) onboardingOverlay.hidden = true;
      return;
    }
  });

  // A self-contained build carries its run inline; load it through the same
  // path as the file picker, after every control above is wired.
  const embedded = embeddedFiles(document);
  if (embedded) loadRunReportingErrors(embedded);

  // Test hook: lets Playwright force a code path (e.g. floatLinear=false to
  // exercise the manual trilinear fallback) and redraw without a UI event.
  state.render = render;
  window.__viewerState = state; // inspected by Playwright tests
}
