/**
 * Talking-head puppet from a scan photo — real 3D.
 *
 * Pass 1 (same technique as Camera-Effects "Sid Eyes"): a thin-plate-spline warp
 *   through MediaPipe landmark control points moves jaw / lips / eyelids / brows
 *   smoothly (no triangle facets). The open-mouth interior is painted on top.
 * Pass 2: the result becomes the texture of a dense grid whose depth comes from a
 *   second spline fitted through the landmarks' 3D z values. The grid is rotated by
 *   (a) the inverse of the photo's own head orientation, so a tilted / turned scan
 *   is frontalized, and (b) the live head motion, then perspective-projected with a
 *   depth buffer — so the head genuinely turns in 3D.
 */

const TASKS_URL = "/static/vendor/mediapipe";
const WASM = `${TASKS_URL}/wasm`;
const MODEL = "/static/vendor/mediapipe/face_landmarker.task";

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const smoothstep = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

// ---- MediaPipe landmark groups -------------------------------------------
const INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
const INNER_LOWER = [95, 88, 178, 87, 14, 317, 402, 318, 324];
const MOUTH_RING = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const OUTER_UPPER = [61, 40, 37, 267, 270, 291];
const OUTER_LOWER = [146, 84, 17, 314, 375];
const LOWER_LIP = new Set([...INNER_LOWER, ...OUTER_LOWER]);
const UPPER_LIP = new Set([...INNER_UPPER.filter((i) => i !== 78 && i !== 308), 61, 40, 37, 267, 270, 291]);
const CORNERS = new Set([61, 291, 78, 308]);

const EYES = [
  { upper: [246, 161, 160, 159, 158, 157, 173], lower: [7, 163, 144, 145, 153, 154, 155], corners: [33, 133], brow: [70, 105, 107, 46, 52, 55] },
  { upper: [466, 388, 387, 386, 385, 384, 398], lower: [249, 390, 373, 374, 380, 381, 382], corners: [263, 362], brow: [300, 334, 336, 276, 282, 285] },
];
const JAW = [152, 148, 176, 149, 150, 136, 172, 58, 377, 400, 378, 379, 365, 397, 288];
const ANCHORS = [168, 6, 4, 10, 67, 297, 151, 234, 454, 93, 323, 127, 356, 205, 425, 50, 280];

const CONTROL_IDX = [
  ...new Set([
    ...INNER_UPPER, ...INNER_LOWER, ...OUTER_UPPER, ...OUTER_LOWER,
    ...EYES.flatMap((e) => [...e.upper, ...e.lower, ...e.corners, ...e.brow]),
    ...JAW, ...ANCHORS,
  ]),
];
const MAXN = 128;

// ---- Landmark detection ---------------------------------------------------
const landmarkerPromises = {};
/**
 * One landmarker per (delegate, running mode). Browsers differ wildly in which combination works
 * (some privacy setups break still-image mode but allow video mode), so buildMeshFromScans tries
 * several. A landmarker that has thrown is poisoned (later calls can hang), so it is discarded.
 */
function getLandmarker(delegate, mode) {
  const key = `${delegate}-${mode}`;
  if (!landmarkerPromises[key]) {
    landmarkerPromises[key] = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import(`${TASKS_URL}/vision_bundle.mjs`);
      const vision = await FilesetResolver.forVisionTasks(WASM);
      const lmk = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL, delegate },
        runningMode: mode,
        numFaces: 1,
        minFaceDetectionConfidence: 0.2,
        minFacePresenceConfidence: 0.2,
        minTrackingConfidence: 0.2,
        outputFacialTransformationMatrixes: true,
      });
      lmk._ts = 0;
      return lmk;
    })();
  }
  return landmarkerPromises[key];
}
function discardLandmarker(delegate, mode) {
  const key = `${delegate}-${mode}`;
  const p = landmarkerPromises[key];
  delete landmarkerPromises[key];
  p?.then((l) => l.close?.()).catch(() => {});
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image failed: " + url));
    img.src = url;
  });
}

/** Head orientation of a landmark set, from its own 3D points (yaw, pitch, roll magnitudes). */
function restOrientation(lm, IW, IH) {
  const P = (i) => [lm[i].x * IW, lm[i].y * IH, -lm[i].z * IW];
  const a = P(33), b = P(263), top = P(10), chin = P(152);
  const X = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const Y = [chin[0] - top[0], chin[1] - top[1], chin[2] - top[2]];
  const nx = Math.hypot(...X) || 1;
  const ny = Math.hypot(...Y) || 1;
  const yaw = Math.asin(clamp(X[2] / nx, -1, 1));
  const pitch = Math.asin(clamp(Y[2] / ny, -1, 1));
  const roll = Math.atan2(X[1], X[0]);
  return { yaw, pitch, roll };
}


/**
 * MediaPipe's detector expects a face well inside the frame; a close-up that fills most of a
 * circular crop often isn't found. Retry on a padded (zoomed-out) copy and map the
 * landmarks back to the original photo's normalized coordinates.
 */
function detectFace(landmarker, mode, img) {
  // Detection never needs full resolution (landmarks are normalized), and big textures hang or
  // fail on some browser setups — so run on small copies only.
  const IW = img.naturalWidth;
  const IH = img.naturalHeight;
  const variants = [
    { size: 384, pad: 1 },
    { size: 384, pad: 2 }, // zoomed out: close-ups fill the frame and confuse the detector
    { size: 256, pad: 1 },
  ];
  for (const { size, pad } of variants) {
    const k = size / Math.max(IW, IH);
    const w = Math.round(IW * k);
    const h = Math.round(IH * k);
    const c = document.createElement("canvas");
    c.width = w * pad;
    c.height = h * pad;
    const x = c.getContext("2d");
    x.fillStyle = "#3a3a3a";
    x.fillRect(0, 0, c.width, c.height);
    x.imageSmoothingQuality = "high";
    x.drawImage(img, (w * (pad - 1)) / 2, (h * (pad - 1)) / 2, w, h);
    let pts = null;
    if (mode === "IMAGE") {
      pts = landmarker.detect(c).faceLandmarks?.[0] || null;
    } else {
      // VIDEO mode tracks across frames: feed the same still a few times with rising timestamps.
      for (let i = 0; i < 3 && !pts; i++) {
        landmarker._ts += 40;
        pts = landmarker.detectForVideo(c, landmarker._ts).faceLandmarks?.[0] || null;
      }
    }
    if (!pts) continue;
    if (pad === 1) return pts;
    const o = (pad - 1) / 2;
    return pts.map((p) => ({ x: p.x * pad - o, y: p.y * pad - o, z: p.z * pad }));
  }
  return null;
}

/** Use landmarks computed on the server (no in-browser detection, which hangs on some browsers). */
export async function meshFromLandmarks(url, landmarks) {
  const img = await loadImage(url);
  return {
    img,
    lm: landmarks.map(([x, y, z]) => ({ x, y, z })),
    w: img.naturalWidth,
    h: img.naturalHeight,
    url,
    delegate: "server",
  };
}

/** Find landmarks in every candidate photo; return the one closest to facing the camera, level. */
export async function buildMeshFromScans(candidates, onProgress) {
  // Only a frame captioned exactly "Center" is the dedicated straight-on shot (not "Center again" etc.).
  const isHero = (c) => /^\s*center\s*$/i.test(c.caption || "");
  const ordered = [...candidates].sort((a, b) => Number(isHero(b)) - Number(isHero(a)));
  const stats = { total: ordered.length, loaded: 0, loadFail: 0, noFace: 0, errors: [], delegate: "" };
  let combos = [["CPU", "IMAGE"], ["CPU", "VIDEO"], ["GPU", "IMAGE"], ["GPU", "VIDEO"]];
  let best = null;
  let n = 0;
  for (const c of ordered) {
    onProgress?.(++n, ordered.length);
    let img;
    try {
      img = await loadImage(c.url);
      stats.loaded++;
    } catch {
      stats.loadFail++;
      continue;
    }
    let pts = null;
    for (const [d, mode] of combos) {
      const label = `${d}/${mode}`;
      if (stats.dead?.has(label)) continue;
      try {
        const lmk = await getLandmarker(d, mode);
        pts = detectFace(lmk, mode, img);
        if (pts) {
          stats.delegate = label;
          combos = [[d, mode], ...combos.filter((c) => c[0] !== d || c[1] !== mode)]; // remember what works
          break;
        }
      } catch (e) {
        const msg = `${label}: ${e?.message || e}`;
        if (!stats.errors.includes(msg)) stats.errors.push(msg);
        console.warn("detect failed", label, e);
        discardLandmarker(d, mode);
        (stats.dead ||= new Set()).add(label);
        onProgress?.(n, ordered.length, `detector error (${msg.slice(0, 80)}) — trying another mode`);
      }
    }
    if (!pts) {
      stats.noFace++;
      continue;
    }
    const o = restOrientation(pts, img.naturalWidth, img.naturalHeight);
    const score = Math.abs(o.yaw) + Math.abs(o.pitch) + 0.5 * Math.abs(o.roll);
    if (!best || score < best.score) best = { img, pts, score, url: c.url };
    if (isHero(c)) break;
  }
  if (!best) {
    const err = new Error(
      `No face found in the scan photos (photos loaded ${stats.loaded}/${stats.total}, no face in ${stats.noFace}` +
        (stats.errors.length ? `, detector errors: ${stats.errors.join("; ").slice(0, 160)}` : "") + ")"
    );
    throw err;
  }
  return { img: best.img, lm: best.pts, w: best.img.naturalWidth, h: best.img.naturalHeight, url: best.url, delegate: stats.delegate };
}

// ---- Thin-plate spline solve (any number of output channels) --------------
const tpsU = (r2) => (r2 > 1e-14 ? 0.5 * r2 * Math.log(r2) : 0);

/**
 * Fit f(c_i) = values[k][i]. Coordinates normalized.
 * @returns {{w: Float64Array, a: Float64Array}|null} w: K per point, a: 3 per channel [const, x, y]
 */
function solveTPS(cx, cy, values, n, lambda) {
  const K = values.length;
  const m = n + 3;
  const cols = m + K;
  const A = new Float64Array(m * cols);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const dx = cx[i] - cx[j];
      const dy = cy[i] - cy[j];
      A[i * cols + j] = tpsU(dx * dx + dy * dy);
    }
    A[i * cols + i] += lambda;
    A[i * cols + n] = 1;
    A[i * cols + n + 1] = cx[i];
    A[i * cols + n + 2] = cy[i];
    A[n * cols + i] = 1;
    A[(n + 1) * cols + i] = cx[i];
    A[(n + 2) * cols + i] = cy[i];
    for (let k = 0; k < K; k++) A[i * cols + m + k] = values[k][i];
  }
  for (let col = 0; col < m; col++) {
    let piv = col;
    let big = Math.abs(A[col * cols + col]);
    for (let r = col + 1; r < m; r++) {
      const v = Math.abs(A[r * cols + col]);
      if (v > big) {
        big = v;
        piv = r;
      }
    }
    if (big < 1e-12) return null;
    if (piv !== col) {
      for (let c = 0; c < cols; c++) {
        const t = A[col * cols + c];
        A[col * cols + c] = A[piv * cols + c];
        A[piv * cols + c] = t;
      }
    }
    const d = A[col * cols + col];
    for (let r = col + 1; r < m; r++) {
      const f = A[r * cols + col] / d;
      if (f === 0) continue;
      for (let c = col; c < cols; c++) A[r * cols + c] -= f * A[col * cols + c];
    }
  }
  const sol = new Float64Array(m * K);
  for (let k = 0; k < K; k++) {
    for (let r = m - 1; r >= 0; r--) {
      let s = A[r * cols + m + k];
      for (let c = r + 1; c < m; c++) s -= A[r * cols + c] * sol[c * K + k];
      sol[r * K + k] = s / A[r * cols + r];
    }
  }
  const w = new Float64Array(n * K);
  for (let i = 0; i < n * K; i++) w[i] = sol[i];
  const a = new Float64Array(3 * K);
  for (let k = 0; k < K; k++) {
    a[k * 3] = sol[n * K + k];
    a[k * 3 + 1] = sol[(n + 1) * K + k];
    a[k * 3 + 2] = sol[(n + 2) * K + k];
  }
  return { w, a };
}

// ---- Shaders ----------------------------------------------------------------
const WARP_VERT = `
attribute vec2 aIn;
uniform vec2 uImg;
uniform float uScale;
varying vec2 vP;
void main() {
  gl_Position = vec4(aIn, 0.0, 1.0);
  vP = vec2(aIn.x * 0.5 + 0.5, 0.5 - aIn.y * 0.5) * uImg / uScale;
}`;

const WARP_FRAG = `
precision highp float;
varying vec2 vP;
uniform sampler2D uTex;
uniform vec2 uImg;
uniform float uScale;
uniform int uN;
uniform vec4 uC[${MAXN}];
uniform vec3 uAu;
uniform vec3 uAv;
void main() {
  float u = uAu.x + uAu.y * vP.x + uAu.z * vP.y;
  float v = uAv.x + uAv.y * vP.x + uAv.z * vP.y;
  for (int i = 0; i < ${MAXN}; i++) {
    if (i >= uN) break;
    vec2 d = vP - uC[i].xy;
    float r2 = dot(d, d);
    float U = r2 > 1e-14 ? 0.5 * r2 * log(r2) : 0.0;
    u += uC[i].z * U;
    v += uC[i].w * U;
  }
  gl_FragColor = texture2D(uTex, vec2(u, v) * uScale / uImg);
}`;

const HEAD_VERT = `
attribute vec2 aIn;
uniform vec2 uImg;
uniform float uScale;
uniform int uN;
uniform vec3 uCz[${MAXN}];
uniform vec3 uAz;
uniform mat3 uM;
uniform vec3 uPivot;
uniform float uCam;
uniform float uOut;
uniform vec2 uOff;
uniform float uDepthRange;
varying vec2 vUV;
void main() {
  vec2 p = aIn * uImg / uScale;
  float z = uAz.x + uAz.y * p.x + uAz.z * p.y;
  for (int i = 0; i < ${MAXN}; i++) {
    if (i >= uN) break;
    vec2 d = p - uCz[i].xy;
    float r2 = dot(d, d);
    z += uCz[i].z * (r2 > 1e-14 ? 0.5 * r2 * log(r2) : 0.0);
  }
  vec3 v = vec3(p * uScale, z * uScale) - uPivot;
  v = uM * v;
  float s = uCam / (uCam - v.z);
  vec2 q = v.xy * s;
  gl_Position = vec4(-q.x * uOut + uOff.x, -q.y * uOut + uOff.y, -v.z / uDepthRange, 1.0);
  vUV = aIn;
}`;

const HEAD_FRAG = `
precision mediump float;
varying vec2 vUV;
uniform sampler2D uTex;
void main() { gl_FragColor = texture2D(uTex, vUV); }`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader error");
  return s;
}
function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "aIn");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || "link error");
  return p;
}


/** True when the photo's four corners are (nearly) black, i.e. it is a circular crop on black. */
function hasBlackCorners(img, IW, IH) {
  try {
    const c = document.createElement("canvas");
    c.width = IW;
    c.height = IH;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.drawImage(img, 0, 0);
    const n = Math.max(4, Math.round(Math.min(IW, IH) * 0.04));
    let dark = 0;
    for (const [px, py] of [[0, 0], [IW - n, 0], [0, IH - n], [IW - n, IH - n]]) {
      const d = x.getImageData(px, py, n, n).data;
      let sum = 0;
      for (let i = 0; i < d.length; i += 4) sum += (d[i] + d[i + 1] + d[i + 2]) / 3;
      if (sum / (d.length / 4) < 14) dark++;
    }
    return dark >= 3;
  } catch {
    return true; // cannot read pixels: keep the old behaviour
  }
}

/**
 * Scan photos are circular crops with black corners. When the head is rotated those
 * corners swing into view, so replace everything outside the circle with a blurred,
 * enlarged copy of the photo itself and feather the real edge into it.
 */
function fillOutsideCircle(img, IW, IH) {
  // Only circular scan crops (black corners) need the fill. An ordinary rectangular photo (a movie
  // still, a drawing) already has real picture there, and blurring it away leaves a smeared band.
  if (!hasBlackCorners(img, IW, IH)) return img;
  const c = document.createElement("canvas");
  c.width = IW;
  c.height = IH;
  const x = c.getContext("2d");
  x.filter = "blur(14px)";
  x.drawImage(img, IW / 2 - IW * 0.8, IH / 2 - IH * 0.8, IW * 1.6, IH * 1.6);
  x.filter = "none";
  const fg = document.createElement("canvas");
  fg.width = IW;
  fg.height = IH;
  const f = fg.getContext("2d");
  f.drawImage(img, 0, 0);
  f.globalCompositeOperation = "destination-in";
  const r = Math.min(IW, IH) / 2;
  const g = f.createRadialGradient(IW / 2, IH / 2, r * 0.975, IW / 2, IH / 2, r * 0.997);
  g.addColorStop(0, "rgba(0,0,0,1)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  f.fillStyle = g;
  f.fillRect(0, 0, IW, IH);
  x.drawImage(fg, 0, 0);
  return c;
}

// ---- 3×3 helpers (row-major) -------------------------------------------------
const mul3 = (A, B) => {
  const R = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) R[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
  return R;
};
const rotX = (a) => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
const rotY = (a) => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
const rotZ = (a) => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];
const transpose3 = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
const colMajor = (A) => new Float32Array([A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]);

/**
 * @param {HTMLCanvasElement} canvas visible canvas
 * @param {Awaited<ReturnType<typeof buildMeshFromScans>>} mesh
 */
export function createMeshPuppet(canvas, mesh) {
  const { img, lm, w: IW, h: IH } = mesh;
  const SCALE = Math.max(IW, IH);
  const k = 640 / SCALE; // flat-texture pixels per image pixel
  const FW = Math.round(IW * k);
  const FH = Math.round(IH * k);
  const OUT = 640; // visible canvas is square
  canvas.width = OUT;
  canvas.height = OUT;
  const ctx = canvas.getContext("2d");

  const flat = document.createElement("canvas"); // pass-1 result + painted mouth
  flat.width = FW;
  flat.height = FH;
  const fctx = flat.getContext("2d");

  const glCanvas = document.createElement("canvas");
  glCanvas.width = OUT;
  glCanvas.height = OUT;
  const gl = glCanvas.getContext("webgl", { alpha: false, antialias: true, depth: true });
  if (!gl) throw new Error("WebGL unavailable");

  const warpProg = program(gl, WARP_VERT, WARP_FRAG);
  const headProg = program(gl, HEAD_VERT, HEAD_FRAG);
  const U = (p, n) => gl.getUniformLocation(p, n);

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

  const GRID = 64;
  const gridV = new Float32Array((GRID + 1) * (GRID + 1) * 2);
  for (let j = 0; j <= GRID; j++) for (let i = 0; i <= GRID; i++) {
    const o = (j * (GRID + 1) + i) * 2;
    gridV[o] = i / GRID;
    gridV[o + 1] = j / GRID;
  }
  const gridI = new Uint16Array(GRID * GRID * 6);
  let q = 0;
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    const a = j * (GRID + 1) + i;
    gridI.set([a, a + 1, a + GRID + 1, a + 1, a + GRID + 2, a + GRID + 1], q);
    q += 6;
  }
  const gridBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
  gl.bufferData(gl.ARRAY_BUFFER, gridV, gl.STATIC_DRAW);
  const gridIdx = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gridIdx);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, gridI, gl.STATIC_DRAW);

  const srcTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, srcTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, fillOutsideCircle(img, IW, IH));
  for (const [p, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, p, v);
  const flatTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, flatTex);
  for (const [p, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, p, v);

  const wU = { img: U(warpProg, "uImg"), scale: U(warpProg, "uScale"), n: U(warpProg, "uN"), c: U(warpProg, "uC"), au: U(warpProg, "uAu"), av: U(warpProg, "uAv"), tex: U(warpProg, "uTex") };
  const hU = { img: U(headProg, "uImg"), scale: U(headProg, "uScale"), n: U(headProg, "uN"), cz: U(headProg, "uCz"), az: U(headProg, "uAz"), m: U(headProg, "uM"), pivot: U(headProg, "uPivot"), cam: U(headProg, "uCam"), out: U(headProg, "uOut"), off: U(headProg, "uOff"), dr: U(headProg, "uDepthRange"), tex: U(headProg, "uTex") };

  // ---- Rest pose (image px; zd = depth toward the viewer) ----
  const L = (i) => ({ x: lm[i].x * IW, y: lm[i].y * IH, z: -lm[i].z * IW });
  const P = {};
  [...new Set([...CONTROL_IDX, 10, 152, 454, 234, 13, 14, 61, 291, 33, 263])].forEach((i) => (P[i] = L(i)));
  const idxs = Object.keys(P);
  const zMean = idxs.reduce((s, i) => s + P[i].z, 0) / idxs.length;
  for (const i of idxs) P[i].z -= zMean;

  // Face frame: X across the eyes, Y chin-ward, Z out of the face.
  const norm = (v) => {
    const l = Math.hypot(...v) || 1;
    return v.map((c) => c / l);
  };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const X = norm([P[263].x - P[33].x, P[263].y - P[33].y, P[263].z - P[33].z]);
  let Yr = [P[152].x - P[10].x, P[152].y - P[10].y, P[152].z - P[10].z];
  const yx = dot(Yr, X);
  const Y = norm([Yr[0] - yx * X[0], Yr[1] - yx * X[1], Yr[2] - yx * X[2]]);
  const Z = [X[1] * Y[2] - X[2] * Y[1], X[2] * Y[0] - X[0] * Y[2], X[0] * Y[1] - X[1] * Y[0]];
  const Rrest = [X[0], Y[0], Z[0], X[1], Y[1], Z[1], X[2], Y[2], Z[2]]; // columns X,Y,Z
  const unfront = transpose3(Rrest); // rotates the photo's head to face the camera, level
  const X2 = norm([X[0], X[1], 0]).slice(0, 2);
  const Y2 = norm([Y[0], Y[1], 0]).slice(0, 2);

  const faceH = Math.abs(P[152].y - P[10].y) || IH * 0.5;
  const faceW = Math.abs(P[454].x - P[234].x) || IW * 0.5;
  const mouthW = Math.abs(P[291].x - P[61].x) || faceW * 0.35;
  const mouthCx = (P[61].x + P[291].x) / 2;
  const mouthCy = (P[13].y + P[14].y) / 2;
  const faceCx = (P[234].x + P[454].x) / 2;
  const faceCy = (P[10].y + P[152].y) / 2;
  const jawDrop = faceH * 0.085;
  const along = (px, py, ox, oy, ax) => (px - ox) * ax[0] + (py - oy) * ax[1];

  const pts = [];
  const addPt = (x, y, z, idx, jawW, fixed = false) => {
    const u = along(x, y, mouthCx, mouthCy, X2);
    const v = along(x, y, mouthCx, mouthCy, Y2);
    const d = Math.hypot(u, v) / (mouthW * 0.9);
    pts.push({ x, y, z, idx, jaw: jawW, mg: Math.exp(-d * d), u, fixed });
  };
  for (const i of CONTROL_IDX) {
    const p = P[i];
    const v = along(p.x, p.y, mouthCx, mouthCy, Y2);
    const side = 1 - smoothstep(faceW * 0.28, faceW * 0.5, Math.abs(along(p.x, p.y, faceCx, faceCy, X2)));
    let wgt = side * smoothstep(-0.02 * faceH, 0.12 * faceH, v);
    if (LOWER_LIP.has(i)) wgt = 1;
    if (UPPER_LIP.has(i)) wgt = 0;
    if (CORNERS.has(i)) wgt = 0.3;
    else if (!LOWER_LIP.has(i)) wgt *= 0.7; // cheeks / chin follow the jaw only partly
    addPt(p.x, p.y, p.z, i, wgt);
  }
  const chin = P[152];
  // Room left in the photo below the chin (in face-heights); anchors must stay INSIDE the picture,
  // otherwise they fight the border pins and smear the bottom edge when the jaw moves.
  const room = Math.max(0.08, (IH - chin.y) / faceH);
  const clampX = (v) => Math.min(IW * 0.985, Math.max(IW * 0.015, v));
  const clampY = (v) => Math.min(IH * 0.985, Math.max(IH * 0.015, v));
  const neck = [[0, Math.min(0.12, room * 0.3), 0.3], [-0.3 * faceW, Math.min(0.1, room * 0.25), 0.08], [0.3 * faceW, Math.min(0.1, room * 0.25), 0.08]];
  neck.forEach(([ox, oy, wj], n) => addPt(clampX(chin.x + X2[0] * ox + Y2[0] * oy * faceH), clampY(chin.y + X2[1] * ox + Y2[1] * oy * faceH), 0, -1 - n, wj));
  // Pin the area below the chin (neck, collar, chest) so only the jaw moves, not the whole lower body.
  [[-0.9, 0.55], [-0.45, 0.55], [0, 0.55], [0.45, 0.55], [0.9, 0.55], [0, 0.9]].forEach(([ox, k], n) =>
    addPt(clampX(chin.x + X2[0] * ox * faceW + Y2[0] * k * room * faceH), clampY(chin.y + X2[1] * ox * faceW + Y2[1] * k * room * faceH), 0, -20 - n, 0, true));
  for (const t of [0, 1 / 3, 2 / 3]) {
    addPt(t * IW, 0, 0, -10, 0, true);
    addPt(IW, t * IH, 0, -10, 0, true);
    addPt((1 - t) * IW, IH, 0, -10, 0, true);
    addPt(0, (1 - t) * IH, 0, -10, 0, true);
  }
  const N = Math.min(pts.length, MAXN);
  const byIdx = new Map(pts.map((p, n) => [p.idx, n]));
  const lowerNear = (idxs2, uWanted) => {
    let best = idxs2[0];
    let bd = 1e9;
    for (const i of idxs2) {
      const d = Math.abs(along(P[i].x, P[i].y, mouthCx, mouthCy, X2) - uWanted);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    return P[best];
  };

  const dst = pts.map((p) => ({ x: p.x, y: p.y }));
  const cxN = new Float64Array(N), cyN = new Float64Array(N);
  const vxN = new Float64Array(N), vyN = new Float64Array(N), vzN = new Float64Array(N);
  const cData = new Float32Array(MAXN * 4);
  const czData = new Float32Array(MAXN * 3);
  for (let n = 0; n < N; n++) {
    vxN[n] = pts[n].x / SCALE;
    vyN[n] = pts[n].y / SCALE;
    vzN[n] = (pts[n].z * 0.35) / SCALE; // much flatter than MediaPipe's raw depth: a rough depth map bends the jaw when the head turns
  }

  function deform(pose) {
    const { open = 0, wide = 0, blinkL = 0, blinkR = 0, brow = 0 } = pose;
    for (let n = 0; n < N; n++) {
      const p = pts[n];
      let x = p.x;
      let y = p.y;
      if (!p.fixed) {
        // Jaw drops along the FACE's down axis; mouth spreads along its sideways axis.
        const dj = open * jawDrop * p.jaw;
        x += Y2[0] * dj;
        y += Y2[1] * dj;
        const dw = p.u * wide * 0.2 * p.mg;
        x += X2[0] * dw;
        y += X2[1] * dw;
        if (CORNERS.has(p.idx)) {
          const lift = open * 0.01 * faceH - Math.max(0, wide) * 0.025 * faceH;
          x += Y2[0] * lift;
          y += Y2[1] * lift;
        }
      }
      dst[n].x = x;
      dst[n].y = y;
    }
    EYES.forEach((eye, e) => {
      const b = e === 0 ? blinkL : blinkR;
      for (const i of eye.brow) {
        const n = byIdx.get(i);
        if (n === undefined || n >= N) continue;
        const dy = -(brow * 0.012 * faceH - b * 0.004 * faceH);
        dst[n].x += Y2[0] * dy;
        dst[n].y += Y2[1] * dy;
      }
    });
  }

  // Skin colour just above each eye (sampled once from the photo) for the painted eyelid.
  const sampleCanvas = document.createElement("canvas");
  sampleCanvas.width = IW;
  sampleCanvas.height = IH;
  const sctx = sampleCanvas.getContext("2d", { willReadFrequently: true });
  sctx.drawImage(img, 0, 0);
  const eyeSkin = EYES.map((eye) => {
    const lid = P[eye.upper[3]];
    const eyeW = Math.hypot(P[eye.corners[0]].x - P[eye.corners[1]].x, P[eye.corners[0]].y - P[eye.corners[1]].y);
    let r = 0, g = 0, bl = 0, c = 0;
    for (const off of [0.07, 0.1, 0.13]) {
      for (const sx of [-0.2, 0, 0.2]) {
        const x = Math.round(lid.x - Y2[0] * off * eyeW + X2[0] * sx * eyeW);
        const y = Math.round(lid.y - Y2[1] * off * eyeW + X2[1] * sx * eyeW);
        if (x < 0 || y < 0 || x >= IW || y >= IH) continue;
        const d = sctx.getImageData(x, y, 1, 1).data;
        r += d[0]; g += d[1]; bl += d[2]; c++;
      }
    }
    c = c || 1;
    return { r: r / c, g: g / c, b: bl / c, eyeW };
  });

  function paintEyelids(blinks) {
    EYES.forEach((eye, e) => {
      const b = blinks[e];
      if (b < 0.04) return;
      const sk = eyeSkin[e];
      const D = (i) => dst[byIdx.get(i)];
      const up = eye.upper.map((i) => D(i));
      const lo = eye.lower.map((i) => D(i));
      const c0 = D(eye.corners[0]);
      const c1 = D(eye.corners[1]);
      const lift = 0.004 * faceH;
      const top = up.map((p) => ({ x: (p.x - Y2[0] * lift) * k, y: (p.y - Y2[1] * lift) * k }));
      const edge = up.map((p, i) => ({
        x: (p.x + (lo[i].x - p.x) * b) * k,
        y: (p.y + (lo[i].y - p.y) * b) * k,
      }));
      fctx.save();
      fctx.beginPath();
      fctx.moveTo(c0.x * k, c0.y * k);
      top.forEach((p) => fctx.lineTo(p.x, p.y));
      fctx.lineTo(c1.x * k, c1.y * k);
      for (let i = edge.length - 1; i >= 0; i--) fctx.lineTo(edge[i].x, edge[i].y);
      fctx.closePath();
      const mid = edge[3];
      const g = fctx.createLinearGradient(top[3].x, top[3].y, mid.x, mid.y);
      g.addColorStop(0, `rgb(${sk.r * 0.9 | 0},${sk.g * 0.87 | 0},${sk.b * 0.86 | 0})`);
      g.addColorStop(1, `rgb(${sk.r * 0.72 | 0},${sk.g * 0.66 | 0},${sk.b * 0.64 | 0})`);
      fctx.fillStyle = g;
      fctx.shadowColor = `rgba(${sk.r * 0.85 | 0},${sk.g * 0.8 | 0},${sk.b * 0.78 | 0},0.9)`;
      fctx.shadowBlur = Math.max(2, sk.eyeW * k * 0.07);
      fctx.fill();
      fctx.shadowBlur = 0;
      // Lash line along the closing edge.
      if (b > 0.3) {
        fctx.strokeStyle = `rgba(28,17,14,${clamp((b - 0.3) / 0.4, 0, 0.92)})`;
        fctx.lineWidth = Math.max(1.6, sk.eyeW * k * 0.06);
        fctx.lineJoin = "round";
        fctx.lineCap = "round";
        fctx.beginPath();
        fctx.moveTo(c0.x * k, c0.y * k);
        edge.forEach((p) => fctx.lineTo(p.x, p.y));
        fctx.lineTo(c1.x * k, c1.y * k);
        fctx.stroke();
      }
      fctx.restore();
    });
  }

  function paintMouth(open) {
    if (open < 0.03) return;
    const ring = MOUTH_RING.map((i) => {
      const d = dst[byIdx.get(i)];
      return { x: d.x * k, y: d.y * k };
    });
    const top = dst[byIdx.get(13)];
    const bot = dst[byIdx.get(14)];
    const topX = top.x * k, topY = top.y * k;
    const gap = ((bot.x - top.x) * Y2[0] + (bot.y - top.y) * Y2[1]) * k;
    const ang = Math.atan2(X2[1], X2[0]);
    fctx.save();
    fctx.beginPath();
    ring.forEach((p, n) => (n ? fctx.lineTo(p.x, p.y) : fctx.moveTo(p.x, p.y)));
    fctx.closePath();
    // Dark cavity, deepest in the middle.
    const g = fctx.createLinearGradient(topX, topY, topX + Y2[0] * gap, topY + Y2[1] * gap);
    g.addColorStop(0, "#1a0709");
    g.addColorStop(0.6, "#2e0e12");
    g.addColorStop(1, "#3d1418");
    fctx.fillStyle = g;
    fctx.fill();
    fctx.clip();
    // inner shadow so the opening reads as depth, not a flat cut-out
    fctx.strokeStyle = "rgba(12,2,3,0.7)";
    fctx.lineWidth = Math.max(3, mouthW * k * 0.05);
    fctx.lineJoin = "round";
    fctx.stroke();
    fctx.translate(topX, topY);
    fctx.rotate(ang);
    const halfW = mouthW * k * 0.5;
    // Tongue: small, low, only when the mouth is open enough.
    if (gap > faceH * k * 0.05) {
      fctx.fillStyle = "rgba(150,70,78,0.85)";
      fctx.beginPath();
      fctx.ellipse(0, gap * 0.95, halfW * 0.42, gap * 0.3, 0, 0, Math.PI * 2);
      fctx.fill();
    }
    // Upper teeth: a band that follows the curve of the upper lip, narrowing to nothing at the corners.
    // (Drawn in image-plane coordinates again, not the rotated frame.)
    fctx.rotate(-ang);
    fctx.translate(-topX, -topY);
    if (open > 0.12 && gap > faceH * k * 0.025) {
      const upIdx = INNER_UPPER; // 78 … 13 … 308, left → right
      const base = upIdx.map((i) => {
        const d = dst[byIdx.get(i)];
        return { x: d.x * k, y: d.y * k };
      });
      const th = Math.min(gap * 0.32, faceH * k * 0.026);
      const band = base.map((p, i) => {
        const t = i / (base.length - 1);
        const taper = Math.pow(Math.sin(Math.PI * t), 0.7) * 0.9;
        return { x: p.x + Y2[0] * th * taper, y: p.y + Y2[1] * th * taper };
      });
      const at = (arr, f) => {
        const x = clamp(f, 0, 1) * (arr.length - 1);
        const i0 = Math.floor(x);
        const i1 = Math.min(arr.length - 1, i0 + 1);
        const w = x - i0;
        return { x: arr[i0].x + (arr[i1].x - arr[i0].x) * w, y: arr[i0].y + (arr[i1].y - arr[i0].y) * w };
      };
      fctx.save();
      fctx.beginPath();
      base.forEach((p, i) => (i ? fctx.lineTo(p.x, p.y) : fctx.moveTo(p.x, p.y)));
      for (let i = band.length - 1; i >= 0; i--) fctx.lineTo(band[i].x, band[i].y);
      fctx.closePath();
      const mid = base[5];
      const tg = fctx.createLinearGradient(mid.x, mid.y, mid.x + Y2[0] * th, mid.y + Y2[1] * th);
      tg.addColorStop(0, "rgba(120,104,92,0.95)"); // shadow under the lip
      tg.addColorStop(0.25, "rgba(214,203,186,0.95)");
      tg.addColorStop(0.8, "rgba(226,216,200,0.95)");
      tg.addColorStop(1, "rgba(190,176,160,0.9)");
      fctx.fillStyle = tg;
      fctx.shadowColor = "rgba(20,5,5,0.55)";
      fctx.shadowBlur = Math.max(1, th * 0.25);
      fctx.fill();
      fctx.shadowBlur = 0;
      fctx.clip();
      // Gaps between teeth (centre pair a touch wider, like real incisors).
      fctx.strokeStyle = "rgba(96,78,68,0.55)";
      fctx.lineWidth = Math.max(0.8, th * 0.07);
      const TEETH = 8;
      for (let j = 1; j < TEETH; j++) {
        const f = 0.1 + (0.8 * j) / TEETH;
        const p0 = at(base, f);
        const p1 = at(band, f);
        fctx.beginPath();
        fctx.moveTo(p0.x, p0.y);
        fctx.lineTo(p1.x, p1.y);
        fctx.stroke();
      }
      fctx.restore();
    }
    fctx.restore();
  }

  const pivot = [faceCx, faceCy, -0.3 * faceW];
  const camD = 6 * faceW; // long lens = little perspective distortion
  // Show the WHOLE photo circle (hair, ears, chin), centred on the photo — not a tight crop of the
  // face. Zoom in only if the face would otherwise be tiny.
  // Circular scan crops (black corners) keep the old framing. An ordinary rectangular picture is
  // zoomed in so the circle is completely filled with picture, centred lower on the face/chin.
  const circular = hasBlackCorners(img, IW, IH);
  const scaleOut = circular
    ? Math.max(OUT / (0.97 * Math.min(IW, IH)), (0.45 * OUT) / faceW)
    : Math.max(OUT / (0.8 * Math.min(IW, IH)), (0.62 * OUT) / faceW);
  const viewR = OUT / 2 / scaleOut;
  let vcx = faceCx;
  let vcy = faceCy + (circular ? 0 : faceH * 0.1);
  if (circular) {
    // The circular view must stay inside the photo's solid circle: no further from the photo
    // centre than the spare room allows.
    const spare = Math.max(0, 0.9 * (Math.min(IW, IH) / 2) - viewR);
    const dxv = vcx - IW / 2;
    const dyv = vcy - IH / 2;
    const dv = Math.hypot(dxv, dyv);
    if (dv > spare && dv > 0) {
      vcx = IW / 2 + (dxv / dv) * spare;
      vcy = IH / 2 + (dyv / dv) * spare;
    }
  } else {
    // Keep the whole circle inside the picture rectangle so no edge or smear shows.
    vcx = Math.min(Math.max(vcx, viewR), IW - viewR);
    vcy = Math.min(Math.max(vcy, viewR), IH - viewR);
  }

  function render(pose = {}) {
    deform(pose);
    for (let n = 0; n < N; n++) {
      cxN[n] = dst[n].x / SCALE;
      cyN[n] = dst[n].y / SCALE;
    }
    const sol = solveTPS(cxN, cyN, [vxN, vyN, vzN], N, 1e-6);
    if (!sol) return;
    for (let n = 0; n < N; n++) {
      cData[n * 4] = cxN[n];
      cData[n * 4 + 1] = cyN[n];
      cData[n * 4 + 2] = sol.w[n * 3];
      cData[n * 4 + 3] = sol.w[n * 3 + 1];
      czData[n * 3] = cxN[n];
      czData[n * 3 + 1] = cyN[n];
      czData[n * 3 + 2] = sol.w[n * 3 + 2];
    }

    // ---- Pass 1: spline warp of the photo into the flat texture ----
    gl.viewport(0, 0, FW, FH);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(warpProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1i(wU.tex, 0);
    gl.uniform2f(wU.img, IW, IH);
    gl.uniform1f(wU.scale, SCALE);
    gl.uniform1i(wU.n, N);
    gl.uniform4fv(wU.c, cData);
    gl.uniform3f(wU.au, sol.a[0], sol.a[1], sol.a[2]);
    gl.uniform3f(wU.av, sol.a[3], sol.a[4], sol.a[5]);
    gl.clearColor(0.05, 0.07, 0.09, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // GL's viewport origin is bottom-left, so the FW×FH render sits in the canvas's bottom-left corner.
    fctx.setTransform(1, 0, 0, 1, 0, 0);
    fctx.drawImage(glCanvas, 0, OUT - FH, FW, FH, 0, 0, FW, FH);
    paintEyelids([pose.blinkL || 0, pose.blinkR || 0]);
    paintMouth(pose.open || 0);

    // ---- Pass 2: texture a depth-displaced grid and rotate it in 3D ----
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, flatTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, flat);
    gl.viewport(0, 0, OUT, OUT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(headProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gridIdx);
    const { yaw = 0, pitch = 0, roll = 0, dx = 0, dy = 0, zoom = 1 } = pose;
    const M = mul3(mul3(rotZ(roll), mul3(rotX(pitch), rotY(yaw))), unfront);
    gl.uniform1i(hU.tex, 1);
    gl.uniform2f(hU.img, IW, IH);
    gl.uniform1f(hU.scale, SCALE);
    gl.uniform1i(hU.n, N);
    gl.uniform3fv(hU.cz, czData);
    gl.uniform3f(hU.az, sol.a[6], sol.a[7], sol.a[8]);
    gl.uniformMatrix3fv(hU.m, false, colMajor(M));
    gl.uniform3f(hU.pivot, pivot[0], pivot[1], pivot[2]);
    gl.uniform1f(hU.cam, camD);
    gl.uniform1f(hU.out, ((2 * scaleOut) / OUT) * zoom);
    const uo = ((2 * scaleOut) / OUT) * zoom;
    gl.uniform2f(hU.off, (2 * dx * scaleOut) / OUT - (faceCx - vcx) * uo, (-2 * dy * scaleOut) / OUT - (faceCy - vcy) * uo);
    gl.uniform1f(hU.dr, 2 * faceW);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawElements(gl.TRIANGLES, gridI.length, gl.UNSIGNED_SHORT, 0);

    ctx.drawImage(glCanvas, 0, 0);
  }

  render({});
  return { render, restOrientation: restOrientation(lm, IW, IH) };
}
