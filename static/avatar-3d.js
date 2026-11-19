/**
 * The scanned face as a real 3D head.
 *
 *  - Face  : the 478 face landmarks as a 3D mesh, textured with skin_stitched.png (the seamless skin built from the scan).
 *  - Eyes  : real eyeballs (sclera, iris, pupil) behind eyelid openings; the lids close to blink.
 *  - Mouth : a dark cavity with upper and lower teeth, so an open mouth looks like a mouth.
 *  - Head  : a skull shell (front textured from the front-facing scan photo, back in the hair colour), ears from the photo, and a neck.
 *
 * Animation uses the `pose` values the avatar already produces: {open, wide, blinkL, blinkR, brow, yaw, pitch, roll, zoom}.
 * Files written by the face scan: <scan folder>/skin_atlas.json and skin_stitched.png.
 */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
const INNER_LOWER = [95, 88, 178, 87, 14, 317, 402, 318, 324];
const MOUTH_RING = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const OUTER_LOWER = [146, 84, 17, 314, 375];
const LOWER_LIP = new Set([...INNER_LOWER, ...OUTER_LOWER]);
const UPPER_LIP = new Set([...INNER_UPPER.filter((i) => i !== 78 && i !== 308), 61, 40, 37, 267, 270, 291]);
const CORNERS = new Set([61, 291, 78, 308]);
const UPPER_ORDER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
const LOWER_ORDER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
const EYES = [
  { upper: [246, 161, 160, 159, 158, 157, 173], lower: [7, 163, 144, 145, 153, 154, 155], brow: [70, 105, 107, 46, 52, 55], corners: [33, 133], iris: [468, 469, 470, 471, 472] },
  { upper: [466, 388, 387, 386, 385, 384, 398], lower: [249, 390, 373, 374, 380, 381, 382], brow: [300, 334, 336, 276, 282, 285], corners: [263, 362], iris: [473, 474, 475, 476, 477] },
];
const smooth = (a, b, x) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function loadTexture(url) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(url, resolve, undefined, () => reject(new Error("texture not found: " + url)));
  });
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("image not found: " + url));
    im.src = url;
  });
}

/** Average colour (0..1 RGB) of a small square in a photo; u, v are 0..1 of the photo's width and height. */
function makeSampler(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return (u, v, rPx, lumMin = 20, lumMax = 235) => {
    const cx = Math.round(u * c.width);
    const cy = Math.round(v * c.height);
    const r = Math.max(2, Math.round(rPx));
    const x0 = clamp(cx - r, 0, c.width - 1);
    const y0 = clamp(cy - r, 0, c.height - 1);
    const w = clamp(2 * r, 1, c.width - x0);
    const h = clamp(2 * r, 1, c.height - y0);
    const d = ctx.getImageData(x0, y0, w, h).data;
    let R = 0, G = 0, B = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) {
      const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
      if (l < lumMin || l > lumMax) continue;
      R += d[i];
      G += d[i + 1];
      B += d[i + 2];
      n++;
    }
    return n ? [R / n / 255, G / n / 255, B / n / 255] : null;
  };
}

/** A sphere whose vertex colours draw an eye: sclera, limbal ring, iris, pupil and a small highlight. */
function makeEyeball(r, iris) {
  const g = new THREE.SphereGeometry(r, 72, 54);
  const p = g.attributes.position;
  const col = new Float32Array(p.count * 3);
  const irisA = 0.5;
  const pupilA = 0.19;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i) / r;
    const y = p.getY(i) / r;
    const z = p.getZ(i) / r;
    const ang = Math.acos(clamp(z, -1, 1));
    let c;
    if (ang < pupilA) c = [0.02, 0.02, 0.025];
    else if (ang < irisA) {
      const t = (ang - pupilA) / (irisA - pupilA); // 0 at the pupil edge, 1 at the limbus
      const k = 1.15 - 0.55 * t * t; // lighter near the pupil, darker toward the rim
      c = [iris[0] * k, iris[1] * k, iris[2] * k];
      if (t > 0.88) c = c.map((v) => v * 0.45); // dark limbal ring
    } else c = [0.93, 0.9, 0.87];
    const shade = 0.84 + 0.16 * clamp(z, 0, 1) + 0.0 * (x + y);
    // small glint up and to the side
    const gd = Math.hypot(x - 0.2, y - 0.24, z - 0.94);
    if (gd < 0.1) c = [1, 1, 1];
    col[i * 3] = c[0] * (gd < 0.1 ? 1 : shade);
    col[i * 3 + 1] = c[1] * (gd < 0.1 ? 1 : shade);
    col[i * 3 + 2] = c[2] * (gd < 0.1 ? 1 : shade);
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true }));
}

/** A dynamic strip of quads between two rows of points (used for the teeth). */
function makeStrip(n, color) {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 2 * 3);
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
  const idx = [];
  for (let i = 0; i < n - 1; i++) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  g.setIndex(idx);
  return { geo: g, pos, mesh: new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide })) };
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {string} base  URL folder of the scan, e.g. /data/media/jeff_molofee/scan
 * @param {{url:string, landmarks:number[][], w:number, h:number}|null} hero  the front-facing scan photo and its landmarks
 */
export async function createSkinHeadPuppet(canvas, base, hero = null) {
  const stamp = Date.now();
  const [layout, tex, heroImg] = await Promise.all([
    fetch(`${base}/skin_atlas.json?t=${stamp}`).then((r) => {
      if (!r.ok) throw new Error("no scan layout");
      return r.json();
    }),
    loadTexture(`${base}/skin_stitched.png?t=${stamp}`),
    hero?.url ? loadImage(hero.url).catch(() => null) : Promise.resolve(null),
  ]);
  tex.flipY = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;

  // ---- 3D landmark positions: from the front-facing photo if we have it (same picture as the head shell), else from the scan ----
  let L;
  let ar = 1;
  if (hero?.landmarks?.length >= 478 && hero.w && hero.h) {
    ar = hero.w / hero.h;
    L = hero.landmarks.slice(0, 478).map((p) => [p[0], p[1] / ar, p[2]]);
  } else if (layout.neutral3d) L = layout.neutral3d;
  else throw new Error("scan layout has no 3D points (rescan once)");
  const heroPhoto = heroImg && hero?.landmarks?.length >= 478 ? heroImg : null;

  const allTris = layout.triangles;
  const NV = layout.points.length;
  const nReal = 478;
  const N0 = new Float32Array(NV * 3);
  for (let i = 0; i < nReal; i++) {
    N0[i * 3] = L[i][0];
    N0[i * 3 + 1] = L[i][1];
    N0[i * 3 + 2] = L[i][2] * 0.85;
  }
  const centreOf = new Map();
  for (const t of allTris)
    for (const v of t)
      if (v >= nReal) {
        if (!centreOf.has(v)) centreOf.set(v, new Set());
        for (const o of t) if (o < nReal) centreOf.get(v).add(o);
      }
  const setCentres = (arr) => {
    for (const [c, set] of centreOf) {
      let x = 0, y = 0, z = 0;
      for (const o of set) {
        x += arr[o * 3];
        y += arr[o * 3 + 1];
        z += arr[o * 3 + 2];
      }
      const n = set.size || 1;
      arr[c * 3] = x / n;
      arr[c * 3 + 1] = y / n;
      arr[c * 3 + 2] = z / n;
    }
  };
  setCentres(N0);
  // Eye openings are real holes (the eyeballs show through); everything else becomes skin.
  const skinTris = allTris.filter((t) => t.every((v) => v < nReal));

  // ---- normalise: face centred, 1.0 tall (forehead 10 to chin 152) ----
  const faceHU = Math.abs(N0[152 * 3 + 1] - N0[10 * 3 + 1]) || 1;
  let cxU = 0, cyU = 0, czU = 0;
  for (let i = 0; i < nReal; i++) {
    cxU += N0[i * 3];
    cyU += N0[i * 3 + 1];
    czU += N0[i * 3 + 2];
  }
  cxU /= nReal;
  cyU /= nReal;
  czU /= nReal;
  for (let i = 0; i < NV; i++) {
    N0[i * 3] = (N0[i * 3] - cxU) / faceHU;
    N0[i * 3 + 1] = (N0[i * 3 + 1] - cyU) / faceHU;
    N0[i * 3 + 2] = (N0[i * 3 + 2] - czU) / faceHU;
  }
  const faceW = Math.abs(N0[454 * 3] - N0[234 * 3]) || 0.8;
  const faceCx = (N0[454 * 3] + N0[234 * 3]) / 2;
  const mouthY = (N0[13 * 3 + 1] + N0[14 * 3 + 1]) / 2;
  const mouthCx = (N0[61 * 3] + N0[291 * 3]) / 2;
  const mouthW = Math.abs(N0[291 * 3] - N0[61 * 3]) || 0.35;

  // ---- colours sampled from the front photo (hair, skin, iris) ----
  let hairRGB = [0.12, 0.09, 0.07];
  let skinRGB = [0.78, 0.58, 0.48];
  const irisRGB = [[0.30, 0.19, 0.11], [0.30, 0.19, 0.11]];
  let photoUV = null; // maps a normalised face position (X, Y up) to the photo's u, v
  if (heroPhoto) {
    const samp = makeSampler(heroPhoto);
    const W = heroPhoto.naturalWidth;
    const lmP = hero.landmarks;
    skinRGB = samp((lmP[205][0] + lmP[425][0]) / 2, (lmP[205][1] + lmP[425][1]) / 2, W * 0.015, 40, 245) || skinRGB;
    const topY = lmP[10][1] - 0.075 * (hero.w / hero.h > 1 ? 1 : 1);
    hairRGB = samp(lmP[10][0], clamp(topY, 0.03, 0.97), W * 0.018, 5, 140) || hairRGB;
    EYES.forEach((e, k) => {
      const ctr = lmP[e.iris[0]];
      const rr = Math.abs(lmP[e.iris[1]][0] - lmP[e.iris[3]][0]) * W * 0.5;
      const c = samp(ctr[0], ctr[1], Math.max(3, rr * 0.55), 25, 105);
      if (c && Math.max(...c) < 0.34) {
        irisRGB[k] = c.map((v) => clamp(v * 1.1, 0.03, 0.5));
      }
    });
    photoUV = (X, Y) => [X * faceHU + cxU, (-Y * faceHU + cyU) * ar];
  }

  // ---- per-vertex jaw weights ----
  const jawW = new Float32Array(NV);
  const mouthG = new Float32Array(NV);
  for (let i = 0; i < nReal; i++) {
    const x = N0[i * 3];
    const y = N0[i * 3 + 1];
    const side = 1 - smooth(faceW * 0.3, faceW * 0.52, Math.abs(x - faceCx));
    let w = side * smooth(-0.004, 0.03, y - mouthY); // the lip band and everything below it move rigidly with the jaw (no stretched lip)
    if (LOWER_LIP.has(i)) w = 1;
    if (UPPER_LIP.has(i)) w = 0;
    if (CORNERS.has(i)) w = 0.3;
    jawW[i] = w;
    const d = Math.hypot(x - mouthCx, y - mouthY) / (mouthW * 0.9);
    mouthG[i] = Math.exp(-d * d);
  }

  // ---- portrait framing (forehead..chin + room for hair/ears) ----
  const topY3 = -N0[10 * 3 + 1];
  const chinY3 = -N0[152 * 3 + 1];
  const headH = Math.max(0.55, chinY3 - topY3);
  const portraitAr = hero?.w && hero?.h ? hero.w / hero.h : ar || 0.75;
  const portraitH = headH * 1.72;
  const portraitW = portraitH * portraitAr;
  const portraitCy = (topY3 + chinY3) / 2 + headH * 0.1;

  // ---- three.js scene ----
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const size = Math.max(200, canvas.clientWidth || 320);
  renderer.setSize(size, size, false);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.05, 50);
  const lookY = (topY3 + chinY3) / 2;
  camera.position.set(0, lookY + 0.02, 2.45);
  camera.lookAt(0, lookY, 0);
  camera.updateProjectionMatrix();
  const group = new THREE.Group();
  scene.add(group);

  const pos = new Float32Array(NV * 3);
  const uv = new Float32Array(NV * 2);
  for (let i = 0; i < NV; i++) {
    uv[i * 2] = layout.points[i][0];
    uv[i * 2 + 1] = layout.points[i][1];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos.slice(), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(skinTris.flat());

  // three.js-space helpers for the fixed (neutral) face
  const T = (i) => [N0[i * 3], -N0[i * 3 + 1], -N0[i * 3 + 2]];
  const zNasion = T(168)[2];
  const zContour = (T(234)[2] + T(454)[2]) / 2;
  const topY = topY3;
  const chinY = chinY3;
  const portraitZ = zContour - 0.22;

  // ---- full portrait plane: real hair, ears, and shoulders from the center scan photo ----
  let heroTex = null;
  if (heroPhoto) {
    heroTex = new THREE.Texture(heroPhoto);
    heroTex.flipY = false;
    heroTex.colorSpace = THREE.SRGBColorSpace;
    heroTex.wrapS = heroTex.wrapT = THREE.ClampToEdgeWrapping;
    heroTex.needsUpdate = true;
    const portrait = new THREE.Mesh(
      new THREE.PlaneGeometry(portraitW, portraitH),
      new THREE.MeshBasicMaterial({ map: heroTex, transparent: true, depthWrite: true, side: THREE.FrontSide }),
    );
    portrait.position.set(0, portraitCy, portraitZ);
    portrait.renderOrder = 0;
    group.add(portrait);
  }

  const faceMat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    alphaTest: 0.42,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });
  const faceMesh = new THREE.Mesh(geo, faceMat);
  faceMesh.renderOrder = 2;
  faceMesh.position.z = portraitZ + 0.018;
  group.add(faceMesh);

  // ---- head shell: sides/back fill when the head rotates (no separate front texture if we have the portrait) ----
  const rx = faceW * 0.6;
  const ry = 0.52;
  const rz = 0.6;
  const shellCy = topY * 0.72 + chinY * 0.28 + 0.02;
  const shellCz = Math.min(zNasion, zContour + 0.1) - 0.04 - rz * 0.9;
  const shellMat = (map, color) =>
    new THREE.MeshBasicMaterial({ map: map || null, color: map ? 0xffffff : color, polygonOffset: true, polygonOffsetFactor: 3, polygonOffsetUnits: 3 });
  if (!heroPhoto) {
    heroTex = null;
  }
  const back = new THREE.SphereGeometry(1, 56, 40, Math.PI, Math.PI, 0, Math.PI);
  {
    const p = back.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const X = p.getX(i) * rx;
      const Y = p.getY(i) * ry + shellCy;
      p.setXYZ(i, X, Y, p.getZ(i) * rz + shellCz);
    }
    p.needsUpdate = true;
  }
  group.add(new THREE.Mesh(back, shellMat(null, new THREE.Color(...hairRGB))));
  if (!heroPhoto) {
    const front = new THREE.SphereGeometry(1, 56, 40, 0, Math.PI, 0, Math.PI);
    const p = front.attributes.position;
    const uvs = new Float32Array(p.count * 2);
    for (let i = 0; i < p.count; i++) {
      const X = p.getX(i) * rx;
      const Y = p.getY(i) * ry + shellCy;
      p.setXYZ(i, X, Y, p.getZ(i) * rz + shellCz);
    }
    front.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    p.needsUpdate = true;
    group.add(new THREE.Mesh(front, shellMat(null, new THREE.Color(...skinRGB))));
  }

  // neck: tapered, and fading out toward the bottom (no hard cut-off)
  const neckTopY = chinY + 0.1;
  const neckGeo = new THREE.CylinderGeometry(faceW * 0.25, faceW * 0.3, 0.6, 28, 6, true);
  {
    const np = neckGeo.attributes.position;
    const nc = new Float32Array(np.count * 4);
    const nb = [skinRGB[0] * 0.78, skinRGB[1] * 0.76, skinRGB[2] * 0.74];
    for (let i = 0; i < np.count; i++) {
      const t = (np.getY(i) + 0.3) / 0.6; // 0 at the bottom, 1 at the top
      nc.set([nb[0], nb[1], nb[2], clamp(t * 1.6 - 0.1, 0, 1)], i * 4);
    }
    neckGeo.setAttribute("color", new THREE.BufferAttribute(nc, 4));
  }
  const neck = new THREE.Mesh(neckGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, side: THREE.DoubleSide }));
  neck.position.set(0, neckTopY - 0.3, shellCz + 0.3);
  if (!heroPhoto) group.add(neck);

  // ---- eyes: real eyeballs behind the lid openings ----
  const eyeballs = EYES.map((e, k) => {
    const ci = [...centreOf.keys()][k] ?? null;
    const lc = ci !== null ? T(ci) : T(e.corners[0]);
    const eyeW = Math.abs(N0[e.corners[0] * 3] - N0[e.corners[1] * 3]) || 0.18;
    const r = eyeW * 0.52;
    const mesh = makeEyeball(r, irisRGB[k]);
    mesh.position.set(lc[0], lc[1], lc[2] - r * 1.04);
    group.add(mesh);
    return mesh;
  });

  // ---- mouth: cavity and teeth ----
  const cavPos = new Float32Array((MOUTH_RING.length + 1) * 3);
  const cavGeo = new THREE.BufferGeometry();
  cavGeo.setAttribute("position", new THREE.BufferAttribute(cavPos, 3).setUsage(THREE.DynamicDrawUsage));
  const cavIdx = [];
  for (let i = 0; i < MOUTH_RING.length; i++) cavIdx.push(MOUTH_RING.length, i, (i + 1) % MOUTH_RING.length);
  cavGeo.setIndex(cavIdx);
  group.add(new THREE.Mesh(cavGeo, new THREE.MeshBasicMaterial({ color: 0x1b0708, side: THREE.DoubleSide })));
  const tongue = makeStrip(LOWER_ORDER.length, 0xa8454a);
  group.add(tongue.mesh);
  const upperTeeth = makeStrip(UPPER_ORDER.length, 0xe4dfd3);
  const lowerTeeth = makeStrip(LOWER_ORDER.length, 0xd6d0c4);
  group.add(upperTeeth.mesh, lowerTeeth.mesh);

  // Jaw hinge (the joint in front of the ear): the jaw swings on it, so the chin and lower lip travel on an arc, down and back.
  const hingeY = (N0[234 * 3 + 1] + N0[454 * 3 + 1]) / 2 + 0.05;
  const hingeZ = (N0[234 * 3 + 2] + N0[454 * 3 + 2]) / 2;
  const LIPRING = new Set([...MOUTH_RING, 61, 291, 40, 37, 267, 270, 146, 84, 17, 314, 375, 0, 185, 183, 42, 41, 38, 12, 268, 271, 272, 407, 408, 409]);
  function deform(pose) {
    let { open = 0, wide = 0, blinkL = 0, blinkR = 0, brow = 0, press = 0 } = pose;
    // wide < 0 is a rounded "oo" mouth, wide > 0 is a spread "ee" mouth
    const funnel = clamp(Math.max(0, -wide), 0, 1);
    const spread = clamp(Math.max(0, wide), 0, 1);
    press = clamp(press, 0, 1);
    open = clamp(open * (1 - press), 0, 1);
    const theta = open * 0.3; // jaw rotation about the hinge, up to ~17 degrees
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    for (let i = 0; i < nReal; i++) {
      let x = N0[i * 3];
      let y = N0[i * 3 + 1];
      let z = N0[i * 3 + 2];
      const w = jawW[i];
      if (w > 0 && theta > 0) {
        const dy = y - hingeY;
        const dz = z - hingeZ;
        const ry2 = hingeY + dy * ct - dz * st;
        const rz2 = hingeZ + dy * st + dz * ct;
        y += (ry2 - y) * w;
        z += (rz2 - z) * w;
      }
      const g = mouthG[i];
      // spread ("ee"): mouth widens; funnel ("oo"): mouth narrows and the lips push forward
      x = mouthCx + (x - mouthCx) * (1 + 0.2 * spread * g - 0.38 * funnel * g);
      if (LIPRING.has(i)) z -= 0.055 * funnel * g;
      if (CORNERS.has(i)) y -= spread * 0.022 + open * 0.008;
      if (UPPER_LIP.has(i)) y -= open * 0.012;
      if (LOWER_LIP.has(i)) y += open * 0.02;
      // pressed lips ("mm", "b", "p"): upper lip comes down, lower lip comes up, and they meet
      if (press > 0) {
        if (UPPER_LIP.has(i)) y += 0.006 * press;
        if (LOWER_LIP.has(i)) y -= 0.012 * press;
      }
      pos[i * 3] = x;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = z;
    }
    EYES.forEach((eye, e) => {
      const b = clamp(e === 0 ? blinkL : blinkR, 0, 1);
      eye.upper.forEach((u, k) => {
        const l = eye.lower[k];
        const gap = pos[l * 3 + 1] - pos[u * 3 + 1];
        pos[u * 3 + 1] -= gap * 0.45; // resting eyes wider open than the half-closed scan photo
        pos[l * 3 + 1] += gap * 0.1;
        pos[u * 3 + 1] += (pos[l * 3 + 1] - pos[u * 3 + 1]) * b * 1.02;
      });
      for (const i of eye.brow) pos[i * 3 + 1] -= brow * 0.012 - b * 0.004;
    });
    const a = geo.attributes.position.array;
    for (let i = 0; i < nReal; i++) {
      a[i * 3] = pos[i * 3];
      a[i * 3 + 1] = -pos[i * 3 + 1];
      a[i * 3 + 2] = -pos[i * 3 + 2];
    }
    geo.attributes.position.needsUpdate = true;

    // mouth cavity
    let mx = 0, my = 0, mz = 0;
    MOUTH_RING.forEach((v, k) => {
      cavPos[k * 3] = a[v * 3];
      cavPos[k * 3 + 1] = a[v * 3 + 1];
      cavPos[k * 3 + 2] = a[v * 3 + 2] - 0.035;
      mx += cavPos[k * 3];
      my += cavPos[k * 3 + 1];
      mz += cavPos[k * 3 + 2];
    });
    const n = MOUTH_RING.length;
    cavPos[n * 3] = mx / n;
    cavPos[n * 3 + 1] = my / n;
    cavPos[n * 3 + 2] = mz / n - 0.02;
    cavGeo.attributes.position.needsUpdate = true;

    // upper teeth hang from the upper lip, lower teeth rise from the lower lip: thin, with smooth edges, so the dark mouth shows
    const gap = Math.max(0, a[13 * 3 + 1] - a[14 * 3 + 1]);
    const hUp = clamp(gap * 0.3, 0, 0.024);
    const hLow = gap > 0.055 ? clamp(gap * 0.14, 0, 0.014) : 0;
    const meanY = (order) => order.reduce((acc, v) => acc + a[v * 3 + 1], 0) / order.length;
    const upBase = meanY(UPPER_ORDER.slice(1, -1));
    const loBase = meanY(LOWER_ORDER.slice(1, -1));
    UPPER_ORDER.forEach((v, k) => {
      const x = a[v * 3];
      const z = a[v * 3 + 2] - 0.012;
      const edge = Math.abs(k - (UPPER_ORDER.length - 1) / 2) / ((UPPER_ORDER.length - 1) / 2);
      const top = Math.min(a[v * 3 + 1], upBase + 0.01);
      upperTeeth.pos.set([x, top, z, x, top - (top - (upBase - hUp)) * (1 - edge * edge), z], k * 6);
    });
    LOWER_ORDER.forEach((v, k) => {
      const x = a[v * 3];
      const edge = Math.abs(k - (LOWER_ORDER.length - 1) / 2) / ((LOWER_ORDER.length - 1) / 2);
      const bot = Math.max(a[v * 3 + 1], loBase - 0.01);
      lowerTeeth.pos.set([x, bot, a[v * 3 + 2] - 0.016, x, bot + (loBase + hLow - bot) * (1 - edge * edge), a[v * 3 + 2] - 0.016], k * 6);
      const th = clamp(gap * 0.42, 0, 0.055) * (1 - edge * edge);
      tongue.pos.set([x, bot, a[v * 3 + 2] - 0.022, x, loBase + th, a[v * 3 + 2] - 0.022], k * 6);
    });
    tongue.geo.attributes.position.needsUpdate = true;
    tongue.mesh.visible = gap > 0.045;
    upperTeeth.geo.attributes.position.needsUpdate = true;
    lowerTeeth.geo.attributes.position.needsUpdate = true;
    upperTeeth.mesh.visible = gap > 0.012;
    lowerTeeth.mesh.visible = hLow > 0;
  }

  const resize = () => {
    const w = Math.max(200, canvas.clientWidth || 320);
    const h = Math.max(200, canvas.clientHeight || w);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  const ro = new ResizeObserver(resize);
  if (canvas.parentElement) ro.observe(canvas.parentElement);

  // gentle eye movement: look at the camera, with small saccades
  const gaze = { x: 0, y: 0, tx: 0, ty: 0, next: 0 };
  return {
    debug: { zNasion, zContour, topY, chinY, shellCy, shellCz, faceW, hairRGB, skinRGB, irisRGB },
    render(pose = {}) {
      deform(pose);
      const now = performance.now() / 1000;
      if (now > gaze.next) {
        gaze.tx = (Math.random() - 0.5) * 0.18;
        gaze.ty = (Math.random() - 0.5) * 0.1;
        gaze.next = now + 1.2 + Math.random() * 2.8;
      }
      gaze.x += (gaze.tx - gaze.x) * 0.2;
      gaze.y += (gaze.ty - gaze.y) * 0.2;
      const yaw = clamp(-(pose.yaw || 0), -0.6, 0.6);
      const pitch = clamp(pose.pitch || 0, -0.45, 0.45);
      eyeballs.forEach((m) => m.rotation.set(-pitch * 0.6 + gaze.y, -yaw * 0.6 + gaze.x, 0, "YXZ"));
      group.rotation.set(pitch, yaw, clamp(pose.roll || 0, -0.3, 0.3), "YXZ");
      group.position.y = 0;
      group.scale.setScalar(pose.zoom || 1);
      renderer.render(scene, camera);
    },
    dispose() {
      ro.disconnect();
      geo.dispose();
      cavGeo.dispose();
      tex.dispose();
      heroTex?.dispose();
      renderer.dispose();
    },
  };
}
