/**
 * The person's photo: upload one, or use the webcam. Either way the server finds the face, levels it, centres it, zooms and crops it to
 * a square, and says in plain words whether the photo is good (and why not when it is not).
 *
 * Webcam: the page watches your face live (MediaPipe, in the browser) and takes a picture by itself whenever you look straight at the
 * camera with your eyes open, held still and in focus. At the end the server keeps only the best one.
 */
const MP = "/static/vendor/mediapipe";
const MAX_SNAPS = 6;
const SNAP_GAP_MS = 700;
const TARGET_FACE_PX = 280; // what the zoom aims for
const MIN_FACE_PX = 170; // forehead to chin, in camera pixels: about what you get sitting normally at a desk (the server accepts 120+ and prefers 200+)
const GOOD_FRAMES = 3; // checks must hold this many frames in a row before a picture is taken
const GIVE_UP_HINT_MS = 30000;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** An uploaded file -> a JPEG data URL, no larger than 3200 px on the long side (keeps the upload reasonable). */
async function fileToDataUrl(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const k = Math.min(1, 3200 / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * k);
  c.height = Math.round(bmp.height * k);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.93);
}

export async function runPhotoStep(root, canonicalName, api) {
  const hb = document.getElementById("logout-btn");
  if (hb) hb.style.display = "none"; // this step has its own Cancel
  const wrap = document.createElement("section");
  wrap.className = "card photo-card";
  let stream = null;
  let landmarker = null;
  let raf = 0;
  let finished = false;
  let resolveStep = () => {};

  function stopCamera() {
    cancelAnimationFrame(raf);
    raf = 0;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  function leave(value) {
    if (finished) return;
    finished = true;
    stopCamera();
    try { landmarker?.close(); } catch { /* ignore */ }
    if (hb) hb.style.display = "";
    resolveStep(value);
  }
  function show(html) {
    wrap.innerHTML = html;
  }

  // ---------------------------------------------------------------- choose
  async function choose() {
    stopCamera();
    let photos = [];
    let profile = null;
    try {
      const got = await (await fetch(`/api/photo/list?canonical_name=${encodeURIComponent(canonicalName)}`)).json();
      photos = got.photos || [];
      profile = got.profile || null;
    } catch { /* no gallery */ }
    const gallery = photos.length
      ? `<p class="photo-label">Your photos. Click one to use it for your avatar.</p>
         <div class="photo-gallery">${photos
           .map(
             (p) => `<div class="photo-thumb${p.active ? " active" : ""}" data-name="${esc(p.name)}" title="${p.active ? "In use" : "Use this photo"}">
               <img src="${esc(p.url)}" alt="Your photo" />${p.active ? '<span class="photo-tag">In use</span>' : ""}<button type="button" class="photo-del" data-del="${esc(p.name)}" title="Delete this photo"></button>
             </div>`
           )
           .join("")}</div>`
      : "";
    show(`
      <h2>Your Photo</h2>
      ${gallery}
      <p class="photo-label">${photos.length ? "Add another photo" : "Add a photo of your face"}</p>
      <div class="photo-options">
        <button type="button" class="photo-option" id="ph-cam"><strong>Use your webcam</strong><span>Look at the camera. It takes the pictures by itself when you are looking straight ahead, eyes open and in focus, and keeps the best.</span></button>
        <button type="button" class="photo-option" id="ph-up"><strong>Upload a photo</strong><span>Choose a picture of your face, or drop one here. It is centred, zoomed and cropped for you.</span></button>
      </div>
      <input type="file" id="ph-file" accept="image/*" hidden />
      <div class="step-actions step-actions--row"><button type="button" class="btn-ghost" id="ph-cancel">${photos.length ? "Done" : "Cancel"}</button></div>`);
    const file = wrap.querySelector("#ph-file");
    wrap.querySelector("#ph-cancel").onclick = () => leave(photos.length ? { profile } : null);
    wrap.querySelector("#ph-cam").onclick = webcam;
    wrap.querySelector("#ph-up").onclick = () => file.click();
    file.onchange = () => file.files[0] && uploaded(file.files[0]);
    wrap.ondragover = (e) => e.preventDefault();
    wrap.ondrop = (e) => {
      e.preventDefault();
      const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
      if (f) uploaded(f);
    };
    wrap.querySelectorAll(".photo-thumb").forEach((el) => {
      el.onclick = async (e) => {
        if (e.target.closest(".photo-del")) return;
        await api("/api/photo/select", { canonical_name: canonicalName, name: el.dataset.name });
        choose();
      };
    });
    wrap.querySelectorAll(".photo-del").forEach((el) => {
      el.onclick = async () => {
        const inUse = el.closest(".photo-thumb")?.classList.contains("active");
        if (!confirm(inUse ? "Delete the photo your avatar is using? Another photo will be used instead (or you'll need to add one)." : "Delete this photo?")) return;
        await api("/api/photo/delete", { canonical_name: canonicalName, name: el.dataset.del });
        choose();
      };
    });
  }

  // ---------------------------------------------------------------- upload
  async function uploaded(file) {
    show(`<h2>Your Photo</h2><p class="photo-label">Looking at the photo…</p>`);
    try {
      const url = await fileToDataUrl(file);
      review(await api("/api/photo/check", { images: [url] }));
    } catch (err) {
      review({ status: "error", message: "That file could not be opened as a picture." });
    }
  }

  // ---------------------------------------------------------------- review
  function review(r) {
    if (r.status !== "ok") {
      show(`<h2>Your Photo</h2><p class="photo-verdict bad">${esc(r.message || "Something went wrong.")}</p>
        <div class="step-actions step-actions--row"><button type="button" class="btn-primary" id="ph-again">Try another</button></div>`);
      wrap.querySelector("#ph-again").onclick = choose;
      return;
    }
    const head = { good: "This photo looks good.", usable: "This photo will work, but could be better.", bad: "This photo can't be used." }[r.verdict];
    const list = r.issues.map((i) => `<li class="${i.level}">${esc(i.text)}</li>`).join("");
    show(`
      <h2>Your Photo</h2>
      <div class="photo-review">
        <div class="photo-preview">${r.image ? `<img src="${esc(r.image)}" alt="Prepared photo" />` : '<div class="photo-none">No face found</div>'}</div>
        <div class="photo-side">
          <p class="photo-verdict ${r.verdict}">${head}${r.count > 1 ? ` (the best of ${r.count} pictures)` : ""}</p>
          ${list ? `<ul class="photo-issues">${list}</ul>` : ""}
          <p class="photo-hint">${r.image ? "Your face was found, levelled, centred, zoomed and cropped." : ""}</p>
          <div class="step-actions step-actions--row">
            <button type="button" class="btn-ghost" id="ph-again">Try another</button>
            <button type="button" class="btn-primary" id="ph-save" ${r.ok ? "" : "disabled"}>Save</button>
          </div>
        </div>
      </div>`);
    wrap.querySelector("#ph-again").onclick = choose;
    const save = wrap.querySelector("#ph-save");
    save.onclick = async () => {
      save.disabled = true;
      const out = await api("/api/photo/save", { canonical_name: canonicalName, image: r.image });
      if (out.status === "ok") leave(out);
      else {
        alert(out.message || "Could not save the photo.");
        save.disabled = false;
      }
    };
  }

  // ---------------------------------------------------------------- webcam
  const CHECKS = [
    ["face", "Face found"],
    ["size", "Close enough"],
    ["centre", "Centred"],
    ["straight", "Looking straight at the camera"],
    ["eyes", "Eyes open"],
    ["steady", "Held still and in focus"],
    ["light", "Good light"],
  ];

  async function loadTracker(status) {
    const { FaceLandmarker, FilesetResolver } = await import(`${MP}/vision_bundle.mjs`);
    const vision = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
    let lastErr = null;
    for (const delegate of ["GPU", "CPU"]) {
      status(`Loading face tracking (${delegate})…`);
      try {
        return await Promise.race([
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: `${MP}/face_landmarker.task`, delegate },
            runningMode: "VIDEO",
            numFaces: 2,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
            minFaceDetectionConfidence: 0.4,
            minFacePresenceConfidence: 0.4,
            minTrackingConfidence: 0.4,
          }),
          new Promise((_, rej) => setTimeout(() => rej(new Error("timed out")), 25000)),
        ]);
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error("face tracking could not start");
  }

  async function webcam() {
    show(`
      <h2>Your Photo</h2>
      <div class="photo-cam">
        <div class="photo-video"><video id="ph-video" playsinline muted autoplay></video><div class="photo-oval"></div><div class="photo-flash" id="ph-flash"></div></div>
        <div class="photo-side">
          <p class="photo-status" id="ph-status">Starting the camera…</p>
          <ul class="photo-checks" id="ph-checks">${CHECKS.map(([k, t]) => `<li data-k="${k}"><span class="dot"></span>${t}</li>`).join("")}</ul>
          <p class="photo-hint" id="ph-count">No pictures yet.</p>
          <div class="step-actions step-actions--row">
            <button type="button" class="btn-ghost" id="ph-cancel">Cancel</button>
            <button type="button" class="btn-ghost" id="ph-now" hidden>Take it now</button>
            <button type="button" class="btn-primary" id="ph-finish" disabled>Done</button>
          </div>
        </div>
      </div>`);
    const video = wrap.querySelector("#ph-video");
    const status = wrap.querySelector("#ph-status");
    const countEl = wrap.querySelector("#ph-count");
    const btnFinish = wrap.querySelector("#ph-finish");
    const btnNow = wrap.querySelector("#ph-now");
    const flash = wrap.querySelector("#ph-flash");
    wrap.querySelector("#ph-cancel").onclick = choose;

    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 3840 }, height: { ideal: 2160 } }, audio: false });
    } catch {
      status.textContent = "The camera is blocked or not available. Allow camera access, or upload a photo instead.";
      return;
    }
    video.srcObject = stream;
    await video.play().catch(() => {});
    // Cameras that can zoom (many webcams can) are zoomed in for you, so you do not have to lean toward the screen.
    const track = stream.getVideoTracks()[0];
    const zoomCap = track?.getCapabilities?.().zoom || null;
    let zoom = zoomCap ? zoomCap.min : 1;
    let lastZoomAt = 0;
    // keep the picture in focus by itself, where the camera allows it
    if ((track?.getCapabilities?.().focusMode || []).includes?.("continuous")) track.applyConstraints({ advanced: [{ focusMode: "continuous" }] }).catch(() => {});
    function autoZoom(fhPx, vh) {
      if (!zoomCap || performance.now() - lastZoomAt < 700) return;
      const want = fhPx < TARGET_FACE_PX * 0.85 ? 1.12 : fhPx > TARGET_FACE_PX * 1.6 || (fhPx > 0.62 * vh) ? 1 / 1.12 : 1;
      if (want === 1) return;
      const next = Math.max(zoomCap.min, Math.min(zoomCap.max, zoom * want));
      if (Math.abs(next - zoom) < 1e-3) return;
      zoom = next;
      lastZoomAt = performance.now();
      track.applyConstraints({ advanced: [{ zoom }] }).catch(() => {});
    }
    try {
      landmarker = await loadTracker((m) => (status.textContent = m));
    } catch (e) {
      status.textContent = "Face tracking could not start. You can upload a photo instead.";
      return;
    }

    const snaps = [];
    const started = performance.now();
    let good = 0;
    let lastSnap = 0;
    let lastTime = -1;
    let prevNose = null;
    const sharpHist = [];
    const small = document.createElement("canvas");
    small.width = small.height = 160;
    const sctx = small.getContext("2d", { willReadFrequently: true });

    /** Sharpness (variance of the Laplacian) and brightness of the face, from a small copy of it. */
    function faceStats(box) {
      sctx.filter = "blur(1.2px)"; // webcam noise looks like detail to a sharpness test; a light blur leaves only real edges
      sctx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, 160, 160);
      const d = sctx.getImageData(0, 0, 160, 160).data;
      const g = new Float32Array(160 * 160);
      let sum = 0;
      for (let i = 0; i < g.length; i++) {
        g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
        sum += g[i];
      }
      let s = 0;
      let s2 = 0;
      let n = 0;
      for (let y = 1; y < 159; y++) {
        for (let x = 1; x < 159; x++) {
          const i = y * 160 + x;
          const l = g[i - 1] + g[i + 1] + g[i - 160] + g[i + 160] - 4 * g[i];
          s += l;
          s2 += l * l;
          n++;
        }
      }
      return { sharp: s2 / n - (s / n) ** 2, bright: sum / g.length };
    }

    function snap(force = false) {
      const c = document.createElement("canvas");
      c.width = video.videoWidth;
      c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
      snaps.push({ url: c.toDataURL("image/jpeg", 0.93) });
      lastSnap = performance.now();
      flash.classList.add("on");
      setTimeout(() => flash.classList.remove("on"), 120);
      countEl.textContent = `${snaps.length} picture${snaps.length === 1 ? "" : "s"} taken${snaps.length < MAX_SNAPS ? ". Keep looking at the camera." : "."}`;
      btnFinish.disabled = false;
      if (snaps.length >= MAX_SNAPS || force) finish();
    }

    async function finish() {
      if (finished || !snaps.length) return;
      stopCamera();
      show(`<h2>Your Photo</h2><p class="photo-label">Choosing the best of ${snaps.length}…</p>`);
      try {
        review(await api("/api/photo/check", { images: snaps.map((s) => s.url) }));
      } catch {
        review({ status: "error", message: "Could not check the pictures." });
      }
    }
    btnFinish.onclick = finish;
    btnNow.onclick = () => snap(true);

    function setChecks(state, hint) {
      for (const [k] of CHECKS) {
        const li = wrap.querySelector(`#ph-checks [data-k="${k}"]`);
        if (li) li.className = state[k] === undefined ? "" : state[k] ? "ok" : "no";
      }
      status.textContent = hint;
    }

    function tick() {
      if (finished || !stream) return;
      raf = requestAnimationFrame(tick);
      if (video.readyState < 2 || video.currentTime === lastTime) return;
      lastTime = video.currentTime;
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      let res;
      try {
        res = landmarker.detectForVideo(video, performance.now());
      } catch {
        return;
      }
      const lm = res?.faceLandmarks?.[0];
      const st = {};
      if (!lm) {
        good = 0;
        setChecks({ face: false }, "I can't see your face. Look at the camera.");
        return;
      }
      st.face = res.faceLandmarks.length === 1;
      // size and position (landmark coordinates are 0-1 across the picture)
      const top = lm[10].y;
      const chin = lm[152].y;
      const fhPx = (chin - top) * vh;
      const cx = (lm[234].x + lm[454].x) / 2;
      const cy = (top + chin) / 2;
      autoZoom(fhPx, vh);
      st.size = fhPx >= MIN_FACE_PX && chin - top <= 0.8;
      const inside = Math.min(...lm.map((p) => p.x)) > 0.02 && Math.max(...lm.map((p) => p.x)) < 0.98 && top > 0.02 && chin < 0.98;
      st.centre = Math.abs(cx - 0.5) < 0.12 && Math.abs(cy - 0.5) < 0.15 && inside;
      // head turn (degrees) from the face transform
      const m = res.facialTransformationMatrixes?.[0]?.data;
      let yaw = 0;
      let pitch = 0;
      if (m) {
        yaw = (Math.atan2(m[8], m[0]) * 180) / Math.PI;
        pitch = (Math.asin(Math.max(-1, Math.min(1, -m[9]))) * 180) / Math.PI;
      }
      const roll = (Math.atan2((lm[263].y - lm[33].y) * vh, (lm[263].x - lm[33].x) * vw) * 180) / Math.PI;
      st.straight = Math.abs(yaw) < 8 && Math.abs(pitch) < 10 && Math.abs(roll) < 8;
      const bs = Object.fromEntries((res.faceBlendshapes?.[0]?.categories || []).map((c) => [c.categoryName, c.score]));
      const blink = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) / 2;
      st.eyes = blink < 0.3 && (bs.jawOpen || 0) < 0.3;
      // focus / steadiness / light on the face itself
      const x0 = Math.max(0, Math.min(...lm.map((p) => p.x)) * vw);
      const x1 = Math.min(vw, Math.max(...lm.map((p) => p.x)) * vw);
      const y0 = Math.max(0, top * vh);
      const y1 = Math.min(vh, chin * vh);
      const fs = faceStats({ x: x0, y: y0, w: Math.max(8, x1 - x0), h: Math.max(8, y1 - y0) });
      sharpHist.push(fs.sharp);
      if (sharpHist.length > 45) sharpHist.shift();
      const best = Math.max(...sharpHist);
      const move = prevNose ? Math.hypot((lm[1].x - prevNose.x) * vw, (lm[1].y - prevNose.y) * vh) / (fhPx || 1) : 1;
      prevNose = { x: lm[1].x, y: lm[1].y };
      st.steady = move < 0.012 && fs.sharp >= 0.85 * best;
      st.light = fs.bright > 60 && fs.bright < 215;

      const order = ["face", "size", "centre", "straight", "eyes", "steady", "light"];
      const failing = order.find((k) => !st[k]);
      const hints = {
        face: "Only one face should be in view.",
        size: fhPx < MIN_FACE_PX ? "Move a little closer." : "Move back a little.",
        centre: "Move to the middle of the picture, with your whole head in view.",
        straight: "Look straight at the camera and level your head.",
        eyes: "Open your eyes and relax your mouth.",
        steady: "Hold still for a moment.",
        light: fs.bright <= 60 ? "Add some light on your face." : "Too much light on your face.",
      };
      setChecks(st, failing ? hints[failing] : "Perfect. Hold it…");
      if (failing) {
        good = 0;
      } else {
        good += 1;
        const now = performance.now();
        if (good >= GOOD_FRAMES && now - lastSnap > SNAP_GAP_MS && snaps.length < MAX_SNAPS) snap();
      }
      if (!snaps.length && performance.now() - started > GIVE_UP_HINT_MS) btnNow.hidden = false;
    }
    status.textContent = "Look straight at the camera.";
    raf = requestAnimationFrame(tick);
  }

  return new Promise((resolve) => {
    resolveStep = resolve;
    root.innerHTML = "";
    root.appendChild(wrap);
    choose();
  });
}
