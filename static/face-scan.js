import {
  CENTER_SLOT,
  RING_SEGMENTS,
  TOTAL_SLOTS,
  estimateHeadPose,
  facingLabel,
  facingVector,
  scoreCapture,
  segmentCaption,
  rawCompass,
  segmentFromNoseOffset,
  WEDGE_ORIGIN,
  WEDGE_STEP,
} from "/static/face-pose.js?v=25";

const HOLD_FRAMES = 2;
const MIN_SLOTS = TOTAL_SLOTS; // Done only when every direction is captured
const CAPTURE_PX = 640; // output size of each saved wedge frame (was ~250 px)
const FRONTAL_PX = 1024; // straight-on source image for the talking head
const BASELINE_FRAMES = 30; // also gives auto-centering time to settle before it is locked
const YAW_SCALE = 0.3; // raw units of head turn that count as "full" deflection
const PITCH_SCALE = 0.1;
const CENTER_RADIUS = 0.45;
const CAPTURE_INTERVAL_MS = 120;
const QUALITY_BUMP = 0.002;

export async function runFaceScan(root, canonicalName, api) {
  let stream = null;
  let framing = null;
  let finishing = false;
  let settled = false;
  let captureBusy = false;
  let framesSeen = 0;

  /** @type {Array<{jpeg:string, quality:number, caption:string, jitter:number}|null>} */
  const slots = Array(TOTAL_SLOTS).fill(null);
  let activeSeg = -1;
  let holdCount = 0;
  let lastCaptureAt = 0;
  let frontal = null;
  const baseline = { sx: 0, sy: 0, n: 0, x: 0, y: 0, ready: false };

  const wrap = document.createElement("section");
  wrap.className = "card face-scan-card";
  wrap.innerHTML = `
    <h2>Face scan</h2>
    <p class="scan-hint" id="scan-hint">Starting camera…</p>
    <p class="scan-progress" id="scan-progress">0 of ${TOTAL_SLOTS}</p>
    <p class="scan-live" id="scan-live" aria-live="polite"></p>
    <div class="scan-stage" id="scan-stage">
      <svg class="scan-ring" viewBox="0 0 200 200" aria-hidden="true">
        ${Array.from({ length: RING_SEGMENTS }, (_, i) => {
          const a1 = WEDGE_ORIGIN + i * WEDGE_STEP;
          const a2 = a1 + WEDGE_STEP;
          const r = 92;
          const cx = 100,
            cy = 100;
          const x1 = cx + r * Math.cos(a1);
          const y1 = cy + r * Math.sin(a1);
          const x2 = cx + r * Math.cos(a2);
          const y2 = cy + r * Math.sin(a2);
          return `<path class="ring-seg" data-seg="${i}" d="M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2} Z" />`;
        }).join("")}
      </svg>
      <div class="scan-center-mark" id="scan-center-mark" aria-hidden="true"></div>
      <div class="scan-video-wrap" id="scan-video-wrap">
        <video id="scan-video" playsinline muted autoplay></video>
      </div>
    </div>
    <p class="scan-step-label" id="scan-step-label">Facing: —</p>
    <div class="btn-row">
      <button type="button" class="btn-ghost" id="scan-cancel">Back</button>
      <button type="button" class="btn-primary" id="scan-done" disabled>Done scanning</button>
    </div>
  `;

  const video = wrap.querySelector("#scan-video");
  const videoWrap = wrap.querySelector("#scan-video-wrap");
  const scanStage = wrap.querySelector("#scan-stage");
  const centerMark = wrap.querySelector("#scan-center-mark");
  const hint = wrap.querySelector("#scan-hint");
  const liveEl = wrap.querySelector("#scan-live");
  const progressEl = wrap.querySelector("#scan-progress");
  const stepLabel = wrap.querySelector("#scan-step-label");
  const btnCancel = wrap.querySelector("#scan-cancel");
  const btnDone = wrap.querySelector("#scan-done");
  const segs = wrap.querySelectorAll(".ring-seg");

  function filledCount() {
    return slots.filter(Boolean).length;
  }

  function refreshProgress() {
    const n = filledCount();
    progressEl.textContent =
      n >= TOTAL_SLOTS
        ? `${n} of ${TOTAL_SLOTS} — keep moving to improve each shot`
        : `${n} of ${TOTAL_SLOTS}`;
    btnDone.disabled = n < MIN_SLOTS || finishing;
    if (!framing?.isModelReady) {
      hint.textContent =
        "Loading face tracking (first time may take 10–20s)… Video is live; tracking starts when ready.";
    } else if (!baseline.ready) {
      hint.textContent = "Look straight at the camera for a moment…";
    } else if (n < TOTAL_SLOTS) {
      hint.textContent = "Slowly turn your head toward each unfilled wedge (up, right, down, left and the diagonals).";
    } else {
      hint.textContent = `All ${TOTAL_SLOTS} saved — keep moving to improve each shot, then tap Done.`;
    }
  }

  function updateRing(current) {
    segs.forEach((seg, i) => {
      seg.classList.remove("active", "done");
      if (slots[i]) seg.classList.add("done");
      else if (current === i) seg.classList.add("active");
    });
    const centerDone = Boolean(slots[CENTER_SLOT]);
    centerMark.classList.toggle("done", centerDone);
    centerMark.classList.toggle("active", current === CENTER_SLOT && !centerDone);
    scanStage.classList.toggle("center-pose", current === CENTER_SLOT);
    refreshProgress();
  }

  function maybeStore(seg, jpeg, quality, jitter) {
    if (!jpeg) return false;
    const prev = slots[seg];
    const sharper =
      prev &&
      quality >= prev.quality - 0.004 &&
      jitter < prev.jitter - 0.015;
    if (prev && quality <= prev.quality + QUALITY_BUMP && !sharper) return false;

    slots[seg] = {
      jpeg,
      quality,
      jitter,
      caption: segmentCaption(seg),
    };
    updateRing(seg);
    return true;
  }

  function queueCapture(seg, box, jitter, mag) {
    if (captureBusy || settled || !framing) return;
    captureBusy = true;
    const quality = scoreCapture(box, jitter, mag, seg);
    requestAnimationFrame(() => {
      try {
        const jpeg = framing.captureFramedJpeg(0.92, CAPTURE_PX);
        maybeStore(seg, jpeg, quality, jitter);
      } finally {
        captureBusy = false;
      }
    });
  }

  function onFrame(lm, matrix) {
    if (!lm) {
      liveEl.textContent = `No face detected — tilt back toward the camera a little (frames seen: ${framesSeen}, calibrated: ${baseline.ready ? "yes" : baseline.n + "/" + BASELINE_FRAMES})`;
      return;
    }
    framesSeen += 1;

    const { box } = estimateHeadPose(lm, matrix);
    const raw = rawCompass(lm);

    // Calibrate "neutral" from the first frames while the user looks at the camera.
    if (!baseline.ready) {
      // Wait until the picture has centered and zoomed on the face, hold still, then calibrate.
      if (!framing?.isSettled?.()) {
        baseline.sx = 0;
        baseline.sy = 0;
        baseline.n = 0;
        hint.textContent = "Centering your face — look at the camera and hold still…";
        liveEl.textContent = "Centering…";
        return;
      }
      baseline.sx += raw.x;
      baseline.sy += raw.y;
      baseline.n += 1;
      hint.textContent = "Look straight at the camera for a moment…";
      liveEl.textContent = `Calibrating ${baseline.n}/${BASELINE_FRAMES}`;
      if (baseline.n >= BASELINE_FRAMES) {
        baseline.x = baseline.sx / baseline.n;
        baseline.y = baseline.sy / baseline.n;
        baseline.ready = true;
        // Face is centered, zoomed and facing the camera right now: grab the hero frame.
        frontal = { jpeg: framing?.captureFramedJpeg?.(0.95, FRONTAL_PX) ?? null };
        framing?.lockFrame?.(); // freeze pan/zoom so the picture stays put while you turn
        refreshProgress();
      }
      return;
    }

    const cx = Math.max(-1.5, Math.min(1.5, (raw.x - baseline.x) / YAW_SCALE));
    const cy = Math.max(-1.5, Math.min(1.5, (raw.y - baseline.y) / PITCH_SCALE));
    const mag = Math.hypot(cx, cy);
    const seg = segmentFromNoseOffset(cx, cy, CENTER_RADIUS);
    const yaw = cx;
    const pitch = cy;
    const jitter = framing?.getJitter?.() ?? 0;

    if (seg === CENTER_SLOT) {
      activeSeg = seg;
      holdCount = 0;
      updateRing(seg);
      stepLabel.textContent = "Facing: Center — turn toward a wedge";
      liveEl.textContent = `Tracking · frame ${framesSeen} · x ${yaw.toFixed(2)} y ${pitch.toFixed(2)}`;
      return;
    }

    if (seg === activeSeg) holdCount += 1;
    else {
      activeSeg = seg;
      holdCount = 0;
    }

    updateRing(seg);

    stepLabel.textContent = `Facing: ${facingLabel(seg)}${
      slots[seg] ? ` · quality ${(slots[seg].quality * 100) | 0}%` : ""
    }`;

    liveEl.textContent = `Tracking · frame ${framesSeen} · compass yaw ${yaw.toFixed(2)} pitch ${pitch.toFixed(2)} (move head — pitch should change when you look up/down)`;

    const refining = Boolean(slots[seg]);
    const jitterCap = refining ? 0.45 : 1;
    const steadyEnough =
      holdCount >= (refining ? HOLD_FRAMES : 1) && jitter < jitterCap;
    const now = performance.now();
    const due = now - lastCaptureAt >= CAPTURE_INTERVAL_MS;

    if (steadyEnough && due && !captureBusy) {
      lastCaptureAt = now;
      queueCapture(seg, box, jitter, mag);
    }
  }

  let resolveScan = null;

  function completeAndExit(value) {
    if (settled) return;
    settled = true;
    stopCamera();
    resolveScan?.(value);
  }

  async function finishScan() {
    if (finishing || settled) return null;
    finishing = true;
    btnDone.disabled = true;

    const frames = slots
      .map((s, i) =>
        s
          ? {
              image_base64: s.jpeg,
              caption: s.caption,
              angle_index: i,
              quality_score: s.quality,
            }
          : null
      )
      .filter(Boolean);

    if (frames.length < MIN_SLOTS) {
      alert(`Need at least ${MIN_SLOTS} angles — only got ${frames.length}. Keep turning slowly.`);
      finishing = false;
      btnDone.disabled = filledCount() < MIN_SLOTS;
      return null;
    }

    // Straight-on "Center" frame (sharpest of the calibration frames) — the source
    // image for the talking-head avatar. Index TOTAL_SLOTS keeps it out of the wedge set.
    if (frontal?.jpeg) {
      frames.push({
        image_base64: frontal.jpeg,
        caption: "Center",
        angle_index: TOTAL_SLOTS,
        quality_score: 1,
      });
    }

    hint.textContent = "Saving your best angles…";
    liveEl.textContent = "Saving (camera off until save finishes)…";
    stopCamera();
    try {
      const result = await api("/api/capture/scan-frames", {
        canonical_name: canonicalName,
        frames,
      });
      if (result.status === "error") throw new Error(result.message || "Save failed");
      completeAndExit(result);
      return result;
    } catch (e) {
      alert("Could not save scan: " + e.message);
      finishing = false;
      btnDone.disabled = filledCount() < MIN_SLOTS;
      settled = false;
      framesSeen = 0;
      await startCamera();
      return null;
    }
  }

  async function startCamera() {
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
    }

    const { createCameraFraming } = await import("/static/camera-framing.js?v=29");
    framing = await createCameraFraming(video, videoWrap, null, {
      mode: "face",
      steadyFrame: false,
      scanMode: true,
      onLandmarks: onFrame,
      onStatus: (msg) => {
        liveEl.textContent = msg;
      },
    });

    const modelPoll = setInterval(() => {
      if (settled) {
        clearInterval(modelPoll);
        return;
      }
      if (framing?.isModelReady) {
        refreshProgress();
        clearInterval(modelPoll);
      }
    }, 400);

    updateRing(CENTER_SLOT);
    stepLabel.textContent = "Facing: Center";
    refreshProgress();
  }

  function stopCamera() {
    framing?.stop();
    framing = null;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  return new Promise((resolve) => {
    root.innerHTML = "";
    root.appendChild(wrap);
    resolveScan = resolve;

    btnCancel.onclick = () => completeAndExit(null);
    btnDone.onclick = () => void finishScan();

    (async () => {
      try {
        await startCamera();
      } catch (err) {
        hint.textContent = "Camera blocked or unavailable.";
        liveEl.textContent = "";
        console.error(err);
      }
    })();
  });
}