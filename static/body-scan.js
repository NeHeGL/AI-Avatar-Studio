const BODY_STEPS = [
  { caption: "Body — front", hint: "Sit back so your shoulders are in frame. Face the camera." },
  { caption: "Turn 90° right", hint: "Turn your chair about 90° to your right, then capture." },
  { caption: "Turn 90° more", hint: "Another 90° — your back or over-shoulder toward camera." },
  { caption: "Last side", hint: "One more 90° — left side toward camera." },
];

export async function runBodyScan(root, canonicalName, api) {
  let stream = null;
  let framing = null;
  let stepIndex = 0;
  const captured = [];

  root.classList.add("step-enter");
  root.innerHTML = `
    <section class="card face-scan-card body-scan-card">
      <h2>Body scan</h2>
      <p class="scan-hint" id="scan-hint">Optional — helps with shoulders and how you look on camera.</p>
      <div class="scan-stage scan-stage--body">
        <div class="scan-video-wrap scan-video-wrap--body" id="scan-video-wrap">
          <video id="scan-video" playsinline muted autoplay></video>
        </div>
      </div>
      <p class="scan-step-label" id="scan-step-label"></p>
      <div class="btn-row">
        <button type="button" class="btn-ghost" id="scan-cancel">Back</button>
        <button type="button" class="btn-primary" id="scan-capture" disabled>Capture</button>
        <button type="button" class="btn-primary hidden" id="scan-finish" disabled>Continue</button>
      </div>
    </section>
  `;

  const video = root.querySelector("#scan-video");
  const videoWrap = root.querySelector("#scan-video-wrap");
  const hint = root.querySelector("#scan-hint");
  const stepLabel = root.querySelector("#scan-step-label");
  const btnCapture = root.querySelector("#scan-capture");
  const btnFinish = root.querySelector("#scan-finish");
  const btnCancel = root.querySelector("#scan-cancel");

  function setStep() {
    if (stepIndex >= BODY_STEPS.length) {
      stepLabel.textContent = `Captured ${captured.length} views.`;
      btnCapture.classList.add("hidden");
      btnFinish.classList.remove("hidden");
      btnFinish.disabled = captured.length < 2;
      return;
    }
    const step = BODY_STEPS[stepIndex];
    stepLabel.textContent = `Step ${stepIndex + 1} of ${BODY_STEPS.length}: ${step.hint}`;
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      const { createCameraFraming } = await import("/static/camera-framing.js");
      framing = await createCameraFraming(video, videoWrap, hint, { mode: "body" });
      btnCapture.disabled = false;
      setStep();
    } catch (err) {
      stepLabel.textContent = "Camera unavailable. Use Skip on the previous screen.";
      console.error(err);
    }
  }

  function captureFrame() {
    const step = BODY_STEPS[stepIndex];
    const dataUrl = framing
      ? framing.captureFramedJpeg(0.88)
      : (() => {
          const canvas = document.createElement("canvas");
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          canvas.getContext("2d").drawImage(video, 0, 0);
          return canvas.toDataURL("image/jpeg", 0.88);
        })();
    captured.push({
      image_base64: dataUrl,
      caption: step.caption,
      angle_index: stepIndex,
      quality_score: 0.88,
    });
    stepIndex += 1;
    setStep();
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
    btnCapture.onclick = () => captureFrame();
    btnCancel.onclick = () => {
      stopCamera();
      resolve(null);
    };
    btnFinish.onclick = async () => {
      btnFinish.disabled = true;
      try {
        const result = await api("/api/capture/body-frames", {
          canonical_name: canonicalName,
          frames: captured,
        });
        stopCamera();
        resolve(result);
      } catch (e) {
        alert("Could not save body scan: " + e.message);
        btnFinish.disabled = false;
      }
    };
    startCamera();
  });
}