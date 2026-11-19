const VOICE_SCRIPT =
  "Hello, I'm setting up my avatar. This is how I sound when I speak naturally.";

export async function runVoiceCapture(root, canonicalName, api) {
  let recorder = null;
  let chunks = [];
  let stream = null;
  let recordedBlob = null;

  const wrap = document.createElement("section");
  wrap.className = "card";
  wrap.innerHTML = `
    <h2>Voice capture</h2>
    <p>Read this aloud when you press Record (about 5 seconds):</p>
    <blockquote class="voice-script">${VOICE_SCRIPT}</blockquote>
    <div class="voice-meter" id="voice-meter" aria-hidden="true"></div>
    <p id="voice-status" class="voice-status">Microphone not started yet.</p>
    <audio id="voice-playback" controls class="hidden"></audio>
    <div class="btn-row">
      <button type="button" class="btn-ghost" id="voice-cancel">Back</button>
      <button type="button" class="btn-primary" id="voice-record">Record</button>
      <button type="button" class="btn-primary hidden" id="voice-save" disabled>Continue</button>
    </div>
  `;

  const status = wrap.querySelector("#voice-status");
  const btnRecord = wrap.querySelector("#voice-record");
  const btnSave = wrap.querySelector("#voice-save");
  const btnCancel = wrap.querySelector("#voice-cancel");
  const playback = wrap.querySelector("#voice-playback");
  const meter = wrap.querySelector("#voice-meter");

  let recording = false;
  let meterId = null;

  async function ensureMic() {
    if (stream) return stream;
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return stream;
  }

  function stopMeter() {
    if (meterId) cancelAnimationFrame(meterId);
    meterId = null;
    meter.innerHTML = "";
  }

  function animateMeter(analyser) {
    const data = new Uint8Array(analyser.frequencyBinCount);
    const draw = () => {
      analyser.getByteFrequencyData(data);
      const level = data.reduce((a, b) => a + b, 0) / data.length / 255;
      const bars = 12;
      meter.innerHTML = Array.from({ length: bars }, (_, i) => {
        const h = Math.max(8, Math.min(100, level * 120 + Math.random() * 20 * level));
        return `<span style="height:${h}%"></span>`;
      }).join("");
      meterId = requestAnimationFrame(draw);
    };
    draw();
  }

  function stopAll() {
    stopMeter();
    if (recorder && recorder.state !== "inactive") recorder.stop();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  return new Promise((resolve) => {
    root.innerHTML = "";
    root.appendChild(wrap);

    btnCancel.onclick = () => {
      stopAll();
      resolve(null);
    };

    btnRecord.onclick = async () => {
      try {
        await ensureMic();
        if (!recording) {
          chunks = [];
          recordedBlob = null;
          playback.classList.add("hidden");
          btnSave.classList.add("hidden");

          const ctx = new AudioContext();
          const source = ctx.createMediaStreamSource(stream);
          const analyser = ctx.createAnalyser();
          source.connect(analyser);
          animateMeter(analyser);

          recorder = new MediaRecorder(stream);
          recorder.ondataavailable = (e) => chunks.push(e.data);
          recorder.onstop = () => {
            stopMeter();
            recordedBlob = new Blob(chunks, { type: "audio/webm" });
            playback.src = URL.createObjectURL(recordedBlob);
            playback.classList.remove("hidden");
            btnSave.classList.remove("hidden");
            btnSave.disabled = false;
            status.textContent = "Listen back, then Save or Record again.";
            btnRecord.textContent = "Record again";
            recording = false;
          };
          recorder.start();
          recording = true;
          btnRecord.textContent = "Stop";
          status.textContent = "Recording… read the script naturally.";
          setTimeout(() => {
            if (recorder && recorder.state === "recording") recorder.stop();
          }, 6000);
        } else if (recorder && recorder.state === "recording") {
          recorder.stop();
        }
      } catch (e) {
        status.textContent = "Microphone blocked. Allow mic access and try again.";
        console.error(e);
      }
    };

    btnSave.onclick = async () => {
      if (!recordedBlob) return;
      btnSave.disabled = true;
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const result = await api("/api/capture/voice", {
            canonical_name: canonicalName,
            audio_base64: reader.result,
            label: "Setup read",
          });
          stopAll();
          resolve(result);
        } catch (err) {
          alert("Could not save voice: " + err.message);
          btnSave.disabled = false;
        }
      };
      reader.readAsDataURL(recordedBlob);
    };
  });
}