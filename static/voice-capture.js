// About 40 seconds read at a normal pace: long enough to pick the cleanest stretches from, with a mix of sounds, a question and numbers.
const VOICE_SCRIPT =
  "Hello, I'm setting up my avatar, and this is how I sound when I speak naturally. " +
  "The rainbow is a division of white light into many beautiful colors. When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. " +
  "Have you ever stopped to watch one? I usually do. " +
  "Some days I talk quickly, and some days I take my time, so I'll read this the way I'd say it to a friend. " +
  "Now a few numbers: one, two, three, forty-five, and a hundred and twelve. " +
  "Thanks for listening. That's all I needed to say.";
const MIN_SECONDS = 20; // shorter than this and there is not enough clean speech to copy the voice from
const MAX_SECONDS = 75;

export async function runVoiceCapture(root, canonicalName, api) {
  let recorder = null;
  let chunks = [];
  let stream = null;
  let recordedBlob = null;

  const wrap = document.createElement("section");
  wrap.className = "card";
  wrap.innerHTML = `
    <h2>Voice Scan</h2>
    <p>Press Record and read this aloud in one go, at your normal pace and volume (about 40 seconds):</p>
    <blockquote class="voice-script">${VOICE_SCRIPT}</blockquote>
    <div class="voice-meter" id="voice-meter" aria-hidden="true"></div>
    <p id="voice-status" class="voice-status">Microphone not started yet.</p>
    <audio id="voice-playback" controls class="hidden"></audio>
    <div class="step-actions step-actions--row">
      <button type="button" class="btn-ghost" id="voice-cancel">Cancel</button>
      <button type="button" class="btn-primary" id="voice-record">Record</button>
      <button type="button" class="btn-primary hidden" id="voice-save" disabled>Save</button>
    </div>
  `;

  const status = wrap.querySelector("#voice-status");
  const btnRecord = wrap.querySelector("#voice-record");
  const btnSave = wrap.querySelector("#voice-save");
  const btnCancel = wrap.querySelector("#voice-cancel");
  const playback = wrap.querySelector("#voice-playback");
  const meter = wrap.querySelector("#voice-meter");

  let recording = false;
  let t0 = 0;
  let timer = null;
  let meterId = null;

  async function ensureMic() {
    if (stream) return stream;
    // no echo cancelling, noise suppression or automatic volume: they change the sound of the voice that is being copied
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
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
    const hb = document.getElementById("logout-btn");
    if (hb) hb.style.display = "none"; // this screen has its own Cancel and Save
    const done = resolve;
    resolve = (v) => { if (hb) hb.style.display = ""; done(v); };
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

          recorder = new MediaRecorder(stream, { audioBitsPerSecond: 128000 });
          recorder.ondataavailable = (e) => chunks.push(e.data);
          recorder.onstop = () => {
            stopMeter();
            clearInterval(timer);
            const secs = (performance.now() - t0) / 1000;
            recordedBlob = new Blob(chunks, { type: "audio/webm" });
            playback.src = URL.createObjectURL(recordedBlob);
            playback.classList.remove("hidden");
            btnSave.classList.remove("hidden");
            if (secs < MIN_SECONDS) {
              btnSave.disabled = true;
              status.textContent = `That was only ${Math.round(secs)} seconds. Please read the whole passage, then try again.`;
            } else {
              btnSave.disabled = false;
              status.textContent = "Listen back, then Save or Record again.";
            }
            btnRecord.textContent = "Record again";
            recording = false;
          };
          recorder.start();
          recording = true;
          t0 = performance.now();
          btnRecord.textContent = "Stop";
          const tick = () => {
            const s = Math.round((performance.now() - t0) / 1000);
            status.textContent = `Recording… ${s} s. Read all of it, then press Stop.`;
            if (s >= MAX_SECONDS && recorder && recorder.state === "recording") recorder.stop();
          };
          tick();
          timer = setInterval(tick, 500);
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