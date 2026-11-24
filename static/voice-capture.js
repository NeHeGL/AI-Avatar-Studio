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
  wrap.className = "card voice-flow";
  wrap.innerHTML = `
    <h2>Voice Scan</h2>
    <p>Press Record and read this aloud in one go, at your normal pace and volume (about 40 seconds):</p>
    <blockquote class="voice-script">${VOICE_SCRIPT}</blockquote>
    <div class="voice-meter" id="voice-meter" aria-hidden="true"></div>
    <p id="voice-status" class="voice-status">Microphone not started yet.</p>
    <audio id="voice-playback" class="hidden"></audio>
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
            if (secs < MIN_SECONDS) {
              btnSave.disabled = true;
              status.textContent = `That was only ${Math.round(secs)} seconds. Please read the whole passage, then try again.`;
            } else {
              btnSave.disabled = false;
              status.textContent = "Listen back, then Save or Record again.";
            }
            btnRecord.textContent = "Record again";
            recording = false;
            // Go straight to the check screen (it has the playback and the Save), so there is only one Save.
            if (secs >= MIN_SECONDS) btnSave.onclick();
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
        const out = await reviewVoice(root, canonicalName, api, reader.result);
        if (out) {
          stopAll();
          resolve(out);
        } else {
          // back to the recorder to try again
          root.innerHTML = "";
          root.appendChild(wrap);
          btnSave.disabled = false;
        }
      };
      reader.readAsDataURL(recordedBlob);
    };
  });
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** Check a recording (made in the app or uploaded) and show what the program found, in plain words. Resolves {profile} once it is saved,
 * or null when the person chooses to try another. */
export function reviewVoice(root, canonicalName, api, dataUrl) {
  return new Promise(async (resolve) => {
    const wrap = document.createElement("section");
    wrap.className = "card voice-card voice-flow";
    wrap.innerHTML = `<h2>Voice Scan</h2><p class="voice-note">Listening to the recording…</p>`;
    root.innerHTML = "";
    root.appendChild(wrap);
    let r;
    try {
      r = await api("/api/voice/check", { audio_base64: dataUrl });
    } catch {
      r = { status: "error", message: "Could not check the recording." };
    }
    if (r.status !== "ok") {
      wrap.innerHTML = `<h2>Voice Scan</h2><p class="photo-verdict bad">${esc(r.message || "Something went wrong.")}</p>
        <div class="step-actions step-actions--row"><button type="button" class="btn-primary" id="vr-again">Try another</button></div>`;
      wrap.querySelector("#vr-again").onclick = () => resolve(null);
      return;
    }
    const head = { good: "This recording sounds good.", usable: "This recording will work, but could be better.", bad: "This recording can't be used." }[r.verdict];
    wrap.innerHTML = `
      <h2>Voice Scan</h2>
      <p class="photo-verdict ${r.verdict}">${head}</p>
      <audio controls src="${esc(dataUrl)}" class="voice-review-audio"></audio>
      <p class="photo-hint">${r.seconds} seconds, ${r.speech_seconds} seconds of clear speech.</p>
      ${r.issues.length ? `<ul class="photo-issues">${r.issues.map((i) => `<li class="${i.level}">${esc(i.text)}</li>`).join("")}</ul>` : ""}
      <div class="step-actions step-actions--row">
        <button type="button" class="btn-ghost" id="vr-again">Try another</button>
        <button type="button" class="btn-primary" id="vr-save" ${r.verdict === "bad" ? "disabled" : ""}>Save</button>
      </div>`;
    wrap.querySelector("#vr-again").onclick = () => resolve(null);
    const save = wrap.querySelector("#vr-save");
    save.onclick = async () => {
      save.disabled = true;
      const out = await api("/api/voice/save", { canonical_name: canonicalName, token: r.token });
      if (out.status === "ok") resolve(out);
      else {
        alert(out.message || "Could not save the recording.");
        save.disabled = false;
      }
    };
  });
}

const MAX_CLIPS = 5; // shown as tiles, like the photos

/** "Nov 21, 4:11 AM" (the year too when it is not this year): when the recording was made or added. */
function stamp(sec) {
  if (!sec) return "";
  const d = new Date(sec * 1000);
  const opts = { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" };
  if (d.getFullYear() !== new Date().getFullYear()) opts.year = "numeric";
  return d.toLocaleString(undefined, opts);
}

/** The Voice screen, laid out like the Photo screen: your recordings as tiles (click one to play it, x to delete it), then two ways to add
 * another: record a new one (you read a short passage) or upload an audio file. */
export async function runVoiceStep(root, canonicalName, api, opts = {}) {
  const hb = document.getElementById("logout-btn");
  if (hb) hb.style.display = "none"; // this screen has its own Done
  document.body.classList.add("wide-stage");
  const player = new Audio();
  let profile = null;
  return new Promise((resolve) => {
    const finish = () => {
      player.pause();
      if (hb) hb.style.display = "";
      document.body.classList.remove("wide-stage");
      resolve(profile ? { profile } : opts.skippable ? { skipped: true } : null);
    };
    async function show() {
      player.pause();
      let data = { clips: [] };
      try {
        data = await (await fetch(`/api/voice/list?canonical_name=${encodeURIComponent(canonicalName)}`)).json();
        profile = data.profile || profile;
      } catch { /* empty list */ }
      const clips = [...(data.clips || [])].reverse(); // oldest first, numbered 1..n
      const full = clips.length >= MAX_CLIPS;
      const wrap = document.createElement("section");
      wrap.className = "card photo-card voice-step";
      wrap.innerHTML = `
        <h2>Voice Scan</h2>
        ${clips.length ? `<p class="photo-label">Your recordings. Click one to listen to it.</p>
        <div class="photo-gallery" style="--n:${clips.length}">${clips
          .map(
            (c, i) => `<div class="photo-thumb voice-tile" data-url="${esc(c.url)}" data-name="${esc(c.name)}" title="Play">
              <span class="voice-play"></span>
              <span class="voice-tile-title">${c.locked ? "Locked voice" : `Recording ${i + 1}`}</span>
              <span class="voice-tile-len">${c.seconds ? `${c.seconds} s` : ""}</span>
              <span class="voice-tile-date">${stamp(c.recorded)}</span>
              <button type="button" class="photo-del" data-del="${esc(c.name)}" title="Delete this recording"></button>
            </div>`
          )
          .join("")}</div>` : ""}
        <p class="photo-label">${clips.length ? `You can have up to ${MAX_CLIPS} recordings per profile. You can delete a recording by clicking the <span class="photo-x" aria-label="X"></span>.` : "Add a recording of your voice"}</p>
        <div class="photo-options${full ? " full" : ""}">
          <button type="button" class="photo-option" id="vs-rec"${full ? " disabled" : ""}><strong>Record new audio</strong><span>You will read a short passage aloud, about 40 seconds. The program picks the clearest speech to build your voice.</span></button>
          <button type="button" class="photo-option" id="vs-up"${full ? " disabled" : ""}><strong>Upload audio</strong><span>Choose a recording of your voice (wav, mp3, m4a, ogg). Clear speech with little background noise works best.</span></button>
        </div>
        <input type="file" id="vs-file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac" hidden />
        <div class="step-actions step-actions--row"><button type="button" class="btn-ghost" id="vs-done">${clips.length ? "Done" : opts.skippable ? "Skip for now" : "Cancel"}</button></div>`;
      root.innerHTML = "";
      root.appendChild(wrap);
      wrap.querySelector("#vs-done").onclick = finish;
      const addDone = async (r) => {
        if (r?.profile) profile = r.profile;
        show();
      };
      wrap.querySelector("#vs-rec").onclick = async () => {
        document.body.classList.remove("wide-stage");
        const r = await runVoiceCapture(root, canonicalName, api);
        document.body.classList.add("wide-stage");
        addDone(r);
      };
      const file = wrap.querySelector("#vs-file");
      wrap.querySelector("#vs-up").onclick = () => file.click();
      file.onchange = async () => {
        const f = file.files[0];
        if (!f) return;
        const dataUrl = await new Promise((res) => {
          const rd = new FileReader();
          rd.onload = () => res(rd.result);
          rd.readAsDataURL(f);
        });
        const r = await reviewVoice(root, canonicalName, api, dataUrl);
        addDone(r);
      };
      wrap.ondragover = (e) => e.preventDefault();
      wrap.ondrop = async (e) => {
        e.preventDefault();
        const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("audio/"));
        if (!f || full) return;
        const dataUrl = await new Promise((res) => {
          const rd = new FileReader();
          rd.onload = () => res(rd.result);
          rd.readAsDataURL(f);
        });
        addDone(await reviewVoice(root, canonicalName, api, dataUrl));
      };
      // click a tile to play it (click again to stop)
      let playing = null;
      const clear = () => wrap.querySelectorAll(".voice-tile").forEach((x) => x.classList.remove("playing"));
      player.onended = clear;
      wrap.querySelectorAll(".voice-tile").forEach((tile) => {
        tile.onclick = (e) => {
          if (e.target.closest(".photo-del")) return;
          if (playing === tile && !player.paused) {
            player.pause();
            clear();
            return;
          }
          clear();
          playing = tile;
          player.src = tile.dataset.url;
          player.play().catch(() => {});
          tile.classList.add("playing");
        };
      });
      wrap.querySelectorAll(".photo-del").forEach((b) => {
        b.onclick = async () => {
          if (!confirm("Delete this recording?")) return;
          const out = await api("/api/voice/delete", { canonical_name: canonicalName, name: b.dataset.del });
          if (out.profile) profile = out.profile;
          show();
        };
      });
    }
    show();
  });
}
