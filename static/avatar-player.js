import { createVoiceChat } from "/static/voice-chat.js?v=6";
import { loadCinematic } from "/static/cinematic-clips.js?v=26";

/**
 * The chat avatar: a photo-real face (live idle motion, pre-made clips, live reply videos) plus the voice chat.
 * If the photo-real layer is not available for a person, the still photo stays on screen and the voice still works.
 *
 * @param {HTMLElement} stageEl .avatar-stage
 * @param {object} profile profile_summary from API
 * @returns {() => void} teardown
 */
export function mountAvatarPlayer(stageEl, profile, hooks = {}) {
  const name = profile.display_name || profile.canonical_name || "Avatar";
  const frames = (profile.scan_frames?.length ? profile.scan_frames : profile.scan_previews) || [];
  const stillPhoto = profile.avatar_face_url || frames[0]?.url || "";

  // The first picture on screen is the person's prepared photo (the rest frame once their animations exist), never the raw scan
  // photo, so nothing in a different crop flashes up before the avatar loads.
  const mediaBase = (stillPhoto.match(/^(.*\/media\/[^/]+)\//) || [])[1] || "";
  const firstPic = mediaBase ? `${mediaBase}/cinematic/idle_bank/rest.jpg?t=${Date.now()}` : stillPhoto;

  stageEl.innerHTML = `
    <div class="avatar-figure mesh has-face" id="avatar">
      ${stillPhoto ? `<img id="avatar-still" src="${escapeAttr(firstPic)}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;" />` : ""}
    </div>
    <h2>${escapeHtml(name)}</h2>
    <p id="avatar-status" class="avatar-note muted">Starting…</p>
    <div class="chat-meter" title="Your microphone level"><div class="chat-meter-bar" id="chat-meter-bar"></div><div class="chat-meter-thr" id="chat-meter-thr"></div></div>
    <div class="chat-log" id="chat-log" aria-live="polite"></div>
    <form class="chat-type" id="chat-type" autocomplete="off">
      <input id="chat-input" type="text" placeholder="…or type a message" />
      <button type="submit" class="btn-ghost">Send</button>
    </form>
    <audio id="avatar-audio" preload="auto"></audio>
    <div class="avatar-actions btn-row" style="justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
      <button type="button" class="btn-primary" id="chat-mute" disabled>Mute mic</button>
      <button type="button" class="btn-ghost" id="back-profile">Back to profile</button>
    </div>
  `;

  const stillEl = stageEl.querySelector("#avatar-still");
  if (stillEl && mediaBase) {
    // rest frame -> prepared photo -> raw scan photo, whichever exists first
    const chain = [`${mediaBase}/cinematic/source.jpg`, stillPhoto];
    stillEl.addEventListener("error", () => {
      const next = chain.shift();
      if (next) stillEl.src = next;
    });
  }
  const audio = stageEl.querySelector("#avatar-audio");
  const statusEl = stageEl.querySelector("#avatar-status");
  const muteBtn = stageEl.querySelector("#chat-mute");
  const logEl = stageEl.querySelector("#chat-log");
  const meterBar = stageEl.querySelector("#chat-meter-bar");
  const meterThr = stageEl.querySelector("#chat-meter-thr");
  let chat = null;
  let cinema = null; // photo-real layer: live idle face + pre-made clips + reply videos
  let destroyed = false;
  let chatBrainFoot = "";

  // Load the photo-real layer straight away so the live face shows while the chat models start.
  const cinemaReady = loadCinematic(stageEl, profile, { onNote: (m) => !destroyed && (statusEl.textContent = m) }).catch(() => null);
  cinemaReady.then((c) => {
    if (destroyed) c?.dispose();
    else cinema = c;
  });

  /** Play one spoken sentence: a video of the person saying it when there is one, otherwise just the voice. */
  async function playClip(blob, videoUrl = "") {
    if (destroyed) return;
    await cinema?.whenIdle();
    if (cinema && videoUrl) {
      statusEl.textContent = "Speaking…";
      const played = await cinema.playReply(videoUrl, blob);
      // a reply video is single-use: once it has played, the server throws it away
      setTimeout(() => {
        fetch("/api/chat/reply-played", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_url: videoUrl }),
        }).catch(() => {});
      }, 3000);
      if (played) return;
    }
    cinema?.release(); // no video for this sentence: back to the live idle face while the voice plays
    audio.src = URL.createObjectURL(blob);
    statusEl.textContent = "Speaking…";
    await new Promise((resolve) => {
      const done = () => {
        audio.removeEventListener("ended", done);
        audio.removeEventListener("error", done);
        resolve();
      };
      audio.addEventListener("ended", done);
      audio.addEventListener("error", done);
      audio.play().catch((err) => {
        console.error(err);
        statusEl.textContent = `Could not play audio (${err?.name || "error"}) — click the page once, then talk again.`;
        done();
      });
    });
    URL.revokeObjectURL(audio.src);
  }

  function addLine(who, text) {
    const row = document.createElement("p");
    row.className = `chat-line ${who}`;
    row.textContent = (who === "you" ? "You: " : `${name}: `) + text;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  /** Play a pre-made clip over the avatar; the mic is muted meanwhile so the chat does not hear it. */
  async function playCinematic(kind) {
    if (!cinema || destroyed) return false;
    const wasMuted = chat ? chat.muted : false;
    chat?.setMuted(true);
    try {
      return await cinema.play(kind);
    } finally {
      if (!wasMuted && !destroyed) chat?.setMuted(false);
    }
  }

  async function startChat() {
    statusEl.textContent = "Loading the voice and speech models (the first time takes a minute)…";
    try {
      const r = await fetch("/api/chat/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canonical_name: profile.canonical_name }),
      });
      const j = await r.json();
      if (j.status !== "ok") throw new Error(j.message || "chat could not start");
      if (destroyed) return;
      chatBrainFoot = j.llm_model && j.llm_mode ? ` (brain: ${j.llm_model}, ${j.llm_mode})` : "";
      chat = createVoiceChat({
        name: profile.canonical_name,
        token: j.token,
        playClip,
        onSuperseded: () => {
          audio.pause();
          statusEl.textContent = "Chat moved to another window — close the extra avatar tabs, then reload this one.";
          muteBtn.disabled = true;
        },
        onStatus: (t) => {
          if (destroyed) return;
          statusEl.textContent = t + chatBrainFoot;
          if (/^Thinking/.test(t)) cinema?.settle(true); // the live face goes still so the reply video fades in from a match
        },
        onLevel: (v, thr) => {
          meterBar.style.width = `${Math.min(100, v * 600)}%`;
          meterThr.style.left = `${Math.min(100, thr * 600)}%`;
        },
        onLine: addLine,
      });
      await chat.start();
      muteBtn.disabled = false;
      cinema = await cinemaReady;
      if (!cinema) statusEl.textContent = "No photo-real face for this person yet (showing the still photo); the voice chat still works.";
      if (!(cinema && (await playCinematic("greeting")))) await chat.say(`Hello, I'm ${j.display_name || name}.`);
      chat.touch?.(); // the silence clock starts after the greeting, not while the models were loading
    } catch (e) {
      console.error(e);
      statusEl.textContent =
        e?.name === "NotAllowedError" || e?.name === "NotFoundError"
          ? "I can't use the microphone — allow it in the browser's address bar, then reload."
          : `Chat problem: ${e.message || e}`;
    }
  }

  stageEl.querySelector("#back-profile")?.addEventListener("click", () => hooks.onBackProfile?.());
  muteBtn?.addEventListener("click", () => {
    if (!chat) return;
    chat.setMuted(!chat.muted);
    muteBtn.textContent = chat.muted ? "Unmute mic" : "Mute mic";
  });
  stageEl.querySelector("#chat-type")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const input = stageEl.querySelector("#chat-input");
    const t = input.value.trim();
    if (t && chat) {
      input.value = "";
      void chat.sendText(t);
    }
  });

  void startChat();

  return () => {
    destroyed = true;
    audio?.pause();
    chat?.stop();
    cinema?.dispose();
  };
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}
