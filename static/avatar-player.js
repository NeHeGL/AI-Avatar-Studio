import { createVoiceChat } from "/static/voice-chat.js?v=40";
import { loadAvatarFace } from "/static/avatar-face.js?v=42";

/**
 * The chat avatar: the person's photo with its waiting animations and lip-sync reply videos, plus the voice chat.
 * If the prepared photo is not available for a person, the still photo stays on screen and the voice still works.
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
  const personBase = (stillPhoto.match(/^(.*\/profiles\/[^/]+)\/photo\//) || [])[1] || "";
  const mediaBase = personBase ? personBase + "/cache" : "";
  const firstPic = mediaBase ? `${mediaBase}/face/idle_bank/rest.jpg?t=${Date.now()}` : stillPhoto;

  stageEl.innerHTML = `
    <div class="stage-cols">
    <div class="stage-left">
    <div class="avatar-figure mesh has-face" id="avatar">
      ${stillPhoto ? `<img id="avatar-still" src="${escapeAttr(firstPic)}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;" />` : ""}
    </div>
    <h2>${escapeHtml(name)}</h2>
    <p id="avatar-status" class="avatar-note muted attention">Permissions: please allow your microphone</p>
    <div class="chat-meter" title="Your microphone level"><div class="chat-meter-bar" id="chat-meter-bar"></div><div class="chat-meter-thr" id="chat-meter-thr"></div></div>
    </div>
    <div class="stage-right">
    <div class="chat-log" id="chat-log" aria-live="polite"></div>
    <form class="chat-type" id="chat-type" autocomplete="off">
      <input id="chat-input" type="text" placeholder="…or type a message" />
      <button type="submit" class="btn-ghost">Send</button>
    </form>
    <audio id="avatar-audio" preload="auto"></audio>
    <div class="avatar-actions btn-row" style="justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
      <button type="button" class="btn-primary" id="chat-mute" disabled>Mute mic</button>
      <button type="button" class="btn-ghost" id="back-profile">Back to profile settings</button>
    </div>
    </div>
    </div>
  `;
  document.body.classList.add("wide-stage"); // this screen uses the wider frame

  const stillEl = stageEl.querySelector("#avatar-still");
  if (stillEl && mediaBase) {
    // rest frame -> prepared photo -> raw scan photo, whichever exists first
    const chain = [`${mediaBase}/face/source.jpg`, stillPhoto];
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
  let avatarFace = null; // the photo, its waiting animations and the reply videos
  let destroyed = false;
  let chatBrainFoot = ""; // the model that does the thinking, shown while it thinks
  let chatVoiceFoot = ""; // the model that does the speaking, shown while it speaks
  let chatEarsFoot = ""; // the model that does the listening, shown while it listens

  // Load the face layer straight away so the photo and its animations show while the chat models start.
  const avatarFaceReady = loadAvatarFace(stageEl, profile, { onNote: (m) => !destroyed && (statusEl.textContent = m) }).catch(() => null);
  avatarFaceReady.then((c) => {
    if (destroyed) c?.dispose();
    else avatarFace = c;
  });

  /** Play one spoken sentence: a video of the person saying it when there is one, otherwise just the voice. */
  async function playClip(blob, videoUrl = "", onSpeak = null) {
    if (destroyed) return;
    await avatarFace?.whenIdle();
    if (avatarFace && videoUrl) {
      statusEl.textContent = "Speaking…" + chatVoiceFoot;
      const played = await avatarFace.playReply(videoUrl, blob, (secs) => onSpeak?.(secs));
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
    avatarFace?.release(); // no video for this sentence: the voice plays over the still picture
    audio.src = URL.createObjectURL(blob);
    statusEl.textContent = "Speaking…" + chatVoiceFoot;
    audio.addEventListener("playing", () => onSpeak?.(audio.duration), { once: true });
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

  function addLine(who, text, replacePrevious = false, durSec = 0) {
    if (replacePrevious) {
      const olds = logEl.querySelectorAll(".chat-line.you");
      olds[olds.length - 1]?.remove(); // the earlier part of what you said: now part of this longer line
    }
    const row = document.createElement("p");
    row.className = `chat-line ${who}`;
    const prefix = who === "you" ? "You: " : who === "note" ? "" : `${name}: `;
    logEl.appendChild(row);
    if (who === "note") {
      row.textContent = text;
      logEl.scrollTop = logEl.scrollHeight;
    } else if (who !== "you") {
      // the words appear as they are spoken: revealed across the length of the speech
      const words = text.split(/\s+/).filter(Boolean);
      const total = (Number.isFinite(durSec) && durSec > 0.3 ? durSec : words.length * 0.33) * 940;
      const t0 = performance.now();
      const tick = () => {
        const f = Math.min(1, (performance.now() - t0) / total);
        row.textContent = prefix + words.slice(0, Math.max(1, Math.round(words.length * f))).join(" ");
        logEl.scrollTop = logEl.scrollHeight;
        if (f < 1 && !destroyed) setTimeout(tick, 60);
      };
      tick();
    } else {
      row.textContent = prefix + text;
      logEl.scrollTop = logEl.scrollHeight;
    }
  }

  async function startChat() {
    try {
      statusEl.textContent = "Permissions: please allow your microphone";
      statusEl.classList.add("attention");
      const r = await fetch("/api/chat/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canonical_name: profile.canonical_name }),
      });
      const j = await r.json();
      if (j.status !== "ok") throw new Error(j.message || "chat could not start");
      if (destroyed) return;
      chatBrainFoot = j.llm_name ? ` (${j.llm_name}${j.llm_mode ? `, ${j.llm_mode}` : ""})` : "";
      chatEarsFoot = j.ears_name ? ` (${j.ears_name})` : "";
      chatVoiceFoot = j.voice_name ? ` (${j.voice_name})` : "";
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
          const foot = /^Thinking/.test(t) ? chatBrainFoot : /listening|^Listening/.test(t) ? chatEarsFoot : "";
          statusEl.textContent = t + foot;
          statusEl.title = t + foot; // the whole line, when it is too long to fit
          if (/^Thinking/.test(t)) avatarFace?.settle(true); // the next waiting animation follows at once and the reply video takes over from a calm frame
        },
        onPartial: (t) => {
          // what you are saying appears in the message box as you talk; when you stop it goes (the turn is sent) and the
          // line shows up in the conversation above
          const input = stageEl.querySelector("#chat-input");
          if (!input) return;
          if (!t) {
            if (input.dataset.live === "1") {
              input.value = "";
              input.dataset.live = "";
            }
            return;
          }
          if (input.value && input.dataset.live !== "1") return; // never overwrite something typed
          input.value = t;
          input.dataset.live = "1";
          input.scrollLeft = input.scrollWidth; // keep the newest words in view, like typing
        },
        onState: ({ listening, canType }) => {
          const input = stageEl.querySelector("#chat-input");
          const send = stageEl.querySelector("#chat-type button");
          if (input) {
            input.disabled = !canType;
            input.placeholder = canType ? "…or type a message" : "Wait for your turn…";
          }
          if (send) send.disabled = !canType;
          stageEl.querySelector(".chat-meter")?.style.setProperty("outline", listening ? "2px solid #3ddc84" : "none");
        },
        onLevel: (v, thr) => {
          meterBar.style.width = `${Math.min(100, v * 600)}%`;
          meterThr.style.left = `${Math.min(100, thr * 600)}%`;
        },
        onLine: (who, text, replace) => addLine(who, text, replace),
        onSpeak: (text, secs) => addLine("avatar", text, false, secs),
      });
      await chat.start();
      statusEl.classList.remove("attention"); // the microphone was allowed
      muteBtn.disabled = false;
      avatarFace = await avatarFaceReady;
      if (!avatarFace) statusEl.textContent = "No photo-real face for this person yet (showing the still photo); the voice chat still works.";
      await chat.greet();
      chat.touch?.(); // the silence clock starts after the greeting, not while the models were loading
    } catch (e) {
      console.error(e);
      statusEl.classList.remove("attention");
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
    if (t && chat && !input.disabled) {
      input.value = "";
      void chat.sendText(t);
    }
  });

  // "Noted: ...": when the AI writes something it learned into its notes (the AI Learned screen), the conversation says so
  const newsTimer = setInterval(async () => {
    if (destroyed || !profile.canonical_name) return;
    try {
      const j = await (await fetch(`/api/learned/news?name=${encodeURIComponent(profile.canonical_name)}`, { cache: "no-store" })).json();
      for (const f of j.facts || []) addLine("note", `Noted: ${f}`);
    } catch {
      /* the server is busy: try again in a moment */
    }
  }, 3000);

  void startChat();

  return () => {
    destroyed = true;
    clearInterval(newsTimer);
    document.body.classList.remove("wide-stage");
    audio?.pause();
    chat?.stop();
    avatarFace?.dispose();
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
