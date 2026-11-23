const root = document.getElementById("screen-root");
const phaseLabel = document.getElementById("phase-label");
const statusHint = document.getElementById("status-hint");

const state = {
  enteredName: "",
  profile: null,
  canonicalName: "",
};

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const logoutBtn = document.getElementById("logout-btn");

/** The Change profile button shows whenever someone is signed in (a name has been chosen). */
function updateLogout() {
  if (logoutBtn) logoutBtn.hidden = !state.canonicalName;
}

function setPhase(label, hint = "") {
  phaseLabel.textContent = label;
  statusHint.textContent = hint;
  updateLogout();
}

/** Leave the current person: stop the chat and the microphone, forget who was chosen, go back to the start (name) screen. */
function logOut() {
  if (talkInterval) clearInterval(talkInterval);
  if (avatarTeardown) {
    avatarTeardown();
    avatarTeardown = null;
  }
  state.enteredName = "";
  state.canonicalName = "";
  state.profile = null;
  setOnboardingView(false);
  updateLogout();
  showProfileChooser(); // straight back to the Profile Management screen (no Hello screen)
}
logoutBtn?.addEventListener("click", logOut);

async function submitName(name, fromList = false) {
  state.enteredName = name.trim();
  if (!state.enteredName) {
    alert("Please enter a name.");
    return;
  }
  const result = await api("/api/walkthrough/check-name", { name: state.enteredName });
  if (result.status === "exists") {
    setOnboardingView(false);
    state.profile = result.profile_summary;
    if (fromList) {
      await confirmExisting(true); // picked from the list: nothing to confirm, open it
    } else {
      renderExistsPrompt(); // typed a name that is already taken
    }
  } else if (result.status === "new") {
    state.canonicalName = state.enteredName;
    state.profile = emptyProfileSummary(state.canonicalName);
    await runSetupPipeline(
      { root, phaseLabel, statusHint, api, state, escapeHtml, escapeAttr },
      { isNewUser: true }
    );
    await saveNewProfile(true);
  } else {
    alert(result.message || "Invalid name.");
  }
}

function renderExistsPrompt() {
  setPhase("", "");
  root.innerHTML = `
    <section class="card">
      <p class="prompt">A profile named &ldquo;${escapeHtml(state.enteredName)}&rdquo; already exists.</p>
      <ul class="sections-list" id="sections-overview"></ul>
      <div class="btn-row">
        <button type="button" id="yes-you">Open this profile</button>
        <button type="button" class="secondary" id="no-you">Use a different name</button>
      </div>
    </section>
  `;
  renderSectionsList(state.profile);
  document.getElementById("yes-you").onclick = () => confirmExisting(true);
  document.getElementById("no-you").onclick = () => renderUniqueName();
}

/** What a step shows: just a green check when it has what the app needs, or a short word when something is missing. Nothing says
 * "complete": a scan or a set of notes can always take more. The amount is in the tooltip, not on the screen. */
const plural = (n, one, many) => `${n.toLocaleString()} ${n === 1 ? one : many}`;
function sectionBadge(kind, s) {
  if (kind === "scan") {
    if (s.scan.complete) return { cls: "done", text: "✓", tip: plural(s.scan.count, "photo", "photos") };
    return { cls: "pending", text: "Required", tip: "A photo of your face is needed" };
  }
  if (kind === "voice") {
    if (s.voice.complete) return { cls: "done", text: "✓", tip: plural(s.voice.count, "recording", "recordings") };
    return { cls: "pending", text: "Required", tip: "The avatar needs a voice recording" };
  }
  if (s.other.complete) return { cls: "done", text: "✓", tip: s.other.words ? plural(s.other.words, "word", "words") + " of knowledge" : "Display name set" };
  return { cls: "optional", text: "Optional", tip: "Tell the AI about this person so it knows them better" };
}
const badgeHtml = (b) => `<span class="badge ${b.cls}" title="${b.tip}">${b.text}</span>`;

function renderSectionsList(profile) {
  const ul = document.querySelector("#sections-overview");
  if (!ul || !profile) return;
  const s = profile.sections;
  const bs = sectionBadge("scan", s);
  const bv = sectionBadge("voice", s);
  const bo = sectionBadge("other", s);
  const stored = (n, one, many) => (n ? plural(n, one, many) : "None");
  ul.classList.add("stats-grid");
  ul.innerHTML = `
    <li class="stats-head"><span>Step</span><span>Records Stored</span><span>Contains Data</span></li>
    <li class="row-link" data-section="scan" role="button" tabindex="0" title="See and change your photos"><span>Photo</span><span>${stored(s.scan.count, "photo", "photos")}</span>${badgeHtml(bs)}</li>
    <li class="row-link" data-section="voice" role="button" tabindex="0" title="Listen to and change your recordings"><span>Voice</span><span>${stored(s.voice.count, "recording", "recordings")}</span>${badgeHtml(bv)}</li>
    <li class="row-link" data-section="other" role="button" tabindex="0" title="Read and change your knowledge"><span>Knowledge</span><span>${stored(s.other.words || 0, "word", "words")}</span>${badgeHtml(bo)}</li>
  `;
}

function profileCoreComplete(profile) {
  return profile?.core_complete === true;
}

function renderUniqueName() {
  setPhase("New profile", "Use a unique or full name.");
  root.innerHTML = `
    <section class="card">
      <p class="prompt">Please enter a unique or full name (firstname lastname).</p>
      <input type="text" id="full-name" placeholder="Firstname Lastname" />
      <div class="btn-row">
        <button type="button" id="full-continue">Continue</button>
      </div>
    </section>
  `;
  const input = document.getElementById("full-name");
  input.focus();
  document.getElementById("full-continue").onclick = () =>
    confirmExisting(false, input.value);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") confirmExisting(false, input.value);
  });
}

async function confirmExisting(isYou, uniqueFullName = null) {
  const result = await api("/api/walkthrough/confirm", {
    name: state.enteredName,
    is_you: isYou,
    unique_full_name: uniqueFullName,
  });
  if (result.status === "need_unique" || result.status === "collision") {
    alert(result.message);
    if (!isYou) renderUniqueName();
    return;
  }
  if (result.status === "returning") {
    state.profile = result.profile;
    state.canonicalName = result.profile.canonical_name;
    await handleReturningUser();
  } else if (result.status === "new") {
    state.canonicalName = result.canonical_name || uniqueFullName?.trim() || state.enteredName;
    state.profile = emptyProfileSummary(state.canonicalName);
    await runSetupPipeline(
      { root, phaseLabel, statusHint, api, state, escapeHtml, escapeAttr },
      { isNewUser: true }
    );
    await saveNewProfile(true);
  }
}

async function handleReturningUser() {
  if (!profileCoreComplete(state.profile)) {
    setOnboardingView(true);
    root.innerHTML = `
      <section class="step-card step-card--center">
        <p class="step-eyebrow">Welcome back</p>
        <h2 class="step-title">Let's finish your setup</h2>
        <p class="step-copy">We'll only ask for what's still missing — photo, voice, or both.</p>
        <div class="step-actions step-actions--solo">
          <button type="button" class="btn-primary" id="resume-continue">Continue</button>
        </div>
      </section>`;
    await new Promise((r) => {
      document.getElementById("resume-continue").onclick = r;
    });
    await runSetupPipeline(
      { root, phaseLabel, statusHint, api, state, escapeHtml, escapeAttr },
      { isNewUser: false, skipFinishCard: true }
    );
    await saveNewProfile(false);
    const refreshed = await api("/api/walkthrough/skip-updates", {
      canonical_name: state.canonicalName,
    });
    state.profile = refreshed.profile;
  }
  renderProfileHome();
}

function renderProfileHome() {
  setOnboardingView(false);
  const name = state.profile?.display_name || state.canonicalName;
  setPhase("", ""); // the profile name is already shown in the card below
  const p = state.profile;
  root.innerHTML = `
    <section class="card home-card">
      <h2>${escapeHtml(name)}</h2>
      <ul class="sections-list" id="sections-overview"></ul>
      <p class="home-hint">Click a row to see or change it.</p>
      <div class="home-actions">
        <button type="button" class="btn-primary" id="btn-preview">View Your Avatar</button>
        <button type="button" class="btn-ghost btn-danger" id="btn-delete">Delete and Start Fresh</button>
      </div>
    </section>
  `;
  renderSectionsList(p);
  document.getElementById("btn-preview").onclick = () => previewAvatar();
  document.querySelectorAll("#sections-overview .row-link").forEach((li) => {
    const go = () => runUpdateSection(li.dataset.section);
    li.onclick = go;
    li.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), go());
  });
  document.getElementById("btn-delete").onclick = () => deleteExistingProfile();
}

async function previewAvatar() {
  const result = await api("/api/walkthrough/skip-updates", {
    canonical_name: state.canonicalName,
  });
  if (result.status === "error") {
    renderAvatar({
      status: "avatar",
      profile: state.profile,
      avatar_mode: "placeholder",
    });
    return;
  }
  state.profile = result.profile;
  renderAvatar(result);
}

async function runUpdateSection(section) {
  setOnboardingView(true);
  if (section === "scan") {
    const { runPhotoStep } = await import("/static/photo-capture.js?v=16");
    const result = await runPhotoStep(root, state.canonicalName, api);
    if (result?.profile) state.profile = result.profile;
  } else if (section === "voice") {
    const { runVoiceStep } = await import("/static/voice-capture.js?v=11");
    const result = await runVoiceStep(root, state.canonicalName, api);
    if (result?.profile) state.profile = result.profile;
  } else if ((await openOtherEditor()) === false) {
    setOnboardingView(false); // cancelled: nothing was changed, so nothing is saved
    renderProfileHome();
    return;
  }
  await saveNewProfile(false);
  setOnboardingView(false);
  renderProfileHome();
}

/** Deleting a profile takes three separate confirmations: read the warning, type the profile's name, then wait out a short countdown. */
let deleteTimer = 0;
function deleteExistingProfile() {
  renderDeleteStep(1);
}

function renderDeleteStep(step) {
  clearInterval(deleteTimer);
  const name = state.profile?.display_name || state.canonicalName;
  const safe = escapeHtml(name);
  const cancel = `<button type="button" class="btn-primary" id="del-cancel">Cancel</button>`;
  if (step === 1) {
    root.innerHTML = `
      <section class="card home-card">
        <h2>Delete &ldquo;${safe}&rdquo;?</h2>
        <p class="home-hint">This removes the saved profile: its details, notes, photos and voice. It can't be undone.</p>
        <p class="home-hint">To confirm, type the profile's name exactly: <strong>${safe}</strong></p>
        <input type="text" class="step-input" id="del-name" placeholder="${escapeAttr(name)}" autocomplete="off" />
        <div class="step-actions step-actions--row">
          ${cancel}
          <button type="button" class="btn-ghost btn-danger" id="del-next" disabled>Continue</button>
        </div>
      </section>`;
    const input = document.getElementById("del-name");
    const next = document.getElementById("del-next");
    input.focus();
    input.addEventListener("input", () => {
      next.disabled = input.value.trim().toLowerCase() !== name.trim().toLowerCase();
    });
    next.onclick = () => !next.disabled && renderDeleteStep(2);
  } else {
    let left = 5;
    root.innerHTML = `
      <section class="card home-card">
        <h2>Last chance</h2>
        <p class="home-hint">&ldquo;${safe}&rdquo; will be deleted permanently the moment you press the red button.</p>
        <div class="step-actions step-actions--row">
          ${cancel}
          <button type="button" class="btn-ghost btn-danger" id="del-final" disabled>Delete permanently (${left})</button>
        </div>
      </section>`;
    const fin = document.getElementById("del-final");
    deleteTimer = setInterval(() => {
      left -= 1;
      if (left > 0) {
        fin.textContent = `Delete permanently (${left})`;
      } else {
        clearInterval(deleteTimer);
        fin.disabled = false;
        fin.textContent = "Delete permanently";
      }
    }, 1000);
    fin.onclick = () => !fin.disabled && finishDeleteProfile();
  }
  document.getElementById("del-cancel").onclick = () => {
    clearInterval(deleteTimer);
    renderProfileHome();
  };
}

async function finishDeleteProfile() {
  const result = await api("/api/walkthrough/delete-profile", {
    canonical_name: state.canonicalName,
  });
  if (result.status === "error") {
    alert(result.message);
    renderProfileHome();
    return;
  }
  state.enteredName = "";
  state.canonicalName = "";
  state.profile = null;
  updateLogout();
  showProfileChooser(); // the deleted profile is gone from the list: straight back to Profile Management
}

async function openOtherEditor() {
  const o = state.profile?.other_preview || {};
  root.innerHTML = `
    <section class="step-card step-card--center step-card--knowledge">
      <h2 class="step-title">Knowledge</h2>
      <p class="muted knowledge-blurb">This is what the AI knows about ${escapeHtml(o.display_name || state.canonicalName)}. Write anything you want it to know and remember: who they are or were, their family and friends, the things they love or loved, their way of speaking, the stories they tell or told. The AI keeps this as background and brings it up naturally in conversation, the way a person recalls things. The more you write, the more it will feel like them.</p>
      <textarea id="other-notes" class="knowledge-text" rows="12" placeholder="Born in ... Grew up in ... Loves (or loved) ... Often says (or said) ... Best friend ...">${escapeHtml(o.notes || "")}</textarea>
      <div class="step-actions step-actions--row">
        <button type="button" class="btn-ghost" id="other-cancel">Cancel</button>
        <button type="button" class="btn-primary" id="other-save">Save</button>
      </div>
    </section>`;
  return new Promise((resolve) => {
    const hb = document.getElementById("logout-btn");
    if (hb) hb.style.display = "none"; // this screen has its own Cancel and Save
    const done = resolve;
    resolve = (v) => { if (hb) hb.style.display = ""; done(v); };
    document.getElementById("other-cancel").onclick = () => resolve(false); // leave without changing anything
    document.getElementById("other-save").onclick = async () => {
      const result = await api("/api/capture/other", {
        canonical_name: state.canonicalName,
        display_name: o.display_name || state.canonicalName, // the name is set when the profile is made; this screen only changes the knowledge
        notes: document.getElementById("other-notes").value,
        clear: true, // the editor is where the person chooses what the notes say, including nothing
      });
      state.profile = result.profile;
      resolve(true);
    };
  });
}

function emptyProfileSummary(canonicalName) {
  return {
    canonical_name: canonicalName,
    display_name: canonicalName,
    core_complete: false,
    sections: {
      scan: { complete: false, count: 0 },
      voice: { complete: false, count: 0 },
      other: { complete: false },
    },
    scan_previews: [],
    voice_phases: [],
    other_preview: {},
    avatar_ready: false,
  };
}
let talkInterval = null;
let avatarTeardown = null;
async function renderAvatar(result, afterUpdate = false) {
  if (talkInterval) clearInterval(talkInterval);
  if (avatarTeardown) {
    avatarTeardown();
    avatarTeardown = null;
  }
  setPhase("", afterUpdate ? "Profile refined — playing avatar." : ""); // the avatar screen needs no caption under the title or in the footer
  const profile = result?.profile || state.profile;
  const mode = result?.avatar_mode || (profile?.avatar_ready ? "live" : "placeholder");
  root.innerHTML = `<section class="card avatar-stage" id="avatar-stage"></section>`;
  const stage = document.getElementById("avatar-stage");
  const { mountAvatarPlayer } = await import("/static/avatar-player.js?v=141");
  avatarTeardown = mountAvatarPlayer(stage, profile || {}, {
    onBackProfile: () => {
      avatarTeardown?.();
      avatarTeardown = null;
      renderProfileHome();
    },
  });
  if (mode === "placeholder" && !profile?.avatar_voice_url) {
    const note = document.createElement("p");
    note.className = "avatar-note muted";
    note.textContent =
      "Add your photo and voice for a live preview with your photo and recording.";
    stage.appendChild(note);
  }
}
async function saveNewProfile(thenAvatar = true) {
  const result = await api("/api/walkthrough/save-profile", {
    canonical_name: state.canonicalName,
  });
  if (result.status === "error" || result.status === "collision") {
    alert(result.message);
    return;
  }
  state.profile = result.profile;
  if (thenAvatar) renderAvatar(result);
}
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/'/g, "&#39;");
}

import {
  renderHelloIntro,
  renderNameStep,
  runSetupPipeline,
  setOnboardingView,
  setPhaseLabels,
} from "/static/onboarding-flow.js?v=25";

/** Startup screen: the chat brain, ears, speech engine and video animation load once, here, so everything is ready when a person is picked. */
const STARTUP_ROWS = [
  ["brain", "Brain", "Chat", "Gemma 3 12B"],
  ["speech", "Ears", "Speech to Text", "Whisper large-v3 turbo"],
  ["voice", "Voice", "Speech Engine", "Qwen3-TTS"],
  ["video", "Video", "Animation Engine", "SoulX FlashHead"],
];
const STARTUP_WORDS = { wait: "Starting…", download: "Downloading…", setup: "Setting up…", ok: "Ready", failed: "Could not install" };

async function waitForStartup() {
  const get = async () => {
    try {
      return await (await fetch("/api/chat/ready", { cache: "no-store" })).json();
    } catch {
      return null;
    }
  };
  let r = await get();
  if (r?.all) return;
  setPhaseLabels(phaseLabel, statusHint, "", "");
  root.innerHTML = `
    <div class="startup-card">
      <p class="startup-title">Starting AI Avatar Studio</p>
      <p class="startup-blurb">AI Avatar Studio runs several large AI models on your own computer: one to listen, one to think, one to speak and one to animate the face. The first start can take a few minutes while any missing models download. Once they are on your machine, starting is much quicker, but loading them into memory still takes a little while.</p>
      ${STARTUP_ROWS.map(
        ([k, label, what, real]) => (real = r?.names?.[k] || real) && `
        <div class="startup-row" data-part="${k}">
          <span class="startup-dot"></span>
          <span class="startup-name">${label} / <b>${what}</b> <span class="startup-real">(${real})</span></span>
          <span class="startup-state">Starting…</span>
          <div class="startup-bar"><i></i></div>
        </div>`,
      ).join("")}
      <p class="avatar-loading-note" id="startup-note"></p>
    </div>`;
  fetch("/api/chat/warmup", { method: "POST" }).catch(() => {});
  const t0 = Date.now();
  while (!r?.all) {
    if (r) {
      for (const [k] of STARTUP_ROWS) {
        const row = root.querySelector(`[data-part="${k}"]`);
        if (!row) continue;
        const st = r[k] || "wait";
        const pct = r.pct?.[k] ?? 0;
        if (r.names?.[k]) row.querySelector(".startup-real").textContent = `(${r.names[k]})`;
        row.classList.toggle("done", st === "ok");
        row.classList.toggle("downloading", st === "download");
        row.querySelector(".startup-state").textContent = st === "download" ? `Downloading… ${pct}%` : STARTUP_WORDS[st] || "Starting…";
        row.querySelector(".startup-bar i").style.width = st === "download" ? `${pct}%` : "0";
      }
      const note = document.getElementById("startup-note");
      if (note && Date.now() - t0 > 90000 && !Object.values(r.pct || {}).some((v) => v > 0 && v < 100)) note.textContent = "This is taking a long time. If it never finishes, run install.bat and start again.";
    }
    await new Promise((res) => setTimeout(res, 500));
    r = await get();
  }
}

/** Bottom left: how much graphics memory was found and which mode the models were chosen for. */
async function showHardware() {
  const el = document.getElementById("hw-mode");
  if (!el) return;
  try {
    const h = await (await fetch("/api/hardware", { cache: "no-store" })).json();
    const used = h.used_gb ? `${h.used_gb.toFixed(1)}` : "?";
    el.textContent = `VRAM: ${used} / ${h.total_gb.toFixed(1)} GB · ${h.mode}${h.simulated ? " (simulated)" : ""}`;
    el.style.whiteSpace = "nowrap";
    const over = h.used_gb && h.used_gb > h.total_gb;
    el.style.color = over ? "#ff6b6b" : "";
    if (over) el.textContent += " · OVER";
    el.title = `Chat brain: ${h.brain}. Ears: ${h.ears}.`;
  } catch {
    /* not available: leave it empty */
  }
}

let hwTimer = 0;
async function boot() {
  void showHardware();
  clearInterval(hwTimer);
  hwTimer = setInterval(showHardware, 3000); // keeps the memory figure live
  updateLogout();
  await waitForStartup();
  renderHelloIntro(root, showProfileChooser);
}

/** The Profile Management screen: pick a profile, or start a new one. "Change profile" comes straight back here. */
function showProfileChooser() {
  setPhaseLabels(phaseLabel, statusHint, "", "");
  renderNameStep(root, (name, fromList) => submitName(name, fromList));
}

boot();

// Tell the user when the front-end files changed on disk, instead of leaving an old page running.
(function watchBuild() {
  let first = null;
  setInterval(async () => {
    try {
      const r = await fetch("/api/build", { cache: "no-store" });
      const j = await r.json();
      if (first === null) first = j.build;
      else if (j.build !== first && !document.getElementById("new-build")) {
        const b = document.createElement("button");
        b.id = "new-build";
        b.textContent = "A new version is ready — click here to reload";
        b.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:99999;padding:.7rem;background:#2f7d4f;color:#fff;border:0;font-size:1rem;cursor:pointer";
        b.onclick = () => location.reload();
        document.body.appendChild(b);
      }
    } catch {
      /* server restarting: ignore */
    }
  }, 4000);
})();
