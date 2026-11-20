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

/** The Log out button shows whenever someone is signed in (a name has been chosen). */
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
  setCinematicMode(false);
  updateLogout();
  boot();
}
logoutBtn?.addEventListener("click", logOut);

async function submitName(name) {
  state.enteredName = name.trim();
  if (!state.enteredName) {
    alert("Please enter a name.");
    return;
  }
  const result = await api("/api/walkthrough/check-name", { name: state.enteredName });
  if (result.status === "exists") {
    setCinematicMode(false);
    state.profile = result.profile_summary;
    renderExistsPrompt(result.message);
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

function renderExistsPrompt(message) {
  setPhase("Returning user?", message);
  root.innerHTML = `
    <section class="card">
      <p class="prompt">${escapeHtml(message)}</p>
      <ul class="sections-list" id="sections-overview"></ul>
      <div class="btn-row">
        <button type="button" id="yes-you">Yes, that's me</button>
        <button type="button" class="secondary" id="no-you">No</button>
      </div>
    </section>
  `;
  renderSectionsList(state.profile);
  document.getElementById("yes-you").onclick = () => confirmExisting(true);
  document.getElementById("no-you").onclick = () => renderUniqueName();
}

function renderSectionsList(profile) {
  const ul = document.querySelector("#sections-overview");
  if (!ul || !profile) return;
  const s = profile.sections;
  const scanLabel = s.scan.complete
    ? `Complete (${s.scan.count} face angles)`
    : s.scan.count
      ? `Incomplete (${s.scan.count}/${4} face angles)`
      : "Not started";
  const voiceLabel = s.voice.complete
    ? `Complete (${s.voice.count})`
    : s.voice.count
      ? `Incomplete (${s.voice.count})`
      : "Not started";
  const otherLabel = s.other.complete ? "Complete" : "Not started";
  ul.innerHTML = `
    <li>Face scan <span class="badge ${s.scan.complete ? "done" : "pending"}">${scanLabel}</span></li>
    <li>Voice <span class="badge ${s.voice.complete ? "done" : "pending"}">${voiceLabel}</span></li>
    <li>Other <span class="badge ${s.other.complete ? "done" : "pending"}">${otherLabel}</span></li>
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
    setCinematicMode(true);
    root.innerHTML = `
      <section class="step-card step-card--center">
        <p class="step-eyebrow">Welcome back</p>
        <h2 class="step-title">Let's finish your setup</h2>
        <p class="step-copy">We'll only ask for what's still missing — face scan, voice, or both.</p>
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
  setCinematicMode(false);
  const name = state.profile?.display_name || state.canonicalName;
  setPhase(`Hi, ${name}`, "");
  const p = state.profile;
  root.innerHTML = `
    <section class="card home-card">
      <h2>${escapeHtml(name)}</h2>
      <ul class="sections-list" id="sections-overview"></ul>
      <p class="home-hint">Choose what you'd like to do.</p>
      <div class="home-actions">
        <button type="button" class="btn-primary" id="btn-preview">Preview avatar</button>
        <button type="button" class="btn-ghost" id="btn-update">Update your avatar</button>
        <button type="button" class="btn-ghost btn-danger" id="btn-delete">Delete and start fresh</button>
      </div>
    </section>
  `;
  renderSectionsList(p);
  document.getElementById("btn-preview").onclick = () => previewAvatar();
  document.getElementById("btn-update").onclick = () => renderUpdateAvatarScreen();
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

function renderUpdateAvatarScreen() {
  setPhase("Update", "Pick what to refresh");
  const p = state.profile;
  const s = p.sections;
  root.innerHTML = `
    <section class="card">
      <h2>Update your avatar</h2>
      <p class="step-copy" style="text-align:left;">Each step saves when you finish it.</p>
      <div class="update-menu">
        <button type="button" class="update-menu-item" id="upd-scan">
          <span class="update-menu-title">Face scan</span>
          <span class="badge ${s.scan.complete ? "done" : "pending"}">${s.scan.complete ? "Complete" : "Update"}</span>
        </button>
        <button type="button" class="update-menu-item" id="upd-voice">
          <span class="update-menu-title">Voice</span>
          <span class="badge ${s.voice.complete ? "done" : "pending"}">${s.voice.complete ? "Complete" : "Update"}</span>
        </button>
        <button type="button" class="update-menu-item" id="upd-other">
          <span class="update-menu-title">Other</span>
          <span class="badge ${s.other.complete ? "done" : "pending"}">${s.other.complete ? "Complete" : "Update"}</span>
        </button>
      </div>
      <div class="btn-row" style="margin-top:1.25rem;flex-direction:column;">
        <button type="button" class="btn-ghost" id="upd-back">Back</button>
      </div>
    </section>
  `;
  document.getElementById("upd-scan").onclick = () => runUpdateSection("scan");
  document.getElementById("upd-voice").onclick = () => runUpdateSection("voice");
  document.getElementById("upd-other").onclick = () => runUpdateSection("other");
  document.getElementById("upd-back").onclick = () => renderProfileHome();
}

async function runUpdateSection(section) {
  setCinematicMode(true);
  if (section === "scan") {
    const { runFaceScan } = await import("/static/orbit-scan.js?v=41");
    const result = await runFaceScan(root, state.canonicalName, api);
    if (result?.profile) state.profile = result.profile;
  } else if (section === "voice") {
    const { runVoiceCapture } = await import("/static/voice-capture.js");
    const result = await runVoiceCapture(root, state.canonicalName, api);
    if (result?.profile) state.profile = result.profile;
  } else {
    await openOtherEditor();
  }
  await saveNewProfile(false);
  setCinematicMode(false);
  renderUpdateAvatarScreen();
}

async function deleteExistingProfile() {
  const ok = confirm(
    "Delete your saved profile on this computer? You'll start fresh next time."
  );
  if (!ok) return;
  const result = await api("/api/walkthrough/delete-profile", {
    canonical_name: state.canonicalName,
  });
  if (result.status === "error") {
    alert(result.message);
    return;
  }
  state.enteredName = "";
  state.canonicalName = "";
  state.profile = null;
  boot();
}

async function openOtherEditor() {
  const o = state.profile?.other_preview || {};
  root.innerHTML = `
    <section class="step-card step-card--center">
      <h2 class="step-title">Other details</h2>
      <label class="field-label">Display name</label>
      <input type="text" class="step-input" id="other-display" value="${escapeAttr(o.display_name || state.canonicalName)}" />
      <label class="field-label">Notes</label>
      <textarea id="other-notes" rows="4">${escapeHtml(o.notes || "")}</textarea>
      <div class="step-actions step-actions--solo">
        <button type="button" class="btn-primary" id="other-save">Save</button>
      </div>
    </section>`;
  await new Promise((resolve) => {
    document.getElementById("other-save").onclick = async () => {
      const result = await api("/api/capture/other", {
        canonical_name: state.canonicalName,
        display_name: document.getElementById("other-display").value,
        notes: document.getElementById("other-notes").value,
      });
      state.profile = result.profile;
      resolve();
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

function renderReturningDashboard() {
  renderProfileHome();
}

function renderUpdateChooser() {
  const p = state.profile;
  setPhase("Update profile", "Check sections to refresh; low-quality data is dropped automatically.");
  root.innerHTML = `
    <section class="card">
      <h2>Select what to update</h2>
      <div class="update-grid">
        <div class="update-block" data-section="scan">
          <label class="check-head">
            <input type="checkbox" id="chk-scan" /> Scan
          </label>
          <div class="preview" id="prev-scan">
            <div class="scan-thumbs">${renderScanThumbs(p.scan_previews)}</div>
          </div>
        </div>
        <div class="update-block" data-section="voice">
          <label class="check-head">
            <input type="checkbox" id="chk-voice" /> Voice capture
          </label>
          <div class="preview" id="prev-voice">
            <div class="voice-phases" id="voice-list"></div>
          </div>
        </div>
        <div class="update-block" data-section="other">
          <label class="check-head">
            <input type="checkbox" id="chk-other" /> Other
          </label>
          <div class="preview" id="prev-other">
            <div class="other-preview">${escapeHtml(formatOther(p.other_preview))}</div>
          </div>
        </div>
      </div>
      <div class="btn-row">
        <button type="button" class="secondary" id="cancel-update">Cancel</button>
        <button type="button" id="apply-update">Update selected</button>
        <button type="button" class="secondary" id="skip-avatar">Skip — show avatar</button>
      </div>
    </section>
  `;

  wireUpdateCheckboxes();
  bindVoicePreviews(p.voice_phases);
  document.getElementById("cancel-update").onclick = () => renderReturningDashboard();
  document.getElementById("skip-avatar").onclick = () => skipUpdates();
  document.getElementById("apply-update").onclick = () => applyUpdates();
}

function wireUpdateCheckboxes() {
  document.querySelectorAll(".update-block").forEach((block) => {
    const cb = block.querySelector('input[type="checkbox"]');
    cb.addEventListener("change", () => {
      block.classList.toggle("expanded", cb.checked);
    });
  });
}

function renderScanThumbs(previews) {
  if (!previews?.length) return `<span class="muted">No scan images yet.</span>`;
  return previews
    .map(
      (x) =>
        `<img src="${escapeAttr(x.url)}" alt="${escapeAttr(x.caption)}" title="${escapeAttr(x.caption)}" />`
    )
    .join("");
}

function bindVoicePreviews(phases) {
  const list = document.getElementById("voice-list");
  if (!list) return;
  list.innerHTML = phases
    .map(
      (ph, i) => `
    <div class="voice-phase" data-index="${i}" role="button" tabindex="0">
      <span>${escapeHtml(ph.label)}</span>
      <div class="waveform">${(ph.peaks || []).map((h) => `<span style="height:${Math.round(h * 100)}%"></span>`).join("")}</div>
      <span class="badge">${ph.duration_sec?.toFixed(1) ?? "—"}s</span>
    </div>`
    )
    .join("");

  list.querySelectorAll(".voice-phase").forEach((el) => {
    const idx = Number(el.dataset.index);
    el.onclick = () => playVoicePhase(phases[idx], el);
    el.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") playVoicePhase(phases[idx], el);
    };
  });
}

let audioCtx = null;
let talkInterval = null;
let avatarTeardown = null;

function playVoicePhase(phase, el) {
  document.querySelectorAll(".voice-phase.playing").forEach((n) => n.classList.remove("playing"));
  el.classList.add("playing");

  if (phase.url) {
    const a = new Audio(phase.url);
    a.play();
    a.onended = () => el.classList.remove("playing");
    return;
  }

  const duration = (phase.duration_sec || 2) * 1000;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.frequency.value = 180 + Math.random() * 80;
  gain.gain.value = 0.08;
  osc.start();
  setTimeout(() => {
    osc.stop();
    el.classList.remove("playing");
  }, Math.min(duration, 2500));
}

async function skipUpdates() {
  const result = await api("/api/walkthrough/skip-updates", {
    canonical_name: state.canonicalName,
  });
  renderAvatar(result);
}

async function applyUpdates() {
  const sections = [];
  if (document.getElementById("chk-scan")?.checked) sections.push("scan");
  if (document.getElementById("chk-voice")?.checked) sections.push("voice");
  if (document.getElementById("chk-other")?.checked) sections.push("other");
  if (!sections.length) {
    await skipUpdates();
    return;
  }
  const result = await api("/api/walkthrough/apply-updates", {
    canonical_name: state.canonicalName,
    sections,
    merge_existing: true,
  });
  state.profile = result.profile;
  renderAvatar(result, true);
}

async function renderAvatar(result, afterUpdate = false) {
  if (talkInterval) clearInterval(talkInterval);
  if (avatarTeardown) {
    avatarTeardown();
    avatarTeardown = null;
  }
  setPhase("Avatar", afterUpdate ? "Profile refined — playing avatar." : "Your avatar");
  const profile = result?.profile || state.profile;
  const mode = result?.avatar_mode || (profile?.avatar_ready ? "live" : "placeholder");
  root.innerHTML = `<section class="card avatar-stage" id="avatar-stage"></section>`;
  const stage = document.getElementById("avatar-stage");
  const { mountAvatarPlayer } = await import("/static/avatar-player.js?v=95");
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
      "Finish face scan and voice capture for a live preview with your photo and recording.";
    stage.appendChild(note);
  }
}

function renderCapturePreview(profile) {
  const scans = profile?.scan_previews || [];
  const voices = profile?.voice_phases || [];
  const other = profile?.other_preview;
  const scanHtml = scans.length
    ? `<div class="scan-thumbs">${scans
        .map(
          (x) =>
            `<img src="${escapeAttr(x.url)}" alt="${escapeAttr(x.caption)}" title="${escapeAttr(x.caption)}" />`
        )
        .join("")}</div>`
    : "<p class='capture-empty'>No face angles yet — start Face scan.</p>";
  const voiceHtml = voices.length
    ? voices
        .map(
          (v) => `
        <div class="voice-preview-row">
          <span>${escapeHtml(v.label)}</span>
          ${v.url ? `<audio controls src="${escapeAttr(v.url)}" preload="none"></audio>` : "<span class='badge'>Recorded</span>"}
        </div>`
        )
        .join("")
    : "<p class='capture-empty'>No voice yet — start Voice capture.</p>";
  const otherHtml =
    other?.notes || other?.display_name
      ? `<p>${escapeHtml(other.display_name || state.canonicalName)} — ${escapeHtml(other.notes || "")}</p>`
      : "<p class='capture-empty'>Optional notes not set yet.</p>";
  return `
    <section class="capture-preview">
      <h3>What you've captured (draft)</h3>
      <div class="capture-block"><strong>Face</strong>${scanHtml}</div>
      <div class="capture-block"><strong>Voice</strong><div class="voice-preview-list">${voiceHtml}</div></div>
      <div class="capture-block"><strong>Other</strong>${otherHtml}</div>
    </section>
  `;
}

function renderNewUserFlow() {
  setPhase("New profile", "Complete scan, voice, and other settings.");
  const p = state.profile;
  root.innerHTML = `
    <section class="card">
      <h2>Let's set up ${escapeHtml(state.canonicalName)}</h2>
      <p>Use your camera and microphone below. Progress updates as you finish each step.</p>
      <ul class="sections-list" id="sections-overview"></ul>
      ${renderCapturePreview(p)}
      <div class="btn-row">
        <button type="button" id="cap-scan">Face scan</button>
        <button type="button" id="cap-voice">Voice capture</button>
        <button type="button" id="cap-other">Other details</button>
      </div>
      <p class="muted" style="margin-top:1rem;font-size:0.9rem;color:var(--muted)">
        Saved to a draft on this PC until you click Save profile at the end.
      </p>
      <div class="btn-row" style="margin-top: 1rem;">
        <button type="button" id="finish-new">Save profile &amp; show avatar</button>
      </div>
    </section>
  `;
  renderSectionsList(p);

  document.getElementById("cap-scan").onclick = () => runScanFlow();
  document.getElementById("cap-voice").onclick = () => runVoiceFlow();
  document.getElementById("cap-other").onclick = () => runOtherFlow();
  document.getElementById("finish-new").onclick = () => saveNewProfile();
}

async function runScanFlow() {
  const { runFaceScan } = await import("/static/orbit-scan.js?v=41");
  const result = await runFaceScan(root, state.canonicalName, api);
  if (result?.profile) {
    state.profile = result.profile;
    renderNewUserFlow();
    setPhase("Face scan", "Angles saved to your draft.");
  } else {
    renderNewUserFlow();
  }
}

async function runVoiceFlow() {
  const { runVoiceCapture } = await import("/static/voice-capture.js");
  const result = await runVoiceCapture(root, state.canonicalName, api);
  if (result?.profile) {
    state.profile = result.profile;
    renderNewUserFlow();
    setPhase("Voice", "Recording saved to your draft.");
  } else {
    renderNewUserFlow();
  }
}

function runOtherFlow() {
  const o = state.profile?.other_preview || {};
  root.innerHTML = `
    <section class="card">
      <h2>Other details</h2>
      <p>Optional: how your avatar should introduce itself, tone, etc.</p>
      <label class="field-label">Display name</label>
      <input type="text" id="other-display" value="${escapeAttr(o.display_name || state.canonicalName)}" />
      <label class="field-label">Notes</label>
      <textarea id="other-notes" rows="4" placeholder="e.g. casual tone, slower speech">${escapeHtml(o.notes || "")}</textarea>
      <div class="btn-row">
        <button type="button" class="secondary" id="other-back">Back</button>
        <button type="button" id="other-save">Save</button>
      </div>
    </section>
  `;
  document.getElementById("other-back").onclick = () => renderNewUserFlow();
  document.getElementById("other-save").onclick = async () => {
    const result = await api("/api/capture/other", {
      canonical_name: state.canonicalName,
      display_name: document.getElementById("other-display").value,
      notes: document.getElementById("other-notes").value,
    });
    state.profile = result.profile;
    renderNewUserFlow();
  };
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

async function captureStub(section) {
  const result = await api("/api/walkthrough/capture-stub", {
    canonical_name: state.canonicalName,
    section,
  });
  state.profile = result.profile;
  renderNewUserFlow();
  setPhase("Capture", `${section} saved.`);
}

function formatOther(o) {
  if (!o) return "No extra data yet.";
  return `Display: ${o.display_name || "—"}\nNotes: ${o.notes || "—"}\nPreferences: ${JSON.stringify(o.preferences || {}, null, 2)}`;
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
  setCinematicMode,
  setPhaseLabels,
} from "/static/onboarding-flow.js";

function boot() {
  updateLogout();
  renderHelloIntro(root, () => {
    setPhaseLabels(phaseLabel, statusHint, "You", "");
    renderNameStep(root, (name) => submitName(name));
  });
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
