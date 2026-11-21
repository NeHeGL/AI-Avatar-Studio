/**
 * Apple-style linear setup: one screen, one primary action.
 */

function transition(root, html) {
  root.classList.remove("step-enter");
  root.innerHTML = html;
  requestAnimationFrame(() => root.classList.add("step-enter"));
}

export function setCinematicMode(on) {
  document.body.classList.toggle("cinematic", on);
}

export function setPhaseLabels(phaseLabel, statusHint, label, hint) {
  if (phaseLabel) phaseLabel.textContent = label;
  if (statusHint) statusHint.textContent = hint || "";
}

function waitClick(id) {
  return new Promise((resolve) => {
    const el = document.getElementById(id);
    const go = () => {
      el.removeEventListener("click", go);
      resolve();
    };
    el.addEventListener("click", go);
  });
}

function waitClickEither(idA, idB) {
  return new Promise((resolve) => {
    const a = document.getElementById(idA);
    const b = document.getElementById(idB);
    const done = (which) => {
      a.removeEventListener("click", onA);
      b.removeEventListener("click", onB);
      resolve(which);
    };
    const onA = () => done("body");
    const onB = () => done("skip");
    a.addEventListener("click", onA);
    b.addEventListener("click", onB);
  });
}

function sectionNeeds(profile, key) {
  if (!profile?.sections) return true;
  return !profile.sections[key]?.complete;
}

/**
 * @param {object} opts
 * @param {boolean} [opts.isNewUser] offer optional body scan
 * @param {boolean} [opts.needFace] force face on/off
 * @param {boolean} [opts.needVoice] force voice on/off
 * @param {boolean} [opts.skipFinishCard] skip final “nice to meet you” (returning users)
 */
export async function runSetupPipeline(ctx, opts = {}) {
  const { root, phaseLabel, statusHint, api, state } = ctx;
  const prof = state.profile;
  const needFace = opts.needFace ?? sectionNeeds(prof, "scan");
  const needVoice = opts.needVoice ?? sectionNeeds(prof, "voice");
  const offerBody = opts.isNewUser === true && needFace;
  const skipFinish = opts.skipFinishCard === true;

  async function faceIntro() {
    setPhaseLabels(phaseLabel, statusHint, "Face", "");
    transition(
      root,
      `
      <section class="step-card step-card--center">
        <p class="step-eyebrow">Face</p>
        <h2 class="step-title">We'll scan your face</h2>
        <p class="step-copy">Move your head slowly around the circle so we can see you from every angle — like Face&nbsp;ID.</p>
        <div class="step-actions">
          <button type="button" class="btn-primary" id="ob-continue">Continue</button>
          <button type="button" class="btn-ghost" id="ob-skip-face">No face to scan — skip</button>
        </div>
      </section>`
    );
    // "body" = Continue, "skip" = no face (waitClickEither names its two answers that way)
    return waitClickEither("ob-continue", "ob-skip-face");
  }

  // A cartoon character, a pet or a voice-only avatar has no face to scan: remember that choice and move on.
  async function skipFaceScan() {
    const r = await api("/api/capture/skip-scan", { canonical_name: state.canonicalName });
    if (r?.profile) state.profile = r.profile;
  }

  let faceWasSkipped = false;
  if (needFace) {
    let skipFace = (await faceIntro()) === "skip";
    if (!skipFace) {
      const { runFaceScan } = await import("/static/orbit-scan.js?v=41");
      let faceResult = await runFaceScan(root, state.canonicalName, api);
      while (!faceResult?.profile) {
        // Back / cancel from the scan: offer the retry, or going on without a face scan (it used to dead-end here).
        if (confirm("Face scan wasn't saved. Open face scan again?\n\nCancel to continue without a face scan.")) {
          faceResult = await runFaceScan(root, state.canonicalName, api);
        } else {
          skipFace = true;
          break;
        }
      }
      if (!skipFace) state.profile = faceResult.profile;
    }
    if (skipFace) {
      await skipFaceScan();
      faceWasSkipped = true;
    }
  }

  if (offerBody && !faceWasSkipped) {
    setPhaseLabels(phaseLabel, statusHint, "Body", "Optional");
    transition(
      root,
      `
      <section class="step-card step-card--center">
        <p class="step-eyebrow">Optional</p>
        <h2 class="step-title">Fuller look?</h2>
        <p class="step-copy">Sit back from the camera and turn in your chair about 90° for each shot — front, right, back, left. Skip if you only need your face.</p>
        <div class="step-actions">
          <button type="button" class="btn-primary" id="ob-body">Scan body</button>
          <button type="button" class="btn-ghost" id="ob-skip-body">Skip</button>
        </div>
      </section>`
    );
    if ((await waitClickEither("ob-body", "ob-skip-body")) === "body") {
      const { runBodyScan } = await import("/static/body-scan.js");
      const bodyResult = await runBodyScan(root, state.canonicalName, api);
      if (bodyResult?.profile) state.profile = bodyResult.profile;
    }
  }

  if (needVoice) {
    setPhaseLabels(phaseLabel, statusHint, "Voice Scan", "");
    transition(
      root,
      `
      <section class="step-card step-card--center">
        <p class="step-eyebrow">Voice Scan</p>
        <h2 class="step-title">Teach it your voice</h2>
        <p class="step-copy">You'll read a short line aloud so your avatar can sound like you.</p>
        <div class="step-actions step-actions--solo">
          <button type="button" class="btn-primary" id="ob-voice">Continue</button>
        </div>
      </section>`
    );
    await waitClick("ob-voice");

    const { runVoiceCapture } = await import("/static/voice-capture.js");
    let voiceResult = await runVoiceCapture(root, state.canonicalName, api);
    while (!voiceResult?.profile) {
      transition(
        root,
        `
        <section class="step-card step-card--center">
          <p class="step-eyebrow">Voice Scan</p>
          <h2 class="step-title">Teach it your voice</h2>
          <p class="step-copy">You'll read a short line aloud so your avatar can sound like you.</p>
          <div class="step-actions step-actions--solo">
            <button type="button" class="btn-primary" id="ob-voice-retry">Continue</button>
          </div>
        </section>`
      );
      await waitClick("ob-voice-retry");
      voiceResult = await runVoiceCapture(root, state.canonicalName, api);
    }
    state.profile = voiceResult.profile;
  }

  if (needFace || needVoice) {
    await api("/api/capture/other", {
      canonical_name: state.canonicalName,
      display_name: state.canonicalName,
      notes: "",
    });
  }

  if (!skipFinish) {
    setPhaseLabels(phaseLabel, statusHint, "Done", "");
    const first = state.canonicalName.split(/\s+/)[0] || state.canonicalName;
    transition(
      root,
      `
      <section class="step-card step-card--center">
        <p class="step-eyebrow">All set</p>
        <h2 class="step-title">Nice to meet you, ${ctx.escapeHtml(first)}</h2>
        <p class="step-copy">We'll save your profile and bring your avatar to life.</p>
        <div class="step-actions step-actions--solo">
          <button type="button" class="btn-primary" id="ob-save">Continue</button>
        </div>
      </section>`
    );
    await waitClick("ob-save");
  }

  setCinematicMode(false);
  return true;
}

export function renderHelloIntro(root, onContinue) {
  setCinematicMode(true);
  transition(
    root,
    `
    <section class="hello-screen">
      <div class="hello-ambient" aria-hidden="true"></div>
      <div class="hello-layers">
        <span class="hello-ghost hello-ghost--1" aria-hidden="true">Hello</span>
        <span class="hello-ghost hello-ghost--2" aria-hidden="true">Hello</span>
        <h1 class="hello-text" aria-label="Hello">
          <span class="hello-text-inner">Hello</span>
        </h1>
      </div>
      <div class="step-actions step-actions--solo hello-actions">
        <button type="button" class="btn-primary btn-pill" id="hello-continue">Continue</button>
      </div>
    </section>`
  );
  document.getElementById("hello-continue").onclick = onContinue;
}

export async function renderNameStep(root, onSubmit) {
  let users = [];
  try {
    users = (await (await fetch("/api/users")).json()).users || [];
  } catch (_) {}
  const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const hasProfiles = users.length > 0;
  // With profiles: a real drop-down (pick one, or choose "New profile..." to type a name). Without: just the name box.
  transition(
    root,
    `
    <section class="step-card step-card--center">
      <h2 class="step-title">Profile Management</h2>
      ${
        hasProfiles
          ? `<select class="step-input step-select" id="profile-select">
        <option value="" disabled selected>Select a profile…</option>
        ${users.map((u) => `<option value="${esc(u)}">${esc(u)}</option>`).join("")}
        <option value="__new__">＋ New profile…</option>
      </select>`
          : ""
      }
      <input type="text" class="step-input" id="name-input" placeholder="${hasProfiles ? "Type the new profile's full name" : "What's your name?"}" autocomplete="off" ${hasProfiles ? "hidden" : ""} />
      <div class="step-actions step-actions--solo">
        <button type="button" class="btn-primary" id="name-ok">OK</button>
      </div>
    </section>`
  );
  const input = document.getElementById("name-input");
  const select = document.getElementById("profile-select");
  const chosen = () => (select && select.value !== "__new__" ? select.value : input.value.trim());
  const submit = () => {
    const name = chosen();
    const fromList = Boolean(select && select.value && select.value !== "__new__"); // picked from the list, not typed
    if (name) onSubmit(name, fromList);
    else (select && select.value !== "__new__" ? select : input).focus();
  };
  if (select) {
    select.focus();
    select.addEventListener("change", () => {
      const makingNew = select.value === "__new__";
      input.hidden = !makingNew;
      if (makingNew) input.focus();
    });
  } else {
    input.focus();
  }
  document.getElementById("name-ok").onclick = submit;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submit();
  });
  select?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && select.value) submit();
  });
}
