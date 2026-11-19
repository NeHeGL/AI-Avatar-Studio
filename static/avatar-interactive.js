/**
 * Prototype: Jeff in a small 3D room — walk (WASD), dance (Space), talk (voice chat).
 * Head is your center scan; body is a simple stand-in until a real rig exists.
 */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { createVoiceChat } from "/static/voice-chat.js?v=3";

function mountChatChrome(stageEl, profile, name) {
  const portrait = profile.avatar_face_url || "";
  stageEl.innerHTML = `
    <div class="avatar-figure interactive-room" id="avatar">
      <canvas class="avatar-canvas" id="avatar-world-canvas"></canvas>
      <p class="world-hud">WASD move · Space dance · Mouse drag orbit</p>
    </div>
    <h2>${escapeHtml(name)}</h2>
    <p id="avatar-status" class="avatar-note muted">Interactive prototype — voice chat still works.</p>
    <div class="chat-meter" title="Your microphone level"><div class="chat-meter-bar" id="chat-meter-bar"></div><div class="chat-meter-thr" id="chat-meter-thr"></div></div>
    <div class="chat-log" id="chat-log" aria-live="polite"></div>
    <form class="chat-type" id="chat-type" autocomplete="off">
      <input id="chat-input" type="text" placeholder="…or type a message" />
      <button type="submit" class="btn-ghost">Send</button>
    </form>
    <audio id="avatar-audio" preload="auto"></audio>
    <div class="avatar-actions btn-row" style="justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
      <button type="button" class="btn-primary" id="chat-mute" disabled>Mute mic</button>
      <button type="button" class="btn-ghost" id="back-portrait">Portrait chat</button>
      <button type="button" class="btn-ghost" id="back-profile">Back to profile</button>
    </div>
  `;
  return { portrait };
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * @param {HTMLElement} stageEl
 * @param {object} profile
 * @param {{ onPortrait?: () => void, onBackProfile?: () => void }} hooks
 */
export function mountInteractivePlayer(stageEl, profile, hooks = {}) {
  const name = profile.display_name || profile.canonical_name || "Avatar";
  const { portrait } = mountChatChrome(stageEl, profile, name);
  const canvas = stageEl.querySelector("#avatar-world-canvas");
  const statusEl = stageEl.querySelector("#avatar-status");
  const muteBtn = stageEl.querySelector("#chat-mute");
  const logEl = stageEl.querySelector("#chat-log");
  const meterBar = stageEl.querySelector("#chat-meter-bar");
  const meterThr = stageEl.querySelector("#chat-meter-thr");
  const audio = stageEl.querySelector("#avatar-audio");

  let destroyed = false;
  let chat = null;
  let speaking = false;
  let dancing = false;
  let danceUntil = 0;
  const keys = new Set();

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121820);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 1.55, 3.2);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.MeshStandardMaterial({ color: 0x2a3544, roughness: 0.85 }),
  );
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  scene.add(new THREE.GridHelper(12, 24, 0x3d5a80, 0x243040));
  scene.add(new THREE.HemisphereLight(0xc8e0ff, 0x2a2830, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 0.65);
  key.position.set(2, 4, 3);
  scene.add(key);

  const avatar = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.35, 0.42, 1.1, 20),
    new THREE.MeshStandardMaterial({ color: 0x2e3744, roughness: 0.7 }),
  );
  body.position.y = 0.55;
  avatar.add(body);

  const headGeo = new THREE.PlaneGeometry(0.55, 0.72);
  const headMat = new THREE.MeshBasicMaterial({ transparent: true, side: THREE.DoubleSide });
  const head = new THREE.Mesh(headGeo, headMat);
  head.position.set(0, 1.35, 0.01);
  avatar.add(head);
  if (portrait) {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      const tex = new THREE.Texture(im);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      headMat.map = tex;
      headMat.needsUpdate = true;
    };
    im.src = portrait;
  }
  scene.add(avatar);

  const vel = { x: 0, z: 0 };
  const pos = { x: 0, z: 0 };
  let yaw = 0;
  let orbit = { dragging: false, lx: 0, ly: 0, ay: 0.3, ap: 0.15, dist: 3.2 };

  function resize() {
    const w = canvas.clientWidth || 440;
    const h = Math.max(280, Math.round(w * 0.72));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(canvas.parentElement);

  function onKey(e, down) {
    const k = e.key.toLowerCase();
    if (["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(k)) {
      if (down) keys.add(k);
      else keys.delete(k);
      e.preventDefault();
    }
    if (down && e.code === "Space") {
      dancing = true;
      danceUntil = performance.now() + 2800;
      e.preventDefault();
    }
  }
  const onDown = (e) => onKey(e, true);
  const onUp = (e) => onKey(e, false);
  window.addEventListener("keydown", onDown);
  window.addEventListener("keyup", onUp);
  canvas.addEventListener("pointerdown", (e) => {
    orbit.dragging = true;
    orbit.lx = e.clientX;
    orbit.ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!orbit.dragging) return;
    orbit.ay -= (e.clientX - orbit.lx) * 0.006;
    orbit.ap = Math.max(-0.2, Math.min(0.55, orbit.ap - (e.clientY - orbit.ly) * 0.004));
    orbit.lx = e.clientX;
    orbit.ly = e.clientY;
  });
  canvas.addEventListener("pointerup", () => {
    orbit.dragging = false;
  });

  let raf = 0;
  const t0 = performance.now();
  function frame() {
    if (destroyed) return;
    const t = (performance.now() - t0) / 1000;
    const speed = 1.6;
    let mx = 0;
    let mz = 0;
    if (keys.has("w") || keys.has("arrowup")) mz -= 1;
    if (keys.has("s") || keys.has("arrowdown")) mz += 1;
    if (keys.has("a") || keys.has("arrowleft")) mx -= 1;
    if (keys.has("d") || keys.has("arrowright")) mx += 1;
    if (mx || mz) {
      const len = Math.hypot(mx, mz) || 1;
      mx /= len;
      mz /= len;
      yaw = Math.atan2(mx, mz);
      pos.x += mx * speed * 0.016;
      pos.z += mz * speed * 0.016;
      pos.x = Math.max(-5.5, Math.min(5.5, pos.x));
      pos.z = Math.max(-5.5, Math.min(5.5, pos.z));
    }
    avatar.position.set(pos.x, 0, pos.z);
    avatar.rotation.y = yaw;

    const dance = dancing && performance.now() < danceUntil;
    if (!dance && danceUntil && performance.now() >= danceUntil) dancing = false;
    const breathe = Math.sin(t * 1.4) * 0.012;
    const walkBob = mx || mz ? Math.sin(t * 9) * 0.04 : 0;
    const talkBob = speaking ? Math.sin(t * 11) * 0.035 : 0;
    const danceY = dance ? Math.abs(Math.sin(t * 8)) * 0.25 : 0;
    const danceSpin = dance ? t * 2.2 : 0;
    avatar.position.y = breathe + walkBob + talkBob + danceY;
    head.position.y = 1.35 + talkBob * 0.5;
    if (dance) avatar.rotation.y = yaw + danceSpin;

    const cx = pos.x + Math.sin(orbit.ay) * Math.cos(orbit.ap) * orbit.dist;
    const cz = pos.z + Math.cos(orbit.ay) * Math.cos(orbit.ap) * orbit.dist;
    const cy = 1.35 + Math.sin(orbit.ap) * orbit.dist;
    camera.position.set(cx, cy, cz);
    camera.lookAt(pos.x, 1.2, pos.z);

    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  async function playClip(blob, videoUrl = "") {
    if (destroyed) return;
    if (videoUrl) {
      const vid = document.createElement("video");
      vid.src = videoUrl;
      vid.playsInline = true;
      await vid.play().catch(() => {});
      await new Promise((r) => vid.addEventListener("ended", r, { once: true }));
      return;
    }
    speaking = true;
    statusEl.textContent = "Speaking…";
    audio.src = URL.createObjectURL(blob);
    await audio.play().catch(() => {});
    await new Promise((r) => audio.addEventListener("ended", r, { once: true }));
    speaking = false;
    URL.revokeObjectURL(audio.src);
  }

  function addLine(who, text) {
    const row = document.createElement("p");
    row.className = `chat-line ${who}`;
    row.textContent = (who === "you" ? "You: " : `${name}: `) + text;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function startChat() {
    statusEl.textContent = "Loading voice models…";
    try {
      const r = await fetch("/api/chat/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canonical_name: profile.canonical_name }),
      });
      const j = await r.json();
      if (j.status !== "ok") throw new Error(j.message || "chat could not start");
      chat = createVoiceChat({
        name: profile.canonical_name,
        token: j.token,
        playClip,
        onSuperseded: () => {
          audio.pause();
          muteBtn.disabled = true;
        },
        onStatus: (t) => !destroyed && (statusEl.textContent = t),
        onLevel: (v, thr) => {
          meterBar.style.width = `${Math.min(100, v * 600)}%`;
          meterThr.style.left = `${Math.min(100, thr * 600)}%`;
        },
        onLine: addLine,
      });
      await chat.start();
      muteBtn.disabled = false;
      statusEl.textContent = "Use WASD to move · Space to dance · mic or type to talk";
      await chat.say(`Hello, I'm ${j.display_name || name}.`);
    } catch (e) {
      statusEl.textContent = `Chat problem: ${e.message || e}`;
    }
  }

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
  stageEl.querySelector("#back-portrait")?.addEventListener("click", () => hooks.onPortrait?.());
  stageEl.querySelector("#back-profile")?.addEventListener("click", () => hooks.onBackProfile?.());

  void startChat();

  return () => {
    destroyed = true;
    cancelAnimationFrame(raf);
    window.removeEventListener("keydown", onDown);
    window.removeEventListener("keyup", onUp);
    audio.pause();
    chat?.stop();
    renderer.dispose();
  };
}