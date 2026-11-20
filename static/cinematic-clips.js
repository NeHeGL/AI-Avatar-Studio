/**
 * Pre-rendered full-quality clips ("cinematic" library) played over the live avatar.
 *
 * ml/cinematic_build.py renders each line (greeting, thinking, reactions, closing) with the best talking-video
 * model and writes data/media/<user>/cinematic/manifest.json. If that manifest exists, this module plays the
 * clips on top of the avatar; if it doesn't, everything silently falls back to the normal avatar.
 *
 *   const cinema = await loadCinematic(stageEl, profile);   // null when there is no library
 *   await cinema?.play("greeting");                          // resolves when the clip has finished
 *   cinema?.playIfIdle("thinking");                          // fire-and-forget
 *   await cinema?.whenIdle();                                // wait for a running clip before speaking
 */
export async function loadCinematic(stageEl, profile) {
  const frames = (profile.scan_frames?.length ? profile.scan_frames : profile.scan_previews) || [];
  const sample = profile.avatar_face_url || frames[0]?.url || "";
  const base = (sample.match(/^(.*\/media\/[^/]+)\//) || [])[1];
  if (!base) return null;

  let manifest;
  try {
    const r = await fetch(`${base}/cinematic/manifest.json`, { cache: "no-store" });
    if (!r.ok) return null;
    manifest = await r.json();
  } catch {
    return null;
  }
  const clips = (manifest.clips || []).filter((c) => c.video);
  if (!clips.length) return null;

  const figure = stageEl.querySelector("#avatar");
  if (!figure) return null;
  figure.style.position = "relative";
  const video = document.createElement("video");
  video.playsInline = true;
  video.preload = "auto";
  video.style.cssText =
    "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;opacity:0;pointer-events:none;transition:opacity .18s ease;";
  figure.appendChild(video);

  let current = null; // promise of the running clip
  const lastByKind = {};

  function pick(kind) {
    const pool = clips.filter((c) => c.kind === kind);
    if (!pool.length) return null;
    const fresh = pool.filter((c) => c.id !== lastByKind[kind]);
    const c = (fresh.length ? fresh : pool)[Math.floor(Math.random() * (fresh.length || pool.length))];
    lastByKind[kind] = c.id;
    return c;
  }

  function play(kind) {
    const clip = pick(kind);
    if (!clip) return Promise.resolve(false);
    if (current) return current;
    current = new Promise((resolve) => {
      let finished = false;
      const finish = (ok) => {
        if (finished) return;
        finished = true;
        video.removeEventListener("ended", onEnd);
        video.removeEventListener("error", onErr);
        video.style.opacity = "0";
        setTimeout(() => {
          try {
            video.pause();
          } catch {
            /* ignore */
          }
        }, 200);
        current = null;
        resolve(ok);
      };
      const onEnd = () => finish(true);
      const onErr = () => finish(false);
      video.addEventListener("ended", onEnd);
      video.addEventListener("error", onErr);
      video.src = `${base}/cinematic/${clip.video}`;
      video.currentTime = 0;
      video
        .play()
        .then(() => {
          video.style.opacity = "1";
        })
        .catch(() => finish(false)); // e.g. the browser blocked sound before any click: fall back silently
      setTimeout(() => finish(true), Math.max(4000, (clip.seconds || 3) * 1000 + 2500)); // never get stuck
    });
    return current;
  }

  return {
    clips,
    play,
    playIfIdle: (kind) => (current ? Promise.resolve(false) : play(kind)),
    whenIdle: () => current || Promise.resolve(),
    get busy() {
      return Boolean(current);
    },
    dispose() {
      try {
        video.pause();
        video.remove();
      } catch {
        /* ignore */
      }
    },
  };
}
