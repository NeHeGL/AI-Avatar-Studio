/**
 * Pre-rendered full-quality clips ("cinematic" library) played over the live avatar.
 *
 * ml/cinematic_build.py renders each line (greeting, thinking, reactions, closing) with the best talking-video
 * model and writes data/media/<user>/cinematic/manifest.json (+ the spoken line as src/<id>.wav).
 * If that manifest exists, this module plays the clips on top of the avatar; if not, everything falls back
 * silently to the normal avatar.
 *
 * The picture plays MUTED (browsers always allow that) and the voice plays from the clip's own WAV through a
 * normal audio element, the same way the chat's own voice is played, so it does not depend on the browser's
 * video-codec or autoplay-with-sound rules. If sound is blocked, `onNote` says so.
 *
 *   const cinema = await loadCinematic(stageEl, profile, { onNote });   // null when there is no library
 *   await cinema?.play("greeting");                                    // resolves when the clip has finished
 *   cinema?.playIfIdle("thinking");                                    // fire-and-forget
 *   await cinema?.whenIdle();                                          // wait for a running clip
 */
export async function loadCinematic(stageEl, profile, opts = {}) {
  const note = (m) => {
    try {
      console.warn("[cinematic]", m);
      opts.onNote?.(m);
    } catch {
      /* ignore */
    }
  };
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
  video.muted = true; // picture only; the voice comes from the WAV below
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
    const list = fresh.length ? fresh : pool;
    const c = list[Math.floor(Math.random() * list.length)];
    lastByKind[kind] = c.id;
    return c;
  }

  function play(kind) {
    const clip = pick(kind);
    if (!clip) return Promise.resolve(false);
    if (current) return current;
    current = new Promise((resolve) => {
      let finished = false;
      let voice = null;
      let tailTimer = 0;
      const finish = (ok) => {
        if (finished) return;
        finished = true;
        clearTimeout(tailTimer);
        video.removeEventListener("ended", onVideoEnd);
        video.removeEventListener("error", onVideoErr);
        video.style.opacity = "0";
        try {
          voice?.pause();
        } catch {
          /* ignore */
        }
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
      const onVideoEnd = () => finish(true);
      const onVideoErr = () => {
        note(`The "${clip.id}" clip could not be played by this browser.`);
        finish(false);
      };
      video.addEventListener("ended", onVideoEnd);
      video.addEventListener("error", onVideoErr);
      video.src = `${base}/cinematic/${clip.video}`;
      video.currentTime = 0;

      voice = new Audio(`${base}/cinematic/src/${clip.id}.wav`);
      voice.preload = "auto";
      // end shortly after the words end: the rendered video is padded with a few seconds of silent idling
      voice.addEventListener("ended", () => {
        tailTimer = setTimeout(() => finish(true), 350);
      });
      voice.addEventListener("error", () => note(`The voice for "${clip.id}" could not be loaded.`));

      const vp = video.play();
      const ap = voice.play();
      vp.then(() => {
        video.style.opacity = "1";
      }).catch((e) => {
        note(`The "${clip.id}" picture could not start (${e?.name || e}).`);
        finish(false);
      });
      ap.catch((e) => note(`Clip sound was blocked by the browser (${e?.name || e}) — click the page once and try again.`));
      setTimeout(() => finish(true), Math.max(5000, (clip.seconds || 3) * 1000 + 3500)); // never get stuck
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
