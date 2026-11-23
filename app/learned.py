"""What the avatar picks up on its own while talking: data/profiles/<name>/facts/learned.txt, one short fact per line.

Facts (what the person wrote) is the source of truth; this file is what the avatar noted in conversation, such as a birthday it was told or
something the person it talks to likes. The person can read and edit it. If the two ever disagree, Facts wins (the prompt says so, and a new
fact that contradicts Facts is never saved).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

MAX_LINES = 2000  # the whole file is searched before each reply and only the matching lines go into the prompt (see recall.py), so it can grow large
MAX_LINE_CHARS = 200
MAX_NEW_PER_TURN = 3

_lock = threading.Lock()
_news: dict[str, list[str]] = {}  # facts noted since the page last asked, so it can say "Noted: ..."


def path(name: str) -> Path:
    from app.storage import profile_dir

    return profile_dir(name) / "facts" / "learned.txt"


def _key(name: str) -> str:
    """The same person under any spelling of the name ("Jane Doe", "jane_doe") is one key."""
    from app.storage import profile_dir

    return profile_dir(name).name


def read(name: str) -> list[str]:
    try:
        return [l.strip() for l in path(name).read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []


def write(name: str, lines: list[str]) -> list[str]:
    """Replace the whole file (the person edited it). Blank lines and exact repeats are dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for l in lines:
        l = " ".join(str(l).split())[:MAX_LINE_CHARS]
        k = _norm(l)
        if l and k not in seen:
            seen.add(k)
            out.append(l)
    p = path(name)
    with _lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out[-MAX_LINES:]) + ("\n" if out else ""), encoding="utf-8")
    return out[-MAX_LINES:]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def add(name: str, facts: list[str]) -> list[str]:
    """Add new facts (skipping ones already there). Returns the ones that were really added."""
    with _lock:
        have = read(name)
        seen = {_norm(l) for l in have}
        added: list[str] = []
        for f in facts:
            f = " ".join(str(f).split())[:MAX_LINE_CHARS]
            if f and _norm(f) not in seen:
                seen.add(_norm(f))
                added.append(f)
        if added:
            p = path(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join((have + added)[-MAX_LINES:]) + "\n", encoding="utf-8")
            _news.setdefault(_key(name), []).extend(added)
    return added


def take_news(name: str) -> list[str]:
    with _lock:
        return _news.pop(_key(name), [])


_SYSTEM = (
    "You keep notes for an avatar. You read a short piece of conversation between the avatar and the person talking to it, and decide whether "
    "the PERSON told it any lasting facts worth remembering. Reply with ONLY a JSON list of strings, and [] when there is nothing to note."
)


def extract(ask, display: str, facts: str, learned: list[str], dialogue: list[tuple[str, str]]) -> list[str]:
    """Ask the chat model for the facts in this conversation. `ask(system, user)` returns the model's text; `dialogue` is [(avatar said, person
    said), ...]. Returns up to MAX_NEW_PER_TURN short sentences, each starting with who it is about."""
    convo = "\n".join(f'The avatar said: "{a}"\nThe person said: "{p}"' if a else f'The person said: "{p}"' for a, p in dialogue)
    prompt = (
        f"The avatar plays {display}.\n\n"
        f"WHAT THE AVATAR KNOWS (this is the truth, it outranks anything else):\n{facts.strip()[:2500] or '(nothing written)'}\n\n"
        f"WHAT IT HAS ALREADY NOTED:\n{chr(10).join('- ' + l for l in learned[-40:]) or '(nothing yet)'}\n\n"
        f"THE CONVERSATION:\n{convo}\n\n"
        "RULES:\n"
        "- Note only facts the PERSON stated. The avatar's own words are only there so you can understand a short answer such as \"yes\" or \"June fifth\".\n"
        "- Lasting facts only: a name, a birthday, family, friends, pets, a job, where they live, history, things they like or dislike, habits, "
        "beliefs. Not moods, plans for today, small talk or questions.\n"
        f"- Write each fact as one short plain sentence that starts with who it is about: the person's own name (\"Jeff likes pop tarts.\"), or "
        f"\"{display}\" when they are telling you about the life of {display}. Use the person's name whenever it appears anywhere above, in the notes "
        f"or the conversation, even if they did not say it in this latest message. Only if their name is not known anywhere, write \"The person I'm talking to\".\n"
        "- Leave out anything already known or already noted.\n"
        "- Leave out anything that CONTRADICTS what the avatar knows: what it knows is right and the person may be mistaken or joking.\n"
        f"- At most {MAX_NEW_PER_TURN} facts. If the person only chatted or asked something, answer [].\n"
        "Answer with the JSON list only."
    )
    raw = ask(_SYSTEM, prompt)
    return parse(raw)


def parse(raw: str) -> list[str]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        s = " ".join(str(it).split()).strip(" -•\"")
        if 6 <= len(s) <= MAX_LINE_CHARS:
            out.append(s)
    return out[:MAX_NEW_PER_TURN]
