"""Choosing which notes go into the prompt.

The chat model reads a fixed window (about 4,000 tokens) that cannot grow without more graphics memory, and everything shares it: the standing
instructions, the Facts, what the avatar has learned, the conversation and the reply. Facts and AI Learned can be as long as the person likes,
so before each reply they are searched here, on the processor (a few milliseconds, no graphics memory), and only the pieces that matter for what
was just said go in, within a fixed number of tokens. Short notes that already fit go in whole, unchanged.
"""
from __future__ import annotations

import math
import re

CHARS_PER_TOKEN = 3.0  # measured about 3.6 for English on the chat model; 3.0 overestimates a little, so a prompt never ends up too long

_STOP = frozenset(
    "a an and are as at be but by do does did for from had has have he her his i if in is it its me my of on or our she so than that the their them "
    "then there they this to was we were what when where which who why will with you your yours about can could just like not no yes".split()
)


# Everyday words that mean nearly the same thing count as one, so "any holiday plans?" finds "is planning a trip to Iceland".
_SAME = [
    "trip travel vacation holiday journey flight abroad visit voyage",
    "job work career employed profession occupation company office boss",
    "kid child children son daughter baby grandkid grandchild",
    "born birth birthday age",
    "food eat meal dinner lunch breakfast cook recipe snack dessert",
    "music song band guitar sing singer concert piano",
    "movie film cinema show series tv",
    "school class teacher college university study grade homework",
    "home house live apartment address city town neighbourhood neighborhood",
    "sibling sister brother",
    "parent mom mum mother dad father",
    "dog puppy",
    "cat kitten",
    "car drive truck bike",
    "game gaming videogame",
    "sick ill illness health doctor hospital",
    "die death dead passed funeral grave",
    "married wife husband spouse wedding partner",
]
_CANON = {}
for _g in _SAME:
    _ws = _g.split()
    for _w in _ws:
        _CANON[_w] = _ws[0]


def tokens_est(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")):
        if w not in _STOP and len(w) > 1:
            st = _stem(w)
            out.append(_CANON.get(w) or _CANON.get(st) or st)
    return out


def split_chunks(text: str, max_words: int = 45) -> list[str]:
    """Free text into pieces: one per line, and long paragraphs cut at sentence ends into pieces of at most `max_words` words."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line.split()) <= max_words:
            out.append(line)
            continue
        cur: list[str] = []
        n = 0
        for sent in re.split(r"(?<=[.!?])\s+", line):
            k = len(sent.split())
            if cur and n + k > max_words:
                out.append(" ".join(cur))
                cur, n = [], 0
            cur.append(sent)
            n += k
        if cur:
            out.append(" ".join(cur))
    return out


def select(chunks: list[str], query: str, budget_tokens: int, core_words: int = 0, prefer_recent: bool = False) -> list[str]:
    """The pieces to include, in their original order, within `budget_tokens`.

    Everything that fits is returned whole. Otherwise: the first `core_words` words (who the person is) always go in, then the pieces that best match
    `query` (a BM25-style keyword score), and if the query matches little, the newest pieces (`prefer_recent`) or the next ones in order."""
    if sum(tokens_est(c) for c in chunks) <= budget_tokens:
        return list(chunks)
    n = len(chunks)
    chosen: set[int] = set()
    used = 0

    def take(i: int) -> bool:
        nonlocal used
        t = tokens_est(chunks[i]) + 2
        if i in chosen or used + t > budget_tokens:
            return False
        chosen.add(i)
        used += t
        return True

    got = 0
    for i in range(n):  # the core: who they are
        if got >= core_words:
            break
        got += len(chunks[i].split())
        take(i)
    docs = [words(c) for c in chunks]
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    avg = (sum(len(d) for d in docs) / n) or 1.0
    q = set(words(query))
    scores = []
    for i, d in enumerate(docs):
        s = 0.0
        if d:
            tf: dict[str, int] = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            for w in q:
                if w in tf:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append(s)
    for i in sorted((i for i in range(n) if scores[i] > 0), key=lambda i: (-scores[i], -i if prefer_recent else i)):
        take(i)
    order = range(n - 1, -1, -1) if prefer_recent else range(n)  # fill what is left: the newest notes, or the next ones in order
    for i in order:
        if used >= budget_tokens:
            break
        take(i)
    return [chunks[i] for i in sorted(chosen)]


def fit_history(history: list[dict], budget_tokens: int, keep_at_least: int = 2) -> list[dict]:
    """The most recent messages that fit in `budget_tokens` (the oldest go first). The first kept message is always the person's."""
    kept: list[dict] = []
    used = 0
    for m in reversed(history):
        t = tokens_est(m.get("content", "")) + 6
        if len(kept) >= keep_at_least and used + t > budget_tokens:
            break
        kept.append(m)
        used += t
    kept.reverse()
    while len(kept) > 1 and kept[0].get("role") != "user":
        kept.pop(0)
    return kept
