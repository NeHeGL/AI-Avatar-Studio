"""A wide bank of everyday things to talk about, so the avatar has plenty to say that is NOT a recital of its notes.

pick() hands back a few ideas from different areas, plus a hint about the season or an upcoming holiday. They are offered as options,
never as orders, so the conversation stays natural and different every time.
"""
from __future__ import annotations

import datetime
import random

TOPICS: dict[str, list[str]] = {
    "food": [
        "what they would cook if they had all evening", "the best sandwich they have ever had", "a comfort food that always works",
        "pizza toppings and which ones are crimes", "breakfast for dinner", "a snack they could eat every day", "coffee or tea, and how they take it",
        "a dish they have never been brave enough to try", "the perfect burger", "soup weather", "street food", "baking, and whether it ever works out",
    ],
    "music": [
        "a song stuck in their head", "the first album they ever loved", "music for driving", "a band everyone loves that they never got into",
        "music to work to", "a song that instantly changes their mood", "learning an instrument", "a concert they would love to have seen",
    ],
    "movies and shows": [
        "something good they watched recently", "a movie they can quote the whole way through", "a show they keep rewatching",
        "a sequel that was better than the original", "a movie that was overrated", "what makes a great villain", "cosy movies versus scary ones",
        "an old show they miss", "who they would want to play them in a film",
    ],
    "games and play": [
        "a game they could play forever", "board game nights", "a childhood game", "puzzles and riddles", "a video game with a great soundtrack",
        "cards, and who cheats", "a game they were secretly terrible at",
    ],
    "tech": [
        "the gadget they would not give up", "how fast phones changed everything", "robots and what they would actually be useful for",
        "a piece of technology that never worked right", "retro tech and old computers", "what the future of cars might look like", "smart homes: clever or creepy",
    ],
    "nature and weather": [
        "what the sky is doing right now", "their favourite kind of weather", "the first snowfall or the first warm day", "storms and thunder",
        "a place in nature they find peaceful", "birds and what they are up to", "stars and the night sky", "a tree they love", "the sound of rain",
    ],
    "travel and places": [
        "a place they would love to see", "the best trip they have ever taken", "a small town worth visiting", "driving versus flying", "a city they could live in",
        "somewhere cold or somewhere hot", "a road trip dream", "the view they would pick for a window",
    ],
    "everyday life": [
        "their morning routine", "something small that made today better", "the best part of the week", "a chore they secretly enjoy",
        "how they like to unwind", "a perfect lazy day", "something they are looking forward to", "tidying up and the satisfaction of it",
        "early mornings versus late nights", "a little treat they allow themselves",
    ],
    "hobbies and making things": [
        "something they like making with their hands", "a hobby they would take up with more time", "gardening, even a few pots", "fixing things around the house",
        "woodworking or tinkering", "drawing, painting or doodling", "writing, even just lists", "photography and taking a good picture", "collecting something odd",
    ],
    "work and learning": [
        "a skill they would love to learn", "the best advice they ever got", "what makes a good boss", "a job they would never want", "a lesson that took them too long to learn",
        "something they got better at over time", "the thing they always procrastinate on",
    ],
    "animals and pets": [
        "pets, and the odd things they do", "dogs versus cats", "an animal they find hilarious", "the smartest animal they can think of",
        "a pet from their past", "wildlife they have spotted", "what it would be like to have a pet owl",
    ],
    "sports and moving": [
        "a sport they like to watch", "a walk they enjoy", "an underdog moment", "stretching, and why nobody does it", "learning to swim or skate or ride",
        "a team they cheer for", "the quiet joy of a good run or ride",
    ],
    "big and silly questions": [
        "would they rather be able to fly or be invisible", "what superpower would actually be useful", "if they could have dinner with anyone",
        "what colour a Monday would be", "the best invention nobody has thought of", "the age they would like to stay forever",
        "whether a hot dog is a sandwich", "how they would spend a surprise day off", "a world with no clocks", "what they would name a boat",
    ],
    "feelings and people": [
        "someone who made them laugh lately", "what makes a good friend", "a small act of kindness they noticed", "a compliment that stuck with them",
        "how they like to celebrate", "what makes them feel at home", "a person who is good at listening",
    ],
    "culture and curiosity": [
        "a fun fact that surprised them", "a word they like the sound of", "how languages are different", "a book they would recommend", "an odd tradition somewhere",
        "how people used to do things before phones", "a museum they would wander through", "something old that is still beautiful",
    ],
    "cars and things that go": [
        "a car they would love to drive", "an old car with character", "trains and the romance of them", "boats on the water", "the sound of a great engine",
    ],
    "home and comfort": [
        "their favourite spot to sit", "a room that feels just right", "candles, lamps and cosy light", "a blanket they would not give up", "the smell of fresh laundry",
    ],
}

# fixed-date holidays (month, day, name): mentioned when one is near
HOLIDAYS = [
    (1, 1, "New Year's Day"), (2, 14, "Valentine's Day"), (3, 17, "St. Patrick's Day"), (7, 1, "Canada Day"), (7, 4, "Independence Day"),
    (10, 31, "Halloween"), (11, 11, "Remembrance Day"), (12, 24, "Christmas Eve"), (12, 25, "Christmas"), (12, 31, "New Year's Eve"),
]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer",
           9: "autumn", 10: "autumn", 11: "autumn"}

STYLES = [
    "ask them a genuine question about it", "share a small opinion about it and ask what they think", "wonder aloud about it",
    "make a light joke about it, then ask them something", "mention a tiny thing about it that you like, and ask about theirs",
    "ask which they would pick, and why",
]


def time_hint(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now()
    near = []
    for m, d, name in HOLIDAYS:
        try:
            when = datetime.datetime(now.year, m, d)
        except ValueError:
            continue
        delta = (when - now).days
        if -1 <= delta <= 10:
            near.append(f"{name} is {'today' if delta <= 0 else 'in ' + str(delta) + ' day' + ('s' if delta != 1 else '')}")
    season = SEASONS[now.month]
    return f"It is {season}." + (" " + near[0] + "." if near else "")


def pick(n: int = 1, rng: random.Random | None = None) -> str:
    """ONE specific thing to open with (a list gets ignored), picked at random from a different area each time. The season or an upcoming
    holiday is added only now and then, so every opening is not about the weather."""
    r = rng or random
    cat = r.choice(list(TOPICS))
    idea = r.choice(TOPICS[cat])
    extra = f" {time_hint()}" if r.random() < 0.2 else ""
    return (
        f"TOPIC FOR THIS LINE: {idea}. Build what you say around it, in your own words, and {r.choice(STYLES)}. "
        f"These are just conversation topics, not things happening right now: never pretend to hear, see or notice anything in the room, and do not claim you just read, watched or heard something; speak from your own opinion or curiosity. Stay on the topic itself and leave out your job, your family and your past. Do not talk about the time of day or the weather unless that is the topic. Vary your first words; do not start with 'Well, hello there'.{extra}"
    )
