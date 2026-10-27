# Labelling Guide — Comment Intent

File to label: `data/labelled/comments_to_label.csv`
Fill the `label` column with **exactly one** of the five values below (spelling and case as shown).
Do not edit `comment_id`, `video_id` or `text`.

| Label | Use when the comment… | Examples |
|---|---|---|
| `Question` | asks something answerable — a factual, conceptual or "how/why/which" question — **or** reports being stuck, an error, something not working, or not understanding part of the video (a viewer in trouble is asking for help) | "What's the difference between let and var?" · "Sir ye JEE Mains me aayega kya?" · "Getting ModuleNotFoundError at 12:30" · "Samajh nahi aaya step 3 🤔" |
| `Request` | asks the creator to make, upload or share something (topic, video, notes, playlist, course) | "Please make a video on Django REST" · "Notes ka link de do sir" |
| `Praise` | expresses thanks, appreciation or positive feedback, with no question or request | "Best explanation on YouTube ❤️" · "Thank you bhaiya, placement ho gaya!" |
| `Spam` | is promotional, a scam, link-dropping, self-promotion, bot-like or irrelevant repetition | "Earn ₹5000 daily, DM me on Telegram" · "Check out my channel!!" |
| `Other` | doesn't fit any of the above — general remarks, jokes, attendance ("Day 12 ✅"), timestamps-only, off-topic chat | "Who's watching in 2026?" · "Day 5 done" |

## Tie-break rules (comments with several intents)

Pick the most **actionable** intent for the creator, in this order:

1. `Spam` — if it's promotional/scam, that overrides everything.
2. `Question` — the viewer is stuck or asks something ("Great video but my code throws an error" → Question).
3. `Request` — asks for new content ("Amazing! Please make one on graphs" → Request).
4. `Praise`
5. `Other`

## Notes

- "Can you make…?" phrased as a question is still a `Request`.
- Label Hindi, Hinglish and emoji-only comments by meaning, the same as English. Emoji-only praise (🔥🔥❤️) → `Praise`.
- If you genuinely can't tell, use `Other` rather than guessing.

## Conventions applied in `comments_labelled.csv`

- Personal career or study advice ("BCA ke baad kya karu?", "should I choose CSE?") → `Question`.
- Viewers correcting the video ("12:03 it's MCP not MVC") and general criticism → `Other`.
- Joke "next video" asks ("nuclear plant next", "area 51 next") → `Other`; sincere topic asks → `Request`.
- Generic encouragement ("keep it up", "more of this!") → `Praise`; a named topic or format → `Request`.
- Asking to be pinned, replied to or given a job → `Other` (not content).
- Self-promotion, affiliate or course links, piracy (mod APKs), chain letters, and one account flooding the same text → `Spam`.
- Templated bot praise (identical lines from many 0-like accounts) is labelled `Praise` by its text and flagged in `review_note`.

## Change log

- 2026-10-24: `Confusion/Problem` (19 examples) merged into `Question`. The original 6-class label is kept in the `label_6class` column of `comments_labelled.csv`.
