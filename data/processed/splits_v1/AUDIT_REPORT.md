# CreatorPulse — Dataset Audit and Split Report

Audit date: 2026-10-24. Files audited (unchanged by this audit):
`data/raw/educational_comments.csv`, `data/labelled/comments_to_label.csv`,
`data/labelled/comments_labelled.csv`, `data/processed/training_comments.csv`,
`data/processed/preprocessed_comments.csv`. SHA-256 of the split inputs are in
`manifest.json`.

## 1. Original text preservation — PASS

- `text` was compared character-for-character by `comment_id` across all five
  files: **0 mismatches** (1,000 / 1,000 / 1,000 / 900 / 900 rows).
- All files are valid UTF-8 (with BOM for Excel). No U+FFFD replacement
  characters, no mojibake (`Ã`, `â€`, `à¤`, `ðŸ`), no HTML entities or tags,
  no lone surrogates.
- Preserved content: 262 comments with emoji (18 with ZWJ / variation-selector
  sequences), 9 with Devanagari Hindi, 6 in other scripts (Arabic, Chinese,
  Japanese, Korean, Russian, Bengali), and Romanised Hinglish throughout.
- Line breaks inside comments are intact: 110 use `\n` as returned by the API;
  5 contain `\r\n` in all files consistently (as posted, not a CSV artefact).
- Limitation: the raw file is the API's `textDisplay` in `plainText` format.
  It could not be re-checked against YouTube from this environment.

## 2. `duplicate_count` and repeated comments

**How it is calculated** (`src/prepare_training_data.py`): after creator
comments are removed, comments are grouped when their normalised text is
identical or char-n-gram TF-IDF cosine ≥ 0.80. Within each group the first
comment per label is kept; `duplicate_count` = number of comments in that
(group, label), including the kept one. Rebuilding from the code reproduces
`training_comments.csv` exactly; counts sum to 993 (= 1,000 − 7 creator).

**Are repeats legitimate?** Yes — there are no accidental duplicate records.
Every `comment_id` is unique, and every repeat has its own ID and its own
`published_at`. The 93 collapsed comments fall into 25 groups:

| Kind | Groups | Comments | Examples |
|---|---|---|---|
| Different accounts, same sentiment | 17 | 79 | 20× "Congratulations Harry bhai", 21× ❤, templated bot praise |
| One account flooding | 3 | 29 | "Hiii" ×14 in 2 min, "Y" ×11 in 1 min, one request pasted 4× across 3 videos |
| One account posting twice | 5 | 10 | the same DSA request twice, 6 s apart |

**Caveat:** `duplicate_count` mixes "many people said this" with "one person
spammed this". Using it as a sample weight would weight one flooder 14×.
Do not use it as a weight; treat it as descriptive only.

## 3. Labels and row validity — PASS (1 invalid row)

All of these pass: unique, well-formed `comment_id` (`Ug…AaABAg`) and 11-char
`video_id` in every file; labelling sheet and labelled file in the same order
as raw; `video_id` agrees with raw everywhere; no missing labels; every label
is one of the 5 classes; `label` equals `label_6class` with
Confusion/Problem → Question; training and preprocessed labels match the
labelled file; identical normalised text never has different labels;
timestamps ISO-8601; counts are integers; the labelling sheet is still blank.

Similar-but-different comments with different labels (cosine ≥ 0.6) were
reviewed by hand, e.g. "Sir timing?" (Question) vs "what a timing" (Other).
All are genuinely different intents.

**Invalid:** comment `UgyRNepYLAu82z4YiO14AaABAg` has empty text and is in
`training_comments.csv`. It is excluded from the splits (`excluded.csv`).
Two other content-free comments ("21:27", ".") were already removed as
near-duplicates of it.

**Note:** `data/processed/removed_comments.csv` is no longer in the project
folder. `python -m src.prepare_training_data --force` regenerates it, and
reproduces `training_comments.csv` byte-for-byte.

## 4. Hinglish suitability of preprocessing

Checked against every comment, not just examples.

**Negations — PASS after fixes.** 156 comments contain a negation (237
occurrences: not/no/never/n't/dont…, nahi/nhi/nahin/ni/mat/bina/na, नहीं).
With the fixed pipeline, **0 are lost and 0 invented**, in `tokens`, in
`lemmas`, and with `--remove-stopwords`.

**Problems found in the previous `preprocessed_comments.csv`** (left
unchanged as instructed; the split files carry corrected columns):

| Issue | Example | Fix |
|---|---|---|
| `nai` → `nahi` invented a negation | "aap **nai** is dialogue" (nai = ne) | mapping removed; `nai` kept as written |
| `r` → `are` | "J.A.R.V.I.S.", maths variables | mapping removed |
| Digits squeezed as elongation | "10000000" → "100", "111" → "11" | digits never squeezed |
| "IT" became the pronoun "it" | "worked in IT sector" | `IT` → `info_tech` unless the comment is all caps |
| Unused risky shorthand | `hn`, `ty`, `sm`, `tq` | removed |

Also added: `he` (= hai), `nai`, `ni` to the Hinglish list, and the variants
bhut/bhaut → bahut, chaiye/chahie → chahiye, krte → karte, lakin → lekin,
phele/pahle → pehle. Each mapping was checked against its corpus occurrences:
`ni` → `nahi` 2/2 correct; `u` → `you` correct in 12 of 13 comments. The
exception is a maths proof where `u` is a variable ("P * u * (10^n)"); it was
kept because it affects one comment.

**Lemmatisation — suitable.** In the 170 Hinglish comments, every lemma
change is an English word (videos → video, skills → skill, was → be). No
Hinglish word is changed; unprotected Hinglish words (hota, kabhi, batao)
are unknown to WordNet and pass through.

**Limits:** Hinglish spelling varies far beyond any list (krte/karte,
bhut/bahut, kese/kaise). Word tokens split these; character n-gram features
(e.g. TF-IDF `char_wb` 2–5) are recommended alongside word features.

## 5. Splits — `data/processed/splits_v1/`

899 comments (900 − 1 empty). Reproduce with `python -m src.make_splits`
(seed 42). A second run reproduces every file byte-for-byte.

**Leakage groups** join comments that must stay on one side: near-duplicates
at cosine ≥ 0.70 (stricter than the 0.80 used for dedup) **and** comments by
the same author. 841 groups, largest 4.

| | Random (stratified group) | Video held-out |
|---|---|---|
| Train / test | 719 / 180 | 705 / 183 (+11 purged) |
| Test: Other / Praise / Question / Request / Spam | 89 / 33 / 27 / 29 / 2 | 90 / 36 / 26 / 29 / 2 |
| Shared authors | 0 | 0 |
| Shared leakage groups | 0 | 0 |
| Shared videos | 23 (by design) | 0 |
| Identical normalised text across split | 0 | 0 |
| Test rows with a train neighbour ≥ 0.70 | 0 | 0 |
| Max cross-split similarity | 0.69 | 0.69 |

- **Random split:** `StratifiedGroupKFold(5)`; fold 0 is the test set.
  `cv_fold` (0–4) supports cross-validation with the same guarantees.
- **Video split:** one video per channel held out (CodeWithHarry
  `-UZPUkl_KwE`, freeCodeCamp `JGly6PSK5iw`, 3Blue1Brown `ausLKMojXaY`,
  Physics Wallah `kc6J2sQR8VM`, Apna College `mS5lTKaOt_4`), chosen from
  5,000 seeded draws for class balance. Grouping by video directly would chain
  567 of 899 comments into one component (15 authors comment on several
  videos), so the 11 training comments linked to test comments are purged.
- Closest remaining pairs (0.60–0.69) were read by hand and are different
  comments sharing a word ("Thank you" / "Thank you for help me").
- On the random split, the "learn hacking and hack PW" joke family appears on
  both sides as different comments on a shared theme. The video split holds
  that video out entirely and is the better generalisation test.

## Recommendations

1. Report results on **both** splits; the gap between them estimates how much
   a model relies on video-specific topics.
2. **Spam has 10 examples (2 per test set).** Spam precision/recall on 2 items
   is not meaningful. Report macro-F1 over the other four classes as well, or
   fold Spam into Other until more spam is collected.
3. Fit every vectoriser, vocabulary or scaler on the train file only.
4. Do not use `duplicate_count` as a sample weight (see section 2).
5. Use the `tokens` / `lemmas` in the split files, not the older
   `preprocessed_comments.csv`, which has the issues in section 4.
