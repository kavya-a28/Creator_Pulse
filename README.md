# CreatorPulse

**An NLP-Based Audience Intelligence and Content Opportunity Mining System for YouTube Creators.**

CreatorPulse analyzes public YouTube comments to uncover audience needs, detect recurring questions, and recommend content opportunities — helping creators understand what their audience really wants.

## What it does

Paste a YouTube video or channel link into the CreatorPulse web app and it:

- **Collects** the most recent public comments with the YouTube Data API v3
- **Classifies** each comment as Question, Request, Confusion/Problem, Praise, Spam or Other (TF-IDF + Linear SVM)
- **Discovers topics** — groups comments about the same need, names them from the comments themselves, and separates viewer check-ins
- **Recommends video ideas**, each backed by real comments and a transparent 0–100 priority score
- Shows everything in a dashboard: Overview, Comments Explorer (filters + search), Insights and Recommendations

---

## Prerequisites

- **Python 3.8+** (tested on Python 3.10+)
- **Git** (for version control)
- **Google Cloud Platform account** (for YouTube API key)

---

## Setup

### 1. Clone the Repository

```powershell
git clone <your-repo-url>
cd creator_pulse
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.venv\Scripts\activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## YouTube API Key Setup

### 5. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a project** → **New Project**
3. Name it (e.g., "CreatorPulse") and click **Create**

### 6. Enable YouTube Data API v3

1. In your GCP project, go to **APIs & Services** → **Library**
2. Search for **"YouTube Data API v3"**
3. Click on it and press **Enable**

### 7. Create an API Key

1. Go to **APIs & Services** → **Credentials**
2. Click **+ Create Credentials** → **API key**
3. Copy the generated key
4. *(Optional but recommended)* Click **Edit API key** and restrict it to only the YouTube Data API v3

### 8. Configure `.env`

```powershell
copy .env.example .env
```

Open `.env` in a text editor and paste your API key:

```
YOUTUBE_API_KEY=AIzaSy...your_actual_key_here
```

> ⚠️ **Never share or commit your `.env` file.** It is already in `.gitignore`.

---

## Run the CreatorPulse App (URL → insights)

The app lets you paste a YouTube video or channel link and get the full analysis in your browser: comment categories, topics, insights and video ideas. You don't need to run any scripts or upload CSV files.

### 1. Activate the existing virtual environment and install the app dependency

From the project folder in PowerShell (use the `.venv` you already have; don't create a new one):

```powershell
cd C:\Users\KAVYA\Desktop\creator_pulse
.venv\Scripts\activate
pip install -r requirements.txt
```

The only new package is `streamlit` (it brings `altair` for charts). If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in Command Prompt.

### 2. Configure your YouTube API key

```powershell
copy .env.example .env
notepad .env
```

Set `YOUTUBE_API_KEY=<your key>` and save. `.env` is listed in `.gitignore`, so never commit it. The app only checks that a key is present and never shows it. To create a key, see [YouTube API Key Setup](#youtube-api-key-setup).

### 3. Launch

```powershell
streamlit run app.py
```

Your browser opens at http://localhost:8501 on **New Analysis**. Paste a link and click **Analyze**. A 7-step tracker shows progress, and the app opens **Overview** when the analysis finishes. Use the sidebar for **Overview**, **Comments**, **Insights** and **Recommendations** of the selected analysis, and **Past Runs** to reopen older ones. Click a category, topic or idea on the Overview to open the matching comments. To check your API key and network, run `python -m src.check_connection`.

- **Supported links:** `youtube.com/watch?v=…`, `youtu.be/…`, `/shorts/…`, `/live/…`, a bare video ID, and channel links `youtube.com/@handle`, `/channel/UC…`, `/c/…`, `/user/…`, `@handle`.
- **Not supported:** playlists, search pages and non-YouTube links. The app explains why when you paste one.
- **Options (Advanced Options on New Analysis):** one video per analysis; a channel link analyzes the channel's latest upload. **Most recent comments to collect** (50-2000) takes the newest top-level comments first; replies aren't collected. **Exclude spam from topics** is on by default. There's no date filter, because the YouTube API can't filter comments by date.
- **API cost:** the app shows the estimated cost before you run. A video costs about 2 units (video details and channel picture) plus 1 per 100 comments, out of 10,000 units per day.
- **Progress:** the analysis runs in the background through 8 stages: validate, collect, preprocess, classify, topics, insights, recommendations, dashboard. Each stage is shown only when it has actually started or finished. You can keep using the page while it runs, and nothing is collected twice.
- **Duplicate check:** if the same link and settings were analyzed in the last 30 minutes, the app offers to open that result instead of using quota again.
- **Errors:** invalid links, a missing or invalid key, the API not being enabled, exhausted quota, comments turned off, private or deleted videos, and empty results each show a plain message and a fix. If a stage fails, the outputs of earlier stages are kept.
- **Model:** the saved model is only loaded, never retrained or refitted on new comments.

### Where results are saved

| What | Where |
|---|---|
| Raw comments of each analysis | `data\raw\app_runs\comments_<run_id>.csv` (never overwritten) |
| Status, timestamps, settings, stage progress, errors, output paths | `data\runs\<run_id>\metadata.json` |
| Stage outputs (predictions, topics, insights, recommendations) | `data\runs\<run_id>\processed\` |

The sidebar lists previous analyses, so you can reopen them at any time. Both folders are in `.gitignore`. Your original `data\raw\*.csv` and labelled files are only read.

### App code

| File | Responsibility |
|---|---|
| `app.py` | Streamlit pages (New Analysis with live progress, Overview, Comments, Topics, Insights, Recommendations, Past Runs) |
| `src\ui_components.py` | Theme, CSS animations, cards, charts and small display helpers |
| `src\collect.py` | URL validation, collection through the existing collectors, API error messages |
| `src\analysis.py` | The 8-stage pipeline for one URL, using `batch_predict`, `topics`, `insights` and `recommend` |
| `src\job_runner.py` | Runs an analysis in a background thread and blocks duplicate submissions |
| `src\runs.py` | Run folders and `metadata.json` |
| `src\dashboard_data.py` | Loads a run's outputs for the dashboard |
| `.streamlit\config.toml` | Light theme with an indigo accent; hides developer menus and raw error details |

---

## Usage

### Fetch Comments for a Single Video

Using a video ID:
```powershell
python -m src.fetch_comments --video dQw4w9WgXcQ
```

Using a full URL:
```powershell
python -m src.fetch_comments --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Supported URL formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/shorts/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

Limit the number of pages fetched (useful for testing):
```powershell
python -m src.fetch_comments --video dQw4w9WgXcQ --max-pages 2
```

### Fetch Recent Videos from a Channel

You can pass a **channel URL**, a **handle (`@name`)**, or a raw **channel ID (`UC...`)**:

Using a channel URL (handle or channel ID format):
```powershell
python -m src.fetch_channel_videos --channel https://www.youtube.com/@LifeOfLimbachiyaas --limit 20
```

Using a channel handle directly:
```powershell
python -m src.fetch_channel_videos --channel @LifeOfLimbachiyaas --limit 20
```

Using a raw channel ID:
```powershell
python -m src.fetch_channel_videos --channel UCiKNX0TIMyKgGqQznhR1Xig --limit 20
```

Fetch videos **and** automatically collect comments for each:
```powershell
python -m src.fetch_channel_videos --channel https://www.youtube.com/@LifeOfLimbachiyaas --limit 5 --fetch-comments
```

Limit comment pages per video (to conserve quota):
```powershell
python -m src.fetch_channel_videos --channel UC_CHANNEL_ID --limit 10 --fetch-comments --max-comment-pages 3
```

Enable verbose logging:
```powershell
python -m src.fetch_comments --video dQw4w9WgXcQ -v
```

---

### Build the NLP Labelling Dataset

Collects ~1,000 comments from 5 educational channels (CodeWithHarry, Apna College, freeCodeCamp, 3Blue1Brown, Physics Wallah), removes duplicate `comment_id`s and creates a labelling sheet:

```powershell
python -m src.build_labelling_dataset
```

Options: `--target 1000`, `--channels @handle1 @handle2 ...`, `--videos-per-channel 15`, `--per-video-cap 50`, `--force` (overwrite a labelling file that already has labels).

| File | Description |
|------|-------------|
| `data/raw/educational_comments.csv` | Original, unmodified comments (emojis, Hindi, Hinglish preserved) plus channel/video metadata |
| `data/labelled/comments_to_label.csv` | `comment_id`, `video_id`, `text`, empty `label` |
| `data/labelled/LABELLING_GUIDE.md` | Definitions for the five labels: Question, Request, Praise, Spam, Other |

Cost: roughly 100 API quota units per run.

---

### Build the Training Set

Creates the training data from `data/labelled/comments_labelled.csv`:

```powershell
python -m src.prepare_training_data
```

- **Creator-authored comments are excluded.** Any comment whose `author_channel_id` equals the uploading channel's `channel_id` (pinned promos, announcements) is dropped, whatever its label.
- **Near-duplicates are collapsed.** Comments with identical normalised text, or char n-gram TF-IDF cosine similarity ≥ 0.80 (`--threshold`), are grouped, and the first comment per label in each group is kept. `duplicate_count` records how many comments each kept row stands for.

| File | Description |
|------|-------------|
| `data/processed/training_comments.csv` | `comment_id`, `video_id`, `text`, `label`, `duplicate_count` |
| `data/processed/removed_comments.csv` | Every dropped comment with its `reason` and the `kept_comment_id` it duplicates |

---

### Preprocess Text for NLP

Adds normalised text, tokens and lemmas to the training set. The original `text` column is never changed.

```powershell
python -m nltk.downloader wordnet omw-1.4   # once
python -m src.preprocess
```

Options: `--no-lemmatize`, `--remove-stopwords`, `--output PATH`, `--force` (the script will not overwrite an existing output otherwise) (off by default; negations, question words, "please" and Hinglish words are never removed).

- **Normalisation:** lowercase; URLs, `@mentions` and timestamps become `<url>`, `<user>`, `<time>`; elongations squeezed (`sooooo` → `soo`); contractions expanded so negations stay (`didn't` → `did not`, `dont` → `do not`); common spelling variants unified (`plz` → `please`, `nhi` → `nahi`, `kese` → `kaise`). Emoji are kept.
- **Tokenisation:** Devanagari and other scripts stay whole; `?` and `!` are kept as tokens; each emoji is a token.
- **Lemmatisation:** NLTK WordNet, English words only. Negations, question words, request words and a Hinglish word list are protected (WordNet would otherwise turn Hinglish `us` into `u`). Words that are already base words, and intent-bearing adjectives like `confused` or `amazing`, are kept as written.

| File | Description |
|------|-------------|
| `data/processed/preprocessed_comments.csv` | Training columns plus `text_norm`, `tokens`, `lemmas` (space-separated) |

---

### Create Train/Test Splits

```powershell
python -m src.make_splits
```

Writes a new folder `data/processed/splits_v1/` (it refuses to overwrite an existing one; use `--out` for another seed or `--force`). Comments that are near-duplicates (cosine ≥ 0.70) or share an author are kept on the same side of every split.

| File | Description |
|------|-------------|
| `random_train.csv` / `random_test.csv` | Stratified group split (80/20). `cv_fold` gives 5 leakage-safe CV folds |
| `video_train.csv` / `video_test.csv` | One whole video per channel held out; linked training comments are purged |
| `assignments.csv` | Every comment's leakage group, fold and split |
| `excluded.csv` | Rows not usable (empty text) |
| `manifest.json` | Seed, thresholds, input SHA-256, library versions, class balance, leakage checks |
| `AUDIT_REPORT.md` | Data audit: text preservation, duplicates, labels, Hinglish preprocessing, splits |

Fit vectorisers and other learned preprocessing on the train file only.

---

### Train the Baseline Classifier (TF-IDF + Linear SVM)

> ⚠️ **Label provenance.** The labels in `comments_labelled.csv` were generated automatically by Claude (an AI model) and have not been fully human-verified, so they may contain errors. Every metric below measures **agreement with those AI-generated labels**, not accuracy against ground truth. A human-verified test set is needed before claiming real-world accuracy.

```powershell
python -m src.train_svm
```

- **Classes (6):** `Question`, `Request`, `Confusion/Problem`, `Praise`, `Spam`, `Other`, read from the `label_6class` column. The original CSV is only read, never changed.
- **Splits:** reuses `data/processed/splits_v1/` (near-duplicates and same-author comments never cross train/test). The **primary** evaluation is the held-out-video split. The random group split and grouped 5-fold CV (`cv_fold`) are reported as well. The script stops if a split shares a comment ID or leakage group, or if its text or label differs from the labelled file.
- **Text:** the pipeline takes the original `text` and applies `src.preprocess.normalize` (identical to `text_norm`, checked at run time) and `src.preprocess.tokenize` inside `TfidfVectorizer`. Negations (`not`, `nahi`, `नहीं`), question words, Hinglish, Devanagari, emoji, `?` and `!` are kept. No stopword removal, and the `lemmas` column is not used.
- **Model:** `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` → `LinearSVC(class_weight="balanced")` in one scikit-learn `Pipeline`, so TF-IDF is fitted on training data only. No oversampling. `duplicate_count` is reported but not used as a weight.

| File | Description |
|------|-------------|
| `models/creatorpulse_svm.joblib` | Fitted pipeline (trained on `video_train.csv`) |
| `reports/classification_report.txt` | Accuracy, macro/weighted F1, per-class P/R/F1, confusion matrices, class distributions, reliability notes |
| `reports/confusion_matrix.png` | Confusion matrix on the held-out-video test set |
| `reports/evaluation_summary.json` | All metrics, bootstrap/Wilson confidence intervals, data audit, versions |
| `reports/test_predictions.csv` | Every test comment with its dataset label and prediction (for error review) |

**Baseline results (held-out videos, 183 test comments):** accuracy 0.667, macro F1 0.427, weighted F1 0.657. For comparison, always predicting `Other` gives accuracy 0.492 and macro F1 0.110. `Confusion/Problem` (4 test / 19 total examples) and `Spam` (2 test / 10 total) are too small to evaluate. The model got none of them right on the test split or in 5-fold CV, and most `Confusion/Problem` comments are predicted as `Question`.

### Classify a Comment

```powershell
python -m src.predict "Sir please make a video on Django REST"
python -m src.predict --examples          # one example comment per class
python -m src.predict --model other.joblib "text"
```

Prints the predicted label and the SVM score for each class. Empty or punctuation-only input is skipped with a message (exit code 2). A missing, corrupt or incompatible model file gives a clear error (exit code 1). The model must be loaded with the same scikit-learn version it was saved with (1.9.1).

### Synthetic Training Data

```powershell
python -m src.generate_synthetic                                        # 10,002 rows, 1,667 per class
python -m src.train_svm --synthetic data/synthetic/synthetic_comments.csv
python -m src.compare_synthetic                                         # real-only vs +synthetic, on real test data
python -m src.predict --model models/creatorpulse_svm_synthetic.joblib "text"
```

- **What it is:** template + slot-filling comments for all 6 classes (64 templates; coding, maths and JEE/NEET topics; English, Hinglish and Devanagari Hindi; typos, shorthand, emoji, casing noise). Labels follow `LABELLING_GUIDE.md`, including the tie-break rules for mixed comments. Fixed seed, so the output is byte-identical on every run.
- **Training only.** Synthetic rows are added to the training data of every evaluation, and every test set stays 100% real. A model trained and tested on synthetic data scores 1.000, which shows the templates are easy to tell apart and says nothing about real comments.
- **Leakage filter:** a synthetic row is dropped if its character TF-IDF cosine similarity to any real labelled comment is ≥ 0.70 (the same threshold `make_splits` uses).
- **Template-writer bias:** the person who wrote the templates read 92 real comments while doing it (listed in `data/synthetic/viewed_while_writing_templates.csv`), and close paraphrases of those comments were then removed. `compare_synthetic` also reports CV results on the 821 comments that were not read. That is the cleanest estimate.
- Outputs go to `models/creatorpulse_svm_synthetic.joblib` and `reports/synthetic/`, so the real-only baseline is never overwritten. `reports/synthetic_comparison.md` has the full comparison.

| Training data | Video acc | Video macro F1 | CV macro F1 (899) | CV macro F1, unread comments (821) |
|---|---|---|---|---|
| real only | 0.667 | 0.427 | 0.456 | 0.452 |
| real + 250/class synthetic | 0.716 | 0.661 | 0.596 | 0.572 |
| real + all 10,002 synthetic | 0.716 | 0.649 | 0.622 | 0.578 |
| synthetic only | 0.508 | 0.462 | 0.439 | 0.396 |

Synthetic data helps when it is added to real data, but the gain levels off after about 250 rows per class. It cannot replace real data. The video-split macro F1 jumps partly because `Confusion/Problem` (4) and `Spam` (2) have so few test comments, so rely on the CV columns.

---

### Audience Insights Pipeline (prediction → topics → insights → recommendations)

Run everything on the collected comments in `data/raw/` (from the project root, with `.venv` active):

```powershell
python -m src.pipeline
```

Or run one stage at a time (each reads the previous stage's output from `data/processed/`):

```powershell
python -m src.batch_predict          # 1. predict a category for every comment
python -m src.topics                 # 2. near-duplicates + topic discovery
python -m src.insights               # 3. insights, trends, video/channel comparison
python -m src.recommend              # 4. video ideas
python -m src.pipeline --stages topics insights recommend   # re-run from stage 2
python -m src.topic_features_eval    # optional: compare word / char features (add --embeddings to include embeddings)
```

Useful options: `--model models\creatorpulse_svm.joblib` (use the real-only model), `--include Praise` (also cluster praise), `--min-evidence 2` (allow topics with 2 commenters). `src.pipeline` exits with code 3 if a consistency check fails.

| Stage | Output files (in `data/processed/`) |
|---|---|
| 1 Batch prediction | `comments_predicted.csv`, `comments_excluded.csv`, `category_summary.csv`, `batch_predict_stats.json` |
| 2 Topics | `topic_clusters.csv` (one row per comment), `topic_summary.csv` (one row per topic, plus `Unclustered`) |
| 3 Insights | `audience_insights.json`, `topic_trends.csv`, `video_topic_summary.csv` |
| 4 Recommendations | `content_recommendations.csv`, `content_recommendations.json` |
| Runner | `pipeline_report.json` (totals and 14 cross-stage checks) |

How it works:

- **Prediction** loads the saved pipeline and only calls `predict`, so new comments get exactly the training-time `normalize` + `tokenize`, and nothing is refitted. The default model is `creatorpulse_svm_synthetic.joblib` if it exists. Comments with no ID, empty text, no usable tokens, or a repeated ID are written to `comments_excluded.csv` with a reason. Missing channel or title becomes `unknown`, and `metadata_missing` lists what was absent. `seen_in_training` marks comments the model was trained on, because its predictions for them are optimistic.
- **Topics** cluster Question, Request and Confusion/Problem comments. Spam is never clustered, and Praise and Other only with `--include`.
  - Near-duplicates (character similarity ≥ 0.85) count once, but their copies are still reported.
  - Intent and filler words ("please make a video", "sir", "kya") are removed first, so topics form around what people ask about.
  - Features are word TF-IDF plus character 3-5-gram TF-IDF (equal weight). Clustering is agglomerative, average linkage, at cosine ≥ 0.30.
  - A group held together by a single shared word is split again.
  - Topics with fewer than 2 unique comments, and members far from their topic centroid, go to `Unclustered`.
  - Names come from each topic's most distinctive terms. The representative comment is the most central one.
- **Insights** report raw, unique and commenter counts separately. Trends compare a topic's share in the last 7 days against the 7 days before, and only on videos whose comment capture was complete for both weeks. With the current data, trends cannot yet be established, and the reason is stated in `audience_insights.json`. `creator_replied` empty is never treated as unanswered.
- **Recommendations** come from topics with at least 3 distinct commenters. A commenter posting several times counts once.
  - Score = 40% frequency + 25% recurrence across videos + 15% recency + 20% intent, where problems weigh more than requests, and requests more than questions.
  - Each part is shown in the output, along with a reason, 3-5 real comments with IDs, and every evidence comment ID.
  - Similar topics are merged into one idea. Titles and formats come from rules; no LLM or paid API is used.

Limitations: topic matching is lexical, so synonyms ("stock market" / "trading"), abbreviations ("ML" / "machine learning"), Hindi script vs English, and very different spellings ("cybersecurity" / "siber security") can end up as separate topics. `--features embeddings` uses multilingual sentence embeddings when `sentence-transformers` and its model are installed. Run `python -m src.topic_features_eval --embeddings` to compare them on your machine before switching (it downloads a roughly 470 MB model the first time).

---

## Output Files

All output is saved in the `data/raw/` directory:

| File | Description |
|------|-------------|
| `data/raw/comments_<video_id>.csv` | Comments for a single video |
| `data/raw/channel_videos_<channel_id>.csv` | Video metadata for a channel |

### Comment CSV Columns

| Column | Description |
|--------|-------------|
| `comment_id` | Unique YouTube comment ID |
| `video_id` | Video the comment belongs to |
| `text` | Comment text (plain text, Unicode preserved) |
| `author_channel_id` | Channel ID of the comment author |
| `published_at` | ISO 8601 timestamp when the comment was posted |
| `like_count` | Number of likes on the comment |
| `reply_count` | Total number of replies to the comment |
| `creator_replied` | `True`, `False`, or empty (see below) |

### Understanding `creator_replied`

| Value | Meaning |
|-------|---------|
| `True` | A reply from the video creator was found |
| `False` | All replies were checked and none were from the creator |
| *(empty/NaN)* | The API returned only a subset of replies (max 5), so we cannot confirm whether the creator replied. The comment has more replies than what was returned. |

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

Tests do **not** require an API key or internet connection.

---

## Common Errors and Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `YouTube API key not configured` | Missing or empty `.env` | Create `.env` with your API key (see step 8) |
| `API access denied (403)` | YouTube Data API v3 not enabled | Enable it at [GCP API Library](https://console.cloud.google.com/apis/library/youtube.googleapis.com) |
| `Comments are disabled` | Video doesn't allow comments | Try a different video |
| `Video not found (404)` | Invalid ID or private video | Check the video ID and ensure it's public |
| `API quota exceeded` | Daily 10,000 unit limit reached | Wait until midnight PT or check [GCP Dashboard](https://console.cloud.google.com/apis/dashboard) |
| `Network error` | Internet connectivity issue | Check your connection; the tool retries automatically |

---

## API Quota Considerations

The YouTube Data API v3 has a free daily quota of **10,000 units** (resets at midnight Pacific Time).

| Operation | Cost per call |
|-----------|---------------|
| Fetch one page of comments | 1 unit |
| Fetch video info | 1 unit |
| Fetch channel info | 1 unit |
| Fetch playlist items (videos) | 1 unit |

**Estimate for collecting comments from one video:**
- 1 unit for video info + 1 unit per 100 comments page
- A video with 500 comments ≈ 6 units total

**Tip:** Use `--max-pages` and `--max-comment-pages` flags to control quota usage during development.

---

## Known Limitations

1. **Incomplete reply data** — The YouTube API returns at most 5 replies per comment thread inline. For threads with more replies, the `creator_replied` field may be uncertain (empty/NaN).

2. **Disabled comments** — Some videos have comments disabled. These are reported and skipped gracefully.

3. **Private/restricted videos** — Videos that are private, age-restricted, or region-locked may not be accessible via the API.

4. **API quota limits** — The free tier allows 10,000 units/day. Large-scale collection may require multiple days or a quota increase.

5. **Rate limiting** — The tool implements exponential backoff for transient errors, but sustained high-frequency requests may still be throttled.

6. **Comment ordering** — The API returns comments in its default order (usually relevance). The order is not guaranteed to be chronological.

---

## Future Features (Not Yet Implemented)

- **Transcript collection** — Video transcript/subtitle extraction using the YouTube Transcript API. Transcript availability depends on the video (auto-generated vs. manually uploaded captions). This is planned as an optional data source, not a prerequisite for comment analysis.

- **Better NLP classification** — A TF-IDF + Linear SVM baseline is implemented (see above). Next: a human-verified test set, char n-gram features for Hinglish spelling variants, more Confusion/Problem and Spam examples, and sentence-embedding models.

- **Trend analysis** — Topic growth needs repeated collections of the same channel over time; the code is in place (`src/insights.py`) but one snapshot cannot show a trend.

---

## Project Structure

```
creator_pulse/
├── app.py                  # Streamlit app:  streamlit run app.py
├── requirements.txt
├── .env.example            # Template for your API key (copy to .env)
├── .streamlit/config.toml  # Light theme, hides developer menus
├── src/
│   ├── collect.py              # App: URL validation + collection, friendly API errors
│   ├── analysis.py             # App: the 8-stage pipeline for one URL
│   ├── job_runner.py, runs.py  # App: background runs + per-run folders/metadata
│   ├── dashboard_data.py       # App: loads a run's outputs for the dashboard
│   ├── ui_components.py        # App: theme, cards, charts, animations
│   ├── check_connection.py     # Diagnose API key / DNS / proxy problems
│   ├── config.py, url_utils.py # API key loading, URL parsing
│   ├── fetch_comments.py, fetch_channel_videos.py   # Collection CLIs
│   ├── build_labelling_dataset.py, prepare_training_data.py, make_splits.py
│   ├── preprocess.py, train_svm.py, predict.py      # NLP + classifier
│   ├── generate_synthetic.py, compare_synthetic.py  # Synthetic training data
│   ├── batch_predict.py        # Stage 1: predict categories
│   ├── topics.py               # Stage 2: topic discovery and naming
│   ├── insights.py             # Stage 3: insights and trends
│   ├── recommend.py            # Stage 4: content recommendations
│   ├── topic_features_eval.py  # Compare topic feature options
│   └── pipeline.py             # Runs stages 1-4 on data/processed + checks
├── models/                 # Trained classifiers (.joblib) loaded by the app
├── reports/                # Evaluation reports, confusion matrices
├── data/
│   ├── labelled/           # Labelled comments + labelling guide (committed)
│   ├── synthetic/          # Template-generated training-only comments (committed)
│   ├── processed/splits_v1 # Train/test splits (committed); other outputs are generated
│   ├── raw/                # Collected comments — local only, not committed
│   └── runs/               # Each app analysis — local only, not committed
└── tests/                  # Unit + UI tests (no API key needed)
```

Not in the repository (see `.gitignore`): your `.env` key, the virtual environment, collected YouTube comments (`data/raw`, `data/runs`) and regenerated pipeline outputs.

---

## License

This project is developed as an academic research project.
