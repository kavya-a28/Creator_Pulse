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


