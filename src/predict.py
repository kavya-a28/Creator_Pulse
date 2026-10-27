"""
CreatorPulse — predict the intent of a YouTube comment with the saved
TF-IDF + LinearSVC pipeline (models/creatorpulse_svm.joblib).

The model was trained on labels generated automatically by Claude (an AI
model) that have not been fully human-verified; predictions reflect those
labels and can be wrong. Confusion/Problem and Spam had too few training
examples (15 and 8) to be learned reliably.

Usage (from the project root):
    python -m src.predict "Sir please make a video on Django REST"
    python -m src.predict "Samajh nahi aaya" "Best explanation ever ❤"
    python -m src.predict --examples
    python -m src.predict --model path/to/model.joblib "text"

Exit codes: 0 ok, 1 model could not be loaded, 2 no valid comment given.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Allow `python src/predict.py` as well as `python -m src.predict`: the saved
# pipeline references src.preprocess, so the project root must be importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = PROJECT_ROOT / "models" / "creatorpulse_svm.joblib"

# One illustrative comment per class, written for this demo (not taken from the
# dataset). The "intended" class is what the example was written to show; the
# printed prediction is whatever the model actually returns.
EXAMPLES = [
    ("Question", "What is the difference between a list and a tuple in Python?"),
    ("Request", "Sir please make a full video on React hooks, ek playlist bana do"),
    ("Confusion/Problem", "I followed every step but my code still shows an error, samajh nahi aa raha"),
    ("Praise", "Best explanation on YouTube, thank you so much bhaiya ❤"),
    ("Spam", "Earn 5000 daily from home!! DM me on telegram, link in my bio"),
    ("Other", "Who is watching this in 2026?"),
]


class ModelLoadError(Exception):
    """The model file is missing, unreadable or not a CreatorPulse pipeline."""


class EmptyCommentError(ValueError):
    """The comment is empty or has no usable text."""


def load_model(path=DEFAULT_MODEL):
    """Load the saved pipeline, raising ModelLoadError with a readable message."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(
            f"Model file not found: {path}\n"
            "Train it first with:  python -m src.train_svm")
    if not path.is_file():
        raise ModelLoadError(f"Model path is not a file: {path}")
    try:
        import joblib
        model = joblib.load(path)
    except Exception as exc:  # corrupt file, wrong format, version mismatch
        raise ModelLoadError(
            f"Could not load model from {path}: {type(exc).__name__}: {exc}\n"
            "The file may be corrupt or saved with an incompatible scikit-learn "
            "version; re-run  python -m src.train_svm") from exc
    if not (hasattr(model, "predict") and hasattr(model, "classes_")):
        raise ModelLoadError(f"{path} does not contain a fitted classifier pipeline.")
    return model


def _check_text(text) -> str:
    if text is None or not isinstance(text, str):
        raise EmptyCommentError("Comment must be a non-empty string.")
    if not text.strip():
        raise EmptyCommentError("Comment is empty.")
    from src.preprocess import normalize, tokenize
    if not tokenize(normalize(text)):
        raise EmptyCommentError(
            "Comment has no words, emoji, '?' or '!' after normalisation, so it cannot be classified.")
    return text


def predict(model, text: str) -> dict:
    """Return the predicted label, SVM decision scores and a vocabulary-coverage note."""
    text = _check_text(text)
    label = str(model.predict([text])[0])
    scores = model.decision_function([text])[0]
    ranked = sorted(zip(model.classes_, scores), key=lambda kv: kv[1], reverse=True)
    result = {
        "text": text,
        "label": label,
        "scores": {str(c): round(float(s), 3) for c, s in ranked},
        "margin": round(float(ranked[0][1] - ranked[1][1]), 3),
        "known_features": None,
    }
    tfidf = getattr(model, "named_steps", {}).get("tfidf")
    if tfidf is not None:
        result["known_features"] = int(tfidf.transform([text]).nnz)
    return result


def predict_many(model, texts: List[str]) -> List[dict]:
    out = []
    for t in texts:
        try:
            out.append(predict(model, t))
        except EmptyCommentError as exc:
            out.append({"text": t, "error": str(exc)})
    return out


def _print_result(r: dict, intended: Optional[str] = None) -> None:
    print(f"Comment:    {r['text']}")
    if "error" in r:
        print(f"  [skipped] {r['error']}\n")
        return
    if intended:
        print(f"  Written as an example of: {intended}")
    print(f"  Predicted:  {r['label']}   (margin over runner-up {r['margin']})")
    print("  Scores:     " + ", ".join(f"{c} {s:+.2f}" for c, s in r["scores"].items()))
    if r["known_features"] == 0:
        print("  Note: no word in this comment is in the model's vocabulary; the prediction "
              "is the model's default and is not meaningful.")
    print()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Classify YouTube comments with the CreatorPulse SVM.")
    parser.add_argument("comments", nargs="*", help="Comment text(s) to classify.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to the .joblib pipeline.")
    parser.add_argument("--examples", action="store_true", help="Classify one example comment per class.")
    args = parser.parse_args(argv)

    if not args.comments and not args.examples:
        parser.print_usage(sys.stderr)
        print("[ERROR] Give at least one comment, or --examples.", file=sys.stderr)
        return 2

    try:
        model = load_model(args.model)
    except ModelLoadError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("Note: model trained on AI-generated (Claude) labels; predictions may be wrong.\n")
    status = 0
    if args.examples:
        for intended, text in EXAMPLES:
            _print_result(predict(model, text), intended)
    for r in predict_many(model, args.comments):
        _print_result(r)
        if "error" in r:
            status = 2
    return status


if __name__ == "__main__":
    sys.exit(main())
