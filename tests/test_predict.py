"""Tests for src.predict (uses a tiny model fitted in a temp folder)."""

import joblib
import pytest

from src import predict as P
from tests.test_train_svm import fitted


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(fitted(), path)
    return path


def test_load_missing_model(tmp_path):
    with pytest.raises(P.ModelLoadError, match="not found"):
        P.load_model(tmp_path / "missing.joblib")


def test_load_directory_is_rejected(tmp_path):
    with pytest.raises(P.ModelLoadError, match="not a file"):
        P.load_model(tmp_path)


def test_load_corrupt_model(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a model")
    with pytest.raises(P.ModelLoadError, match="Could not load"):
        P.load_model(bad)


def test_load_non_classifier(tmp_path):
    path = tmp_path / "dict.joblib"
    joblib.dump({"a": 1}, path)
    with pytest.raises(P.ModelLoadError, match="fitted classifier"):
        P.load_model(path)


@pytest.mark.parametrize("text", ["", "   \n", None, "...", "-- ,,"])
def test_empty_input_rejected(model_path, text):
    model = P.load_model(model_path)
    with pytest.raises(P.EmptyCommentError):
        P.predict(model, text)


def test_predict_returns_label_and_scores(model_path):
    model = P.load_model(model_path)
    r = P.predict(model, "please make a video on django")
    assert r["label"] in model.classes_
    assert set(r["scores"]) == set(model.classes_)
    assert list(r["scores"])[0] == r["label"]
    assert r["margin"] >= 0


def test_predict_many_keeps_going_after_empty(model_path):
    model = P.load_model(model_path)
    out = P.predict_many(model, ["", "thank you best video"])
    assert "error" in out[0] and "label" in out[1]


def test_cli_exit_codes(model_path, tmp_path, capsys):
    assert P.main(["--model", str(tmp_path / "nope.joblib"), "hi"]) == 1
    assert P.main(["--model", str(model_path)]) == 2
    assert P.main(["--model", str(model_path), "   "]) == 2
    assert P.main(["--model", str(model_path), "what is a tuple ?"]) == 0
    assert P.main(["--model", str(model_path), "--examples"]) == 0
    assert len(P.EXAMPLES) == 6
