from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from src.gop_speaker_adapted import (
    MODEL_ID,
    build_phoneme_vocab,
    compute_gop_from_frame_scores,
    compute_phoneme_gop,
    gop_results_to_phonology_issues,
)


class _FakeTokenizer:
    pad_token_id = 4
    all_special_ids = [4]
    word_delimiter_token = "|"

    def get_vocab(self):
        return {"a": 0, "e": 1, "i": 2, "o": 3, "u": 4, "[PAD]": 5, "|": 6, "r": 7}


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def __call__(self, audio, sampling_rate=None, return_tensors=None, padding=None):
        return SimpleNamespace(
            input_values=torch.zeros(1, 16000, dtype=torch.float32),
            attention_mask=None,
        )


def test_smoke_model_vocab_contains_expected_spanish_phonemes():
    if os.getenv("RUN_GOP_MODEL_TESTS") == "1":
        transformers = pytest.importorskip("transformers")
        processor = transformers.Wav2Vec2Processor.from_pretrained(MODEL_ID)
        model = transformers.Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        assert model.config.vocab_size > 0
    else:
        processor = _FakeProcessor()

    vocab = build_phoneme_vocab(processor)
    for phone in ["a", "e", "i", "o", "u", "r", "ɾ"]:
        assert phone in vocab


def test_compute_phoneme_gop_on_synthetic_signal(monkeypatch):
    import src.gop_speaker_adapted as gop_module

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return None

        def __call__(self, input_values, attention_mask=None):
            logits = torch.zeros(1, 10, 7, dtype=torch.float32)
            logits[:, :, 0] = 2.0
            logits[:, :, 1] = 0.5
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        gop_module,
        "_load_audio",
        lambda path: (torch.sin(torch.linspace(0, 100, 16000)).numpy(), 16000),
    )
    monkeypatch.setattr(
        gop_module,
        "_load_adapted_model_and_processor",
        lambda adapter, base: (FakeModel(), _FakeProcessor()),
    )
    monkeypatch.setattr(gop_module, "_detect_device", lambda: "cpu")

    results = compute_phoneme_gop(
        "synthetic.wav",
        [{"phoneme": "a", "t_start": 0.0, "t_end": 1.0}],
        "unused_adapter",
        MODEL_ID,
    )

    assert len(results) == 1
    item = results[0]
    assert {"gop", "gop_normalized", "best_competing", "status"}.issubset(item)
    assert item["status"] == "correct"


def test_reference_like_gop_is_non_negative_after_mocked_adaptation():
    frame_scores = torch.tensor(
        [
            [2.0, 1.4, 0.2],
            [1.8, 1.2, 0.1],
            [2.1, 1.5, 0.0],
        ],
        dtype=torch.float32,
    )

    results = compute_gop_from_frame_scores(
        frame_scores,
        [{"phoneme": "a", "t_start": 0.0, "t_end": 1.0}],
        {"a": 0, "e": 1, "i": 2},
        1.0,
        id_to_label={0: "a", 1: "e", 2: "i"},
        valid_competitor_indices=[0, 1, 2],
    )

    assert results[0]["gop"] >= 0
    assert results[0]["status"] == "correct"


def test_gop_issue_conversion_skips_spanish_b_v_noncontrast():
    issues = gop_results_to_phonology_issues(
        [
            {
                "phoneme": "b",
                "t_start": 0.0,
                "t_end": 0.1,
                "gop": -4.0,
                "gop_normalized": -40.0,
                "best_competing": "v",
                "status": "mispronunciation",
            }
        ]
    )

    assert issues == []


def test_gop_status_thresholds_are_configurable():
    rows = compute_gop_from_frame_scores(
        torch.tensor([[0.0, 1.5]], dtype=torch.float32),
        [{"phoneme": "a", "t_start": 0.0, "t_end": 1.0}],
        {"a": 0, "e": 1},
        1.0,
        id_to_label={0: "a", 1: "e"},
        valid_competitor_indices=[0, 1],
        error_threshold=-2.0,
        warning_threshold=-1.0,
    )

    assert rows[0]["status"] == "near_miss"
