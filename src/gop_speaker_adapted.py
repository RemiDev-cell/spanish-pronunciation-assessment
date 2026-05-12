"""Speaker-adapted GOP scoring with wav2vec2 + LoRA.

The public surface is intentionally small:
fine-tune a per-speaker LoRA adapter on HeyGen reference recordings, then compute
phone-level GOP scores from MFA phone boundaries.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from src.config import get_settings
from src.models import LocalizedErrorType, PhonologyIssue

logger = logging.getLogger(__name__)

MODEL_ID = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-spanish-ep5-944h"
LIGHTWEIGHT_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
SAMPLE_RATE = 16_000

_SILENCE_LABELS = {"", "sil", "sp", "spn", "silence", "<eps>", "<unk>", "unk"}
_SPANISH_VOWELS = {"a", "e", "i", "o", "u"}
_SPANISH_B_V_ALLOPHONES = {"b", "v", "β"}
_SPECIAL_TOKEN_PATTERN = re.compile(r"^\[.*\]$|^<.*>$")

_PHONE_ALIASES: dict[str, tuple[str, ...]] = {
    "a": ("a", "á"),
    "e": ("e", "é"),
    "i": ("i", "í", "y"),
    "o": ("o", "ó"),
    "u": ("u", "ú", "ü", "w"),
    "b": ("b",),
    "β": ("β", "b", "v"),
    "d": ("d",),
    "ð": ("ð", "d"),
    "f": ("f",),
    "g": ("g", "ɡ"),
    "ɣ": ("ɣ", "g", "ɡ"),
    "x": ("x", "j", "g"),
    "j": ("j", "y", "ll"),
    "ʝ": ("ʝ", "j", "y", "ll"),
    "k": ("k", "c", "q"),
    "l": ("l",),
    "ʎ": ("ʎ", "ll", "y"),
    "m": ("m",),
    "n": ("n",),
    "ɲ": ("ɲ", "ñ", "ny"),
    "ŋ": ("ŋ", "n"),
    "p": ("p",),
    "r": ("r", "rr"),
    "ɾ": ("ɾ", "r"),
    "s": ("s", "z", "c"),
    "θ": ("θ", "z", "c"),
    "t": ("t",),
    "tʃ": ("tʃ", "ch"),
}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("missing_dependency:gop_speaker_adapted requires torch") from exc
    return torch


def _require_hf() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            Wav2Vec2ForCTC,
            Wav2Vec2Processor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "missing_dependency:gop_speaker_adapted requires transformers, peft, "
            "accelerate, torch, torchaudio, soundfile"
        ) from exc
    return (
        torch,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        LoraConfig,
        PeftModel,
        get_peft_model,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )


def _detect_device() -> str:
    torch = _require_torch()
    if torch.cuda.is_available():
        logger.info("SAME_SPEAKER_GOP: GPU CUDA detecte.")
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        logger.info("SAME_SPEAKER_GOP: Apple Silicon detecte, utilisation de MPS.")
        return "mps"
    logger.warning(
        "SAME_SPEAKER_GOP: ni CUDA ni MPS disponibles. "
        "Le fine-tuning en CPU peut prendre plusieurs heures. "
        "Considerez l'utilisation de Google Colab ou d'un environnement GPU."
    )
    return "cpu"


def _base_cache_dir(output_model_dir: Path) -> Path:
    return output_model_dir.parent / "base"


def _iter_heygen_pairs(heygen_audio_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for wav in sorted(heygen_audio_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt))
    return pairs


def _load_audio(audio_path: Path, target_sr: int = SAMPLE_RATE) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        try:
            import torch
            import torchaudio.functional as AF

            waveform = torch.tensor(audio, dtype=torch.float32)
            audio = AF.resample(waveform, sample_rate, target_sr).cpu().numpy()
            sample_rate = target_sr
        except Exception as exc:
            raise RuntimeError(f"audio_resample_failed:{audio_path}:{exc}") from exc
    return np.asarray(audio, dtype="float32"), sample_rate


def _read_transcript(txt_path: Path) -> str:
    text = txt_path.read_text(encoding="utf-8").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


class _HeygenCTCDataset:
    def __init__(self, pairs: list[tuple[Path, Path]], processor: Any):
        self.pairs = pairs
        self.processor = processor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wav_path, txt_path = self.pairs[idx]
        audio, sample_rate = _load_audio(wav_path)
        inputs = self.processor(audio, sampling_rate=sample_rate)
        transcript = _read_transcript(txt_path)
        try:
            labels = self.processor(text=transcript).input_ids
        except TypeError:
            with self.processor.as_target_processor():
                labels = self.processor(transcript).input_ids
        return {
            "input_values": inputs.input_values[0],
            "labels": labels,
        }


@dataclass
class _CTCCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        torch = _require_torch()
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        try:
            labels_batch = self.processor.pad(
                labels=label_features, padding=True, return_tensors="pt"
            )
        except TypeError:
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features, padding=True, return_tensors="pt"
                )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels.to(dtype=torch.long)
        return batch


def _make_lora_model(base_model: Any, LoraConfig: Any, get_peft_model: Any) -> Any:
    kwargs = dict(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    try:
        lora_config = LoraConfig(task_type="CTC", **kwargs)
        return get_peft_model(base_model, lora_config)
    except Exception:
        lora_config = LoraConfig(**kwargs)
        return get_peft_model(base_model, lora_config)


def _training_args(
    TrainingArguments: Any,
    output_dir: Path,
    device: str,
    has_eval: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": 10,
        "per_device_train_batch_size": 4,
        "learning_rate": 1e-4,
        "warmup_steps": 50,
        "save_strategy": "epoch",
        "dataloader_num_workers": 2,
        "report_to": [],
        "logging_strategy": "epoch",
    }
    if has_eval:
        kwargs.update(
            {
                "evaluation_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    if device == "cuda":
        kwargs["fp16"] = True
    elif device == "mps":
        kwargs["bf16"] = True
        kwargs["use_mps_device"] = True

    try:
        return TrainingArguments(**kwargs)
    except (TypeError, ValueError):
        kwargs.pop("use_mps_device", None)
        kwargs.pop("bf16", None)
        return TrainingArguments(**kwargs)


def _split_pairs(pairs: list[tuple[Path, Path]]) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if len(pairs) < 2:
        return pairs, []
    n_eval = max(1, math.ceil(len(pairs) * 0.2))
    return pairs[:-n_eval], pairs[-n_eval:]


def _run_fine_tuning(
    *,
    heygen_pairs: list[tuple[Path, Path]],
    output_model_dir: Path,
    base_model_id: str,
    device: str,
) -> dict[str, Any]:
    (
        _torch,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        LoraConfig,
        _PeftModel,
        get_peft_model,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    ) = _require_hf()

    cache_dir = _base_cache_dir(output_model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor = Wav2Vec2Processor.from_pretrained(base_model_id, cache_dir=str(cache_dir))
    base_model = Wav2Vec2ForCTC.from_pretrained(base_model_id, cache_dir=str(cache_dir))
    base_model.config.ctc_loss_reduction = "mean"
    if getattr(processor, "tokenizer", None) is not None:
        base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    if hasattr(base_model, "freeze_feature_encoder"):
        base_model.freeze_feature_encoder()

    model = _make_lora_model(base_model, LoraConfig, get_peft_model)
    model.to(device)

    train_pairs, eval_pairs = _split_pairs(heygen_pairs)
    train_dataset = _HeygenCTCDataset(train_pairs, processor)
    eval_dataset = _HeygenCTCDataset(eval_pairs, processor) if eval_pairs else None
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset else []
    args = _training_args(
        TrainingArguments,
        output_model_dir / "checkpoints",
        device,
        has_eval=eval_dataset is not None,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_CTCCollator(processor),
        tokenizer=processor,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    adapter_dir = output_model_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(output_model_dir / "processor"))

    metrics = dict(getattr(train_result, "metrics", {}) or {})
    if eval_dataset is not None:
        metrics.update({f"final_{k}": v for k, v in trainer.evaluate().items()})
    metrics.update(
        {
            "base_model_id": base_model_id,
            "train_examples": len(train_pairs),
            "eval_examples": len(eval_pairs),
            "device": device,
        }
    )
    return metrics


def fine_tune_on_heygen(
    heygen_audio_dir: str,
    output_model_dir: str,
    base_model_id: str = MODEL_ID,
    force_retrain: bool = False,
) -> str:
    """
    Fine-tune a wav2vec2 CTC model with a speaker-specific LoRA adapter.

    If no HeyGen wav/txt pairs are present, returns an empty adapter path and leaves
    inference to the generic base model.
    """
    heygen_dir = Path(heygen_audio_dir)
    out_dir = Path(output_model_dir)
    adapter_dir = out_dir / "lora_adapter"
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists() and not force_retrain:
        logger.info("SAME_SPEAKER_GOP: adaptateur LoRA existant reutilise: %s", adapter_dir)
        return str(adapter_dir)

    pairs = _iter_heygen_pairs(heygen_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "base_model_id.txt").write_text(base_model_id + "\n", encoding="utf-8")
    if not pairs:
        logger.warning("SAME_SPEAKER: adaptation locuteur desactivee, modele generique utilise.")
        return ""

    device = _detect_device()
    try:
        metrics = _run_fine_tuning(
            heygen_pairs=pairs,
            output_model_dir=out_dir,
            base_model_id=base_model_id,
            device=device,
        )
    except (NotImplementedError, RuntimeError) as exc:
        if device != "mps":
            raise
        logger.warning(
            "SAME_SPEAKER_GOP: operation non supportee sur MPS (%s). "
            "Repli sur CPU pour le fine-tuning.",
            exc,
        )
        metrics = _run_fine_tuning(
            heygen_pairs=pairs,
            output_model_dir=out_dir,
            base_model_id=base_model_id,
            device="cpu",
        )
        metrics["mps_fallback_error"] = str(exc)

    (out_dir / "fine_tuning_log.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(adapter_dir)


def _normalize_phone_label(label: str) -> str:
    phone = str(label or "").strip().lower()
    phone = phone.strip("/")
    phone = phone.replace("ˈ", "").replace("ˌ", "").replace(".", "")
    phone = phone.replace("ɡ", "g")
    return phone


def _processor_token_vocab(processor: Any) -> dict[str, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "get_vocab"):
        return dict(tokenizer.get_vocab())
    if hasattr(tokenizer, "vocab"):
        return dict(tokenizer.vocab)
    if hasattr(tokenizer, "encoder"):
        return dict(tokenizer.encoder)
    raise RuntimeError("gop_vocab_unavailable")


def build_phoneme_vocab(processor: Any) -> dict[str, int]:
    """Build a forgiving MFA-phone -> CTC-token index map."""
    raw_vocab = _processor_token_vocab(processor)
    normalized_token_to_idx = {
        _normalize_phone_label(token): int(idx) for token, idx in raw_vocab.items()
    }
    phone_vocab: dict[str, int] = {}
    for phone, aliases in _PHONE_ALIASES.items():
        candidates = (phone,) + aliases
        for candidate in candidates:
            idx = normalized_token_to_idx.get(_normalize_phone_label(candidate))
            if idx is not None:
                phone_vocab[phone] = idx
                phone_vocab[f"/{phone}/"] = idx
                break
    for token, idx in raw_vocab.items():
        normalized = _normalize_phone_label(token)
        if normalized and normalized not in _SILENCE_LABELS:
            phone_vocab.setdefault(normalized, int(idx))
            phone_vocab.setdefault(f"/{normalized}/", int(idx))
    return phone_vocab


def _id_to_label(processor: Any) -> dict[int, str]:
    vocab = _processor_token_vocab(processor)
    return {int(idx): str(token) for token, idx in vocab.items()}


def _valid_competitor_indices(processor: Any) -> list[int]:
    vocab = _processor_token_vocab(processor)
    tokenizer = getattr(processor, "tokenizer", processor)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    word_delimiter = getattr(tokenizer, "word_delimiter_token", "|")
    valid: list[int] = []
    for token, idx_raw in vocab.items():
        idx = int(idx_raw)
        if idx in special_ids:
            continue
        normalized = _normalize_phone_label(token)
        if token == word_delimiter or normalized in _SILENCE_LABELS:
            continue
        if _SPECIAL_TOKEN_PATTERN.match(str(token)):
            continue
        valid.append(idx)
    return sorted(set(valid))


def compute_gop_from_frame_scores(
    frame_scores: Any,
    phoneme_alignments: list[dict[str, Any]],
    vocab: dict[str, int],
    total_duration_sec: float,
    *,
    id_to_label: Optional[dict[int, str]] = None,
    valid_competitor_indices: Optional[Iterable[int]] = None,
    error_threshold: Optional[float] = None,
    warning_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Compute GOP from raw frame logits or log-probabilities.

    Raw logits are preferred by compute_phoneme_gop because they avoid the
    over-confidence introduced by normalized softmax posteriors.
    """
    torch = _require_torch()
    settings = get_settings()
    err = settings.gop_error_threshold if error_threshold is None else error_threshold
    warn = settings.gop_warning_threshold if warning_threshold is None else warning_threshold
    scores = frame_scores.detach().cpu() if hasattr(frame_scores, "detach") else torch.tensor(frame_scores)
    if scores.ndim != 2:
        raise ValueError("frame_scores must have shape (n_frames, vocab_size)")
    total_duration = max(float(total_duration_sec), 1e-6)
    fps = float(scores.shape[0]) / total_duration
    valid_indices = list(valid_competitor_indices or range(scores.shape[1]))
    reverse_vocab = id_to_label or {idx: label for label, idx in vocab.items()}

    results: list[dict[str, Any]] = []
    for align in phoneme_alignments:
        phone = str(align.get("phoneme") or align.get("label") or "")
        normalized_phone = _normalize_phone_label(phone)
        start = float(align.get("t_start", align.get("start", 0.0)) or 0.0)
        end = float(align.get("t_end", align.get("end", start)) or start)
        duration = max(end - start, 0.0)

        target_idx = vocab.get(phone)
        if target_idx is None:
            target_idx = vocab.get(normalized_phone)
        if normalized_phone in _SILENCE_LABELS:
            target_idx = None

        f_start = max(0, min(int(start * fps), scores.shape[0]))
        f_end = max(0, min(int(math.ceil(end * fps)), scores.shape[0]))
        if f_end <= f_start and f_start < scores.shape[0]:
            f_end = f_start + 1
        segment = scores[f_start:f_end]

        if target_idx is None or target_idx < 0 or target_idx >= scores.shape[1] or segment.numel() == 0:
            results.append(
                {
                    "phoneme": phone,
                    "t_start": start,
                    "t_end": end,
                    "gop": None,
                    "gop_normalized": None,
                    "best_competing": None,
                    "status": "unavailable",
                }
            )
            continue

        mean_scores = segment.mean(dim=0)
        target_score = float(mean_scores[target_idx].item())
        competing = torch.full_like(mean_scores, float("-inf"))
        for idx in valid_indices:
            if 0 <= int(idx) < competing.shape[0]:
                competing[int(idx)] = mean_scores[int(idx)]
        competing[target_idx] = float("-inf")
        best_competing_idx = int(competing.argmax().item())
        best_competing_score = float(competing[best_competing_idx].item())
        if not math.isfinite(best_competing_score):
            best_competing_idx = int(mean_scores.argmax().item())
            best_competing_score = float(mean_scores[best_competing_idx].item())

        gop = target_score - best_competing_score
        if gop < err:
            status = "mispronunciation"
        elif gop < warn:
            status = "near_miss"
        else:
            status = "correct"
        results.append(
            {
                "phoneme": phone,
                "t_start": start,
                "t_end": end,
                "gop": gop,
                "gop_normalized": gop / max(duration, 1e-6),
                "best_competing": reverse_vocab.get(best_competing_idx, str(best_competing_idx)),
                "status": status,
            }
        )
    return results


def _load_adapted_model_and_processor(
    lora_adapter_path: str,
    base_model_id: str,
) -> tuple[Any, Any]:
    (
        _torch,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        _LoraConfig,
        PeftModel,
        _get_peft_model,
        _Trainer,
        _TrainingArguments,
        _EarlyStoppingCallback,
    ) = _require_hf()
    settings = get_settings()
    adapter_path = Path(lora_adapter_path) if lora_adapter_path else Path()
    cache_root = (
        adapter_path.parent.parent / "base"
        if adapter_path.exists()
        else settings.project_root / "models" / "base"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    processor_dir = adapter_path.parent / "processor" if adapter_path.exists() else None
    processor_source = str(processor_dir) if processor_dir and processor_dir.exists() else base_model_id
    processor = Wav2Vec2Processor.from_pretrained(processor_source, cache_dir=str(cache_root))
    model = Wav2Vec2ForCTC.from_pretrained(base_model_id, cache_dir=str(cache_root))
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
    elif lora_adapter_path:
        logger.warning(
            "SAME_SPEAKER_GOP: adaptateur introuvable (%s), modele generique utilise.",
            lora_adapter_path,
        )
    return model, processor


def compute_phoneme_gop(
    audio_path: str,
    phoneme_alignments: list[dict],
    lora_adapter_path: str,
    base_model_id: str = MODEL_ID,
) -> list[dict]:
    """
    Compute per-phone GOP for one recording from MFA phone alignments.

    The GOP score is computed on raw wav2vec2 frame logits, not on softmax
    probabilities, following recent GOP robustness findings.
    """
    torch = _require_torch()
    model, processor = _load_adapted_model_and_processor(lora_adapter_path, base_model_id)
    device = _detect_device()
    model.to(device)
    model.eval()

    audio, sample_rate = _load_audio(Path(audio_path))
    total_duration_sec = float(len(audio)) / float(sample_rate)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        if attention_mask is not None:
            logits = model(input_values, attention_mask=attention_mask).logits
        else:
            logits = model(input_values).logits
    frame_logits = logits[0].detach().cpu()
    vocab = build_phoneme_vocab(processor)
    return compute_gop_from_frame_scores(
        frame_logits,
        phoneme_alignments,
        vocab,
        total_duration_sec,
        id_to_label=_id_to_label(processor),
        valid_competitor_indices=_valid_competitor_indices(processor),
    )


def gop_results_to_phonology_issues(gop_results: list[dict[str, Any]]) -> list[PhonologyIssue]:
    """Convert actionable GOP findings into the existing scoring issue model."""
    issues: list[PhonologyIssue] = []
    for i, item in enumerate(gop_results):
        status = item.get("status")
        if status not in {"mispronunciation", "near_miss"}:
            continue
        phone = _normalize_phone_label(str(item.get("phoneme") or "phoneme"))
        gop = item.get("gop")
        gop_float = float(gop) if gop is not None else -2.0
        best = _normalize_phone_label(str(item.get("best_competing") or "autre phoneme"))
        if phone in _SPANISH_B_V_ALLOPHONES and best in _SPANISH_B_V_ALLOPHONES:
            continue
        error_type = (
            LocalizedErrorType.VOYELLE_MAL_REALISEE
            if phone in _SPANISH_VOWELS
            else LocalizedErrorType.SUBSTITUTION_SEGMENTALE
        )
        severity = min(0.6, 0.16 + abs(min(gop_float, 0.0)) / 7.0)
        confidence = 0.72 if status == "mispronunciation" else 0.55
        priority = "haute" if status == "mispronunciation" else "moyenne"
        issues.append(
            PhonologyIssue(
                error_type=error_type,
                target_unit=phone or str(item.get("phoneme") or "phoneme"),
                precise_location=f"gop_phone_index_{i}|{item.get('t_start')}:{item.get('t_end')}",
                confidence=confidence,
                observation=(
                    "GOP locuteur-adapte sous le seuil pour le phoneme cible "
                    f"({item.get('phoneme')}, GOP={gop_float:.2f})."
                ),
                observed=best,
                expected=str(item.get("phoneme") or phone),
                perceptual_effect=(
                    "Le modele acoustique adapte juge un phoneme concurrent plus vraisemblable."
                    if status == "mispronunciation"
                    else "Le phoneme reste proche mais moins stable que la reference adaptee."
                ),
                correction="Reprendre ce phoneme en imitant la reference HeyGen du meme locuteur.",
                priority=priority,
                score_penalty_hint=severity,
            )
        )
    return issues
