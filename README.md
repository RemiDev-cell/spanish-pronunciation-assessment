# Spanish phonetic assessment MVP

Production-minded **MVP** with two modes:

1. **Learner vs script** — one audio file + expected Spanish text (`comparison_type`: `audio_vs_expected_text`).
2. **Learner vs reference audio** — model recording + learner recording of the **same provided script** (`audio_vs_reference_audio`): MFA aligns both to the shared text, then **heuristic deltas** (durations, pauses, prominence, global tempo, F0 variability) are scored vs the reference.

Both modes chain **Whisper** (reading validation), **Montreal Forced Aligner** (MFA), **Parselmouth/Praat**, and **silabeador**, with **explicit HEURISTIC** comments in code where inference is limited.

## What is actually measured vs heuristic

| Layer | What is measured | What is heuristic |
|------|------------------|-------------------|
| ASR gate | Whisper transcript vs expected text (token-level similarity) | Thresholds for warn/reject; partial match boost |
| MFA | Word/phone time boundaries from acoustic model | Mapping phones → syllable “prominence windows” (not physiological syllabification in the signal) |
| Prosody | F0 and intensity samples inside time intervals; pause lengths between aligned words | “Syllable” sub-intervals inside a word are **proportional to grapheme length** of each syllable (see `src/features.py`) |
| Lexical stress | Text stress from **silabeador** vs **max acoustic prominence** inside those proportional windows | Prominence = weighted mix of duration, mean F0, mean intensity — not a full prominence model |
| Reference pair | Per-word deltas vs reference on the **same script** (durations, pauses, prominence z on tonic syllable, WPM ratio, σ(F0)) | Same syllable-window limits as above; different microphones/gain affect raw intensity/duration |
| Segmental | Phone duration outliers vs global utterance statistics | Vowel/consonne classification from MFA phone **symbols** (approximate) |

## Not implemented yet (honest MVP limits)

- Native syllable-tier alignment or G2P-based syllable–phone grouping.
- Vowel formant tracking, true SNR, or robust VAD-based disfluency detection.
- Reference-free “ground truth” phoneme correctness (MFA aligns expected text, not what was truly produced if the learner deviates strongly while ASR still passes).
- `omission_segmentale` / `ajout_segmental` as separate detectors (taxonomy reserved; rarely emitted without dedicated ASR–align differencing).

## Why phoneme-level scoring is only approximate

MFA provides **time-aligned phone labels under the hypothesis that the expected transcript was read**. It does **not** prove each phone was realized as a native target; duration outliers flag **possible** articulatory or segmentation issues, not categorical substitution errors.

## Why lexical stress estimation is heuristic unless syllable prominence is fully modeled

Lexical stress from **silabeador** is orthographic. Acoustic “stress” is inferred from **coarse proxies** in hand-built syllable-sized windows. A mismatch can reflect true stress errors, **inaccurate window placement**, or coarticulation — always read together with `niveau_confiance`.

## Requirements

- **Python** 3.9+ (3.11+ recommended for performance).
- **ffmpeg** on `PATH` (decode mp3/m4a/mp4 → mono 16 kHz WAV).
- **Montreal Forced Aligner 2.x** with `mfa` on `PATH`, plus Spanish pretrained **dictionary + acoustic** models (names or paths via env — see `.env.example`).
- **Praat** installed where **Parselmouth** expects it (bundled binary with `praat-parselmouth` wheels on many platforms).

### MFA models (example)

```bash
mfa model download dictionary spanish_mfa
mfa model download acoustic spanish_mfa
```

If your MFA version uses different model IDs, set `SPANISH_PHON_MFA_DICTIONARY` and `SPANISH_PHON_MFA_ACOUSTIC` to those identifiers or to **local model paths**.

### Troubleshooting MFA failures on single long files

MFA can fail when aligning **a single long recording**. This repo uses `mfa align_one` and will automatically retry with a more relaxed beam (`SPANISH_PHON_MFA_RETRY_BEAM`). If alignment still fails:

- Ensure the **script text matches** what is spoken (even small omissions can hurt).
- Try a **shorter excerpt** first (10–30 seconds) to validate your setup.
- Check MFA logs under `~/Documents/MFA` (or `MFA_ROOT_DIR`) and the temporary directory mentioned in the error.

## Install

```bash
cd spanish_phon_assessment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

Optional: copy `.env.example` to `.env` and adjust thresholds or MFA model names.

## JSON schema

Regenerate after model changes:

```bash
python scripts/generate_json_schema.py
```

Output: [`json_schema/evaluation_report.schema.json`](json_schema/evaluation_report.schema.json).

## CLI

**Mode A — learner vs expected text**

```bash
python -m src.pipeline \
  --audio path/to/learner.mp3 \
  --text "Texto esperado en español" \
  --output data/output/report.json
```

**Mode B — learner vs reference (same script read twice)**

```bash
python -m src.pipeline \
  --reference-audio path/to/reference_model.wav \
  --audio path/to/learner.wav \
  --text "El mismo guion leído por los dos." \
  --output data/output/report_pair.json
```

The JSON includes `comparison_type`, `reference_audio_path`, `test_audio_path`, and `asr_reference_text` (plus `asr_text` for the learner).

Flags:

- `--strict-text-match` — raises ASR warn/reject thresholds (more `non_evaluable`).
- `--allow-partial-match` — loosens ASR gate only (`rapidfuzz.partial_ratio` blend). **Does not** shorten or alter the text passed to MFA.
- `--debug` — print Python tracebacks for technical failures.

Example with warnings written to stdout:

```bash
python -m src.pipeline --audio data/input/learner.wav --text "Hola mundo"
```

A minimal illustrative JSON (no real audio run) is at [`data/output/sample_report.json`](data/output/sample_report.json).

## Tests

```bash
pytest
```

Heavy integration (Whisper + MFA + Praat on real audio) is intentionally **manual** — keep CI on the lightweight unit tests above.

## Score model (fixed weights)

| Domain | Max points |
|--------|------------|
| segmental_precision | 30 |
| syllabic_structure_and_articulation | 15 |
| lexical_stress | 20 |
| sentence_prosody | 20 |
| fluency_and_voice_control | 10 |
| phonological_intelligibility | 5 |

French interpretation bands are applied per domain `percent_similarity` (see `src/scoring.py`).

## Repository layout

- `src/config.py` — env-driven settings.
- `src/preprocess.py` — ffmpeg + quality gate.
- `src/text_processing.py` — normalization + silabeador.
- `src/transcribe.py` / `src/asr_validation.py` — Whisper + text policy.
- `src/align.py` / `src/textgrid_io.py` — MFA + TextGrid parsing.
- `src/features.py` — Parselmouth features + pause metrics.
- `src/spanish_phonology.py` — issue detectors (taxonomy) for learner-vs-text mode.
- `src/reference_pair_phonology.py` — learner-vs-reference deltas (same script).
- `src/scoring.py` / `src/reporting.py` — points + JSON assembly.
- `src/pipeline.py` — orchestration + CLI.

## Next steps (realistic roadmap)

- Syllable-tier MFA post-processing or neural phoneme-to-syllable alignment.
- Richer disfluency detection (intra-word silence, repetitions from ASR).
- Calibrated thresholds from labeled learner data instead of fixed heuristics.

## License

Add your preferred license file for redistribution (dependencies include LGPL silabeador, MIT/BSD stack, etc.).