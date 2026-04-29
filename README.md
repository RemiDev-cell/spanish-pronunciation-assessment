# Spanish comparative pronunciation assessment MVP

Production-minded **MVP** for comparing two Spanish recordings of the **same
speaker**, same equipment, same acoustic environment, and same provided script.
The speaker first records a personal reference performance, then later records
the production to evaluate.

The pipeline chains **Whisper** for reading validation, **Montreal Forced Aligner**
(MFA) for word/phone timing, **Parselmouth/Praat** for acoustic features, and
**silabeador** for lexical stress expectations. The report is a comparative aid:
it surfaces acoustic and temporal deltas against the speaker's own reference,
with explicit heuristic limits. It is not an absolute judge of phoneme correctness.

## Core use case assumption

This tool assumes a **same-speaker reference pair**:

- the reference and learner/evaluated recordings are made by the same person;
- the microphone, gain, room, and recording workflow are kept stable;
- differences in F0, intensity, timing, and vowel acoustics are therefore treated as production differences, not anatomical speaker differences.

Inter-speaker normalization is intentionally disabled for decision logic. Direct
Hz, dB, and duration comparisons are meaningful under this protocol. Quality
checks such as clipping and silence detection remain guardrails, not bias
correctors.

## What is actually measured vs heuristic

| Layer | What is measured | What is heuristic |
|------|------------------|-------------------|
| ASR gate | Whisper transcripts for reference and evaluated audio vs the shared script | Thresholds for warn/reject; partial match boost |
| MFA | Word/phone time boundaries for both recordings under the same script | MFA aligns to the expected transcript; it does not prove what was truly produced |
| Prosody | F0 and intensity samples inside aligned intervals; pause lengths between aligned words | Assumes same speaker/setup, so direct Hz and dB deltas are interpretable |
| Lexical stress | Text stress from **silabeador** vs same-speaker duration/F0/intensity prominence in word sub-intervals | “Syllable” windows are proportional to grapheme length, not true signal syllable boundaries |
| Reference comparison | Per-word durations, pauses, prominence, global tempo, F0 variability, and vowel F1/F2 deltas vs personal reference | Thresholds are configurable but not yet empirically calibrated |
| Segmental timing | Phone duration distributions and vowel formants from MFA intervals | Duration/formant outliers indicate possible issues, not categorical substitutions or omissions |

## Current limits

- No native syllable-tier alignment or G2P-based syllable-to-phone grouping.
- No true SNR or robust VAD-based disfluency detection.
- Vowel formants are extracted as same-speaker F1/F2 deltas, but remain sensitive to MFA boundaries and short vowels.
- Clipping and mostly-silent audio are flagged with simple signal-level heuristics.
- No reference-free ground truth phoneme correctness.
- No calibrated scoring against human pronunciation judgments.
- `omission_segmentale` and `ajout_segmental` remain taxonomy entries, not robust detectors.

## Requirements

- **Python** 3.9+ (3.11+ recommended for performance).
- **ffmpeg** on `PATH` for decoding to mono 16 kHz WAV.
- **Montreal Forced Aligner 2.x** with `mfa` on `PATH`, plus Spanish pretrained dictionary and acoustic models.
- **Praat** through `praat-parselmouth`.

### MFA models

```bash
mfa model download dictionary spanish_mfa
mfa model download acoustic spanish_mfa
```

If your MFA version uses different model IDs, set `SPANISH_PHON_MFA_DICTIONARY`
and `SPANISH_PHON_MFA_ACOUSTIC` to those identifiers or to local model paths.

## Install

```bash
cd spanish_phon_assessment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Optional: copy `.env.example` to `.env` and adjust thresholds or MFA model names.
Same-speaker comparison thresholds are exposed through `SPANISH_PHON_*` settings
such as `SPANISH_PHON_WORD_DURATION_RATIO_LOW`,
`SPANISH_PHON_STRESS_PROMINENCE_DELTA_THRESHOLD`, and
`SPANISH_PHON_F0_STD_RATIO_LOW`. Vowel-quality thresholds are also configurable:
`SPANISH_PHON_VOWEL_FORMANT_F1_DELTA_THRESHOLD_HZ`,
`SPANISH_PHON_VOWEL_FORMANT_F2_DELTA_THRESHOLD_HZ`, and
`SPANISH_PHON_VOWEL_FORMANT_DISTANCE_THRESHOLD_HZ`.

## CLI

The CLI requires three inputs: the speaker's personal reference audio, the later
evaluated audio, and the shared Spanish script.

```bash
python -m src.pipeline \
  --reference-audio path/to/self_reference.wav \
  --audio path/to/learner.wav \
  --text "El mismo guion leído por los dos." \
  --output data/output/report_pair.json
```

The JSON includes `comparison_type`, `model_audio_path`, `learner_audio_path`,
`asr_model_text`, and `asr_text` for the learner. It also exposes audit fields:

- `alignment_artifacts` with model and learner TextGrid paths when MFA alignment succeeds.
  With `--output`, TextGrids are copied to `aligned_textgrids/` next to the report.
  Without `--output`, they are copied under `data/output/aligned_textgrids/`.
- `audio_quality` with the model and learner quality-gate measurements.
  It includes duration, RMS, high-frequency energy ratio, clipping ratio, and silence ratio.
- `raw_metrics` with model, learner, and delta acoustic/timing measurements used by the heuristic comparison.
  `raw_metrics.assumptions` records that same-speaker mode is active and inter-speaker normalization is disabled.
  Syllable prominence exposes direct duration, F0, and intensity components.
  Vowel formants expose F1/F2 values and deltas for aligned vowel phones.
- `confidence_by_domain` with simple evidence levels for ASR/script match, alignment, prosody, and vowel quality.
  Vowel-quality confidence is low when too few aligned vowels have complete F1/F2 measurements.

Flags:

- `--strict-text-match` raises ASR warn/reject thresholds.
- `--allow-partial-match` loosens the ASR gate only; it does not shorten or alter the text passed to MFA.
- `--debug` prints Python tracebacks for technical failures.

## JSON schema

Regenerate after model changes:

```bash
python scripts/generate_json_schema.py
```

Output: [`json_schema/evaluation_report.schema.json`](json_schema/evaluation_report.schema.json).

## Tests

```bash
pytest
```

Heavy integration tests with Whisper, MFA, and Praat on real audio remain manual.

## Score model

The current score model is still fixed-weight and heuristic. It starts from domain
maximums and subtracts issue penalties. Treat scores as comparative indicators,
not psychometric measurements.

| Domain | Max points |
|--------|------------|
| segmental_precision | 30 |
| syllabic_structure_and_articulation | 15 |
| lexical_stress | 20 |
| sentence_prosody | 20 |
| fluency_and_voice_control | 10 |
| phonological_intelligibility | 5 |

## Repository layout

- `src/config.py` — env-driven settings.
- `src/preprocess.py` — ffmpeg conversion and basic quality gate.
- `src/text_processing.py` — normalization, tokenization, silabeador expectations.
- `src/transcribe.py` / `src/asr_validation.py` — Whisper and text validation.
- `src/align.py` / `src/textgrid_io.py` — MFA and TextGrid parsing.
- `src/features.py` — Parselmouth features and pause metrics.
- `src/reference_pair_phonology.py` — learner-vs-model deltas for the same script.
- `src/scoring.py` / `src/reporting.py` — heuristic points and JSON assembly.
- `src/pipeline.py` — orchestration and CLI.

## Roadmap

Short term:

- Add lightweight diagnostic plots for timing, pauses, and F0.
- Add confidence signals for audio quality and ASR instability in non-evaluable reports.

Medium term:

- Add selected consonant timing/spectral features.
- Replace proportional syllable windows with syllable-to-phone grouping.
- Add diagnostic plots for timing, pauses, F0, and prominence.

Long term:

- Calibrate thresholds and scores against annotated learner data.
- Add per-domain confidence scores.
- Separate expert JSON from learner-facing pedagogical reports.

## License

Add your preferred license file for redistribution.
