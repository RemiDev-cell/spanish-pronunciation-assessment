from src.features import _same_speaker_prominence_score, _zscore_values


def test_zscore_values_handles_missing_and_constant_values():
    assert _zscore_values([None, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_zscore_values_centers_available_values():
    zs = _zscore_values([1.0, 2.0, 3.0])

    assert round(sum(zs), 7) == 0.0
    assert zs[0] < zs[1] < zs[2]


def test_same_speaker_prominence_score_uses_direct_acoustic_values():
    quieter = _same_speaker_prominence_score(0.1, 180.0, 60.0)
    louder = _same_speaker_prominence_score(0.1, 180.0, 70.0)

    assert louder > quieter
