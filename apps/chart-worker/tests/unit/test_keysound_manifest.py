from uuid import UUID

import pytest
from pydantic import ValidationError

from chart_worker.schema.keysound import KeysoundManifest


def _manifest(**overrides) -> KeysoundManifest:
    values = {
        "song_version_id": UUID(int=1),
        "bgm_asset_id": UUID(int=2),
        "keys_asset_id": UUID(int=3),
        "drum_onsets": [100, 220],
    }
    return KeysoundManifest(**(values | overrides))


def test_keysound_defaults_are_the_phase3_contract():
    manifest = _manifest()
    assert manifest.slice_sec == 0.30
    assert manifest.preroll_sec == 0.012
    assert manifest.snap_window_ms == 50


def test_keysound_json_uses_camel_case():
    payload = _manifest().model_dump(by_alias=True)
    assert payload["schemaVersion"] == 1
    assert payload["drumOnsets"] == [100, 220]
    assert "drum_onsets" not in payload


@pytest.mark.parametrize("onsets", [[220, 100], [100, 100], [-1]])
def test_keysound_rejects_unsorted_duplicate_or_negative_onsets(onsets):
    with pytest.raises(ValidationError, match="drumOnsets"):
        _manifest(drum_onsets=onsets)
