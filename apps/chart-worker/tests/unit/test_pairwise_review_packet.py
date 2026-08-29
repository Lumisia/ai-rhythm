import hashlib

import pytest

from chart_worker.validation.pairwise_review_packet import (
    build_blind_osu_payload_v1,
    normalized_playable_sections_sha256,
    playable_sections_sha256,
    replace_song_audio_filename_v1,
)

SOURCE = b"""osu file format v14\r
\r
[General]\r
AudioFilename: secret-song.flac\r
Mode: 3\r
SampleSet: All\r
\r
[Editor]\r
Bookmarks:1000,2000\r
\r
[Metadata]\r
Title:Secret title\r
TitleUnicode:Secret title\r
Artist:Secret artist\r
ArtistUnicode:Secret artist\r
Creator:Mapperatorinator\r
Version:EXPERT PRIMARY\r
Source:private\r
Tags:model=C:\\private\\model difficulty=4.5 descriptors=stream\r
\r
[Difficulty]\r
HPDrainRate:5\r
CircleSize:4\r
OverallDifficulty:8\r
ApproachRate:8\r
SliderMultiplier:1.4\r
SliderTickRate:1\r
\r
[Events]\r
0,0,"secret-background.jpg",0,0\r
\r
[TimingPoints]\r
0,500,4,2,1,60,1,0\r
1000,-100,4,2,1,60,0,0\r
\r
[HitObjects]\r
64,192,1000,1,0,0:0:0:0:\r
192,192,1500,128,0,2500:0:0:0:0:\r
"""


def test_blind_osu_scrubs_identity_but_preserves_playable_sections_exactly():
    before = playable_sections_sha256(SOURCE)

    blinded = build_blind_osu_payload_v1(
        SOURCE,
        audio_filename="game.ogg",
        blind_title="Blind set 0123",
        blind_version="T01-A",
    )

    assert playable_sections_sha256(blinded) == before
    assert hashlib.sha256(blinded).hexdigest() != hashlib.sha256(SOURCE).hexdigest()
    decoded = blinded.decode("utf-8")
    assert "AudioFilename: game.ogg" in decoded
    assert "Title:Blind set 0123" in decoded
    assert "Version:T01-A" in decoded
    for secret in (
        "Secret title",
        "Secret artist",
        "Mapperatorinator",
        "EXPERT",
        "PRIMARY",
        "private",
        "difficulty=",
        "descriptors=",
        "secret-background.jpg",
        "Bookmarks:",
    ):
        assert secret not in decoded
    assert "[TimingPoints]\r\n0,500" in decoded
    assert "[HitObjects]\r\n64,192,1000" in decoded


def test_normalized_playable_hash_ignores_only_line_ending_serialization():
    lf_source = SOURCE.replace(b"\r\n", b"\n")

    assert playable_sections_sha256(lf_source) != playable_sections_sha256(SOURCE)
    assert normalized_playable_sections_sha256(lf_source) == (
        normalized_playable_sections_sha256(SOURCE)
    )


@pytest.mark.parametrize(
    "bad_source",
    [
        b"not an osu file",
        SOURCE.replace(b"[TimingPoints]", b"[Other]"),
        SOURCE.replace(b"[HitObjects]", b"[Other]"),
        SOURCE.replace(b"Mode: 3", b"Mode: 0"),
        SOURCE + b"\r\n[Metadata]\r\nTitle:duplicate\r\n",
    ],
)
def test_blind_osu_fails_closed_on_malformed_or_non_mania_source(bad_source: bytes):
    with pytest.raises((TypeError, ValueError)):
        build_blind_osu_payload_v1(
            bad_source,
            audio_filename="game.flac",
            blind_title="Blind set",
            blind_version="T01-A",
        )


@pytest.mark.parametrize(
    ("audio_filename", "blind_title", "blind_version"),
    [
        ("../game.flac", "Blind", "A"),
        ("game.flac", "", "A"),
        ("game.flac", "Blind", "A\nVersion:EXPERT"),
    ],
)
def test_blind_metadata_rejects_path_traversal_empty_or_line_injection(
    audio_filename: str,
    blind_title: str,
    blind_version: str,
):
    with pytest.raises((TypeError, ValueError)):
        build_blind_osu_payload_v1(
            SOURCE,
            audio_filename=audio_filename,
            blind_title=blind_title,
            blind_version=blind_version,
        )


@pytest.mark.parametrize("audio_filename", ["game.flac", "game.wav", "game.m4a"])
def test_blind_osu_rejects_song_audio_unsupported_by_osu_stable(
    audio_filename: str,
):
    with pytest.raises(ValueError, match="osu! stable song audio"):
        build_blind_osu_payload_v1(
            SOURCE,
            audio_filename=audio_filename,
            blind_title="Blind set",
            blind_version="T01-A",
        )


def test_replace_song_audio_filename_preserves_every_playable_byte():
    blinded = build_blind_osu_payload_v1(
        SOURCE,
        audio_filename="old.mp3",
        blind_title="Blind set",
        blind_version="T01-A",
    )

    migrated = replace_song_audio_filename_v1(blinded, audio_filename="game.ogg")

    assert playable_sections_sha256(migrated) == playable_sections_sha256(blinded)
    assert b"AudioFilename: game.ogg\r\n" in migrated
    assert b"AudioFilename: old.mp3" not in migrated


def test_replace_song_audio_filename_requires_exactly_one_general_field():
    duplicate = SOURCE.replace(
        b"AudioFilename: secret-song.flac\r\n",
        b"AudioFilename: first.mp3\r\nAudioFilename: second.mp3\r\n",
    )

    with pytest.raises(ValueError, match="AudioFilename exactly once"):
        replace_song_audio_filename_v1(duplicate, audio_filename="game.ogg")
