"""Migrate a frozen pairwise review packet to osu! stable-compatible OGG audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from chart_worker.validation.pairwise_review_packet import (
    playable_sections_sha256,
    replace_song_audio_filename_v1,
)

_FFMPEG_AUDIO_ARGS = (
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-map",
    "0:a:0",
    "-vn",
    "-map_metadata",
    "-1",
    "-c:a",
    "libvorbis",
    "-q:a",
    "6",
    "-threads",
    "1",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> str:
    data = _canonical_json(payload)
    path.write_bytes(data)
    return _sha256_bytes(data)


@contextmanager
def _staging_directory(output: Path) -> Iterator[Path]:
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise ValueError(f"staging output already exists: {staging}")
    if staging.parent != output.parent or staging.name != f".{output.name}.building":
        raise ValueError("unsafe staging directory")
    staging.mkdir()
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging)
        raise
    else:
        staging.replace(output)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _ffmpeg_version(ffmpeg: str) -> str:
    first_line = _run([ffmpeg, "-version"]).stdout.splitlines()[0]
    if not first_line.startswith("ffmpeg version "):
        raise ValueError("unexpected ffmpeg version output")
    return first_line


def _probe_audio(ffprobe: str, path: Path) -> dict[str, object]:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,start_time,duration,duration_ts,time_base",
            "-of",
            "json",
            "--",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams")
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        raise ValueError(f"{path} must contain exactly one selected audio stream")
    stream = streams[0]
    required = ("codec_name", "sample_rate", "channels", "start_time", "duration_ts")
    if any(name not in stream for name in required):
        raise ValueError(f"{path} audio probe is missing a required field")
    sample_rate = int(stream["sample_rate"])
    duration_samples = int(stream["duration_ts"])
    if sample_rate <= 0 or duration_samples <= 0:
        raise ValueError(f"{path} has invalid audio duration")
    return {
        "channels": int(stream["channels"]),
        "codec": str(stream["codec_name"]),
        "durationMs": round(duration_samples * 1000 / sample_rate, 6),
        "durationSamples": duration_samples,
        "sampleRate": sample_rate,
        "startTimeMs": round(float(stream["start_time"]) * 1000, 6),
    }


def _safe_review_path(review_root: Path, relative: object, *, suffix: str) -> Path:
    if type(relative) is not str or not relative or "\\" in relative:
        raise ValueError("review path must be a non-empty POSIX relative path")
    candidate = (review_root / relative).resolve()
    if not candidate.is_relative_to(review_root.resolve()) or candidate.suffix.lower() != suffix:
        raise ValueError(f"unsafe review path: {relative!r}")
    return candidate


def _verify_source(source: Path) -> tuple[dict[str, object], dict[str, object]]:
    terminal_path = source / "packet-terminal-v1.json"
    review_manifest_path = source / "review" / "review-packet-v1.json"
    private_bundle_path = source / "private" / "contract" / "private-bundle.json"
    index_path = source / "review" / "index.html"
    terminal = _read_json(terminal_path)
    manifest = _read_json(review_manifest_path)
    checks = (
        (review_manifest_path, terminal.get("reviewManifestSha256")),
        (private_bundle_path, terminal.get("privateBundleSha256")),
        (index_path, terminal.get("reviewHtmlSha256")),
    )
    for path, expected in checks:
        if type(expected) is not str or _sha256_file(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {path}")
    if manifest.get("packetSha256") != terminal.get("packetSha256"):
        raise ValueError("source review manifest packet identity mismatch")
    tasks = manifest.get("tasks")
    if type(tasks) is not list or len(tasks) != terminal.get("taskCount"):
        raise ValueError("source review task count mismatch")
    return terminal, manifest


def _private_audio_by_task(source: Path) -> dict[str, str]:
    bundle = _read_json(source / "private" / "contract" / "private-bundle.json")
    tasks = bundle.get("tasks")
    if type(tasks) is not list:
        raise ValueError("private bundle tasks are missing")
    result: dict[str, str] = {}
    for task in tasks:
        if type(task) is not dict:
            raise ValueError("private task must be an object")
        task_id = task.get("taskId")
        left = task.get("left")
        right = task.get("right")
        if type(task_id) is not str or type(left) is not dict or type(right) is not dict:
            raise ValueError("private task identity is malformed")
        left_audio = left.get("audioSha256")
        right_audio = right.get("audioSha256")
        if type(left_audio) is not str or left_audio != right_audio:
            raise ValueError(f"private task audio identity mismatch: {task_id}")
        result[task_id] = left_audio
    return result


def _task_audio_by_folder(
    manifest: dict[str, object], private_audio: dict[str, str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    tasks = manifest["tasks"]
    assert type(tasks) is list
    for task in tasks:
        if type(task) is not dict:
            raise ValueError("review task must be an object")
        folder = task.get("songFolder")
        task_id = task.get("taskId")
        if type(folder) is not str or not re.fullmatch(r"S[0-9a-f]{12}", folder):
            raise ValueError("review song folder identity is malformed")
        if type(task_id) is not str or task_id not in private_audio:
            raise ValueError("review/private task identity mismatch")
        previous = result.setdefault(folder, private_audio[task_id])
        if previous != private_audio[task_id]:
            raise ValueError(f"one blind folder maps to multiple canonical audio files: {folder}")
    return result


def _transcode_audio(ffmpeg: str, source: Path, target: Path) -> None:
    _run([ffmpeg, *_FFMPEG_AUDIO_ARGS[:4], "-i", str(source), *_FFMPEG_AUDIO_ARGS[4:], str(target)])
    _run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-i",
            str(target),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )


def _rewrite_review_charts(
    source_review: Path,
    target_review: Path,
    manifest: dict[str, object],
) -> None:
    tasks = manifest["tasks"]
    assert type(tasks) is list
    for task in tasks:
        assert type(task) is dict
        for side in ("A", "B"):
            entry = task.get(side)
            if type(entry) is not dict:
                raise ValueError(f"review task {side} entry is malformed")
            source_chart = _safe_review_path(source_review, entry.get("path"), suffix=".osu")
            source_bytes = source_chart.read_bytes()
            if _sha256_bytes(source_bytes) != entry.get("sha256"):
                raise ValueError(f"frozen chart hash mismatch: {source_chart}")
            if playable_sections_sha256(source_bytes) != entry.get("playableSectionsSha256"):
                raise ValueError(f"frozen playable-section hash mismatch: {source_chart}")
            target_chart = _safe_review_path(target_review, entry["path"], suffix=".osu")
            target_chart.parent.mkdir(parents=True, exist_ok=True)
            target_bytes = replace_song_audio_filename_v1(
                source_bytes, audio_filename="game.ogg"
            )
            if playable_sections_sha256(target_bytes) != entry["playableSectionsSha256"]:
                raise ValueError(f"migration changed playable sections: {source_chart}")
            target_chart.write_bytes(target_bytes)
            entry["sha256"] = _sha256_bytes(target_bytes)


def _rewrite_review_html(source_html: Path, target_html: Path, manifest: object) -> str:
    text = source_html.read_text(encoding="utf-8")
    replacement = "const packet=" + _canonical_json(manifest).decode("utf-8") + "; const root="
    rewritten, count = re.subn(
        r"const packet=.*?; const root=",
        lambda _match: replacement,
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("review HTML does not contain one embedded packet")
    target_html.write_text(rewritten, encoding="utf-8", newline="")
    return _sha256_file(target_html)


def migrate(source: Path, output: Path, *, ffmpeg: str, ffprobe: str) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"source packet does not exist: {source}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_terminal, source_manifest = _verify_source(source)
    private_audio = _private_audio_by_task(source)
    audio_by_folder = _task_audio_by_folder(source_manifest, private_audio)
    ffmpeg_version = _ffmpeg_version(ffmpeg)

    with _staging_directory(output) as staging:
        target_review = staging / "review"
        target_songs = target_review / "songs"
        target_songs.mkdir(parents=True)
        shutil.copytree(source / "private", staging / "private")

        migrated_manifest = json.loads(json.dumps(source_manifest))
        migrated_manifest["version"] = "pairwise-blind-review-packet-v2-osu-stable"
        migrated_manifest["sourcePacketSha256"] = source_terminal["packetSha256"]
        _rewrite_review_charts(source / "review", target_review, migrated_manifest)

        audio_entries: list[dict[str, object]] = []
        for folder, expected_source_sha in sorted(audio_by_folder.items()):
            source_audio = source / "review" / "songs" / folder / "game.flac"
            if _sha256_file(source_audio) != expected_source_sha:
                raise ValueError(f"canonical audio hash mismatch: {source_audio}")
            target_audio = target_songs / folder / "game.ogg"
            target_audio.parent.mkdir(parents=True, exist_ok=True)
            source_probe = _probe_audio(ffprobe, source_audio)
            _transcode_audio(ffmpeg, source_audio, target_audio)
            target_probe = _probe_audio(ffprobe, target_audio)
            if target_probe["codec"] != "vorbis":
                raise ValueError(f"unexpected review audio codec: {target_audio}")
            for key in ("sampleRate", "channels"):
                if source_probe[key] != target_probe[key]:
                    raise ValueError(f"review audio {key} changed: {folder}")
            delta_samples = int(target_probe["durationSamples"]) - int(
                source_probe["durationSamples"]
            )
            sample_rate = int(source_probe["sampleRate"])
            start_delta_ms = float(target_probe["startTimeMs"]) - float(
                source_probe["startTimeMs"]
            )
            if abs(delta_samples) > 1 or abs(start_delta_ms) > 1000 / sample_rate:
                raise ValueError(
                    f"review audio timeline changed: {folder}, "
                    f"duration delta {delta_samples} samples, start delta {start_delta_ms} ms"
                )
            audio_entries.append(
                {
                    "durationDeltaMs": round(delta_samples * 1000 / sample_rate, 6),
                    "durationDeltaSamples": delta_samples,
                    "review": {
                        "path": f"review/songs/{folder}/game.ogg",
                        "probe": target_probe,
                        "sha256": _sha256_file(target_audio),
                    },
                    "songFolder": folder,
                    "source": {
                        "path": f"review/songs/{folder}/game.flac",
                        "probe": source_probe,
                        "sha256": expected_source_sha,
                    },
                    "startDeltaMs": round(start_delta_ms, 6),
                }
            )

        audio_manifest = {
            "audioCount": len(audio_entries),
            "encoder": {
                "arguments": list(_FFMPEG_AUDIO_ARGS),
                "version": ffmpeg_version,
            },
            "sourcePacketSha256": source_terminal["packetSha256"],
            "songs": audio_entries,
            "version": "pairwise-osu-stable-audio-v1",
        }
        audio_manifest_sha = _write_json(
            staging / "osu-stable-audio-v1.json", audio_manifest
        )
        review_manifest_sha = _write_json(
            target_review / "review-packet-v2.json", migrated_manifest
        )
        review_html_sha = _rewrite_review_html(
            source / "review" / "index.html",
            target_review / "index.html",
            migrated_manifest,
        )
        readme = (
            "# 난이도 A/B 블라인드 검토 패킷 (osu! stable 호환본)\n\n"
            "1. `review/songs`의 S... 폴더들을 osu!의 Songs 폴더에 복사합니다.\n"
            "2. 같은 이름의 구형 폴더가 있으면 파일 덮어쓰기를 허용합니다. 채보는 `game.ogg`를 사용합니다.\n"
            "3. osu!에서 F5로 beatmap을 재검색하거나 클라이언트를 다시 시작합니다.\n"
            "4. T...-A와 T...-B를 플레이한 뒤 `review/index.html`에서 응답을 저장합니다.\n"
            "5. `private` 폴더는 라벨 결합용이므로 검토를 마치기 전 열지 않습니다.\n\n"
            "원본 v1의 A/B 작업·노트·HOLD·타이밍은 유지하고, 공식 지원 형식인 OGG로만 이식했습니다.\n"
        )
        (staging / "README.md").write_text(readme, encoding="utf-8", newline="\n")

        identity = {
            "audioManifestSha256": audio_manifest_sha,
            "privateBundleSha256": source_terminal["privateBundleSha256"],
            "reviewHtmlSha256": review_html_sha,
            "reviewManifestSha256": review_manifest_sha,
            "sourcePacketSha256": source_terminal["packetSha256"],
        }
        terminal = {
            "audioCount": len(audio_entries),
            "candidateCount": source_terminal["candidateCount"],
            **identity,
            "packetSha256": _sha256_bytes(_canonical_json(identity)),
            "taskCount": source_terminal["taskCount"],
            "version": "pairwise-pilot-packet-terminal-v2-osu-stable",
        }
        _write_json(staging / "packet-terminal-v2.json", terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    terminal = migrate(
        args.source,
        args.output,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    print(json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
