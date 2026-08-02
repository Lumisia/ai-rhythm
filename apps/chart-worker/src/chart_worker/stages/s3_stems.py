"""S3: optional 감산 keysound 자산과 드럼 onset."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from chart_worker.analysis.audio_io import load_audio
from chart_worker.analysis.onset import OnsetBackend, analyze_onsets, librosa_backend
from chart_worker.analysis.stems import StemBackend, demucs_backend, separate_stems
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.hashing import sha256_file
from chart_worker.schema.keysound import KeysoundManifest
from chart_worker.schema.playtest_run import AudioFileRef
from chart_worker.stages.types import AnalysisStageResult, StemStageResult


def _relative(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        raise ValueError(f"asset path must be inside the run directory: {path}") from None


def _write_flac(path: Path, samples, sample_rate_hz: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(path), samples, sample_rate_hz, format="FLAC", subtype="PCM_16")
    except Exception as error:
        path.unlink(missing_ok=True)
        raise WorkerError(
            ErrorCode.STEMS_SEPARATION_FAILED,
            f"could not write stem {path.name}: {error}",
        ) from error


def _require_same_timeline(analysis: AnalysisStageResult, path: Path) -> None:
    signal = load_audio(path)
    expected = (
        analysis.signal.sample_rate_hz,
        analysis.signal.channels,
        analysis.signal.frame_count,
    )
    actual = (signal.sample_rate_hz, signal.channels, signal.frame_count)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise WorkerError(
            ErrorCode.STEMS_SEPARATION_FAILED,
            f"stem timeline {actual} differs from game audio {expected}",
        )


def run_stems(
    analysis: AnalysisStageResult,
    run_dir: Path,
    *,
    enabled: bool,
    stem_backend: StemBackend | None = None,
    onset_backend: OnsetBackend | None = None,
) -> StemStageResult:
    game_ref = AudioFileRef(
        path=_relative(analysis.normalized.path, run_dir),
        sha256=analysis.normalized.sha256,
    )
    if not enabled:
        return StemStageResult(game_ref, None, None, (), None, None)

    stems = separate_stems(
        analysis.signal,
        backend=stem_backend or demucs_backend(),
    )
    no_drums_path = run_dir / "audio" / "no_drums.flac"
    keys_path = run_dir / "audio" / "drums.flac"
    _write_flac(no_drums_path, stems.no_drums, stems.sample_rate_hz)
    _write_flac(keys_path, stems.drums, stems.sample_rate_hz)
    _require_same_timeline(analysis, no_drums_path)
    _require_same_timeline(analysis, keys_path)

    keys_signal = load_audio(keys_path)
    drum_analysis = analyze_onsets(
        keys_signal,
        backend=onset_backend or librosa_backend(),
    )
    audio_sha = analysis.normalized.sha256
    manifest = KeysoundManifest(
        song_version_id=uuid5(NAMESPACE_URL, f"{audio_sha}:song-version"),
        bgm_asset_id=uuid5(NAMESPACE_URL, f"{audio_sha}:bgm"),
        keys_asset_id=uuid5(NAMESPACE_URL, f"{audio_sha}:keys"),
        drum_onsets=list(drum_analysis.onset_ms),
    )
    manifest_path = run_dir / "keysound-manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return StemStageResult(
        game_ref=game_ref,
        no_drums_ref=AudioFileRef(
            path=_relative(no_drums_path, run_dir),
            sha256=sha256_file(no_drums_path),
        ),
        keys_ref=AudioFileRef(
            path=_relative(keys_path, run_dir),
            sha256=sha256_file(keys_path),
        ),
        drum_onsets=tuple(drum_analysis.onset_ms),
        keysound_manifest=manifest,
        keysound_manifest_path=manifest_path,
    )
