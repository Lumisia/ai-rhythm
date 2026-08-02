"""Regenerate the small worker output consumed by the frontend contract test."""

import json
import tempfile
import wave
from pathlib import Path
from shutil import copyfile

import numpy as np

from chart_worker.pipeline import PipelineOptions, run_pipeline
from tests.support import contract_fixture_dependencies

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = (
    REPOSITORY_ROOT / "apps" / "frontend" / "src" / "test" / "fixtures" / "playtest-run"
)


def _write_tone_wav(path: Path) -> None:
    sample_rate_hz = 48_000
    frames = np.arange(sample_rate_hz * 8, dtype=np.float64)
    tone = (0.08 * np.sin(2.0 * np.pi * 220.0 * frames / sample_rate_hz) * 32_767).astype(
        "<i2"
    )
    stereo = np.column_stack((tone, tone))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(stereo.tobytes())


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ai-rhythm-contract-") as temporary:
        temporary_root = Path(temporary)
        source = temporary_root / "contract.wav"
        output_dir = temporary_root / "run"
        _write_tone_wav(source)
        result = run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="contract fixture",
                seed=7,
                worker_version="fixture",
            ),
            dependencies=contract_fixture_dependencies(),
        )

        files = [
            result.manifest_path,
            output_dir / "audio" / "game.flac",
            *result.chart_paths,
        ]
        for source_path in files:
            relative_path = source_path.relative_to(output_dir)
            target_path = FIXTURE_ROOT / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source_path, target_path)

        report = json.loads((output_dir / "generation-report.json").read_text(encoding="utf-8"))
        report["elapsedMsByStage"] = dict.fromkeys(report["elapsedMsByStage"], 0)
        report_path = FIXTURE_ROOT / "generation-report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
