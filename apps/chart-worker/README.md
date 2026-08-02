# chart-worker 로컬 품질 검증

`chart-worker bench`는 한 곡의 4K·6K·7K, 네 난이도 조합을 생성하고 출력
폴더에 `generation-report.json`, `benchmark-report.json`,
`playtest-run-v1.json`을 기록합니다. Mapperatorinator는 상시 서비스가 아니라 이
명령을 실행하는 동안에만 사용됩니다.

## 실행 전 경로 확인과 프로세스 환경 설정

PowerShell의 현재 프로세스에만 Mapperatorinator와 WinGet shared FFmpeg 경로를
설정합니다. 시스템 환경 변수는 바꾸지 않습니다.

```powershell
$mapperRoot = "C:\Users\PC\mapperatorinator"
$mapperPython = "C:\Users\PC\mapperatorinator\.venv\Scripts\python.exe"
$sharedFfmpegBin = "C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg.LGPL.Shared.7.1_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-n7.1.5-1-g7d0e842004-win64-lgpl-shared-7.1\bin"
$songPath = "C:\Users\PC\Desktop\Koe no Yukue (声の行く先) - Take 2.wav"
$outputPath = ".data\playtests\koe-no-yukue-quality-v4"

Test-Path -LiteralPath $mapperRoot -PathType Container
Test-Path -LiteralPath $mapperPython -PathType Leaf
Test-Path -LiteralPath "$sharedFfmpegBin\ffmpeg.exe" -PathType Leaf
Test-Path -LiteralPath "$sharedFfmpegBin\ffprobe.exe" -PathType Leaf
Test-Path -LiteralPath $sharedFfmpegBin -PathType Container
Test-Path -LiteralPath $songPath -PathType Leaf
Test-Path -LiteralPath $outputPath

if (-not (Test-Path -LiteralPath $mapperRoot -PathType Container)) { throw "Mapperatorinator home is missing" }
if (-not (Test-Path -LiteralPath $mapperPython -PathType Leaf)) { throw "Mapperatorinator Python is missing" }
if (-not (Test-Path -LiteralPath "$sharedFfmpegBin\ffmpeg.exe" -PathType Leaf)) { throw "shared FFmpeg is missing" }
if (-not (Test-Path -LiteralPath "$sharedFfmpegBin\ffprobe.exe" -PathType Leaf)) { throw "shared FFprobe is missing" }
if (-not (Test-Path -LiteralPath $songPath -PathType Leaf)) { throw "trial song is missing" }
if (Test-Path -LiteralPath $outputPath) { throw "choose a new output path so existing playtests are preserved" }

$env:MAPPERATORINATOR_HOME = $mapperRoot
$env:MAPPERATORINATOR_PYTHON = $mapperPython
$env:FFMPEG_BIN = "$sharedFfmpegBin\ffmpeg.exe"
$env:FFMPEG_SHARED_BIN_DIR = $sharedFfmpegBin
```

앞의 처음 여섯 `Test-Path` 결과는 `True`, 새 output 확인은 `False`여야 합니다.

## Koe no Yukue 시험곡

기존 결과를 보존하기 위해 비어 있는 새 `--out` 폴더를 사용합니다.

```powershell
uv run --project apps/chart-worker chart-worker bench "C:\Users\PC\Desktop\Koe no Yukue (声の行く先) - Take 2.wav" --out ".data\playtests\koe-no-yukue-quality-v4" --title "Koe no Yukue" --generator mapperatorinator --keysounds
```

사람이 표시한 onset 정답 파일을 함께 측정하려면
`--reference-onsets <reference-onsets-v1.json>`을 추가합니다. 정답 파일이 없는
실행의 `referenceAccuracy`는 숫자 `0`이 아니라 `UNAVAILABLE`입니다. Beat This나
librosa 측정값을 사람 정답 정확도로 대신 해석하면 안 됩니다.

모든 생성 후보가 품질 게이트를 통과하지 못하면 파이프라인은
`CHART_CANDIDATES_EXHAUSTED` 오류를 반환합니다. 이때 실패 원인을 담은
`generation-report.json`과 `status: FAIL`인 `benchmark-report.json`을 남긴 뒤 같은
오류를 반환합니다. 성공을 뜻하는 `playtest-run-v1.json`은 만들지 않습니다.
