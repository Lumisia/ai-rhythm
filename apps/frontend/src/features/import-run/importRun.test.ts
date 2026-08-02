import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import type {
  ChartDocument,
  Difficulty,
  KeyMode,
  KeysoundManifest,
  LaneSemantic,
  PlaytestRunManifest,
  RunChartRef,
} from "../../game/core/types";
import { LocalDirectoryAssetSource } from "../../shared/local-files/LocalDirectoryAssetSource";
import { importRun } from "./importRun";

const encoder = new TextEncoder();
const keyModes: KeyMode[] = [4, 6, 7];
const difficulties: Difficulty[] = ["EASY", "NORMAL", "HARD", "EXPERT"];
const songVersionId = "10000000-0000-4000-8000-000000000001";
const gameAudioAssetId = "20000000-0000-4000-8000-000000000001";

interface RunOptions {
  badChartHash?: boolean;
  chartPath?: string;
  duplicateCombination?: boolean;
  invalidChartSchema?: boolean;
  omitFirstChart?: boolean;
  withKeysounds?: boolean;
}

function digest(body: string | Uint8Array): string {
  return createHash("sha256").update(body).digest("hex");
}

function laneSemantics(keyMode: KeyMode): LaneSemantic[] {
  if (keyMode === 4) return ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"];
  if (keyMode === 6) {
    return ["SIDE_LEFT", "MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4", "SIDE_RIGHT"];
  }
  return ["SIDE_LEFT", "MAIN_1", "MAIN_2", "CENTER", "MAIN_3", "MAIN_4", "SIDE_RIGHT"];
}

function makeFile(relativePath: string, body: string | Uint8Array): File {
  let part: BlobPart;
  if (typeof body === "string") {
    part = body;
  } else {
    const buffer = new ArrayBuffer(body.byteLength);
    new Uint8Array(buffer).set(body);
    part = buffer;
  }
  const file = new File([part], relativePath.split("/").at(-1) ?? relativePath);
  Object.defineProperty(file, "webkitRelativePath", {
    configurable: true,
    value: `selected-run/${relativePath}`,
  });
  return file;
}

function makeChart(index: number, keyMode: KeyMode, difficulty: Difficulty, audioSha256: string) {
  const chart: ChartDocument = {
    schemaVersion: 1,
    chartId: `30000000-0000-4000-8000-${String(index + 1).padStart(12, "0")}`,
    songVersionId,
    gameAudioAssetId,
    audioSha256,
    keyMode,
    difficulty,
    laneSemantics: laneSemantics(keyMode),
    offsetMs: 0,
    durationMs: 8000,
    bpmEvents: [{ timeMs: 0, bpm: 120 }],
    bpmSource: "BEAT_THIS",
    notes: [{ id: 1, lane: 0, timeMs: 1000, type: "TAP", durationMs: null }],
    autoPlayOnsets: [],
    metrics: {
      noteCount: 1,
      holdCount: 0,
      avgNps: 0.125,
      p95Nps: 1,
      peakNps: 1,
      chordRatio: 0,
      maxJack: 1,
      projectRating: index + 1,
      projectTier: difficulty,
      patternEntropy: 0,
      drumCoverage: 0,
      drumPrecision: 0,
      meanAbsErrMs: 0,
      sideNoteRatio: 0,
      sideHoldRatio: 0,
      movedNoteRatio: 0,
    },
    generator: {
      name: "fixture",
      version: "1",
      analysisVersion: "1",
      postprocessVersion: "1",
      seed: 7,
    },
  };
  return chart;
}

function makeRunFiles(options: RunOptions = {}): File[] {
  const gameAudio = new Uint8Array([0x66, 0x4c, 0x61, 0x43, 1, 2, 3]);
  const gameSha = digest(gameAudio);
  const chartFiles: File[] = [];
  const chartRefs: RunChartRef[] = [];

  let index = 0;
  for (const keyMode of keyModes) {
    for (const difficulty of difficulties) {
      const chart = makeChart(index, keyMode, difficulty, gameSha);
      if (options.invalidChartSchema && index === 0) {
        Object.assign(chart, { keyMode: 5 });
      }
      const body = JSON.stringify(chart);
      const path = index === 0 && options.chartPath ? options.chartPath : `charts/${keyMode}k-${difficulty}.json`;
      chartRefs.push({ path, sha256: digest(body), keyMode, difficulty });
      if (!(options.omitFirstChart && index === 0)) {
        chartFiles.push(makeFile(`charts/${keyMode}k-${difficulty}.json`, options.badChartHash && index === 0 ? "{}" : body));
      }
      index += 1;
    }
  }

  if (options.duplicateCombination) {
    chartRefs[1] = { ...chartRefs[1], keyMode: chartRefs[0].keyMode, difficulty: chartRefs[0].difficulty };
  }

  const manifest: PlaytestRunManifest = {
    version: 1,
    runId: "40000000-0000-4000-8000-000000000001",
    title: "fixture song",
    generatedAt: "2026-08-02T00:00:00Z",
    workerVersion: "0.1.0",
    audio: {
      game: { path: "audio/game.flac", sha256: gameSha },
      noDrums: null,
      keys: null,
    },
    charts: chartRefs,
    keysoundManifestPath: null,
    generationReportPath: "generation-report.json",
  };

  const files = [makeFile("audio/game.flac", gameAudio), ...chartFiles];

  if (options.withKeysounds) {
    const noDrums = new Uint8Array([10, 11, 12]);
    const keys = new Uint8Array([20, 21, 22]);
    manifest.audio.noDrums = { path: "audio/no_drums.flac", sha256: digest(noDrums) };
    manifest.audio.keys = { path: "audio/drums.flac", sha256: digest(keys) };
    manifest.keysoundManifestPath = "keysound-manifest.json";
    const keysoundManifest: KeysoundManifest = {
      schemaVersion: 1,
      songVersionId,
      bgmAssetId: "50000000-0000-4000-8000-000000000001",
      keysAssetId: "60000000-0000-4000-8000-000000000001",
      sliceSec: 0.3,
      prerollSec: 0.012,
      snapWindowMs: 50,
      drumOnsets: [1000, 2000],
    };
    files.push(
      makeFile("audio/no_drums.flac", noDrums),
      makeFile("audio/drums.flac", keys),
      makeFile("keysound-manifest.json", JSON.stringify(keysoundManifest)),
    );
  }

  files.push(makeFile("playtest-run-v1.json", JSON.stringify(manifest)));
  return files;
}

describe("importRun", () => {
  it("imports a valid 12-chart run", async () => {
    const imported = await importRun(makeRunFiles());
    expect(imported.manifest.charts).toHaveLength(12);
    expect(imported.charts).toHaveLength(12);
    expect(Array.from(new Uint8Array(imported.audio.game).slice(0, 4))).toEqual(
      Array.from(encoder.encode("fLaC")),
    );
  });

  it("rejects a missing run manifest", async () => {
    const files = makeRunFiles().filter((file) => file.name !== "playtest-run-v1.json");
    await expect(importRun(files)).rejects.toThrow(/playtest-run-v1\.json/);
  });

  it("rejects a missing chart with its file name", async () => {
    await expect(importRun(makeRunFiles({ omitFirstChart: true }))).rejects.toThrow(
      /charts\/4k-EASY\.json/,
    );
  });

  it("rejects a chart that fails its JSON Schema", async () => {
    await expect(importRun(makeRunFiles({ invalidChartSchema: true }))).rejects.toThrow(
      /charts\/4k-EASY\.json.*schema/i,
    );
  });

  it("rejects a chart whose sha256 differs from the run manifest", async () => {
    await expect(importRun(makeRunFiles({ badChartHash: true }))).rejects.toThrow(/SHA-256.*4k-EASY/);
  });

  it("rejects a parent-directory path", async () => {
    await expect(importRun(makeRunFiles({ chartPath: "../chart.json" }))).rejects.toThrow(
      /relative path/,
    );
  });

  it("rejects a duplicate key-mode and difficulty combination", async () => {
    await expect(importRun(makeRunFiles({ duplicateCombination: true }))).rejects.toThrow(
      /duplicate.*4K.*EASY/i,
    );
  });

  it("imports and validates optional keysound assets", async () => {
    const imported = await importRun(makeRunFiles({ withKeysounds: true }));
    expect(imported.keysoundManifest?.drumOnsets).toEqual([1000, 2000]);
    expect(imported.audio.noDrums?.byteLength).toBe(3);
    expect(imported.audio.keys?.byteLength).toBe(3);
  });

  it("rejects case-insensitive duplicate paths", () => {
    expect(
      () =>
        new LocalDirectoryAssetSource([
          makeFile("charts/chart.json", "one"),
          makeFile("Charts/CHART.json", "two"),
        ]),
    ).toThrow(/duplicate path/i);
  });
});
