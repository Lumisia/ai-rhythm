import userEvent from "@testing-library/user-event";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ChartDocument, Difficulty, KeyMode } from "../game/core/types";
import type { ImportedChart, ImportedRun } from "../features/import-run/importRun";
import { LocalPlaytestPage } from "./LocalPlaytestPage";

const difficulties: Difficulty[] = ["EASY", "NORMAL", "HARD", "EXPERT"];
const keyModes: KeyMode[] = [4, 6, 7];

function chart(index: number, keyMode: KeyMode, difficulty: Difficulty): ImportedChart {
  const document: ChartDocument = {
    schemaVersion: 1,
    chartId: `30000000-0000-4000-8000-${String(index + 1).padStart(12, "0")}`,
    songVersionId: "10000000-0000-4000-8000-000000000001",
    gameAudioAssetId: "20000000-0000-4000-8000-000000000001",
    audioSha256: "a".repeat(64),
    keyMode,
    difficulty,
    laneSemantics:
      keyMode === 4
        ? ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"]
        : keyMode === 6
          ? ["SIDE_LEFT", "MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4", "SIDE_RIGHT"]
          : ["SIDE_LEFT", "MAIN_1", "MAIN_2", "CENTER", "MAIN_3", "MAIN_4", "SIDE_RIGHT"],
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
  const path = `charts/${keyMode}k-${difficulty}.json`;
  return {
    ref: { path, sha256: String(index).padStart(64, "0"), keyMode, difficulty },
    document,
    bytes: new ArrayBuffer(1),
  };
}

function importedRun(boundaryAvailable = false): ImportedRun {
  const charts: ImportedChart[] = [];
  let index = 0;
  for (const keyMode of keyModes) {
    for (const difficulty of difficulties) {
      charts.push(chart(index, keyMode, difficulty));
      index += 1;
    }
  }
  return {
    source: {
      has: () => false,
      readBytes: async () => new ArrayBuffer(0),
      readText: async () => "",
    },
    manifest: {
      version: 1,
      runId: "40000000-0000-4000-8000-000000000001",
      title: "Koe no Yukue",
      generatedAt: "2026-08-02T00:00:00Z",
      workerVersion: "fixture",
      audio: {
        game: { path: "audio/game.flac", sha256: "a".repeat(64) },
        noDrums: null,
        keys: null,
      },
      charts: charts.map(({ ref }) => ref),
      keysoundManifestPath: null,
      generationReportPath: "generation-report.json",
    },
    charts,
    publicationState: "PRODUCTION_VERIFIED",
    publicationReasons: [],
    keysoundManifest: null,
    audio: { game: new ArrayBuffer(8), noDrums: null, keys: null },
    boundaryLabelContext: {
      available: boundaryAvailable,
      unavailableReason: boundaryAvailable
        ? null
        : "v1 run manifest does not bind generation-report SHA-256",
      songVersionId: "10000000-0000-4000-8000-000000000001",
      gameAudioAssetId: "20000000-0000-4000-8000-000000000001",
      audioDurationMs: 8000,
      generationReport: boundaryAvailable
        ? { path: "generation-report.json", sha256: "b".repeat(64) }
        : null,
      automaticEvidence: {
        availability: "UNAVAILABLE",
        unavailableReason: "musicBounds must be an object",
        evaluationVersion: null,
        policyState: null,
        policyConfidence: null,
        enforcementMode: null,
        observationSha256: null,
        lastDetectedOnsetMs: null,
        lastActiveRmsEndMs: null,
        lastEvidenceMs: null,
        provisionalMaxNoteStartMs: null,
        provisionalReleaseEndMs: null,
        effectiveMaxNoteStartMs: null,
        effectiveReleaseEndMs: null,
      },
    },
  };
}

describe("LocalPlaytestPage", () => {
  it("moves from a valid imported run to the twelve-chart selector", async () => {
    const user = userEvent.setup();
    const importer = vi.fn(async () => importedRun());
    render(<LocalPlaytestPage importer={importer} />);
    await user.upload(
      screen.getByLabelText("실행 폴더"),
      new File(["manifest"], "playtest-run-v1.json"),
    );
    expect(await screen.findAllByRole("button", { name: /플레이$/ })).toHaveLength(12);
    expect(screen.getByText("Koe no Yukue")).toBeVisible();
  });

  it("keeps importer errors on the import stage", async () => {
    const user = userEvent.setup();
    const importer = vi.fn(async () => {
      throw new Error("charts/4k-EASY.json SHA-256 mismatch");
    });
    render(<LocalPlaytestPage importer={importer} />);
    await user.upload(
      screen.getByLabelText("실행 폴더"),
      new File(["manifest"], "playtest-run-v1.json"),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("charts/4k-EASY.json");
    expect(screen.queryByRole("button", { name: /플레이$/ })).not.toBeInTheDocument();
  });

  it("opens settings for a chart and returns to selection without booting audio", async () => {
    const user = userEvent.setup();
    render(<LocalPlaytestPage importer={async () => importedRun()} />);
    await user.upload(
      screen.getByLabelText("실행 폴더"),
      new File(["manifest"], "playtest-run-v1.json"),
    );
    await user.click(await screen.findByRole("button", { name: "4K EASY 플레이" }));
    expect(screen.getByRole("heading", { name: "플레이 설정" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "채보 목록으로" }));
    expect(await screen.findAllByRole("button", { name: /플레이$/ })).toHaveLength(12);
  });

  it("opens the separate song-end review and returns to all twelve charts", async () => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:game-audio"),
    });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    const user = userEvent.setup();
    render(<LocalPlaytestPage importer={async () => importedRun(true)} />);
    await user.upload(
      screen.getByLabelText("실행 폴더"),
      new File(["manifest"], "playtest-run-v2.json"),
    );

    await user.click(await screen.findByRole("button", { name: "곡 끝 검토" }));
    expect(screen.getByRole("heading", { name: "곡 끝 경계 검토" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "채보 목록으로" }));
    expect(await screen.findAllByRole("button", { name: /플레이$/ })).toHaveLength(12);
  });
});
