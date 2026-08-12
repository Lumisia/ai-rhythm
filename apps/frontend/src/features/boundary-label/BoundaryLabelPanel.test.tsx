import userEvent from "@testing-library/user-event";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ImportedRun } from "../import-run/importRun";
import { BoundaryLabelPanel } from "./BoundaryLabelPanel";

function run(): ImportedRun {
  return {
    source: {
      has: () => false,
      readBytes: async () => new ArrayBuffer(0),
      readText: async () => "",
    },
    manifest: {
      version: 2,
      runId: "40000000-0000-4000-8000-000000000001",
      title: "Fixture Song",
      generatedAt: "2026-08-10T00:00:00Z",
      workerVersion: "fixture",
      audio: {
        game: { path: "audio/game.flac", sha256: "a".repeat(64) },
        noDrums: null,
        keys: null,
      },
      charts: [],
      missingCharts: [],
      keysoundManifestPath: null,
      generationReport: { path: "generation-report.json", sha256: "b".repeat(64) },
      outcome: {
        execution: "SUCCEEDED",
        completeness: "COMPLETE",
        quality: "PASS",
        failureCategory: "NONE",
        publishableStrict: false,
      },
      strictBlockers: ["BOUNDARY_POLICY_UNCALIBRATED"],
      publication: {
        policyVersion: "PUBLICATION_POLICY_V2",
        decision: "PLAYTEST_ONLY",
        reasonCodes: ["BOUNDARY_POLICY_UNCALIBRATED", "STRICT_OUTCOME_FALSE"],
      },
    },
    publicationState: "PLAYTEST_ONLY",
    publicationReasons: ["BOUNDARY_POLICY_UNCALIBRATED"],
    charts: [],
    keysoundManifest: null,
    audio: { game: new ArrayBuffer(16), noDrums: null, keys: null },
    boundaryLabelContext: {
      available: true,
      unavailableReason: null,
      songVersionId: "10000000-0000-4000-8000-000000000001",
      gameAudioAssetId: "20000000-0000-4000-8000-000000000001",
      audioDurationMs: 10_000,
      generationReport: { path: "generation-report.json", sha256: "b".repeat(64) },
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

beforeEach(() => {
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: vi.fn(() => "blob:canonical-game-audio"),
  });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

afterEach(() => vi.restoreAllMocks());

function setRequiredTimes(): void {
  fireEvent.change(screen.getByLabelText("마지막으로 칠 수 있는 타격 — 이른 시각 (ms)"), {
    target: { value: "8200" },
  });
  fireEvent.change(screen.getByLabelText("마지막으로 칠 수 있는 타격 — 늦은 시각 (ms)"), {
    target: { value: "8400" },
  });
  fireEvent.change(screen.getByLabelText("주요 음악·보컬 종료 — 이른 시각 (ms)"), {
    target: { value: "8800" },
  });
  fireEvent.change(screen.getByLabelText("주요 음악·보컬 종료 — 늦은 시각 (ms)"), {
    target: { value: "9000" },
  });
  fireEvent.change(screen.getByLabelText("허용 가능한 잔향·배경음 종료 — 이른 시각 (ms)"), {
    target: { value: "9200" },
  });
  fireEvent.change(screen.getByLabelText("허용 가능한 잔향·배경음 종료 — 늦은 시각 (ms)"), {
    target: { value: "9500" },
  });
}

describe("BoundaryLabelPanel", () => {
  it("uses canonical audio time for a selected human interval endpoint", async () => {
    const user = userEvent.setup();
    render(<BoundaryLabelPanel run={run()} onBack={vi.fn()} />);
    const audio = screen.getByLabelText("곡 끝 검토용 원본 게임 오디오") as HTMLAudioElement;
    audio.currentTime = 8.8;
    fireEvent.timeUpdate(audio);

    const field = screen
      .getByLabelText("주요 음악·보컬 종료 — 이른 시각 (ms)")
      .closest(".boundary-time-field");
    expect(field).not.toBeNull();
    await user.click(within(field as HTMLElement).getByRole("button", { name: "현재 위치 사용" }));

    expect(screen.getByLabelText("주요 음악·보컬 종료 — 이른 시각 (ms)")).toHaveValue(8800);
    expect(screen.getByText("8.800초 / 8800ms")).toBeVisible();
  });

  it("starts a long recording at the last 30 seconds and loops from the real end", async () => {
    const user = userEvent.setup();
    const longRun = run();
    longRun.boundaryLabelContext.audioDurationMs = 40_000;
    render(<BoundaryLabelPanel run={longRun} onBack={vi.fn()} />);
    const audio = screen.getByLabelText("곡 끝 검토용 원본 게임 오디오") as HTMLAudioElement;

    fireEvent.loadedMetadata(audio);
    expect(audio.currentTime).toBe(10);
    expect(screen.getByText("10.000초 / 10000ms")).toBeVisible();

    await user.click(screen.getByRole("checkbox", { name: "끝 구간 반복" }));
    audio.currentTime = 40;
    fireEvent.ended(audio);

    expect(audio.currentTime).toBe(10);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
  });

  it("keeps NOT_AVAILABLE fixed when automatic evidence is unavailable", () => {
    render(<BoundaryLabelPanel run={run()} onBack={vi.fn()} />);

    expect(screen.getByRole("status", { name: "자동 경계 증거 상태" })).toHaveTextContent(
      "musicBounds must be an object",
    );
    expect(screen.getByLabelText("자동 경계와 비교한 판정")).toHaveValue("NOT_AVAILABLE");
    expect(screen.getByLabelText("자동 경계와 비교한 판정")).toBeDisabled();
    expect(screen.getByText(/사람 라벨로 자동 복사되지 않습니다/)).toBeVisible();
  });

  it("exports only after all required human fields are valid", async () => {
    const user = userEvent.setup();
    const download = vi.fn();
    render(<BoundaryLabelPanel run={run()} onBack={vi.fn()} download={download} />);
    const exportButton = screen.getByRole("button", { name: "경계 라벨 JSON 저장" });
    expect(exportButton).toBeDisabled();

    await user.type(screen.getByLabelText("검토자 ID"), "reviewer-a");
    setRequiredTimes();
    await user.click(screen.getByRole("checkbox", { name: "페이드 또는 잔향" }));

    expect(exportButton).toBeEnabled();
    await user.click(exportButton);
    expect(download).toHaveBeenCalledOnce();
    expect(download.mock.calls[0][0]).toMatchObject({
      version: 2,
      reviewerId: "reviewer-a",
      group: {
        groupId: "10000000-0000-4000-8000-000000000001",
        relation: "EXACT_RECORDING",
      },
      annotation: {
        lastPlayableAttack: { earliestMs: 8200, latestMs: 8400 },
        primaryContentEnd: { earliestMs: 8800, latestMs: 9000 },
        acceptableReleaseEnd: { earliestMs: 9200, latestMs: 9500 },
        provisionalBoundaryVerdict: "NOT_AVAILABLE",
      },
    });
    expect(screen.getByRole("status", { name: "저장 결과" })).toHaveTextContent(
      "파일을 다운로드했습니다",
    );
  });

  it("does not claim success when the browser blocks the download", async () => {
    const user = userEvent.setup();
    const download = vi.fn(() => {
      throw new Error("download blocked");
    });
    render(<BoundaryLabelPanel run={run()} onBack={vi.fn()} download={download} />);
    await user.type(screen.getByLabelText("검토자 ID"), "reviewer-a");
    setRequiredTimes();
    await user.click(screen.getByRole("checkbox", { name: "페이드 또는 잔향" }));
    await user.click(screen.getByRole("button", { name: "경계 라벨 JSON 저장" }));

    expect(screen.getByRole("alert")).toHaveTextContent("download blocked");
    expect(screen.queryByText("파일을 다운로드했습니다")).not.toBeInTheDocument();
  });

  it("releases the canonical audio object URL on exit", () => {
    const { unmount } = render(<BoundaryLabelPanel run={run()} onBack={vi.fn()} />);
    unmount();

    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:canonical-game-audio");
  });
});
