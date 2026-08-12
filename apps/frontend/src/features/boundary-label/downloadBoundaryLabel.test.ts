import { afterEach, describe, expect, it, vi } from "vitest";

import type { BoundaryLabelV1, BoundaryLabelV2 } from "../../game/core/types";
import { downloadBoundaryLabel, downloadBoundaryLabelV2 } from "./downloadBoundaryLabel";

function label(): BoundaryLabelV1 {
  return {
    version: 1,
    labelId: "10000000-0000-4000-8000-000000000001",
    createdAt: "2026-08-10T00:00:00.000Z",
    reviewerId: "reviewer-a",
    run: {
      runId: "20000000-0000-4000-8000-000000000001",
      title: "Fixture",
      songVersionId: "30000000-0000-4000-8000-000000000001",
      gameAudioAssetId: "40000000-0000-4000-8000-000000000001",
    },
    audio: { sha256: "a".repeat(64), durationMs: 10_000 },
    generationReport: { path: "generation-report.json", sha256: "b".repeat(64) },
    group: { groupId: "group-a", relation: "UNKNOWN", confirmed: false },
    automaticEvidence: {
      availability: "UNAVAILABLE",
      unavailableReason: "missing",
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
    annotation: {
      lastMeaningfulAttack: { earliestMs: 8800, latestMs: 9000 },
      lastAcceptableRelease: { earliestMs: 9000, latestMs: 9500 },
      provisionalBoundaryVerdict: "NOT_AVAILABLE",
      tailCharacters: ["MIXED_OR_UNCERTAIN"],
      confidence: "LOW",
      comment: "",
    },
  };
}

function labelV2(): BoundaryLabelV2 {
  const v1 = label();
  return {
    ...v1,
    version: 2,
    annotation: {
      lastPlayableAttack: { earliestMs: 8400, latestMs: 8500 },
      primaryContentEnd: { earliestMs: 8800, latestMs: 9000 },
      acceptableReleaseEnd: { earliestMs: 9200, latestMs: 9500 },
      provisionalBoundaryVerdict: "NOT_AVAILABLE",
      tailCharacters: ["MIXED_OR_UNCERTAIN"],
      confidence: "LOW",
      comment: "",
    },
  };
}

afterEach(() => vi.restoreAllMocks());

describe("downloadBoundaryLabel", () => {
  it("downloads the JSON and always cleans up the temporary URL and anchor", () => {
    const createObjectURL = vi.fn(() => "blob:boundary-label");
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectURL });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    downloadBoundaryLabel(label());

    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    const anchor = click.mock.instances[0] as HTMLAnchorElement;
    expect(anchor.download).toBe(
      "20000000-0000-4000-8000-000000000001-boundary-label-v1.json",
    );
    expect(anchor.isConnected).toBe(false);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:boundary-label");
  });

  it("downloads a v2 label with a distinct filename", () => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:boundary-label-v2"),
    });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    downloadBoundaryLabelV2(labelV2());

    const anchor = click.mock.instances[0] as HTMLAnchorElement;
    expect(anchor.download).toBe(
      "20000000-0000-4000-8000-000000000001-boundary-label-v2.json",
    );
  });

  it("revokes the URL and removes the anchor when click throws", () => {
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:boundary-label"),
    });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectURL });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {
      throw new Error("download blocked");
    });

    expect(() => downloadBoundaryLabel(label())).toThrow(/download blocked/i);
    expect(document.querySelector('a[download$="boundary-label-v1.json"]')).toBeNull();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:boundary-label");
  });
});
