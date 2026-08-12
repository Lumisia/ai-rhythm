import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type {
  ImportPublicationReason,
  ImportedChart,
  ImportedRun,
  PublicationState,
} from "../import-run/importRun";
import { ChartSelector } from "./ChartSelector";

function importedRun(
  publicationState: PublicationState,
  publicationReasons: ImportPublicationReason[],
  chartCount: number,
  boundaryAvailable = false,
): ImportedRun {
  const charts = Array.from(
    { length: chartCount },
    () => ({ document: { keyMode: 0 } }) as unknown as ImportedChart,
  );
  return {
    source: {
      has: () => false,
      readBytes: async () => new ArrayBuffer(0),
      readText: async () => "",
    },
    manifest: {
      version: 1,
      runId: "40000000-0000-4000-8000-000000000001",
      title: "fixture",
      generatedAt: "2026-08-02T00:00:00Z",
      workerVersion: "fixture",
      audio: {
        game: { path: "audio/game.flac", sha256: "a".repeat(64) },
        noDrums: null,
        keys: null,
      },
      charts: [],
      keysoundManifestPath: null,
      generationReportPath: "generation-report.json",
    },
    charts,
    keysoundManifest: null,
    audio: { game: new ArrayBuffer(0), noDrums: null, keys: null },
    publicationState,
    publicationReasons,
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

describe("ChartSelector publication status", () => {
  it("shows a visible playtest-only warning and its literal reasons", () => {
    render(
      <ChartSelector
        run={importedRun(
          "PLAYTEST_ONLY",
          [
            "BOUNDARY_POLICY_UNCALIBRATED",
            "INCOMPLETE_CHART_SET",
            "QUALITY_REVIEW_REQUIRED",
            "STRICT_OUTCOME_FALSE",
          ],
          11,
        )}
        onReset={vi.fn()}
        onSelect={vi.fn()}
        onBoundaryReview={vi.fn()}
      />,
    );

    expect(screen.getByText("PLAYTEST ONLY / 11 CHARTS")).toBeVisible();
    expect(screen.getByRole("status", { name: /publication status/i })).toHaveTextContent(
      "INCOMPLETE_CHART_SET",
    );
    expect(screen.getByRole("status", { name: /publication status/i })).toHaveTextContent(
      "BOUNDARY_POLICY_UNCALIBRATED",
    );
    expect(screen.getByRole("status", { name: /publication status/i })).toHaveTextContent(
      "QUALITY_REVIEW_REQUIRED",
    );
  });

  it("distinguishes legacy-unverified input from a production-verified run", () => {
    const { rerender } = render(
      <ChartSelector
        run={importedRun("LEGACY_UNVERIFIED", ["LEGACY_MANIFEST_UNVERIFIED"], 12)}
        onReset={vi.fn()}
        onSelect={vi.fn()}
        onBoundaryReview={vi.fn()}
      />,
    );
    expect(screen.getByText("LEGACY UNVERIFIED / 12 CHARTS")).toBeVisible();

    rerender(
      <ChartSelector
        run={importedRun("PRODUCTION_VERIFIED", [], 12)}
        onReset={vi.fn()}
        onSelect={vi.fn()}
        onBoundaryReview={vi.fn()}
      />,
    );
    expect(screen.getByText("RUN VERIFIED / 12 CHARTS")).toBeVisible();
  });

  it("explains why legacy runs cannot create a bound song-end label", () => {
    render(
      <ChartSelector
        run={importedRun("LEGACY_UNVERIFIED", ["LEGACY_MANIFEST_UNVERIFIED"], 12)}
        onBoundaryReview={vi.fn()}
        onReset={vi.fn()}
        onSelect={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "곡 끝 검토" })).toBeDisabled();
    expect(screen.getByText(/generation-report SHA-256/)).toBeVisible();
  });

  it("opens song-end review from a bound v2 run", async () => {
    const onBoundaryReview = vi.fn();
    const user = (await import("@testing-library/user-event")).default.setup();
    render(
      <ChartSelector
        run={importedRun("PRODUCTION_VERIFIED", [], 12, true)}
        onBoundaryReview={onBoundaryReview}
        onReset={vi.fn()}
        onSelect={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "곡 끝 검토" }));
    expect(onBoundaryReview).toHaveBeenCalledOnce();
  });
});
