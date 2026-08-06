import userEvent from "@testing-library/user-event";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { bindingsFor } from "../../game/input/KeyBindings";
import { createReviewMarker, markerKindForCode } from "./MarkerControls";
import { ReviewResult, type ReviewSession } from "./ReviewResult";

function sessionWithMisses(): ReviewSession {
  return {
    runId: "40000000-0000-4000-8000-000000000001",
    title: "Koe no Yukue",
    chartId: "30000000-0000-4000-8000-000000000001",
    chartSha256: "a".repeat(64),
    audioSha256: "b".repeat(64),
    keyMode: 4,
    difficulty: "NORMAL",
    durationMs: 8000,
    result: {
      score: {
        counts: { PERFECT: 1, GREAT: 0, GOOD: 0, BAD: 0, MISS: 2 },
        combo: 0,
        maxCombo: 1,
        rawCombo: 0,
        rawMaxCombo: 1,
        totalJudgments: 3,
        accuracy: 1 / 3,
        meanErrMs: -10,
        meanAbsoluteErrMs: 10,
        timingSampleCount: 1,
        lanes: {
          0: { judgments: 2, misses: 1 },
          2: { judgments: 1, misses: 1 },
        },
      },
      judgments: [
        { noteId: 1, lane: 0, noteType: "TAP", phase: "HEAD", judgment: "PERFECT", errMs: -10, timeMs: 990, noteTimeMs: 1000 },
        { noteId: 2, lane: 0, noteType: "TAP", phase: "HEAD", judgment: "MISS", errMs: 201, timeMs: 4201, noteTimeMs: 4000 },
        { noteId: 3, lane: 2, noteType: "TAP", phase: "HEAD", judgment: "MISS", errMs: 201, timeMs: 4701, noteTimeMs: 4500 },
      ],
      inputs: [{ action: "DOWN", lane: 0, code: "KeyA", timeMs: 990 }],
      markers: [createReviewMarker("TIMING", 4200, 8000)],
      settings: {
        calibrationMs: -12,
        scrollSpeed: 1,
        judgmentPreset: "lenient",
        keysound: false,
        loopEnabled: false,
        loopStartMs: 0,
        loopEndMs: 8000,
      },
    },
  };
}

describe("marker controls", () => {
  it("maps Digit1 through Digit8 without colliding with play bindings", () => {
    expect(markerKindForCode("Digit1")).toBe("TIMING");
    expect(markerKindForCode("Digit8")).toBe("EXTRA");
    expect(markerKindForCode("KeyA")).toBeNull();
    for (const code of bindingsFor(7).keys()) expect(markerKindForCode(code)).toBeNull();
  });

  it("records a clamped two-second range around the marker", () => {
    expect(createReviewMarker("LANE", 1000, 8000)).toEqual({
      kind: "LANE",
      timeMs: 1000,
      rangeStartMs: 0,
      rangeEndMs: 3000,
    });
  });
});

describe("ReviewResult", () => {
  it("shows timing statistics and requires a subjective verdict before export", () => {
    render(<ReviewResult session={sessionWithMisses()} />);
    expect(screen.getByText(/평균 절대 오차/)).toBeVisible();
    expect(screen.getByRole("button", { name: "리뷰 JSON 저장" })).toBeDisabled();
    expect(screen.getByText("0–5초")).toBeVisible();
  });

  it("builds the review after both subjective fields are selected", async () => {
    const user = userEvent.setup();
    const download = vi.fn();
    render(<ReviewResult download={download} session={sessionWithMisses()} />);
    await user.selectOptions(screen.getByLabelText("체감 난이도"), "TOO_HARD");
    await user.selectOptions(screen.getByLabelText("종합 판정"), "NEEDS_CHANGES");
    await user.type(screen.getByLabelText("검수 메모"), "후렴 패턴 수정 필요");
    await user.click(screen.getByRole("button", { name: "리뷰 JSON 저장" }));
    expect(download).toHaveBeenCalledWith(
      expect.objectContaining({
        perceivedDifficulty: "TOO_HARD",
        verdict: "NEEDS_CHANGES",
        comment: "후렴 패턴 수정 필요",
      }),
    );
  });
});
