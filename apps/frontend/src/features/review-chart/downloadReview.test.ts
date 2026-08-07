import { describe, expect, it, vi } from "vitest";

import type { PlaytestReview } from "./review";
import { downloadReview, reviewFileName } from "./downloadReview";

function review(): PlaytestReview {
  return {
    version: 1,
    runId: "40000000-0000-4000-8000-000000000001",
    chartId: "30000000-0000-4000-8000-000000000001",
    chartSha256: "a".repeat(64),
    audioSha256: "b".repeat(64),
    keyMode: 7,
    difficulty: "HARD",
    calibrationMs: 0,
    judgmentPreset: "lenient",
    perceivedDifficulty: "APPROPRIATE",
    verdict: "PASS",
    feverEnabled: false,
    maxCombo: 0,
    rawMaxCombo: 0,
    events: [],
    judgments: [],
    markers: [],
    comment: "",
  };
}

describe("downloadReview", () => {
  it("uses a stable file name and revokes its temporary object URL", () => {
    const createObjectURL = vi.fn(() => "blob:review");
    const revokeObjectURL = vi.fn();
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectURL });
    try {
      downloadReview(review());
      expect(reviewFileName(review())).toBe(
        "40000000-0000-4000-8000-000000000001-7k-HARD-review-v1.json",
      );
      expect(createObjectURL).toHaveBeenCalledOnce();
      expect(click).toHaveBeenCalledOnce();
      expect(revokeObjectURL).toHaveBeenCalledWith("blob:review");
      expect(document.querySelector('a[download$="review-v1.json"]')).toBeNull();
    } finally {
      Object.defineProperty(URL, "createObjectURL", { configurable: true, value: originalCreate });
      Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: originalRevoke });
      click.mockRestore();
    }
  });
});
