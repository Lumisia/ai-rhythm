import type { PlaytestReview } from "./review";
import { serializeReview } from "./review";

export function reviewFileName(review: PlaytestReview): string {
  return `${review.runId}-${review.keyMode}k-${review.difficulty}-review-v1.json`;
}

export function downloadReview(review: PlaytestReview): void {
  const blob = new Blob([serializeReview(review)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = reviewFileName(review);
  anchor.hidden = true;
  document.body.append(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(url);
  }
}
