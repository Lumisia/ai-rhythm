import type { ReviewMarker, ReviewMarkerKind } from "./review";

export const markerKinds: readonly ReviewMarkerKind[] = [
  "TIMING",
  "DIFFICULTY",
  "LANE",
  "JACK",
  "CHORD",
  "HOLD",
  "MISSING",
  "EXTRA",
];

const labels: Record<ReviewMarkerKind, string> = {
  TIMING: "박자",
  DIFFICULTY: "난이도",
  LANE: "레인",
  JACK: "연타",
  CHORD: "동시치기",
  HOLD: "홀드",
  MISSING: "노트 누락",
  EXTRA: "불필요 노트",
};

export function markerKindForCode(code: string): ReviewMarkerKind | null {
  const match = /^Digit([1-8])$/.exec(code);
  return match ? markerKinds[Number(match[1]) - 1] : null;
}

export function markerKindForSlot(slot: number): ReviewMarkerKind | null {
  return markerKinds[slot - 1] ?? null;
}

/** 플레이 화면에 띄울 마커 이름. */
export function markerLabelForSlot(slot: number): string | null {
  const kind = markerKindForSlot(slot);
  return kind ? labels[kind] : null;
}

export function createReviewMarker(
  kind: ReviewMarkerKind,
  timeMs: number,
  durationMs: number,
): ReviewMarker {
  const clampedTimeMs = Math.min(durationMs, Math.max(0, Math.round(timeMs)));
  return {
    kind,
    timeMs: clampedTimeMs,
    rangeStartMs: Math.max(0, clampedTimeMs - 2000),
    rangeEndMs: Math.min(durationMs, clampedTimeMs + 2000),
  };
}

interface MarkerControlsProps {
  disabled?: boolean;
  markers: readonly ReviewMarker[];
  onAdd: (kind: ReviewMarkerKind) => void;
}

export function MarkerControls({ disabled, markers, onAdd }: MarkerControlsProps) {
  return (
    <section className="marker-controls" aria-labelledby="marker-title">
      <div className="marker-heading">
        <strong id="marker-title">문제 마커</strong>
        <span>{markers.length} REC</span>
      </div>
      <div className="marker-grid">
        {markerKinds.map((kind, index) => (
          <button disabled={disabled} key={kind} onClick={() => onAdd(kind)} type="button">
            <kbd>{index + 1}</kbd><span>{labels[kind]}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
