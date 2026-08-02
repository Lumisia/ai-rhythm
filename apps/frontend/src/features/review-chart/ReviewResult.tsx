import { useMemo, useState } from "react";

import type { Difficulty, KeyMode } from "../../game/core/types";
import type { PlaySessionResult } from "../play-chart/PlayChartPanel";
import { downloadReview } from "./downloadReview";
import type {
  PerceivedDifficulty,
  PlaytestReview,
  ReviewVerdict,
} from "./review";

export interface ReviewSession {
  runId: string;
  title: string;
  chartId: string;
  chartSha256: string;
  audioSha256: string;
  keyMode: KeyMode;
  difficulty: Difficulty;
  durationMs: number;
  result: PlaySessionResult;
}

interface ReviewResultProps {
  session: ReviewSession;
  download?: (review: PlaytestReview) => void;
  onBack?: () => void;
  onSaved?: (review: PlaytestReview) => void;
}

function failingSections(session: ReviewSession): Array<{ startMs: number; count: number }> {
  const bins = new Map<number, number>();
  for (const event of session.result.judgments) {
    if (event.judgment !== "MISS") continue;
    const startMs = Math.floor(event.noteTimeMs / 5000) * 5000;
    bins.set(startMs, (bins.get(startMs) ?? 0) + 1);
  }
  return [...bins.entries()]
    .map(([startMs, count]) => ({ startMs, count }))
    .sort((left, right) => right.count - left.count || left.startMs - right.startMs)
    .slice(0, 5);
}

function secondsLabel(startMs: number): string {
  return `${Math.floor(startMs / 1000)}–${Math.floor((startMs + 5000) / 1000)}초`;
}

export function ReviewResult({
  session,
  download = downloadReview,
  onBack,
  onSaved,
}: ReviewResultProps) {
  const [perceivedDifficulty, setPerceivedDifficulty] = useState<PerceivedDifficulty | "">("");
  const [verdict, setVerdict] = useState<ReviewVerdict | "">("");
  const [comment, setComment] = useState("");
  const sections = useMemo(() => failingSections(session), [session]);
  const { score } = session.result;

  const save = () => {
    if (!perceivedDifficulty || !verdict) return;
    const review: PlaytestReview = {
      version: 1,
      runId: session.runId,
      chartId: session.chartId,
      chartSha256: session.chartSha256,
      audioSha256: session.audioSha256,
      keyMode: session.keyMode,
      difficulty: session.difficulty,
      calibrationMs: session.result.settings.calibrationMs,
      judgmentPreset: session.result.settings.judgmentPreset,
      perceivedDifficulty,
      verdict,
      events: session.result.inputs,
      judgments: session.result.judgments,
      markers: session.result.markers,
      comment,
    };
    download(review);
    onSaved?.(review);
  };

  return (
    <section className="workspace-panel review-panel" aria-labelledby="result-title">
      <header className="workspace-heading">
        <div>
          <p className="eyebrow">SESSION CAPTURED / {session.keyMode}K {session.difficulty}</p>
          <h2 id="result-title">플레이 결과</h2>
          <p>{session.title}</p>
        </div>
        {onBack ? <button className="text-button" onClick={onBack} type="button">채보 목록으로</button> : null}
      </header>

      <div className="result-strip">
        <span>MAX COMBO <strong>{score.maxCombo}</strong></span>
        <span>MISS <strong>{score.counts.MISS}</strong></span>
        <span>정확도 <strong>{(score.accuracy * 100).toFixed(1)}%</strong></span>
      </div>

      <div className="review-grid">
        <section className="review-block" aria-labelledby="timing-stat-title">
          <h3 id="timing-stat-title">판정과 타이밍</h3>
          <dl className="stat-list">
            {Object.entries(score.counts).map(([name, count]) => <div key={name}><dt>{name}</dt><dd>{count}</dd></div>)}
            <div><dt>평균 오차</dt><dd>{score.meanErrMs.toFixed(1)} ms</dd></div>
            <div><dt>평균 절대 오차</dt><dd>{score.meanAbsoluteErrMs.toFixed(1)} ms</dd></div>
          </dl>
        </section>

        <section className="review-block" aria-labelledby="lane-stat-title">
          <h3 id="lane-stat-title">레인별 MISS</h3>
          <dl className="stat-list">
            {Array.from({ length: session.keyMode }, (_, lane) => {
              const value = score.lanes[lane] ?? { judgments: 0, misses: 0 };
              const rate = value.judgments === 0 ? 0 : (value.misses / value.judgments) * 100;
              return <div key={lane}><dt>LANE {lane + 1}</dt><dd>{value.misses} / {value.judgments} · {rate.toFixed(1)}%</dd></div>;
            })}
          </dl>
        </section>

        <section className="review-block" aria-labelledby="failure-title">
          <h3 id="failure-title">MISS 집중 구간</h3>
          {sections.length ? (
            <ol className="failure-list">
              {sections.map((section) => <li key={section.startMs}><span>{secondsLabel(section.startMs)}</span><strong>{section.count} MISS</strong></li>)}
            </ol>
          ) : <p className="empty-state">MISS가 없습니다.</p>}
        </section>

        <section className="review-block" aria-labelledby="recorded-marker-title">
          <h3 id="recorded-marker-title">문제 마커</h3>
          {session.result.markers.length ? (
            <ol className="failure-list">
              {session.result.markers.map((marker, index) => <li key={`${marker.timeMs}-${index}`}><span>{marker.kind}</span><strong>{(marker.timeMs / 1000).toFixed(2)}s</strong></li>)}
            </ol>
          ) : <p className="empty-state">기록한 마커가 없습니다.</p>}
        </section>
      </div>

      <section className="subjective-review" aria-labelledby="subjective-title">
        <h3 id="subjective-title">주관 평가</h3>
        <label>체감 난이도
          <select aria-label="체감 난이도" onChange={(event) => setPerceivedDifficulty(event.currentTarget.value as PerceivedDifficulty | "")} value={perceivedDifficulty}>
            <option value="">선택</option><option value="TOO_EASY">너무 쉬움</option><option value="APPROPRIATE">적절함</option><option value="TOO_HARD">너무 어려움</option>
          </select>
        </label>
        <label>종합 판정
          <select aria-label="종합 판정" onChange={(event) => setVerdict(event.currentTarget.value as ReviewVerdict | "")} value={verdict}>
            <option value="">선택</option><option value="PASS">통과</option><option value="NEEDS_CHANGES">수정 필요</option>
          </select>
        </label>
        <label className="comment-field">검수 메모
          <textarea aria-label="검수 메모" onChange={(event) => setComment(event.currentTarget.value)} rows={4} value={comment} />
        </label>
        <button className="primary-action" disabled={!perceivedDifficulty || !verdict} onClick={save} type="button">리뷰 JSON 저장</button>
      </section>
    </section>
  );
}
