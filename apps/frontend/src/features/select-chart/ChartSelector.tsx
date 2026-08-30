import type { ImportedChart, ImportedRun } from "../import-run/importRun";

interface ChartSelectorProps {
  run: ImportedRun;
  lastReviews?: Readonly<Record<string, string>>;
  onSelect: (chart: ImportedChart) => void;
  onBoundaryReview: () => void;
  onReset: () => void;
}

const keyModes = [4, 6, 7] as const;

function chartTrust(chart: ImportedChart) {
  const provenance = chart.ref.provenance ?? "LEGACY_UNVERIFIED";
  const fallback = new Set([
    "COVERAGE_REPAIR",
    "RAW_UNVERIFIED",
    "SAFE_FALLBACK",
  ]).has(provenance);
  const tier =
    chart.ref.playabilityTier ??
    (fallback ? "DIAGNOSTIC_ONLY" : "MODEL_PLAYABLE");
  const label = {
    MODEL_PLAYABLE: "MODEL PLAYABLE",
    RECOVERY_PLAYABLE: "PLAYABLE RECOVERY",
    DIAGNOSTIC_ONLY: "DIAGNOSTIC ONLY",
  }[tier];
  return { label, provenance, summary: chart.ref.coverageSummary };
}

function seconds(ms: number): string {
  return `${(ms / 1_000).toFixed(1)}초`;
}

export function ChartSelector({
  run,
  lastReviews = {},
  onSelect,
  onBoundaryReview,
  onReset,
}: ChartSelectorProps) {
  const publicationLabel = {
    PRODUCTION_VERIFIED: "RUN VERIFIED",
    PLAYTEST_ONLY: "PLAYTEST ONLY",
    LEGACY_V2_PLAYTEST_ONLY: "V2 LEGACY · PLAYTEST ONLY",
    LEGACY_UNVERIFIED: "LEGACY UNVERIFIED",
  }[run.publicationState];
  return (
    <section className="workspace-panel selector-panel" aria-labelledby="selector-title">
      <header className="workspace-heading">
        <div>
          <p className="eyebrow">{publicationLabel} / {run.charts.length} CHARTS</p>
          <h2 id="selector-title">{run.manifest.title}</h2>
          <p>{run.manifest.workerVersion} · {new Date(run.manifest.generatedAt).toLocaleString("ko-KR")}</p>
          {run.publicationReasons.length > 0 ? (
            <p
              aria-label="publication status"
              className="publication-status"
              data-state={run.publicationState}
              role="status"
            >
              {run.publicationReasons.join(" · ")}
            </p>
          ) : null}
        </div>
        <button className="text-button" onClick={onReset} type="button">다른 실행 불러오기</button>
      </header>

      <aside className="boundary-entry" aria-labelledby="boundary-entry-title">
        <div>
          <p className="eyebrow">SONG-LEVEL EVIDENCE</p>
          <h3 id="boundary-entry-title">곡 끝 경계</h3>
          <p>12개 채보와 별개로, 원본 게임 오디오의 마지막 소리 구간을 한 번만 기록합니다.</p>
          {!run.boundaryLabelContext.available ? (
            <p className="boundary-entry__reason">{run.boundaryLabelContext.unavailableReason}</p>
          ) : (
            <p className="boundary-entry__ready">보고서 SHA-256과 오디오 시간축이 연결되어 있습니다.</p>
          )}
        </div>
        <button
          disabled={!run.boundaryLabelContext.available}
          onClick={onBoundaryReview}
          type="button"
        >
          곡 끝 검토
        </button>
      </aside>

      <div className="chart-groups">
        {keyModes.map((keyMode) => (
          <section className="chart-group" key={keyMode} aria-labelledby={`mode-${keyMode}`}>
            <div className="mode-label" id={`mode-${keyMode}`}><span>{keyMode}K</span><small>LANE MODE</small></div>
            <div className="chart-grid">
              {run.charts
                .filter((chart) => chart.document.keyMode === keyMode)
                .map((chart) => {
                  const trust = chartTrust(chart);
                  return (
                  <article
                    className="chart-card"
                    data-playability={chart.ref.playabilityTier ?? "INFERRED"}
                    key={chart.document.chartId}
                  >
                    <div className="card-topline">
                      <strong>{chart.document.difficulty}</strong>
                      <span>LV {chart.document.metrics.projectRating.toFixed(2)}</span>
                    </div>
                    <div aria-label="chart trust" className="chart-trust">
                      <strong>{trust.label}</strong>
                      <span>{trust.provenance}</span>
                      {trust.summary ? (
                        <>
                          <span>
                            첫 노트 {trust.summary.firstNoteTimeMs === null
                              ? "없음"
                              : seconds(trust.summary.firstNoteTimeMs)}
                          </span>
                          <span>최대 공백 {seconds(trust.summary.maxGapMs)}</span>
                          <span>복구 구간 {trust.summary.repairedGapCount}개</span>
                          {trust.summary.attackRequiredGapCount > 0 ? (
                            <span className="value-fault">
                              활성 공백 {trust.summary.attackRequiredGapCount}개 · {seconds(
                                trust.summary.attackRequiredGapTotalMs,
                              )}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <span className="value-fault">coverage evidence 없음</span>
                      )}
                    </div>
                    <dl>
                      <div><dt>NOTES</dt><dd>{chart.document.metrics.noteCount}</dd></div>
                      <div>
                        <dt>HOLD</dt>
                        <dd className={chart.document.metrics.holdCount === 0 ? "value-fault" : undefined}>
                          {chart.document.metrics.holdCount}
                        </dd>
                      </div>
                      <div><dt>PEAK</dt><dd>{chart.document.metrics.peakNps.toFixed(2)} NPS</dd></div>
                      <div><dt>CHORD</dt><dd>{(chart.document.metrics.chordRatio * 100).toFixed(1)}%</dd></div>
                      {/* 라벨과 실측 티어가 어긋나면 그게 검수 대상이다. 숨기면
                          테스터가 EASY 인 줄 알고 NORMAL 밀도를 친다. */}
                      <div>
                        <dt>측정</dt>
                        <dd
                          className={
                            chart.document.metrics.projectTier === chart.document.difficulty
                              ? undefined
                              : "value-fault"
                          }
                        >
                          {chart.document.metrics.projectTier}
                        </dd>
                      </div>
                      <div><dt>REVIEW</dt><dd>{lastReviews[chart.document.chartId] ?? "—"}</dd></div>
                    </dl>
                    <button onClick={() => onSelect(chart)} type="button">
                      {keyMode}K {chart.document.difficulty} 플레이
                    </button>
                  </article>
                  );
                })}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}
