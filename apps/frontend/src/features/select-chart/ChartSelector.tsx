import type { ImportedChart, ImportedRun } from "../import-run/importRun";

interface ChartSelectorProps {
  run: ImportedRun;
  lastReviews?: Readonly<Record<string, string>>;
  onSelect: (chart: ImportedChart) => void;
  onReset: () => void;
}

const keyModes = [4, 6, 7] as const;

export function ChartSelector({ run, lastReviews = {}, onSelect, onReset }: ChartSelectorProps) {
  return (
    <section className="workspace-panel selector-panel" aria-labelledby="selector-title">
      <header className="workspace-heading">
        <div>
          <p className="eyebrow">RUN VERIFIED / 12 CHARTS</p>
          <h2 id="selector-title">{run.manifest.title}</h2>
          <p>{run.manifest.workerVersion} · {new Date(run.manifest.generatedAt).toLocaleString("ko-KR")}</p>
        </div>
        <button className="text-button" onClick={onReset} type="button">다른 실행 불러오기</button>
      </header>

      <div className="chart-groups">
        {keyModes.map((keyMode) => (
          <section className="chart-group" key={keyMode} aria-labelledby={`mode-${keyMode}`}>
            <div className="mode-label" id={`mode-${keyMode}`}><span>{keyMode}K</span><small>LANE MODE</small></div>
            <div className="chart-grid">
              {run.charts
                .filter((chart) => chart.document.keyMode === keyMode)
                .map((chart) => (
                  <article className="chart-card" key={chart.document.chartId}>
                    <div className="card-topline">
                      <strong>{chart.document.difficulty}</strong>
                      <span>LV {chart.document.metrics.projectRating.toFixed(2)}</span>
                    </div>
                    <dl>
                      <div><dt>NOTES</dt><dd>{chart.document.metrics.noteCount}</dd></div>
                      <div><dt>PEAK</dt><dd>{chart.document.metrics.peakNps.toFixed(2)} NPS</dd></div>
                      <div><dt>CHORD</dt><dd>{(chart.document.metrics.chordRatio * 100).toFixed(1)}%</dd></div>
                      <div><dt>REVIEW</dt><dd>{lastReviews[chart.document.chartId] ?? "—"}</dd></div>
                    </dl>
                    <button onClick={() => onSelect(chart)} type="button">
                      {keyMode}K {chart.document.difficulty} 플레이
                    </button>
                  </article>
                ))}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}
