import { useState } from "react";

import { ImportRunPanel } from "../features/import-run/ImportRunPanel";
import { importRun, type ImportedChart, type ImportedRun } from "../features/import-run/importRun";
import { PlayChartPanel, type PlaySessionResult } from "../features/play-chart/PlayChartPanel";
import { ChartSelector } from "../features/select-chart/ChartSelector";

type Importer = (files: File[]) => Promise<ImportedRun>;

interface LocalPlaytestPageProps {
  importer?: Importer;
}

export function LocalPlaytestPage({ importer = importRun }: LocalPlaytestPageProps) {
  const [run, setRun] = useState<ImportedRun | null>(null);
  const [selectedChart, setSelectedChart] = useState<ImportedChart | null>(null);
  const [result, setResult] = useState<PlaySessionResult | null>(null);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFiles = async (files: File[]) => {
    setImporting(true);
    setError(null);
    try {
      setRun(await importer(files));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setImporting(false);
    }
  };

  if (!run) return <ImportRunPanel error={error} importing={importing} onFiles={(files) => void handleFiles(files)} />;
  if (result && selectedChart) {
    return (
      <section className="workspace-panel result-placeholder" aria-labelledby="result-title">
        <p className="eyebrow">SESSION CAPTURED</p>
        <h2 id="result-title">플레이 결과</h2>
        <div className="result-strip">
          <span>MAX COMBO <strong>{result.score.maxCombo}</strong></span>
          <span>MISS <strong>{result.score.counts.MISS}</strong></span>
          <span>MEAN ABS <strong>{result.score.meanAbsoluteErrMs.toFixed(1)} ms</strong></span>
        </div>
        <button className="primary-action" onClick={() => { setResult(null); setSelectedChart(null); }} type="button">채보 목록으로</button>
      </section>
    );
  }
  if (selectedChart) {
    return <PlayChartPanel chart={selectedChart} onBack={() => setSelectedChart(null)} onComplete={setResult} run={run} />;
  }
  return <ChartSelector onReset={() => { setRun(null); setError(null); }} onSelect={setSelectedChart} run={run} />;
}
