import { useState } from "react";

import { ImportRunPanel } from "../features/import-run/ImportRunPanel";
import { BoundaryLabelPanel } from "../features/boundary-label/BoundaryLabelPanel";
import {
  importPlaytestRun,
  type ImportedChart,
  type ImportedRun,
} from "../features/import-run/importRun";
import { PlayChartPanel, type PlaySessionResult } from "../features/play-chart/PlayChartPanel";
import { ChartSelector } from "../features/select-chart/ChartSelector";
import { ReviewResult } from "../features/review-chart/ReviewResult";
import type { PlaytestReview } from "../features/review-chart/review";

type Importer = (files: File[]) => Promise<ImportedRun>;

interface LocalPlaytestPageProps {
  importer?: Importer;
}

export function LocalPlaytestPage({ importer = importPlaytestRun }: LocalPlaytestPageProps) {
  const [run, setRun] = useState<ImportedRun | null>(null);
  const [selectedChart, setSelectedChart] = useState<ImportedChart | null>(null);
  const [result, setResult] = useState<PlaySessionResult | null>(null);
  const [lastReviews, setLastReviews] = useState<Record<string, string>>({});
  const [reviewingBoundary, setReviewingBoundary] = useState(false);
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
  if (reviewingBoundary) {
    return <BoundaryLabelPanel onBack={() => setReviewingBoundary(false)} run={run} />;
  }
  if (result && selectedChart) {
    return (
      <ReviewResult
        onBack={() => { setResult(null); setSelectedChart(null); }}
        onSaved={(review: PlaytestReview) => setLastReviews((current) => ({ ...current, [review.chartId]: review.verdict }))}
        session={{
          runId: run.manifest.runId,
          title: run.manifest.title,
          chartId: selectedChart.document.chartId,
          chartSha256: selectedChart.ref.sha256,
          audioSha256: run.manifest.audio.game.sha256,
          keyMode: selectedChart.document.keyMode,
          difficulty: selectedChart.document.difficulty,
          durationMs: selectedChart.document.durationMs,
          result,
        }}
      />
    );
  }
  if (selectedChart) {
    return <PlayChartPanel chart={selectedChart} onBack={() => setSelectedChart(null)} onComplete={setResult} run={run} />;
  }
  return (
    <ChartSelector
      lastReviews={lastReviews}
      onBoundaryReview={() => setReviewingBoundary(true)}
      onReset={() => {
        setRun(null);
        setError(null);
        setLastReviews({});
        setReviewingBoundary(false);
      }}
      onSelect={setSelectedChart}
      run={run}
    />
  );
}
