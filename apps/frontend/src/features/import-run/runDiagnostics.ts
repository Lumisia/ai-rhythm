import type { ChartDocument, Difficulty, KeyMode } from "../../game/core/types";

const difficultyOrder: Difficulty[] = ["EASY", "NORMAL", "HARD", "EXPERT"];
const targetRating: Record<Difficulty, number> = {
  EASY: 1.5,
  NORMAL: 2.6,
  HARD: 3.8,
  EXPERT: 5.0,
};
const ratingTolerance = 0.1;

export interface ChartDiagnostic {
  keyMode: KeyMode;
  difficulty: Difficulty;
  noteCount: number;
  holdCount: number;
  projectRating: number;
  projectTier: Difficulty;
}

export interface RunDiagnosticReport {
  charts: ChartDiagnostic[];
  errors: string[];
  warnings: string[];
}

function chartLabel(chart: ChartDocument): string {
  return `${chart.keyMode}K ${chart.difficulty}`;
}

function noteErrors(chart: ChartDocument): string[] {
  const label = chartLabel(chart);
  const errors: string[] = [];

  for (const note of chart.notes) {
    if (note.timeMs < 0 || note.timeMs >= chart.durationMs) {
      errors.push(
        `${label} note ${note.id}: time ${note.timeMs} is outside duration ${chart.durationMs}`,
      );
    }
    if (!Number.isInteger(note.lane) || note.lane < 0 || note.lane >= chart.keyMode) {
      errors.push(`${label} note ${note.id}: lane ${note.lane} is outside ${chart.keyMode}K`);
    }
    if (note.type === "HOLD") {
      const holdDuration = note.durationMs;
      if (
        holdDuration == null ||
        holdDuration <= 0 ||
        note.timeMs + holdDuration > chart.durationMs
      ) {
        errors.push(
          `${label} note ${note.id}: HOLD duration ${String(holdDuration)} exceeds chart duration`,
        );
      }
    }
  }
  return errors;
}

export function diagnoseCharts(charts: readonly ChartDocument[]): RunDiagnosticReport {
  const errors: string[] = [];
  const warnings: string[] = [];

  for (const chart of charts) {
    const label = chartLabel(chart);
    errors.push(...noteErrors(chart));
    if (chart.laneSemantics.length !== chart.keyMode) {
      errors.push(`${label}: laneSemantics length differs from keyMode`);
    }
    if (chart.metrics.noteCount !== chart.notes.length) {
      errors.push(
        `${label}: metrics.noteCount ${chart.metrics.noteCount} differs from ${chart.notes.length}`,
      );
    }
    const holdCount = chart.notes.filter((note) => note.type === "HOLD").length;
    if (chart.metrics.holdCount !== holdCount) {
      errors.push(
        `${label}: metrics.holdCount ${chart.metrics.holdCount} differs from ${holdCount}`,
      );
    }

    const rating = chart.metrics.projectRating;
    const target = targetRating[chart.difficulty];
    if (rating > target + ratingTolerance) {
      warnings.push(
        `${label}: rating ${rating.toFixed(3)} exceeds target ${target.toFixed(3)}`,
      );
    }
    if (chart.metrics.projectTier !== chart.difficulty) {
      warnings.push(`${label}: measured tier is ${chart.metrics.projectTier}`);
    }
  }

  for (const keyMode of [4, 6, 7] as const) {
    const byDifficulty = new Map(
      charts
        .filter((chart) => chart.keyMode === keyMode)
        .map((chart) => [chart.difficulty, chart] as const),
    );
    for (let index = 0; index < difficultyOrder.length - 1; index += 1) {
      const easierDifficulty = difficultyOrder[index];
      const harderDifficulty = difficultyOrder[index + 1];
      const easier = byDifficulty.get(easierDifficulty);
      const harder = byDifficulty.get(harderDifficulty);
      if (!easier || !harder) continue;
      const easierRating = easier.metrics.projectRating;
      const harderRating = harder.metrics.projectRating;
      if (harderRating < easierRating) {
        warnings.push(
          `${keyMode}K rating inversion: ${easierDifficulty} ${easierRating.toFixed(3)} > ` +
            `${harderDifficulty} ${harderRating.toFixed(3)}`,
        );
      }
    }
  }

  return {
    charts: charts.map((chart) => ({
      keyMode: chart.keyMode,
      difficulty: chart.difficulty,
      noteCount: chart.notes.length,
      holdCount: chart.notes.filter((note) => note.type === "HOLD").length,
      projectRating: chart.metrics.projectRating,
      projectTier: chart.metrics.projectTier,
    })),
    errors,
    warnings,
  };
}
