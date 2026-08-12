import { useEffect, useMemo, useRef, useState } from "react";

import type {
  BoundaryGroupRelation,
  BoundaryLabelV2,
  BoundaryTailCharacter,
  BoundaryVerdict,
  HumanLabelConfidence,
} from "../../game/core/types";
import type { ImportedRun } from "../import-run/importRun";
import {
  buildBoundaryLabelV2,
  type BoundaryLabelDraftV2,
} from "./boundaryLabel";
import { downloadBoundaryLabelV2 } from "./downloadBoundaryLabel";

interface BoundaryLabelPanelProps {
  run: ImportedRun;
  onBack: () => void;
  download?: (label: BoundaryLabelV2) => void;
}

interface BoundaryFormState {
  reviewerId: string;
  groupId: string;
  relation: BoundaryGroupRelation;
  groupConfirmed: boolean;
  attackEarliestMs: string;
  attackLatestMs: string;
  contentEarliestMs: string;
  contentLatestMs: string;
  releaseEarliestMs: string;
  releaseLatestMs: string;
  verdict: BoundaryVerdict;
  tailCharacters: BoundaryTailCharacter[];
  confidence: HumanLabelConfidence;
  comment: string;
}

const tailOptions: Array<{ value: BoundaryTailCharacter; label: string }> = [
  { value: "MUSIC", label: "음악" },
  { value: "FADE_OR_REVERB", label: "페이드 또는 잔향" },
  { value: "NOISE", label: "노이즈" },
  { value: "ENCODING_TAIL", label: "인코딩 꼬리" },
  { value: "SILENCE", label: "무음" },
  { value: "MIXED_OR_UNCERTAIN", label: "혼합 또는 불확실" },
];

const verdictOptions: Array<{ value: BoundaryVerdict; label: string }> = [
  { value: "TOO_EARLY", label: "자동 경계가 너무 이름" },
  { value: "ACCEPTABLE", label: "자동 경계가 허용 가능함" },
  { value: "TOO_LATE", label: "자동 경계가 너무 늦음" },
  { value: "UNCERTAIN", label: "판단 불확실" },
  { value: "NOT_AVAILABLE", label: "자동 경계 비교 불가" },
];

function parseRequiredMs(value: string, label: string): number {
  if (!/^\d+$/.test(value)) throw new Error(`${label}을(를) 정수 밀리초로 입력하세요`);
  return Number(value);
}

function formToDraft(form: BoundaryFormState): BoundaryLabelDraftV2 {
  return {
    reviewerId: form.reviewerId,
    groupId: form.groupId,
    relation: form.relation,
    groupConfirmed: form.groupConfirmed,
    lastPlayableAttack: {
      earliestMs: parseRequiredMs(form.attackEarliestMs, "마지막으로 칠 수 있는 타격의 이른 시각"),
      latestMs: parseRequiredMs(form.attackLatestMs, "마지막으로 칠 수 있는 타격의 늦은 시각"),
    },
    primaryContentEnd: {
      earliestMs: parseRequiredMs(form.contentEarliestMs, "주요 음악·보컬 종료의 이른 시각"),
      latestMs: parseRequiredMs(form.contentLatestMs, "주요 음악·보컬 종료의 늦은 시각"),
    },
    acceptableReleaseEnd: {
      earliestMs: parseRequiredMs(form.releaseEarliestMs, "허용 가능한 잔향·배경음 종료의 이른 시각"),
      latestMs: parseRequiredMs(form.releaseLatestMs, "허용 가능한 잔향·배경음 종료의 늦은 시각"),
    },
    provisionalBoundaryVerdict: form.verdict,
    tailCharacters: form.tailCharacters,
    confidence: form.confidence,
    comment: form.comment,
  };
}

function formatTime(ms: number | null): string {
  return ms === null ? "—" : `${(ms / 1000).toFixed(3)}초 / ${ms}ms`;
}

export function BoundaryLabelPanel({
  run,
  onBack,
  download = downloadBoundaryLabelV2,
}: BoundaryLabelPanelProps) {
  const context = run.boundaryLabelContext;
  const evidence = context.automaticEvidence;
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [currentMs, setCurrentMs] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loopTail, setLoopTail] = useState(false);
  const [saveResult, setSaveResult] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [form, setForm] = useState<BoundaryFormState>({
    reviewerId: "",
    groupId: context.songVersionId,
    relation: "EXACT_RECORDING",
    groupConfirmed: true,
    attackEarliestMs: "",
    attackLatestMs: "",
    contentEarliestMs: "",
    contentLatestMs: "",
    releaseEarliestMs: "",
    releaseLatestMs: "",
    verdict: evidence.availability === "AVAILABLE" ? "UNCERTAIN" : "NOT_AVAILABLE",
    tailCharacters: [],
    confidence: "LOW",
    comment: "",
  });

  const identity = useMemo(
    () => ({
      runId: run.manifest.runId,
      title: run.manifest.title,
      audioSha256: run.manifest.audio.game.sha256,
    }),
    [run.manifest],
  );
  const tailStartMs = Math.max(0, context.audioDurationMs - 30_000);

  useEffect(() => {
    try {
      const url = URL.createObjectURL(new Blob([run.audio.game], { type: "audio/flac" }));
      setAudioUrl(url);
      return () => URL.revokeObjectURL(url);
    } catch (error) {
      setAudioError(error instanceof Error ? error.message : String(error));
      return undefined;
    }
  }, [run.audio.game]);

  const validation = useMemo(() => {
    try {
      if (audioError) throw new Error(audioError);
      const label = buildBoundaryLabelV2(formToDraft(form), context, identity, {
        createUuid: () => "00000000-0000-4000-8000-000000000001",
        now: () => new Date("2000-01-01T00:00:00.000Z"),
      });
      return { label, error: null };
    } catch (error) {
      return { label: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [audioError, context, form, identity]);

  const setField = <K extends keyof BoundaryFormState>(key: K, value: BoundaryFormState[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
    setSaveResult(null);
    setSaveError(null);
  };

  const seekToMs = (nextMs: number) => {
    const clamped = Math.min(context.audioDurationMs, Math.max(0, nextMs));
    if (audioRef.current) audioRef.current.currentTime = clamped / 1000;
    setCurrentMs(clamped);
  };

  const togglePlayback = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
      return;
    }
    try {
      await audio.play();
      setIsPlaying(true);
      setAudioError(null);
    } catch (error) {
      setAudioError(error instanceof Error ? error.message : String(error));
    }
  };

  const updateTail = (value: BoundaryTailCharacter, checked: boolean) => {
    if (!checked) {
      setField("tailCharacters", form.tailCharacters.filter((entry) => entry !== value));
      return;
    }
    setField(
      "tailCharacters",
      value === "MIXED_OR_UNCERTAIN"
        ? [value]
        : [...form.tailCharacters.filter((entry) => entry !== "MIXED_OR_UNCERTAIN"), value],
    );
  };

  const handleSave = () => {
    setSaveResult(null);
    setSaveError(null);
    try {
      const label = buildBoundaryLabelV2(formToDraft(form), context, identity);
      download(label);
      setSaveResult("파일을 다운로드했습니다. 자동 경계 정책에는 아직 적용되지 않습니다.");
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <section className="workspace-panel boundary-panel" aria-labelledby="boundary-title">
      <header className="workspace-heading">
        <div>
          <p className="eyebrow">SONG BOUNDARY / HUMAN LABEL</p>
          <h2 id="boundary-title">곡 끝 경계 검토</h2>
          <p>{run.manifest.title} · canonical game audio</p>
        </div>
        <button className="text-button" onClick={onBack} type="button">채보 목록으로</button>
      </header>

      <div className="boundary-layout">
        <section className="boundary-audio-card" aria-labelledby="boundary-audio-title">
          <div className="boundary-section-heading">
            <p className="eyebrow">TAIL WINDOW</p>
            <h3 id="boundary-audio-title">마지막 30초 듣기</h3>
          </div>
          {audioUrl ? (
            <audio
              aria-label="곡 끝 검토용 원본 게임 오디오"
              onEnded={() => {
                if (!loopTail) {
                  setIsPlaying(false);
                  return;
                }
                seekToMs(tailStartMs);
                void audioRef.current?.play().catch((error: unknown) => {
                  setAudioError(error instanceof Error ? error.message : String(error));
                  setIsPlaying(false);
                });
              }}
              onError={() => setAudioError("브라우저가 원본 게임 오디오를 재생하지 못했습니다")}
              onLoadedMetadata={() => seekToMs(tailStartMs)}
              onPause={() => setIsPlaying(false)}
              onPlay={() => setIsPlaying(true)}
              onTimeUpdate={(event) => {
                const audio = event.currentTarget;
                let nextMs = Math.round(audio.currentTime * 1000);
                if (loopTail && nextMs >= context.audioDurationMs) {
                  nextMs = tailStartMs;
                  audio.currentTime = nextMs / 1000;
                  void audio.play().catch(() => undefined);
                }
                setCurrentMs(nextMs);
              }}
              preload="metadata"
              ref={audioRef}
              src={audioUrl}
            />
          ) : null}
          <output className="boundary-clock">{formatTime(currentMs)}</output>
          <div className="boundary-transport" aria-label="곡 끝 재생 제어">
            <button onClick={() => void togglePlayback()} type="button">{isPlaying ? "일시정지" : "재생"}</button>
            {[-10_000, -2_000, 2_000, 10_000].map((delta) => (
              <button key={delta} onClick={() => seekToMs(currentMs + delta)} type="button">
                {delta > 0 ? "+" : "−"}{Math.abs(delta) / 1000}초
              </button>
            ))}
            <button onClick={() => seekToMs(tailStartMs)} type="button">마지막 30초</button>
            <label className="boundary-loop">
              <input checked={loopTail} onChange={(event) => setLoopTail(event.target.checked)} type="checkbox" />
              끝 구간 반복
            </label>
          </div>
          {audioError ? <p role="alert" className="boundary-error">{audioError}</p> : null}
        </section>

        <section className="boundary-evidence-card" aria-labelledby="boundary-evidence-title">
          <div className="boundary-section-heading">
            <p className="eyebrow">READ-ONLY EVIDENCE</p>
            <h3 id="boundary-evidence-title">자동 감지 참고값</h3>
          </div>
          <p
            aria-label="자동 경계 증거 상태"
            className="boundary-evidence-status"
            data-available={evidence.availability === "AVAILABLE"}
            role="status"
          >
            {evidence.availability === "AVAILABLE"
              ? `${evidence.policyState} · ${evidence.policyConfidence} · ${evidence.enforcementMode}`
              : `사용 불가 · ${evidence.unavailableReason}`}
          </p>
          <dl className="boundary-evidence-grid">
            <div><dt>마지막 onset</dt><dd>{formatTime(evidence.lastDetectedOnsetMs)}</dd></div>
            <div><dt>마지막 RMS 활성</dt><dd>{formatTime(evidence.lastActiveRmsEndMs)}</dd></div>
            <div><dt>마지막 통합 증거</dt><dd>{formatTime(evidence.lastEvidenceMs)}</dd></div>
            <div><dt>임시 note start 상한</dt><dd>{formatTime(evidence.provisionalMaxNoteStartMs)}</dd></div>
            <div><dt>실효 note start 상한</dt><dd>{formatTime(evidence.effectiveMaxNoteStartMs)}</dd></div>
          </dl>
          <p className="boundary-evidence-note">자동 참고값은 사람 라벨로 자동 복사되지 않습니다.</p>
        </section>
      </div>

      <form className="boundary-form" onSubmit={(event) => { event.preventDefault(); handleSave(); }}>
        <section className="boundary-form-card" aria-labelledby="human-time-title">
          <div className="boundary-section-heading">
            <p className="eyebrow">HUMAN INTERVALS</p>
            <h3 id="human-time-title">사람이 들은 경계</h3>
          </div>
          <p className="boundary-help">
            세 값은 목적이 다릅니다. 마지막 타격은 실제 노트 시작 후보, 주요 종료는 음악·보컬의 중심 내용이
            끝난 지점, 허용 종료는 페이드·잔향·배경음까지 자연스럽게 끝난 지점입니다. 확신하기 어렵다면
            이른 시각과 늦은 시각을 다르게 적어 범위를 보존하세요.
          </p>
          <div className="boundary-time-grid">
            {([
              ["attackEarliestMs", "마지막으로 칠 수 있는 타격 — 이른 시각 (ms)"],
              ["attackLatestMs", "마지막으로 칠 수 있는 타격 — 늦은 시각 (ms)"],
              ["contentEarliestMs", "주요 음악·보컬 종료 — 이른 시각 (ms)"],
              ["contentLatestMs", "주요 음악·보컬 종료 — 늦은 시각 (ms)"],
              ["releaseEarliestMs", "허용 가능한 잔향·배경음 종료 — 이른 시각 (ms)"],
              ["releaseLatestMs", "허용 가능한 잔향·배경음 종료 — 늦은 시각 (ms)"],
            ] as const).map(([key, label]) => (
              <label className="boundary-time-field" key={key}>
                <span>{label}</span>
                <div>
                  <input
                    aria-label={label}
                    inputMode="numeric"
                    min="0"
                    onChange={(event) => setField(key, event.target.value)}
                    step="1"
                    type="number"
                    value={form[key]}
                  />
                  <button onClick={() => setField(key, String(currentMs))} type="button">현재 위치 사용</button>
                </div>
              </label>
            ))}
          </div>
        </section>

        <section className="boundary-form-card" aria-labelledby="classification-title">
          <div className="boundary-section-heading">
            <p className="eyebrow">CLASSIFICATION</p>
            <h3 id="classification-title">끝부분 분류</h3>
          </div>
          <label className="boundary-select-field">
            <span>자동 경계와 비교한 판정</span>
            <select
              aria-label="자동 경계와 비교한 판정"
              disabled={evidence.availability === "UNAVAILABLE"}
              onChange={(event) => setField("verdict", event.target.value as BoundaryVerdict)}
              value={form.verdict}
            >
              {verdictOptions.map((option) => (
                <option disabled={option.value === "NOT_AVAILABLE" && evidence.availability === "AVAILABLE"} key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <fieldset className="boundary-tail-options">
            <legend>끝부분에서 들린 것</legend>
            {tailOptions.map((option) => (
              <label key={option.value}>
                <input
                  checked={form.tailCharacters.includes(option.value)}
                  onChange={(event) => updateTail(option.value, event.target.checked)}
                  type="checkbox"
                />
                {option.label}
              </label>
            ))}
          </fieldset>
          <label className="boundary-select-field">
            <span>사람 판단 확신도</span>
            <select onChange={(event) => setField("confidence", event.target.value as HumanLabelConfidence)} value={form.confidence}>
              <option value="HIGH">높음</option>
              <option value="MEDIUM">보통</option>
              <option value="LOW">낮음</option>
            </select>
          </label>
        </section>

        <section className="boundary-form-card boundary-identity-card" aria-labelledby="identity-title">
          <div className="boundary-section-heading">
            <p className="eyebrow">IDENTITY & GROUP</p>
            <h3 id="identity-title">검토자와 녹음 그룹</h3>
          </div>
          <label><span>검토자 ID</span><input aria-label="검토자 ID" autoComplete="off" onChange={(event) => setField("reviewerId", event.target.value)} value={form.reviewerId} /></label>
          <label><span>그룹 ID</span><input aria-label="그룹 ID" onChange={(event) => setField("groupId", event.target.value)} value={form.groupId} /></label>
          <label><span>곡 버전 관계</span>
            <select onChange={(event) => setField("relation", event.target.value as BoundaryGroupRelation)} value={form.relation}>
              <option value="EXACT_RECORDING">동일 녹음</option>
              <option value="RELATED_VERSION">관련 버전</option>
              <option value="UNKNOWN">모름</option>
            </select>
          </label>
          <label className="boundary-confirm"><input checked={form.groupConfirmed} onChange={(event) => setField("groupConfirmed", event.target.checked)} type="checkbox" />이 그룹 관계를 확인했습니다</label>
          <label className="boundary-comment"><span>메모</span><textarea maxLength={4000} onChange={(event) => setField("comment", event.target.value)} rows={4} value={form.comment} /></label>
        </section>

        <section className="boundary-export-card" aria-labelledby="export-title">
          <div>
            <p className="eyebrow">LOCAL EXPORT ONLY</p>
            <h3 id="export-title">경계 라벨 저장</h3>
            <p>이 파일은 자동 경계를 활성화하거나 정책을 보정하지 않습니다. 검증된 후속 calibration 입력일 뿐입니다.</p>
            {validation.error ? <p className="boundary-validation">첫 미완료 항목: {validation.error}</p> : <p className="boundary-ready">모든 필수 항목이 유효합니다.</p>}
            {saveError ? <p className="boundary-error" role="alert">{saveError}</p> : null}
            {saveResult ? <p aria-label="저장 결과" className="boundary-ready" role="status">{saveResult}</p> : null}
          </div>
          <button className="primary-action" disabled={!validation.label} type="submit">경계 라벨 JSON 저장</button>
        </section>
      </form>
    </section>
  );
}
