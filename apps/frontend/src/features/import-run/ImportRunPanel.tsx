import { useRef, type ChangeEvent } from "react";

interface ImportRunPanelProps {
  importing: boolean;
  error: string | null;
  onFiles: (files: File[]) => void;
}

const workflow = [
  { index: "01", label: "실행 폴더", detail: "worker 산출물" },
  { index: "02", label: "채보 선택", detail: "4K · 6K · 7K" },
  { index: "03", label: "직접 플레이", detail: "오차와 패턴 기록" },
];

export function ImportRunPanel({ importing, error, onFiles }: ImportRunPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    event.currentTarget.value = "";
    if (files.length > 0) onFiles(files);
  };

  return (
    <section className="entry-panel" aria-labelledby="entry-title">
      <div className="beat-spine" aria-hidden="true">
        <span className="beat-index">000</span>
        {Array.from({ length: 13 }, (_, index) => (
          <span className={index % 4 === 0 ? "beat-tick beat-tick-major" : "beat-tick"} key={index} />
        ))}
        <span className="beat-index">END</span>
      </div>

      <div className="entry-copy">
        <p className="eyebrow">CHART CALIBRATION / MANUAL INPUT</p>
        <h1 id="entry-title">채보 플레이테스트</h1>
        <p className="lede">
          생성된 노트를 직접 연주하면서 박자 오차, 난이도 체감, 잘못된 레인 패턴을 한 타임라인에
          기록합니다.
        </p>

        <input
          aria-label="실행 폴더"
          className="visually-hidden"
          multiple
          onChange={handleFiles}
          ref={inputRef}
          type="file"
          {...({ webkitdirectory: "" } as { webkitdirectory: string })}
        />
        <button
          aria-label="실행 폴더 선택"
          className="folder-button"
          disabled={importing}
          onClick={() => inputRef.current?.click()}
          type="button"
        >
          <span className="button-key">{importing ? "CHECK" : "OPEN"}</span>
          <span>{importing ? "무결성 검사 중" : "실행 폴더 선택"}</span>
          <span className="button-arrow" aria-hidden="true">→</span>
        </button>
        <p className="privacy-note">파일은 브라우저 안에서만 읽으며 서버로 전송하지 않습니다.</p>
        {error ? <p className="import-error" role="alert">{error}</p> : null}
      </div>

      <aside className="workflow-panel" aria-label="플레이테스트 순서">
        <p className="panel-label">SIGNAL ROUTE</p>
        <ol>
          {workflow.map((step) => (
            <li key={step.index}>
              <span className="step-index">{step.index}</span>
              <span><strong>{step.label}</strong><small>{step.detail}</small></span>
            </li>
          ))}
        </ol>
        <div className="format-strip"><span>INPUT</span><strong>playtest-run-v1.json</strong></div>
      </aside>
    </section>
  );
}
