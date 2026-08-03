import { useRef, type ChangeEvent } from "react";

interface ImportRunPanelProps {
  importing: boolean;
  error: string | null;
  onFiles: (files: File[]) => void;
}

const workflow = [
  { label: "FORMAT", detail: "playtest-run-v1" },
  { label: "CHARTS", detail: "4K · 6K · 7K / 12" },
  { label: "MODE", detail: "manual timing review" },
];

export function ImportRunPanel({ importing, error, onFiles }: ImportRunPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    event.currentTarget.value = "";
    if (files.length > 0) onFiles(files);
  };

  return (
    <section className="overlay-screen entry-screen" aria-labelledby="entry-title">
      <div className="menu-card entry-card">
        <p className="eyebrow">MANUAL CHART REVIEW / VERTICAL 4K · 6K · 7K</p>
        <h1 id="entry-title">AI RHYTHM GAME</h1>
        <p className="sub">채보 플레이테스트</p>
        <p className="lede">
          생성된 노트를 직접 연주해 박자 오차, 체감 난이도, 잘못된 레인 패턴을 기록합니다.
        </p>

        <dl className="meta-grid entry-meta">
          {workflow.map((item) => (
            <div key={item.label}><dt>{item.label}</dt><dd>{item.detail}</dd></div>
          ))}
        </dl>

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
        <p className="hint">폴더 안의 manifest·chart·audio 해시를 확인한 뒤 메뉴가 열립니다.</p>
      </div>
    </section>
  );
}
