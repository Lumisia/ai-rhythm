import type { JudgmentPreset } from "../../game/core/types";

/** IIDX 그린넘버 선호 대역 250~330 을 ms 로 옮긴 값.
 *
 * 그린넘버는 (노트가 보이는 프레임 수) × 10 이고 60fps 기준이다.
 * 250 → 25프레임 → 417ms, 330 → 33프레임 → 550ms.
 */
const COMFORTABLE_MIN_MS = 417;
const COMFORTABLE_MAX_MS = 550;

function approachHint(approachMs: number): { label: string; tone: "ok" | "slow" | "fast" } {
  if (approachMs > COMFORTABLE_MAX_MS) return { label: "느림", tone: "slow" };
  if (approachMs < COMFORTABLE_MIN_MS) return { label: "빠름", tone: "fast" };
  return { label: "권장", tone: "ok" };
}

export interface PlaySettingsValue {
  calibrationMs: number;
  scrollSpeed: number;
  judgmentPreset: JudgmentPreset;
  keysound: boolean;
  fever: boolean;
  loopEnabled: boolean;
  loopStartMs: number;
  loopEndMs: number;
  reduceMotion: boolean;
}

interface PlaySettingsProps {
  value: PlaySettingsValue;
  durationMs: number;
  keysoundAvailable: boolean;
  /** 배속 1.0 기준 노출 시간. 씬이 아직 안 뜬 READY 상태에서는 null 이다. */
  approachMsAt1x: number | null;
  disabled?: boolean;
  onChange: (value: PlaySettingsValue) => void;
}

export function PlaySettings({
  value,
  durationMs,
  keysoundAvailable,
  approachMsAt1x,
  disabled,
  onChange,
}: PlaySettingsProps) {
  const update = <K extends keyof PlaySettingsValue>(key: K, next: PlaySettingsValue[K]) =>
    onChange({ ...value, [key]: next });

  return (
    <fieldset className="settings-grid" disabled={disabled}>
      <legend>PLAY PARAMETERS</legend>
      <label>
        <span>입력 보정</span>
        <input max={150} min={-150} onChange={(event) => update("calibrationMs", event.currentTarget.valueAsNumber)} type="number" value={value.calibrationMs} />
        <small>ms</small>
      </label>
      <label>
        <span>스크롤 속도</span>
        <input max={4} min={0.6} onChange={(event) => update("scrollSpeed", event.currentTarget.valueAsNumber)} step={0.1} type="range" value={value.scrollSpeed} />
        {/* 배속보다 "노트가 몇 ms 흐르는가"가 실제로 읽는 값이다. */}
        <small>
          {approachMsAt1x === null
            ? "시작 후 측정"
            : (() => {
                const approachMs = approachMsAt1x / value.scrollSpeed;
                const hint = approachHint(approachMs);
                return (
                  <>
                    {Math.round(approachMs)}ms{" "}
                    <span className={`approach-hint approach-hint--${hint.tone}`}>{hint.label}</span>
                  </>
                );
              })()}
        </small>
      </label>
      <label>
        <span>판정 프리셋</span>
        <select onChange={(event) => update("judgmentPreset", event.currentTarget.value as JudgmentPreset)} value={value.judgmentPreset}>
          <option value="lenient">완화</option><option value="normal">일반</option><option value="strict">엄격</option>
        </select>
      </label>
      <label className="check-setting">
        <input checked={value.keysound && keysoundAvailable} disabled={!keysoundAvailable || disabled} onChange={(event) => update("keysound", event.currentTarget.checked)} type="checkbox" />
        <span>키음 stem 사용</span>
        <small>{keysoundAvailable ? "DRUMS READY" : "UNAVAILABLE"}</small>
      </label>
      <label className="check-setting">
        <input checked={value.fever} onChange={(event) => update("fever", event.currentTarget.checked)} type="checkbox" />
        <span>FEVER</span>
        <small>콤보 ×2 · 정확도 불변</small>
      </label>
      <label className="check-setting">
        <input checked={value.loopEnabled} onChange={(event) => update("loopEnabled", event.currentTarget.checked)} type="checkbox" />
        <span>구간 반복</span>
      </label>
      <label className="check-setting">
        <input
          checked={value.reduceMotion}
          onChange={(event) => update("reduceMotion", event.currentTarget.checked)}
          type="checkbox"
        />
        <span>모션 감소</span>
        <small>파티클·판정선 펄스·콤보 pop 끔 · 레인 플래시와 MISS 표시는 유지</small>
      </label>
      <label>
        <span>구간 시작</span>
        <input disabled={disabled || !value.loopEnabled} max={durationMs} min={0} onChange={(event) => update("loopStartMs", event.currentTarget.valueAsNumber)} step={100} type="number" value={value.loopStartMs} />
        <small>ms</small>
      </label>
      <label>
        <span>구간 끝</span>
        <input disabled={disabled || !value.loopEnabled} max={durationMs} min={0} onChange={(event) => update("loopEndMs", event.currentTarget.valueAsNumber)} step={100} type="number" value={value.loopEndMs} />
        <small>ms</small>
      </label>
    </fieldset>
  );
}
