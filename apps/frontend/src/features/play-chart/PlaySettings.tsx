import type { JudgmentPreset } from "../../game/core/types";

/** 배속 1.0 에서 노트가 화면을 흐르는 시간. 640px 플레이필드 기준값이다. */
const APPROACH_MS_AT_1X = 907;

export interface PlaySettingsValue {
  calibrationMs: number;
  scrollSpeed: number;
  judgmentPreset: JudgmentPreset;
  keysound: boolean;
  loopEnabled: boolean;
  loopStartMs: number;
  loopEndMs: number;
}

interface PlaySettingsProps {
  value: PlaySettingsValue;
  durationMs: number;
  keysoundAvailable: boolean;
  disabled?: boolean;
  onChange: (value: PlaySettingsValue) => void;
}

export function PlaySettings({ value, durationMs, keysoundAvailable, disabled, onChange }: PlaySettingsProps) {
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
        <small>{Math.round(APPROACH_MS_AT_1X / value.scrollSpeed)}ms</small>
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
        <input checked={value.loopEnabled} onChange={(event) => update("loopEnabled", event.currentTarget.checked)} type="checkbox" />
        <span>구간 반복</span>
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
