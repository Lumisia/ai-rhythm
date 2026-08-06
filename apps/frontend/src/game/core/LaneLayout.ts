import type { LaneSemantic } from "./types";

export interface LaneGeometry {
  index: number;
  semantic: LaneSemantic;
  x: number;
  width: number;
  color: number;
  backgroundColor: number;
}

export interface StageGeometry {
  lanes: readonly LaneGeometry[];
  /** 플레이필드 왼쪽 끝. 이 바깥은 무대가 아니다. */
  left: number;
  right: number;
  width: number;
}

/** osu!lazer `Column.COLUMN_WIDTH`. */
const COLUMN_WIDTH = 80;

/** osu!lazer `Column.SPECIAL_COLUMN_WIDTH`. 7키 중앙은 **좁다.**
 *
 * 엄지는 가장 굵은 손가락이지만 컬럼은 좁게 잡는 게 관례다. 중앙이 넓으면
 * 무대 가운데가 벌어져 좌우 손의 기준선이 흐려진다.
 */
const SPECIAL_COLUMN_WIDTH = 70;

/** 새끼손가락 컬럼. MAIN 보다 살짝 좁게 둬서 바깥으로 갈수록 조여 보이게 한다. */
const SIDE_COLUMN_WIDTH = 74;

/** osu!lazer `Stage.COLUMN_SPACING`. */
const COLUMN_SPACING = 1;

function widthFor(semantic: LaneSemantic): number {
  if (semantic === "CENTER") return SPECIAL_COLUMN_WIDTH;
  if (semantic.startsWith("SIDE_")) return SIDE_COLUMN_WIDTH;
  return COLUMN_WIDTH;
}

/** 손가락으로 색을 나눈다. 레인 번호가 아니라 어느 손가락인지가 읽는 단위다. */
function colorFor(semantic: LaneSemantic): number {
  if (semantic.startsWith("SIDE_")) return 0x67e8f9;
  if (semantic === "CENTER") return 0xa78bfa;
  if (semantic === "MAIN_2" || semantic === "MAIN_3") return 0x8b9cf6;
  return 0x5eead4;
}

function backgroundFor(semantic: LaneSemantic): number {
  if (semantic === "CENTER") return 0x181529;
  if (semantic.startsWith("SIDE_")) return 0x0e1522;
  return semantic === "MAIN_2" || semantic === "MAIN_3" ? 0x141626 : 0x101522;
}

/** 무대를 배치한다.
 *
 * **컬럼은 고정 폭이고 무대는 가운데 정렬된다.** 컨테이너를 채우려고 늘리면
 * 레인이 100px 을 넘어가면서 노트가 판때기가 되고, 위에서 아래로 흐르는
 * 리듬게임이 아니라 표처럼 보인다. osu!lazer 의 `Stage` 도 같은 이유로
 * `AutoSizeAxes.X` 다.
 *
 * 컨테이너가 무대보다 좁을 때만 비율을 유지한 채 줄인다.
 */
export function layoutStage(
  totalWidth: number,
  semantics: readonly LaneSemantic[],
): StageGeometry {
  if (!Number.isFinite(totalWidth) || totalWidth <= 0) {
    throw new Error("lane layout width must be positive");
  }
  if (semantics.length === 0) throw new Error("at least one lane semantic is required");

  const naturalWidths = semantics.map(widthFor);
  const spacing = COLUMN_SPACING * (semantics.length - 1);
  const naturalWidth = naturalWidths.reduce((sum, width) => sum + width, 0) + spacing;
  const scale = Math.min(1, totalWidth / naturalWidth);
  const stageWidth = naturalWidth * scale;
  const left = (totalWidth - stageWidth) / 2;

  let x = left;
  const lanes = semantics.map((semantic, index) => {
    const width = naturalWidths[index] * scale;
    const lane: LaneGeometry = {
      index,
      semantic,
      x,
      width,
      color: colorFor(semantic),
      backgroundColor: backgroundFor(semantic),
    };
    x += width + COLUMN_SPACING * scale;
    return lane;
  });

  return { lanes, left, right: left + stageWidth, width: stageWidth };
}

/** 이전 호출부 호환. 레인만 필요할 때 쓴다. */
export function layoutLanes(
  totalWidth: number,
  semantics: readonly LaneSemantic[],
): readonly LaneGeometry[] {
  return layoutStage(totalWidth, semantics).lanes;
}

/** 배속 1.0 에서 노트가 1ms 동안 움직이는 픽셀 수.
 *
 * NoteRenderer 와 설정 화면이 같은 값을 봐야 표시되는 노출 시간이 실제와 맞는다.
 */
export const NOTE_PX_PER_MS = 0.6;

/** 배속 1.0 에서 노트가 화면을 흐르는 시간(ms).
 *
 * 판정선 위 활주로 높이를 속도로 나눈 값이다. 창 크기가 바뀌면 같이 바뀐다.
 * 상수로 박아 두면 창 높이가 다를 때 표시가 거짓말을 하고, 배속을 그 숫자로
 * 맞추는 사용자의 입력 보정이 통째로 어긋난다.
 */
export function approachMsAt1x(judgeLineY: number, pxPerMs: number = NOTE_PX_PER_MS): number {
  return judgeLineY / pxPerMs;
}
