/** Phaser depth 배치.
 *
 * 리터럴이 네 파일에 흩어져 있으면 어느 층이 비었는지 알려고 전부 열어야 한다.
 * HIT_EFFECT 를 NOTE 위, HUD 아래에 두어 이펙트가 노트를 가리지 않고
 * 계기 판독을 덮지도 않게 한다.
 */
export const DEPTH = {
  STAGE_BACKGROUND: 0,
  FEVER_GLOW: 1,
  LANE_HIGHLIGHT: 2,
  KEY_LABEL: 4,
  NOTE: 10,
  HIT_EFFECT: 15,
  HUD_GRAPHICS: 20,
  HUD_COMBO_TEXT: 23,
  HUD_TEXT: 24,
  OVERLAY: 30,
} as const;
