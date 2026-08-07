import Phaser from "phaser";

import { RhythmScene, type RhythmSceneSession } from "./RhythmScene";

export interface CreatedGame {
  game: Phaser.Game;
  /** 씬 인스턴스. 구간 반복 재시작처럼 씬이 소유한 상태를 밖에서
   * 되돌려야 할 때 쓴다. 씬 매니저에서 키로 찾아오면 문자열 의존이 생긴다. */
  scene: RhythmScene;
}

export function createGame(container: HTMLElement, session: RhythmSceneSession): CreatedGame {
  const width = container.clientWidth || 720;
  const height = container.clientHeight || 720;
  const scene = new RhythmScene(session);
  const game = new Phaser.Game({
    type: Phaser.AUTO,
    parent: container,
    width,
    height,
    backgroundColor: "#0d1016",
    banner: false,
    input: false,
    audio: { noAudio: true },
    antialias: true,
    scale: {
      mode: Phaser.Scale.RESIZE,
      autoCenter: Phaser.Scale.CENTER_BOTH,
    },
    scene,
  });
  return { game, scene };
}
