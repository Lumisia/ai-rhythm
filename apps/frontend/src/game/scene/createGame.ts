import Phaser from "phaser";

import { RhythmScene, type RhythmSceneSession } from "./RhythmScene";

export function createGame(container: HTMLElement, session: RhythmSceneSession): Phaser.Game {
  const width = container.clientWidth || 720;
  const height = container.clientHeight || 720;
  return new Phaser.Game({
    type: Phaser.AUTO,
    parent: container,
    width,
    height,
    backgroundColor: "#151923",
    banner: false,
    input: false,
    audio: { noAudio: true },
    antialias: true,
    scale: {
      mode: Phaser.Scale.RESIZE,
      autoCenter: Phaser.Scale.CENTER_BOTH,
    },
    scene: new RhythmScene(session),
  });
}
