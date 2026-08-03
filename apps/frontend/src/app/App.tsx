import "./app.css";
import { LocalPlaytestPage } from "../pages/LocalPlaytestPage";

export function App() {
  return (
    <main className="game-root">
      <div className="ambient-grid" aria-hidden="true" />
      <a className="corner-brand" href="#main-content" aria-label="AI Rhythm 홈">
        <span className="brand-pulse" aria-hidden="true" />
        <span>AI RHYTHM</span>
        <small>LOCAL PLAYTEST</small>
      </a>
      <div id="main-content"><LocalPlaytestPage /></div>
    </main>
  );
}
