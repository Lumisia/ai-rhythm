import "./app.css";
import { LocalPlaytestPage } from "../pages/LocalPlaytestPage";

export function App() {
  return (
    <main className="app-shell">
      <header className="top-rail">
        <a className="wordmark" href="#main-content" aria-label="AI Rhythm 홈">
          <span className="wordmark-mark" aria-hidden="true">
            AR
          </span>
          <span>AI RHYTHM / LAB</span>
        </a>
        <div className="system-state" aria-label="도구 상태">
          <span className="state-lamp" aria-hidden="true" />
          LOCAL REVIEW DECK
        </div>
      </header>

      <div id="main-content"><LocalPlaytestPage /></div>

      <footer className="bottom-rail">
        <span>JUDGMENT PRESET / LENIENT</span>
        <span>AUDIO CLOCK / WEB AUDIO</span>
        <span className="build-number">BUILD 00.1</span>
      </footer>
    </main>
  );
}
