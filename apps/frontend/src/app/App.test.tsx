import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "./App";

describe("App", () => {
  it("renders the local playtest entry point", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "채보 플레이테스트" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "실행 폴더 선택" })).toBeEnabled();
  });
});
