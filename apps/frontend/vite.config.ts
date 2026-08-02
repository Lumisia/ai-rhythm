import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@contracts": fileURLToPath(new URL("../../packages", import.meta.url)),
    },
  },
  server: { fs: { allow: [repositoryRoot] } },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
  },
});
