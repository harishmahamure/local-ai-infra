import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = import.meta.dirname;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://127.0.0.1:8090",
      "/health": "http://127.0.0.1:8090",
      "/ready": "http://127.0.0.1:8090",
    },
  },
  build: {
    outDir: resolve(root, "../control-api/static"),
    emptyOutDir: true,
  },
});
