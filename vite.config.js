import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  root: "./frontend",
  base: "/static/",
  resolve: {
    alias: {
      "@": resolve(__dirname, "frontend"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: Number(process.env.DJANGO_VITE_DEV_SERVER_PORT || 5173),
    watch: {
      usePolling: true,
    },
  },
  build: {
    manifest: "manifest.json",
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        app: resolve(__dirname, "frontend/js/main.jsx"),
      },
    },
  },
});