import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: "../_svelte",
    emptyOutDir: true,
    target: "esnext",
  },
  optimizeDeps: {
    exclude: ["@huggingface/transformers", "@mediapipe/tasks-genai"],
  },
  worker: {
    format: "es",
  },
  // Cross-Origin Isolation enables WebGPU + threaded WASM + SharedArrayBuffer.
  // Required by both transformers.js (ONNX Runtime threaded WASM) and the
  // MediaPipe LLM Inference Web runtime (ADR-2605190824).
  // `credentialless` (vs `require-corp`) lets cross-origin asset URLs (HF
  // model bundles, jsdelivr MediaPipe WASM fileset) load without explicit
  // CORP headers — they're fetched without credentials.
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
});
