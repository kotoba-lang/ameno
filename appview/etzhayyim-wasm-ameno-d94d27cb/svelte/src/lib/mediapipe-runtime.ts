/**
 * mediapipe-runtime.ts — Browser-only Gemma 3n runtime via MediaPipe LLM
 * Inference Web (`@mediapipe/tasks-genai`).
 *
 * Parallel to `inference.ts` (transformers.js / ONNX) but isolated so we can
 * dispatch by kernel without entangling the two runtimes. Shape kept
 * intentionally compatible: loadModel / generate / MODELS plus the same
 * GenerationStats / ChatMessage / InferenceState types re-exported from the
 * canonical module.
 *
 * Authoritative ADR: 90-docs/adr/2605190824-ameno-mediapipe-llm-browser-runtime.md
 */
import type { LlmInference } from "@mediapipe/tasks-genai";
import type { ChatMessage, GenerationStats } from "./inference";

/** MediaPipe `.task` bundle descriptor. */
export interface MediaPipeModelMeta {
  id: string;
  displayName: string;
  /** `.task` bundle URL (Hugging Face LiteRT distribution or self-hosted CDN). */
  bundleUrl: string;
  /** ~bytes, advisory only — UI hint for first-load DL size. */
  approxBytes: number;
  /** MediaPipe accepted decode parameters. */
  defaults: {
    maxTokens: number;
    topK: number;
    temperature: number;
    randomSeed: number;
  };
}

/**
 * Distribution URLs target `litert-community/*-litert-lm/{model}-web.task`,
 * the publicly downloadable LiteRT Web bundle (Apache 2.0, ungated). The
 * older `google/gemma-3n-*-litert-preview` repos are HF-gated and require
 * an access token in the model fetch — left out of the catalog until we
 * stand up an HF token proxy through the ameno Worker.
 */
export const MEDIAPIPE_MODELS: Record<string, MediaPipeModelMeta> = {
  "gemma-4-e2b-mediapipe": {
    id: "gemma-4-e2b-mediapipe",
    displayName: "Gemma 4 E2B (MediaPipe LiteRT, ungated)",
    bundleUrl:
      "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task",
    approxBytes: 2_003_697_664,
    defaults: { maxTokens: 1024, topK: 40, temperature: 0.8, randomSeed: 1 },
  },
  "gemma-4-e4b-mediapipe": {
    id: "gemma-4-e4b-mediapipe",
    displayName: "Gemma 4 E4B (MediaPipe LiteRT, ungated)",
    bundleUrl:
      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.task",
    approxBytes: 2_964_324_352,
    defaults: { maxTokens: 1024, topK: 40, temperature: 0.8, randomSeed: 1 },
  },
  // Note on smaller smoke models: ungated `*-web.task` bundles only exist
  // for the Gemma 4 E2B/E4B (Apache 2.0) repos. All `gemma-3-*` and
  // `Gemma3-*` LiteRT-web repos are HF-gated under the Gemma license, and
  // the litert-community Qwen / Phi / TinyLlama repos only ship the
  // `litert-lm` (Android/Linux runtime) bundle format, which is NOT
  // compatible with MediaPipe LLM Inference Web and aborts during
  // `LlmInference.createFromOptions`. Use Gemma 4 E2B for the smallest
  // working smoke; add gated entries here once an HF token proxy is up.
};

/**
 * Latest URL we resolved for the MediaPipe GenAI WASM fileset. We pin a single
 * version here so OPFS caching is stable across reloads. Bump when upgrading
 * the @mediapipe/tasks-genai pin in package.json.
 */
const GENAI_WASM_FILESET_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.27/wasm";

interface ActiveSession {
  modelId: string;
  llm: LlmInference;
}

let active: ActiveSession | null = null;

export function getLoadedMediapipeModel(): string | null {
  return active?.modelId ?? null;
}

/** Cache API namespace for the persisted `.task` bundles. Bump the version
 *  whenever the bundle format changes incompatibly. */
const TASK_CACHE_NAME = "ameno-mediapipe-task-v1";

/**
 * Fetch a `.task` bundle through the Cache API. On a cache hit we return
 * immediately (no network). On a cache miss we stream the body so we can
 * report byte-level progress to the caller, then persist the assembled
 * buffer to the cache for the next reload.
 *
 * Returns a Uint8Array that we hand to MediaPipe via `modelAssetBuffer`,
 * which keeps the model loader fully driven by us — no double-fetch.
 */
async function fetchTaskWithCache(
  url: string,
  approxBytes: number,
  onByteProgress: (loaded: number, total: number, fromCache: boolean) => void,
): Promise<Uint8Array> {
  const cache = await caches.open(TASK_CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) {
    const buf = new Uint8Array(await cached.arrayBuffer());
    onByteProgress(buf.byteLength, buf.byteLength, true);
    return buf;
  }

  const resp = await fetch(url, { credentials: "omit" });
  if (!resp.ok || !resp.body) {
    throw new Error(`Bundle fetch failed: HTTP ${resp.status} ${resp.statusText}`);
  }
  const headerTotal = Number(resp.headers.get("content-length") || "0");
  const total = headerTotal > 0 ? headerTotal : approxBytes;

  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onByteProgress(loaded, total, false);
  }

  const buf = new Uint8Array(loaded);
  {
    let off = 0;
    for (const c of chunks) {
      buf.set(c, off);
      off += c.byteLength;
    }
  }

  // Persist to Cache API. We must Response.clone semantics — create a new
  // Response from the assembled buffer so the cache owns its own copy.
  // Failures here are non-fatal (e.g. quota exceeded); we still proceed.
  try {
    const cacheable = new Response(buf, {
      headers: {
        "content-type":
          resp.headers.get("content-type") || "application/octet-stream",
        "content-length": String(buf.byteLength),
      },
    });
    await cache.put(url, cacheable);
  } catch (e) {
    console.warn("ameno: Cache API put failed (proceeding without cache):", e);
  }

  return buf;
}

/**
 * Load a `.task` bundle into an `LlmInference` session. Caches the bundle
 * via the Cache API so subsequent loads (after a reload, or after model
 * switch + switch-back) hit local storage instead of the network. Reuses
 * the live in-memory session when the same modelId is requested.
 */
export async function loadMediapipeModel(
  onProgress: (pct: number) => void,
  modelId: string,
): Promise<void> {
  const meta = MEDIAPIPE_MODELS[modelId];
  if (!meta) throw new Error(`Unknown MediaPipe model: ${modelId}`);
  if (active?.modelId === modelId) {
    onProgress(100);
    return;
  }

  // Best-effort: ask the browser to mark this origin as persistent so the
  // multi-GB cached bundles aren't first to be evicted under storage pressure.
  if (navigator.storage?.persist) {
    try {
      await navigator.storage.persist();
    } catch {
      /* ignore — purely opportunistic */
    }
  }

  // Dynamic import keeps the @mediapipe bundle out of the initial chunk so the
  // landing page stays light when the user picks a non-MediaPipe kernel.
  const mod = await import("@mediapipe/tasks-genai");
  const { FilesetResolver, LlmInference } = mod;

  onProgress(2);
  const fileset = await FilesetResolver.forGenAiTasks(GENAI_WASM_FILESET_URL);
  onProgress(5);

  // Stage A — bundle fetch with real byte-level progress (5..85).
  const buf = await fetchTaskWithCache(meta.bundleUrl, meta.approxBytes, (loaded, total, fromCache) => {
    if (fromCache) {
      onProgress(85);
      return;
    }
    const ratio = total > 0 ? Math.min(1, loaded / total) : 0;
    onProgress(5 + Math.round(ratio * 80));
  });

  // Stage B — LlmInference init (85..95 coarse since createFromOptions
  // does not stream progress for the WASM-side load of the buffer).
  const initTimer = startCoarseProgress(onProgress, 85, 95, Math.max(buf.byteLength / 4, 1));
  try {
    const llm = await LlmInference.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: buf },
      maxTokens: meta.defaults.maxTokens,
      topK: meta.defaults.topK,
      temperature: meta.defaults.temperature,
      randomSeed: meta.defaults.randomSeed,
    });
    if (active) active.llm.close();
    active = { modelId, llm };
    onProgress(100);
  } finally {
    initTimer.stop();
  }
}

/**
 * Delete the cached `.task` bundles. Call when the user wants to free disk
 * (each MediaPipe bundle is 2–3 GB).
 */
export async function clearMediapipeCache(): Promise<void> {
  await caches.delete(TASK_CACHE_NAME);
}

/** Total bytes currently sitting in the MediaPipe bundle cache. */
export async function getMediapipeCacheSize(): Promise<number> {
  const cache = await caches.open(TASK_CACHE_NAME);
  const keys = await cache.keys();
  let total = 0;
  for (const req of keys) {
    const resp = await cache.match(req);
    if (!resp) continue;
    const len = Number(resp.headers.get("content-length") || "0");
    total += len;
  }
  return total;
}

/**
 * Gemma 1/2/3/4 chat template. Gemma has no `system` role at training time,
 * so the system prompt is folded into the first user turn with a single
 * blank line, NOT a `[SYSTEM]` marker block (the marker confuses the model
 * into self-prompting).
 */
function renderGemmaChat(messages: ChatMessage[]): string {
  const out: string[] = [];
  let pendingSystem = "";
  for (const m of messages) {
    if (m.role === "system") {
      pendingSystem = pendingSystem ? `${pendingSystem}\n\n${m.content}` : m.content;
      continue;
    }
    if (m.role === "user") {
      const body = pendingSystem ? `${pendingSystem}\n\n${m.content}` : m.content;
      pendingSystem = "";
      out.push(`<start_of_turn>user\n${body}<end_of_turn>\n`);
    } else if (m.role === "assistant") {
      out.push(`<start_of_turn>model\n${m.content}<end_of_turn>\n`);
    }
  }
  out.push(`<start_of_turn>model\n`);
  return out.join("");
}

const GEMMA_STOP_RE = /<\s*(?:end_of_turn|start_of_turn|END_OF_TURN|START_OF_TURN|eos|bos)\b/i;
const GEMMA_STRIP_RE = /<\s*(?:end_of_turn|start_of_turn|END_OF_TURN|START_OF_TURN)\b[^>]*>?/gi;

/**
 * Streaming generate. `onToken` is called once per partial chunk emitted by
 * `generateResponse(prompt, listener)` — MediaPipe emits chunks, not single
 * tokens, but the consumer in App.svelte just concatenates so the granularity
 * difference is invisible.
 */
export async function mediapipeGenerate(
  messages: ChatMessage[],
  onToken: (token: string) => void,
): Promise<GenerationStats> {
  if (!active) throw new Error("MediaPipe model not loaded");
  const prompt = renderGemmaChat(messages);
  const started = performance.now();
  let tokenCount = 0;
  // Stop-token state machine. MediaPipe LLM Inference Web 0.10.x has no
  // stop-sequence API and no mid-stream cancel, so once the model emits
  // any flavor of <(start|end)_of_turn> we silently drop the rest of the
  // stream. The underlying decode still runs until maxTokens — bounding
  // wall-clock via meta.defaults.maxTokens is the real lever.
  let stopped = false;
  let buffered = "";

  await active.llm.generateResponse(prompt, (partial, _done) => {
    if (!partial || stopped) return;
    tokenCount++;
    buffered += partial;
    if (GEMMA_STOP_RE.test(buffered)) {
      const clean = buffered.replace(GEMMA_STRIP_RE, "").trimEnd();
      const already = buffered.length - partial.length;
      const newClean = clean.slice(Math.min(already, clean.length));
      if (newClean) onToken(newClean);
      stopped = true;
      return;
    }
    onToken(partial);
  });

  const durationMs = performance.now() - started;
  // MediaPipe doesn't surface decoded token counts directly in 0.10.x. We
  // approximate via emitted chunk count, which is a lower bound and good
  // enough for ameno's tokens/sec display.
  const tokensPerSecond = tokenCount > 0 ? (tokenCount * 1000) / durationMs : 0;
  return {
    durationMs,
    totalTokens: tokenCount,
    tokensPerSecond,
    ragActive: false,
  };
}

/** Best-effort teardown so the next loadModel call can free GPU buffers. */
export function disposeMediapipe(): void {
  if (active) {
    active.llm.close();
    active = null;
  }
}

/**
 * Synthetic progress between [from, to] over the expected DL duration. Stops
 * advancing at `to - 1` so the caller's final `onProgress(100)` always wins.
 */
function startCoarseProgress(
  onProgress: (pct: number) => void,
  from: number,
  to: number,
  approxBytes: number,
): { stop: () => void } {
  const estMs = Math.max(8_000, (approxBytes / 50_000_000) * 1_000);
  const startedAt = performance.now();
  const id = window.setInterval(() => {
    const elapsed = performance.now() - startedAt;
    const ratio = Math.min(elapsed / estMs, 1);
    const pct = Math.min(to - 1, Math.round(from + (to - from) * ratio));
    onProgress(pct);
  }, 250);
  return { stop: () => window.clearInterval(id) };
}
