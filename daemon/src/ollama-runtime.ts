/**
 * ollama-runtime.ts — LLM kernel for the headless daemon.
 *
 * Talks to a local Ollama server (default http://localhost:11434) via the
 * streaming `/api/chat` endpoint. Function shape mirrors the browser
 * `mediapipeGenerate(messages, onToken)` so the daemon's graph nodes can
 * call `runtimeGenerate(...)` without conditionals.
 *
 * Authoritative ADR: 90-docs/adr/2605191229-ameno-daemon-path-a-bun-langgraph.md
 */
import type { ChatMessage, GenerationStats } from "./types.js";

const OLLAMA_BASE = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
const DEFAULT_MODEL = process.env.AMENO_MODEL ?? "gemma3:4b";

interface OllamaChatLine {
  model?: string;
  message?: { role?: string; content?: string };
  done?: boolean;
  done_reason?: string;
  eval_count?: number;
  eval_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  total_duration?: number;
}

/**
 * `GET /api/tags` probe. Returns true when Ollama responds and the
 * configured model is in its model list. Used by the `/workerInfo`
 * endpoint to surface readiness.
 */
export async function checkOllamaReady(model = DEFAULT_MODEL): Promise<{ reachable: boolean; modelInstalled: boolean }> {
  try {
    const r = await fetch(`${OLLAMA_BASE}/api/tags`);
    if (!r.ok) return { reachable: false, modelInstalled: false };
    const body = (await r.json()) as { models?: Array<{ name?: string }> };
    const installed = (body.models ?? []).some((m) => (m.name ?? "").startsWith(model));
    return { reachable: true, modelInstalled: installed };
  } catch {
    return { reachable: false, modelInstalled: false };
  }
}

export interface OllamaGenerateOptions {
  model?: string;
  /** Decode parameters forwarded to Ollama. */
  temperature?: number;
  top_k?: number;
  /** Hard cap on emitted tokens (Ollama `num_predict`). */
  maxTokens?: number;
}

/**
 * Streaming chat generation. Each line from Ollama is a JSON object with a
 * `message.content` delta; we forward those deltas to `onToken` and tally
 * the total. The returned `GenerationStats` is derived from Ollama's
 * final summary line so the numbers match what the daemon's HTTP clients
 * would see if they pinged Ollama directly.
 */
export async function runtimeGenerate(
  messages: ChatMessage[],
  onToken: (token: string) => void,
  opts: OllamaGenerateOptions = {},
): Promise<GenerationStats> {
  const model = opts.model ?? DEFAULT_MODEL;
  const body = {
    model,
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
    stream: true,
    options: {
      temperature: opts.temperature ?? 0.7,
      top_k: opts.top_k ?? 40,
      num_predict: opts.maxTokens ?? 1024,
    },
  };

  const started = performance.now();
  const resp = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`Ollama chat failed: HTTP ${resp.status} ${resp.statusText}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let tokenCount = 0;
  let evalCount = 0;
  let evalDurationNs = 0;
  let totalDurationNs = 0;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      let obj: OllamaChatLine;
      try {
        obj = JSON.parse(line) as OllamaChatLine;
      } catch {
        continue;
      }
      const piece = obj.message?.content;
      if (piece) {
        tokenCount++;
        onToken(piece);
      }
      if (obj.done) {
        evalCount = obj.eval_count ?? evalCount;
        evalDurationNs = obj.eval_duration ?? evalDurationNs;
        totalDurationNs = obj.total_duration ?? totalDurationNs;
      }
    }
  }

  const wallMs = performance.now() - started;
  const decodeMs = evalDurationNs > 0 ? evalDurationNs / 1_000_000 : wallMs;
  const totalDecoded = evalCount > 0 ? evalCount : tokenCount;
  const tps = totalDecoded > 0 && decodeMs > 0 ? (totalDecoded * 1000) / decodeMs : 0;
  return {
    durationMs: totalDurationNs > 0 ? totalDurationNs / 1_000_000 : wallMs,
    totalTokens: totalDecoded,
    tokensPerSecond: tps,
    ragActive: false,
  };
}

export function getDefaultModel(): string {
  return DEFAULT_MODEL;
}

export function getOllamaBase(): string {
  return OLLAMA_BASE;
}
