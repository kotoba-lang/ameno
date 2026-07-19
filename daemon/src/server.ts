/**
 * server.ts — ameno headless daemon entrypoint.
 *
 * Hono HTTP server hosting the same LangGraph (reflection + active
 * inference + ReAct tools) as the browser appview, but driven by Ollama
 * instead of MediaPipe. State persists to ${AMENO_HOME}/checkpointer.json
 * via FileCheckpointer.
 *
 * ENV:
 *   AMENO_HOME     base directory for state (default: ~/.ameno)
 *   AMENO_PORT     listen port (default: 12480)
 *   AMENO_HOST     listen host (default: 127.0.0.1)
 *   AMENO_MODEL    Ollama model name (default: gemma3:4b)
 *   OLLAMA_BASE_URL  Ollama endpoint (default: http://localhost:11434)
 *
 * Authoritative ADR: 90-docs/adr/2605191229-ameno-daemon-path-a-bun-langgraph.md
 */
import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync } from "node:fs";
import { FileCheckpointer } from "./file-checkpointer.js";
import { getAmenoDaemonGraph, invokeDaemon, type GraphChunk } from "./graph.js";
import { issueNonce, verifyDidSig } from "./did-auth.js";
import { checkOllamaReady, getDefaultModel, getOllamaBase } from "./ollama-runtime.js";
import {
  getDaemonSnapshot,
  getWorkerDid,
  noteBriefProcessed,
  noteError,
} from "./daemon-identity.js";

const AMENO_HOME = process.env.AMENO_HOME ?? join(homedir(), ".ameno");
const CHECKPOINTER_PATH = join(AMENO_HOME, "checkpointer.json");
const DID_PATH = join(AMENO_HOME, "worker-did");
const PORT = Number(process.env.AMENO_PORT ?? "12480");
const HOST = process.env.AMENO_HOST ?? "127.0.0.1";
/** When set, every request except `/healthz` must carry an
 *  `Authorization: Bearer <AMENO_AUTH_TOKEN>` header. Empty / unset =
 *  unauthenticated mode (localhost-only deployments). ADR-2605191407 §sec. */
const AUTH_TOKEN = process.env.AMENO_AUTH_TOKEN ?? "";

mkdirSync(AMENO_HOME, { recursive: true });
const checkpointer = new FileCheckpointer(CHECKPOINTER_PATH);
const workerDid = getWorkerDid(DID_PATH);

const app = new Hono();
app.use("*", cors({ origin: "*", allowHeaders: ["authorization", "content-type"] }));

// Auth middleware. Accepts either:
//   1. `Authorization: Bearer <AMENO_AUTH_TOKEN>` (legacy, ADR-2605191407)
//   2. `Authorization: DIDSig <did:key>:<nonce_id>:<sig>` (ADR-2605191657)
//
// `/healthz` and `/auth/nonce` are exempt — the former for liveness
// probes, the latter because it's the bootstrap step of the DIDSig
// flow.
app.use("*", async (c, next) => {
  const path = c.req.path;
  if (path === "/healthz" || path === "/auth/nonce") return next();
  if (!AUTH_TOKEN) {
    // No bearer token configured — try DIDSig if present, else allow
    // (loopback dev). DIDSig is gracefully ignored when malformed so
    // the loopback path stays painless.
    const auth = c.req.header("authorization");
    if (auth?.startsWith("DIDSig ")) {
      const r = verifyDidSig(auth);
      if (!r.ok) return c.json({ error: r.error ?? "unauthorized" }, 401);
    }
    return next();
  }
  const header = c.req.header("authorization") ?? "";
  if (header.startsWith("DIDSig ")) {
    const r = verifyDidSig(header);
    if (!r.ok) return c.json({ error: r.error ?? "unauthorized" }, 401);
    return next();
  }
  if (header === `Bearer ${AUTH_TOKEN}`) return next();
  return c.json({ error: "unauthorized" }, 401);
});

// Nonce issue — single-use, 60s TTL. ADR-2605191657.
app.get("/auth/nonce", (c) => c.json(issueNonce()));

// ── Health + identity ────────────────────────────────────────────────────

app.get("/healthz", (c) => c.json({ status: "ok", workerDid }));

app.get("/workerInfo", async (c) => {
  const snap = getDaemonSnapshot(workerDid);
  const ollama = await checkOllamaReady(getDefaultModel());
  return c.json({
    ...snap,
    model: getDefaultModel(),
    ollamaBase: getOllamaBase(),
    ollamaReachable: ollama.reachable,
    ollamaModelInstalled: ollama.modelInstalled,
    home: AMENO_HOME,
  });
});

// ── Invoke / stream ──────────────────────────────────────────────────────

interface InvokeBody {
  messages?: unknown;
  maxIterations?: unknown;
  activeInference?: unknown;
  toolsEnabled?: unknown;
}

function parseBody(body: InvokeBody) {
  const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> =
    Array.isArray(body.messages)
      ? (body.messages as Array<{ role?: string; content?: string }>).map((m) => {
          const role: "system" | "user" | "assistant" =
            m.role === "system" || m.role === "user" || m.role === "assistant" ? m.role : "user";
          return {
            role,
            content: typeof m.content === "string" ? m.content : "",
          };
        })
      : [];
  return {
    messages,
    maxIterations: typeof body.maxIterations === "number" ? body.maxIterations : 0,
    activeInference: body.activeInference === true,
    toolsEnabled: body.toolsEnabled !== false,
  };
}

app.post("/threads/:tid/invoke", async (c) => {
  const tid = c.req.param("tid");
  const body = (await c.req.json()) as InvokeBody;
  const opts = parseBody(body);
  let tokens = 0;
  try {
    const finalDraft = await invokeDaemon({
      ...opts,
      threadId: tid,
      checkpointer,
      onChunk: (chunk: GraphChunk) => {
        if (chunk.type === "stats" && chunk.phase === "generate") {
          tokens = chunk.stats.totalTokens;
        }
      },
    });
    noteBriefProcessed(tokens);
    return c.json({ thread_id: tid, draft: finalDraft });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    noteError(msg);
    return c.json({ error: msg }, 500);
  }
});

app.post("/threads/:tid/stream", async (c) => {
  const tid = c.req.param("tid");
  const body = (await c.req.json()) as InvokeBody;
  const opts = parseBody(body);

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (payload: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      };
      let tokens = 0;
      try {
        const finalDraft = await invokeDaemon({
          ...opts,
          threadId: tid,
          checkpointer,
          onChunk: (chunk) => {
            if (chunk.type === "stats" && chunk.phase === "generate") {
              tokens = chunk.stats.totalTokens;
            }
            send(chunk);
          },
        });
        noteBriefProcessed(tokens);
        send({ type: "done", draft: finalDraft });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        noteError(msg);
        send({ type: "error", error: msg });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-accel-buffering": "no",
    },
  });
});

/**
 * Return parsed graph state for a thread. ADR-2605191645 — browser
 * viewer mode pulls this to seed its local message list when the user
 * switches into daemon mode for the first time.
 */
app.get("/threads/:tid/state", async (c) => {
  const tid = c.req.param("tid");
  const graph = getAmenoDaemonGraph(checkpointer);
  const snapshot = await graph.getState({ configurable: { thread_id: tid } });
  if (!snapshot) return c.json({ thread_id: tid, values: null });
  return c.json({ thread_id: tid, values: snapshot.values ?? null });
});

// ── Boot ────────────────────────────────────────────────────────────────

const banner = `ameno-daemon listening on http://${HOST}:${PORT}
  did:        ${workerDid}
  home:       ${AMENO_HOME}
  ollama:     ${getOllamaBase()} (model: ${getDefaultModel()})`;
console.log(banner);

// SIGTERM / SIGINT: flush checkpointer before exit so we don't lose
// debounced writes.
function gracefulShutdown(sig: string): void {
  console.log(`\n${sig} received, flushing checkpointer…`);
  checkpointer.flushNow();
  process.exit(0);
}
process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

serve({ fetch: app.fetch, hostname: HOST, port: PORT });
