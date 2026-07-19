/**
 * viewer-mode.ts — Browser thin-client connector to the ameno daemon.
 *
 * Wraps the daemon's `POST /threads/:tid/stream` SSE endpoint so the
 * svelte appview can route a chat turn through Path A (TS, port 12480)
 * or Path B (Python, port 12481) instead of running LangGraph locally.
 *
 * The chunk shape is intentionally identical to local `invokeAmeno`
 * (ADR-2605191000) — only the transport differs — so App.svelte's
 * onChunk handler stays bit-exact between modes.
 *
 * Authoritative ADR: 90-docs/adr/2605191407-ameno-browser-viewer-mode.md
 */
import type { ChatMessage } from "./inference";
import type { GraphChunk } from "./graph";
import { buildDidSigHeader } from "./did-auth";

/** Daemon `/workerInfo` payload — fields are advisory; tolerate missing keys. */
export interface DaemonWorkerInfo {
  did: string;
  uptimeMs: number;
  briefsPerMinute?: number;
  totalBriefs?: number;
  totalTokensDecoded?: number;
  lastError?: string | null;
  model?: string;
  ollamaBase?: string;
  ollamaReachable?: boolean;
  ollamaModelInstalled?: boolean;
  home?: string;
  kind?: string; // "path-b-python" for Path B; undefined for Path A
}

/** Auth header strategy for daemon calls. ADR-2605191657:
 *  - explicit Bearer token wins when supplied (operator-controlled)
 *  - otherwise we try DIDSig (single-use signed nonce, no shared secret)
 *  - if DIDSig also fails (no /auth/nonce, offline daemon), we send the
 *    request unauthenticated — daemons configured without
 *    `AMENO_AUTH_TOKEN` accept that path.
 *
 *  Callers should `await` this; the DIDSig path makes a network round-
 *  trip to fetch a nonce. */
async function authHeaders(
  baseUrl: string,
  authToken: string | undefined,
  signal?: AbortSignal,
): Promise<Record<string, string>> {
  if (authToken) return { authorization: `Bearer ${authToken}` };
  try {
    const sig = await buildDidSigHeader(baseUrl, signal);
    return { authorization: sig };
  } catch {
    return {};
  }
}

/** Ping daemon `/healthz` to validate URL and CORS reachability. */
export async function pingDaemon(
  baseUrl: string,
  signal?: AbortSignal,
  authToken?: string,
): Promise<boolean> {
  try {
    // /healthz is auth-exempt on both daemons so we deliberately skip
    // the DIDSig round-trip here.
    const r = await fetch(stripTrailingSlash(baseUrl) + "/healthz", {
      method: "GET",
      credentials: "omit",
      headers: authToken ? { authorization: `Bearer ${authToken}` } : {},
      signal,
    });
    return r.ok;
  } catch {
    return false;
  }
}

/**
 * Pull the parsed graph state for `threadId` from the daemon and return
 * its message list (ChatMessage shape). Empty array on any failure or
 * empty thread. ADR-2605191645.
 */
export async function pullThreadMessages(
  baseUrl: string,
  threadId: string,
  signal?: AbortSignal,
  authToken?: string,
): Promise<ChatMessage[]> {
  try {
    const r = await fetch(
      stripTrailingSlash(baseUrl) + `/threads/${encodeURIComponent(threadId)}/state`,
      {
        method: "GET",
        credentials: "omit",
        headers: await authHeaders(baseUrl, authToken, signal),
        signal,
      },
    );
    if (!r.ok) return [];
    const body = (await r.json()) as { values?: { messages?: unknown } | null };
    const raw = body.values?.messages;
    if (!Array.isArray(raw)) return [];
    const out: ChatMessage[] = [];
    for (const m of raw as Array<{ role?: unknown; content?: unknown }>) {
      const role = m.role;
      if (role !== "system" && role !== "user" && role !== "assistant") continue;
      const content = typeof m.content === "string" ? m.content : "";
      out.push({ role, content });
    }
    return out;
  } catch {
    return [];
  }
}

/** Fetch and return the daemon's workerInfo, or null on any failure. */
export async function fetchWorkerInfo(
  baseUrl: string,
  signal?: AbortSignal,
  authToken?: string,
): Promise<DaemonWorkerInfo | null> {
  try {
    const r = await fetch(stripTrailingSlash(baseUrl) + "/workerInfo", {
      method: "GET",
      credentials: "omit",
      headers: await authHeaders(baseUrl, authToken, signal),
      signal,
    });
    if (!r.ok) return null;
    return (await r.json()) as DaemonWorkerInfo;
  } catch {
    return null;
  }
}

/** Options accepted by `invokeAmenoRemote`. Field semantics mirror
 *  `InvokeAmenoOptions` in graph.ts so the App.svelte caller can build
 *  one payload and choose local vs remote dispatch at the last step. */
export interface InvokeAmenoRemoteOptions {
  baseUrl: string;
  threadId?: string;
  messages: ChatMessage[];
  maxIterations: number;
  activeInference?: boolean;
  toolsEnabled?: boolean;
  /** Bearer token for daemons running with AMENO_AUTH_TOKEN. */
  authToken?: string;
  /** Per-chunk callback. */
  onChunk: (chunk: GraphChunk) => void;
  /** Abort control for `Stop generating`. */
  signal?: AbortSignal;
}

/**
 * Drive one user turn through the remote daemon's SSE stream.
 *
 * Returns the final draft string. Each line `data: {...}\n\n` is parsed
 * and forwarded to `onChunk`. A terminal `{type:"done", draft}` chunk
 * is consumed internally; a `{type:"error", error}` chunk is rethrown.
 */
export async function invokeAmenoRemote(opts: InvokeAmenoRemoteOptions): Promise<string> {
  const tid = encodeURIComponent(opts.threadId ?? "viewer");
  const url = stripTrailingSlash(opts.baseUrl) + `/threads/${tid}/stream`;
  const resp = await fetch(url, {
    method: "POST",
    credentials: "omit",
    headers: {
      "content-type": "application/json",
      accept: "text/event-stream",
      ...(await authHeaders(opts.baseUrl, opts.authToken, opts.signal)),
    },
    body: JSON.stringify({
      messages: opts.messages,
      maxIterations: opts.maxIterations,
      activeInference: opts.activeInference ?? false,
      toolsEnabled: opts.toolsEnabled ?? false,
    }),
    signal: opts.signal,
  });
  if (!resp.ok || !resp.body) {
    const text = await safeReadText(resp);
    if (resp.status === 401) {
      // Distinguish "did not in allowlist" so the UI can surface a
      // useful next-step ("ask the operator to add your did:key").
      // ADR-2605191641.
      if (text.includes("did not in allowlist")) {
        throw new Error(
          "daemon rejected DIDSig: this browser's did:key is not in the daemon's AMENO_ALLOWED_DIDS allowlist",
        );
      }
      throw new Error(`daemon auth failed: ${text.slice(0, 200)}`);
    }
    throw new Error(`daemon stream HTTP ${resp.status}: ${text.slice(0, 200)}`);
  }

  let finalDraft = "";
  for await (const event of iterateSSEEvents(resp.body, opts.signal)) {
    const payload = parseChunk(event);
    if (!payload) continue;
    // `done` / `error` are control envelopes not part of GraphChunk.
    if ((payload as { type?: string }).type === "done") {
      finalDraft = (payload as { draft?: string }).draft ?? "";
      break;
    }
    if ((payload as { type?: string }).type === "error") {
      const msg = (payload as { error?: string }).error ?? "(unknown daemon error)";
      throw new Error(msg);
    }
    opts.onChunk(payload as GraphChunk);
  }
  return finalDraft;
}

// ── helpers ────────────────────────────────────────────────────────────

function stripTrailingSlash(s: string): string {
  return s.endsWith("/") ? s.slice(0, -1) : s;
}

async function safeReadText(resp: Response): Promise<string> {
  try {
    return await resp.text();
  } catch {
    return "";
  }
}

function parseChunk(eventData: string): GraphChunk | { type: string; draft?: string; error?: string } | null {
  if (!eventData) return null;
  try {
    return JSON.parse(eventData) as GraphChunk;
  } catch {
    return null;
  }
}

/**
 * Iterate `data: ...\n\n` events from an SSE response body. Reads as
 * UTF-8 via TextDecoder with stream=true so multi-byte chars at chunk
 * boundaries don't corrupt JSON parsing. Yields the `data:` payload
 * string only; SSE comments / id / event lines are dropped (the daemon
 * does not emit any).
 */
async function* iterateSSEEvents(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncIterable<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    while (true) {
      if (signal?.aborted) {
        try {
          await reader.cancel();
        } catch {
          /* ignore */
        }
        return;
      }
      const { done, value } = await reader.read();
      if (done) {
        if (buffer.length > 0) {
          const tail = extractData(buffer);
          if (tail) yield tail;
        }
        return;
      }
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const eventBlock = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const data = extractData(eventBlock);
        if (data !== null) yield data;
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* already released by cancel */
    }
  }
}

function extractData(block: string): string | null {
  // SSE event block: one or more lines like "data: …" / "data:…" / ":heartbeat".
  // For our daemon we only get a single `data:` line per event. We're lenient
  // for forward-compat: concat all `data:` lines with `\n`.
  const lines = block.split("\n");
  const dataLines: string[] = [];
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
  }
  return dataLines.length > 0 ? dataLines.join("\n") : null;
}
