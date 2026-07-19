/**
 * tools.ts — Daemon-side ReAct tool registry.
 *
 * Subset of the browser tool surface (ADR-2605191129). Long-term memory
 * tools (remember / recall_long_term) are intentionally absent here —
 * they're browser-IndexedDB-bound and will be re-added once the
 * substrate-side MstCheckpointSaver lands (ADR-2605171800).
 *
 * Authoritative ADR: 90-docs/adr/2605191229-ameno-daemon-path-a-bun-langgraph.md
 */
import type { ChatMessage } from "./types.js";

export interface ToolContext {
  /** All messages so far this graph turn, INCLUDING the new user message. */
  messages: ChatMessage[];
}

export interface ToolDef {
  name: string;
  description: string;
  argSpec: string;
  execute(args: unknown, ctx: ToolContext): Promise<string>;
}

function trunc(s: string, n = 500): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

const NOW_TOOL: ToolDef = {
  name: "now",
  description: "Returns the current UTC time as ISO 8601.",
  argSpec: "{}",
  async execute() {
    return new Date().toISOString();
  },
};

/**
 * Lexical (token-overlap) recall over recent thread messages. We deliberately
 * stay lexical here — the daemon does not host a sentence-embedding model in
 * v0.1 (Ollama embed and MiniLM 384-d are dim-incompatible with the browser
 * vault). Embedding-based recall is browser-side until substrate sync lands.
 */
const RECALL_TOOL: ToolDef = {
  name: "recall",
  description:
    "Lexical search over PRIOR messages in this thread. Returns up to 3 best matches by token overlap.",
  argSpec: '{"query": string}',
  async execute(rawArgs: unknown, ctx: ToolContext): Promise<string> {
    const args = (rawArgs ?? {}) as { query?: unknown };
    const query = typeof args.query === "string" ? args.query : "";
    if (!query) return "error: missing 'query' argument";
    const history = ctx.messages.slice(0, -1).filter((m) => m.role !== "system");
    if (history.length === 0) return "no prior messages to search";
    const tok = (s: string): Set<string> =>
      new Set(s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);
    const q = tok(query);
    const scored = history.map((m) => {
      const text = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
      const t = tok(text);
      let inter = 0;
      for (const w of q) if (t.has(w)) inter++;
      const union = q.size + t.size - inter;
      const score = union === 0 ? 0 : inter / union;
      return { score, msg: m, text };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored
      .slice(0, 3)
      .map(
        ({ score, msg, text }, i) =>
          `[${i + 1}] (${msg.role}, jaccard=${score.toFixed(3)}) ${trunc(text, 200)}`,
      )
      .join("\n");
  },
};

const WIKIPEDIA_TOOL: ToolDef = {
  name: "wikipedia",
  description:
    "Fetch a short Wikipedia summary by article title. English Wikipedia only. Use underscored or spaced titles.",
  argSpec: '{"title": string}',
  async execute(rawArgs: unknown): Promise<string> {
    const args = (rawArgs ?? {}) as { title?: unknown };
    const title = typeof args.title === "string" ? args.title.trim() : "";
    if (!title) return "error: missing 'title' argument";
    const url =
      "https://en.wikipedia.org/api/rest_v1/page/summary/" +
      encodeURIComponent(title.replace(/ /g, "_"));
    const resp = await fetch(url, { headers: { accept: "application/json" } });
    if (!resp.ok) return `error: HTTP ${resp.status} fetching Wikipedia '${title}'`;
    const body = (await resp.json()) as { extract?: unknown; title?: unknown };
    const extract = typeof body.extract === "string" ? body.extract : "";
    const t = typeof body.title === "string" ? body.title : title;
    if (!extract) return `no extract found for '${t}'`;
    return `${t}\n${trunc(extract, 480)}`;
  },
};

export const TOOLS: Record<string, ToolDef> = {
  now: NOW_TOOL,
  recall: RECALL_TOOL,
  wikipedia: WIKIPEDIA_TOOL,
};

export interface ParsedToolCall {
  raw: string;
  name: string;
  args: unknown;
  parseError: string | null;
}

const TOOL_TAG_RE = /<tool>\s*(\{[\s\S]*?\})\s*<\/tool>/g;

export function parseToolCalls(text: string): ParsedToolCall[] {
  const out: ParsedToolCall[] = [];
  for (const match of text.matchAll(TOOL_TAG_RE)) {
    const raw = match[0];
    const body = match[1];
    try {
      const obj = JSON.parse(body) as { name?: unknown; args?: unknown };
      const name = typeof obj.name === "string" ? obj.name : "";
      out.push({ raw, name, args: obj.args ?? {}, parseError: name ? null : "missing 'name'" });
    } catch (e) {
      out.push({
        raw,
        name: "",
        args: {},
        parseError: e instanceof Error ? e.message : String(e),
      });
    }
  }
  return out;
}

export function stripToolMarkup(text: string): string {
  return text.replace(TOOL_TAG_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

export async function executeToolCall(
  call: ParsedToolCall,
  ctx: ToolContext,
): Promise<string> {
  if (call.parseError) return `error: ${call.parseError}`;
  const tool = TOOLS[call.name];
  if (!tool) return `error: unknown tool '${call.name}'. Available: ${Object.keys(TOOLS).join(", ")}`;
  try {
    return await tool.execute(call.args, ctx);
  } catch (e) {
    return `error: tool '${call.name}' threw: ${e instanceof Error ? e.message : String(e)}`;
  }
}

export function formatToolsForPrompt(): string {
  const lines = [
    "You can call tools by emitting EXACTLY this format on its own:",
    '<tool>{"name":"<tool_name>","args":{...}}</tool>',
    "You may emit a tool call instead of a final answer. After tool results " +
      "come back, you can emit another call or answer directly. Available tools:",
  ];
  for (const t of Object.values(TOOLS)) {
    lines.push(`- ${t.name}: ${t.description} args: ${t.argSpec}`);
  }
  return lines.join("\n");
}

export function formatToolHistory(
  history: Array<{ name: string; args: unknown; result: string }>,
): string {
  if (history.length === 0) return "";
  const lines = ["Tool calls already made this turn:"];
  for (const h of history) {
    const argStr = JSON.stringify(h.args ?? {});
    lines.push(`- ${h.name}(${argStr}) → ${trunc(h.result, 200)}`);
  }
  lines.push(
    "Either emit another <tool>{...}</tool> if you need more information, " +
      "or give the final answer directly. Do not repeat a tool call with the same arguments.",
  );
  return lines.join("\n");
}
