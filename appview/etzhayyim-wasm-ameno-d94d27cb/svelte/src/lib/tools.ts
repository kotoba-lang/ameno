/**
 * tools.ts — Browser-local tool registry for the ameno agent.
 *
 * ReAct format: model emits `<tool>{"name": "...", "args": {...}}</tool>`.
 * We parse with a regex, JSON.parse the body, dispatch to the registry,
 * append the result to state.toolHistory, and re-enter the generate node
 * with a system block describing what happened.
 *
 * Authoritative ADR: 90-docs/adr/2605191129-ameno-browser-tool-use-react.md
 */
import { cosine, embed, isEmbeddingReady } from "./embedding";
import type { ChatMessage } from "./inference";
import { saveMemory, searchMemory } from "./memory-vault";

/** Context passed to a tool's `execute()`. Curated by the graph node. */
export interface ToolContext {
  /** All messages so far this graph turn, INCLUDING the new user message. */
  messages: ChatMessage[];
}

/** Tool definition. `args` and `result` are intentionally `unknown` here so
 *  the registry stays a uniform map; individual tools cast in their body. */
export interface ToolDef {
  name: string;
  /** Short single-line description. Shown to the LLM in the system prompt. */
  description: string;
  /** JSON schema fragment, prose-style, for the prompt. */
  argSpec: string;
  execute(args: unknown, ctx: ToolContext): Promise<string>;
}

/** Truncate to ~500 chars so tool results don't blow the context budget. */
function trunc(s: string, n = 500): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

const NOW_TOOL: ToolDef = {
  name: "now",
  description: "Returns the current UTC time as ISO 8601.",
  argSpec: "{}  // no arguments",
  async execute() {
    return new Date().toISOString();
  },
};

const RECALL_TOOL: ToolDef = {
  name: "recall",
  description:
    "Semantic search over PRIOR messages in this conversation. " +
    "Requires the MiniLM embedding pipeline (turn on 'embedding' surprise mode). " +
    "Returns up to 3 best matches with a similarity score.",
  argSpec: '{"query": string}',
  async execute(rawArgs: unknown, ctx: ToolContext): Promise<string> {
    const args = (rawArgs ?? {}) as { query?: unknown };
    const query = typeof args.query === "string" ? args.query : "";
    if (!query) return "error: missing 'query' argument";
    if (!isEmbeddingReady()) {
      return "error: embedding pipeline not loaded. Tell the user to enable 'embedding' surprise mode first.";
    }
    // Drop the most recent user message (it's "now"; recall is about history).
    const history = ctx.messages.slice(0, -1).filter((m) => m.role !== "system");
    if (history.length === 0) return "no prior messages to search";

    const qv = await embed(query);
    const scored: Array<{ score: number; msg: ChatMessage }> = [];
    for (const m of history) {
      const text = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
      const v = await embed(text);
      scored.push({ score: cosine(qv, v), msg: m });
    }
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, 3);
    return top
      .map(
        ({ score, msg }, i) =>
          `[${i + 1}] (${msg.role}, sim=${score.toFixed(3)}) ${trunc(
            typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
            200,
          )}`,
      )
      .join("\n");
  },
};

const WIKIPEDIA_TOOL: ToolDef = {
  name: "wikipedia",
  description:
    "Fetch a short Wikipedia summary by article title. English Wikipedia only. " +
    "Use underscored or spaced titles, e.g. 'Active_inference' or 'Active inference'.",
  argSpec: '{"title": string}',
  async execute(rawArgs: unknown): Promise<string> {
    const args = (rawArgs ?? {}) as { title?: unknown };
    const title = typeof args.title === "string" ? args.title.trim() : "";
    if (!title) return "error: missing 'title' argument";
    const url =
      "https://en.wikipedia.org/api/rest_v1/page/summary/" +
      encodeURIComponent(title.replace(/ /g, "_"));
    const resp = await fetch(url, { credentials: "omit", headers: { accept: "application/json" } });
    if (!resp.ok) {
      return `error: HTTP ${resp.status} fetching Wikipedia '${title}'`;
    }
    const body = (await resp.json()) as { extract?: unknown; title?: unknown };
    const extract = typeof body.extract === "string" ? body.extract : "";
    const t = typeof body.title === "string" ? body.title : title;
    if (!extract) return `no extract found for '${t}'`;
    return `${t}\n${trunc(extract, 480)}`;
  },
};

const REMEMBER_TOOL: ToolDef = {
  name: "remember",
  description:
    "Persist a fact or note into your long-term encrypted memory vault. " +
    "Use this when the user states something you want to recall across " +
    "future conversations. Requires the MiniLM embedding pipeline.",
  argSpec: '{"content": string, "tags"?: string[]}',
  async execute(rawArgs: unknown): Promise<string> {
    const args = (rawArgs ?? {}) as { content?: unknown; tags?: unknown };
    const content = typeof args.content === "string" ? args.content : "";
    if (!content) return "error: missing 'content' argument";
    const tags = Array.isArray(args.tags)
      ? args.tags.filter((t): t is string => typeof t === "string")
      : [];
    if (!isEmbeddingReady()) {
      return "error: long-term memory requires the MiniLM embedding pipeline. Tell the user to enable 'embedding' surprise mode first.";
    }
    try {
      const id = await saveMemory(content, tags);
      return `saved memory #${id}` + (tags.length ? ` [${tags.join(", ")}]` : "");
    } catch (e) {
      return `error: ${e instanceof Error ? e.message : String(e)}`;
    }
  },
};

const RECALL_LONG_TERM_TOOL: ToolDef = {
  name: "recall_long_term",
  description:
    "Semantic search across the long-term encrypted memory vault. " +
    "Returns top matches with similarity scores and timestamps.",
  argSpec: '{"query": string, "topK"?: number}',
  async execute(rawArgs: unknown): Promise<string> {
    const args = (rawArgs ?? {}) as { query?: unknown; topK?: unknown };
    const query = typeof args.query === "string" ? args.query : "";
    if (!query) return "error: missing 'query' argument";
    const topK = typeof args.topK === "number" ? Math.max(1, Math.min(10, args.topK)) : 3;
    if (!isEmbeddingReady()) {
      return "error: long-term memory recall requires the MiniLM embedding pipeline.";
    }
    try {
      const hits = await searchMemory(query, topK);
      if (hits.length === 0) return "no long-term memories found";
      return hits
        .map((h) => {
          const ageMs = Date.now() - h.createdAt;
          const ago =
            ageMs < 60_000
              ? `${Math.floor(ageMs / 1000)}s`
              : ageMs < 3_600_000
              ? `${Math.floor(ageMs / 60_000)}m`
              : ageMs < 86_400_000
              ? `${Math.floor(ageMs / 3_600_000)}h`
              : `${Math.floor(ageMs / 86_400_000)}d`;
          const tagPart = h.tags.length ? ` [${h.tags.join(",")}]` : "";
          return `[${h.id}] (sim=${h.similarity.toFixed(3)}, ${ago} ago)${tagPart} ${trunc(h.content, 180)}`;
        })
        .join("\n");
    } catch (e) {
      return `error: ${e instanceof Error ? e.message : String(e)}`;
    }
  },
};

/** Public registry. Mutable so apps could register more tools at runtime,
 *  but the default set is frozen here as a sane baseline. */
export const TOOLS: Record<string, ToolDef> = {
  now: NOW_TOOL,
  recall: RECALL_TOOL,
  wikipedia: WIKIPEDIA_TOOL,
  remember: REMEMBER_TOOL,
  recall_long_term: RECALL_LONG_TERM_TOOL,
};

/** Parsed tool call extracted from a model draft. */
export interface ParsedToolCall {
  /** Original `<tool>...</tool>` substring, used to strip from final reply. */
  raw: string;
  name: string;
  args: unknown;
  /** Set when the JSON body could not be parsed. */
  parseError: string | null;
}

const TOOL_TAG_RE = /<tool>\s*(\{[\s\S]*?\})\s*<\/tool>/g;

/**
 * Scan a model draft for `<tool>{...}</tool>` blocks and return them in
 * the order they appear. Malformed JSON bodies are returned with a
 * `parseError` so the executor can report a clean error back to the model
 * instead of silently dropping the call.
 */
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

/** Strip the literal `<tool>...</tool>` blocks from a draft so the visible
 *  reply does not contain markup. */
export function stripToolMarkup(text: string): string {
  return text.replace(TOOL_TAG_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/** Run a single parsed tool call. Always resolves; failures become strings
 *  prefixed `error:` so the model can react gracefully. */
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

/** System-prompt block enumerating available tools. Inserted by graph.ts. */
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

/** System-prompt block summarising tool calls already made this turn. */
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
      "or give the final answer directly. Do not repeat a tool call with " +
      "the same arguments.",
  );
  return lines.join("\n");
}
