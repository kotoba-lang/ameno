/**
 * graph.ts — Daemon-side LangGraph (Pregel) runtime.
 *
 * Clone of the svelte appview's graph.ts (ADR-2605191000 + 191113 + 191129)
 * with browser-only dependencies removed: MediaPipe → Ollama,
 * MiniLM embedding surprise → lexical-only, IndexedDB long-term memory →
 * absent, LocalCheckpointer → FileCheckpointer.
 *
 * Authoritative ADR: 90-docs/adr/2605191229-ameno-daemon-path-a-bun-langgraph.md
 */
import {
  Annotation,
  END,
  START,
  StateGraph,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";
import type { ChatMessage, GenerationStats } from "./types.js";
import { runtimeGenerate } from "./ollama-runtime.js";
import {
  executeToolCall,
  formatToolHistory,
  formatToolsForPrompt,
  parseToolCalls,
  stripToolMarkup,
} from "./tools.js";
import { FileCheckpointer } from "./file-checkpointer.js";

// ── Chunk types ──────────────────────────────────────────────────────────

export type GraphPhase =
  | "surprise_eval"
  | "generate"
  | "execute_tool"
  | "critique"
  | "revise"
  | "finalize"
  | "predict_next";

export type GraphChunk =
  | { type: "token"; phase: GraphPhase; token: string }
  | { type: "phase"; phase: GraphPhase; iteration: number }
  | { type: "stats"; phase: GraphPhase; stats: GenerationStats }
  | { type: "critique"; score: number; feedback: string; iteration: number }
  | { type: "surprise"; prediction: string; actual: string; surprise: number; mode: "lexical" }
  | { type: "prediction"; prediction: string }
  | { type: "tool_call"; name: string; args: unknown; iteration: number }
  | { type: "tool_result"; name: string; result: string; error: boolean; iteration: number };

type ChunkWriter = (chunk: GraphChunk) => void;

function getNodeWriter(config: LangGraphRunnableConfig): ChunkWriter {
  const w = (config?.configurable as { writer?: ChunkWriter } | undefined)?.writer;
  return w ?? (() => {});
}

// ── State annotation ─────────────────────────────────────────────────────

interface CritiqueResult {
  score: number;
  feedback: string;
}

const StateAnnotation = Annotation.Root({
  messages: Annotation<ChatMessage[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  draft: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  critique: Annotation<CritiqueResult | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),
  iteration: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  maxIterations: Annotation<number>({ reducer: (_, b) => b, default: () => 1 }),
  prediction: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  surprise: Annotation<number | null>({ reducer: (_, b) => b, default: () => null }),
  activeInference: Annotation<boolean>({ reducer: (_, b) => b, default: () => false }),
  toolHistory: Annotation<Array<{ name: string; args: unknown; result: string }>>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  toolIteration: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  maxToolIterations: Annotation<number>({ reducer: (_, b) => b, default: () => 3 }),
  toolsEnabled: Annotation<boolean>({ reducer: (_, b) => b, default: () => true }),
});

type GraphState = typeof StateAnnotation.State;

// ── Helper: lexical surprise ─────────────────────────────────────────────

function lexicalSurprise(predicted: string, actual: string): number {
  if (!predicted || !actual) return 5;
  const tok = (s: string): Set<string> =>
    new Set(s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);
  const a = tok(predicted);
  const b = tok(actual);
  if (a.size === 0 && b.size === 0) return 5;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  const union = a.size + b.size - inter;
  const j = union === 0 ? 1 : inter / union;
  return Math.round((1 - j) * 10);
}

function buildActiveInferenceContext(state: GraphState): string {
  if (!state.activeInference) return "";
  if (state.surprise === null || !state.prediction) return "";
  const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
  const actual = lastUser?.content ?? "";
  const lines = [
    `Last turn you predicted the user would say: "${state.prediction}".`,
    `The user actually said: "${actual}".`,
    `Surprise score: ${state.surprise}/10 (lexical Jaccard).`,
  ];
  if (state.surprise >= 7) {
    lines.push("Treat the user's intent as having shifted; ask a short clarifying question.");
  } else if (state.surprise <= 2) {
    lines.push("Your model of the user is on track; proceed confidently.");
  }
  return lines.join("\n");
}

// ── Nodes ────────────────────────────────────────────────────────────────

function surpriseEvalNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Partial<GraphState> {
  const write = getNodeWriter(config);
  write({ type: "phase", phase: "surprise_eval", iteration: 0 });
  const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
  const actual = lastUser?.content ?? "";
  const predicted = state.prediction ?? "";
  const surprise = lexicalSurprise(predicted, actual);
  write({ type: "surprise", prediction: predicted, actual, surprise, mode: "lexical" });
  return { surprise };
}

async function generateNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<Partial<GraphState>> {
  const write = getNodeWriter(config);
  write({ type: "phase", phase: "generate", iteration: state.toolIteration });

  const preamble: ChatMessage[] = [];
  const aiCtx = buildActiveInferenceContext(state);
  if (aiCtx) preamble.push({ role: "system", content: aiCtx });
  if (state.toolsEnabled) {
    preamble.push({ role: "system", content: formatToolsForPrompt() });
    const hist = formatToolHistory(state.toolHistory);
    if (hist) preamble.push({ role: "system", content: hist });
  }
  const prompt: ChatMessage[] = preamble.length > 0
    ? [...preamble, ...state.messages]
    : state.messages;

  let draft = "";
  const stats = await runtimeGenerate(prompt, (tok) => {
    draft += tok;
    write({ type: "token", phase: "generate", token: tok });
  });
  write({ type: "stats", phase: "generate", stats });
  return { draft };
}

async function executeToolNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<Partial<GraphState>> {
  const write = getNodeWriter(config);
  const nextIter = state.toolIteration + 1;
  write({ type: "phase", phase: "execute_tool", iteration: nextIter });

  const calls = parseToolCalls(state.draft);
  const appended: Array<{ name: string; args: unknown; result: string }> = [];
  for (const call of calls) {
    write({
      type: "tool_call",
      name: call.name || "(unnamed)",
      args: call.args,
      iteration: nextIter,
    });
    const result = await executeToolCall(call, { messages: state.messages });
    const isError = result.startsWith("error:");
    write({
      type: "tool_result",
      name: call.name || "(unnamed)",
      result,
      error: isError,
      iteration: nextIter,
    });
    appended.push({ name: call.name || "(unnamed)", args: call.args, result });
  }
  return { toolHistory: appended, toolIteration: nextIter };
}

function decideAfterGenerate(state: GraphState): "execute_tool" | "critic" | "finalize" {
  if (state.toolsEnabled && state.toolIteration < state.maxToolIterations) {
    const calls = parseToolCalls(state.draft);
    const usable = calls.some((c) => c.name && !c.parseError);
    if (usable) return "execute_tool";
  }
  if (state.maxIterations > 0) return "critic";
  return "finalize";
}

async function critiqueNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<Partial<GraphState>> {
  const write = getNodeWriter(config);
  write({ type: "phase", phase: "critique", iteration: state.iteration });

  const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
  const critiquePrompt: ChatMessage[] = [
    {
      role: "system",
      content:
        "You are a strict reviewer. Read the user request and the assistant draft. " +
        "Score the draft from 0 (terrible) to 10 (excellent) and give ONE specific " +
        "actionable improvement. Reply with ONLY this JSON object on a single line, " +
        'no prose: {"score": <int 0-10>, "feedback": "<one sentence>"}',
    },
    {
      role: "user",
      content:
        `# User request\n${lastUser?.content ?? ""}\n\n` +
        `# Assistant draft\n${state.draft}\n\n` +
        `Reply with the JSON object now.`,
    },
  ];

  let raw = "";
  const stats = await runtimeGenerate(critiquePrompt, (tok) => {
    raw += tok;
    write({ type: "token", phase: "critique", token: tok });
  });
  write({ type: "stats", phase: "critique", stats });

  const parsed = parseCritique(raw);
  write({
    type: "critique",
    score: parsed.score,
    feedback: parsed.feedback,
    iteration: state.iteration,
  });
  return { critique: parsed };
}

async function reviseNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<Partial<GraphState>> {
  const write = getNodeWriter(config);
  const nextIter = state.iteration + 1;
  write({ type: "phase", phase: "revise", iteration: nextIter });

  const reviseMessages: ChatMessage[] = [
    ...state.messages,
    {
      role: "system",
      content:
        `Your previous draft was:\n---\n${state.draft}\n---\n\n` +
        `A reviewer scored it ${state.critique?.score ?? "?"}/10 and suggested: ` +
        `${state.critique?.feedback ?? "(no feedback)"}\n\n` +
        `Rewrite the response addressing the suggestion. Output only the improved reply.`,
    },
  ];

  let draft = "";
  const stats = await runtimeGenerate(reviseMessages, (tok) => {
    draft += tok;
    write({ type: "token", phase: "revise", token: tok });
  });
  write({ type: "stats", phase: "revise", stats });
  return { draft, iteration: nextIter };
}

function decideContinue(state: GraphState): "revise" | "finalize" {
  const score = state.critique?.score ?? 10;
  if (score >= 7) return "finalize";
  if (state.iteration >= state.maxIterations) return "finalize";
  return "revise";
}

function finalizeNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Partial<GraphState> {
  const write = getNodeWriter(config);
  write({ type: "phase", phase: "finalize", iteration: state.iteration });
  const visible = state.toolsEnabled ? stripToolMarkup(state.draft) : state.draft;
  return { messages: [{ role: "assistant", content: visible || state.draft }] };
}

async function predictNextNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<Partial<GraphState>> {
  const write = getNodeWriter(config);
  write({ type: "phase", phase: "predict_next", iteration: 0 });

  const predictPrompt: ChatMessage[] = [
    {
      role: "system",
      content:
        "Based on the conversation so far, predict in ONE short sentence " +
        "(<= 20 words) the user's most likely next message. Output ONLY the " +
        "predicted sentence, no quotes, no preamble, no explanation.",
    },
    ...state.messages,
  ];

  let raw = "";
  const stats = await runtimeGenerate(predictPrompt, (tok) => {
    raw += tok;
    write({ type: "token", phase: "predict_next", token: tok });
  });
  write({ type: "stats", phase: "predict_next", stats });

  const prediction = raw.trim().replace(/^["']|["']$/g, "").slice(0, 240);
  write({ type: "prediction", prediction });
  return { prediction };
}

// ── Critic parser ────────────────────────────────────────────────────────

function parseCritique(raw: string): CritiqueResult {
  const match = raw.match(/\{[\s\S]*?\}/);
  if (!match) return { score: 7, feedback: "(critic returned no JSON; accepting)" };
  try {
    const obj = JSON.parse(match[0]) as { score?: unknown; feedback?: unknown };
    const score = typeof obj.score === "number" ? clamp(Math.round(obj.score), 0, 10) : 7;
    const feedback = typeof obj.feedback === "string" ? obj.feedback.slice(0, 240) : "(no feedback)";
    return { score, feedback };
  } catch {
    return { score: 7, feedback: "(critic JSON unparseable; accepting)" };
  }
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

// ── Build & invoke ───────────────────────────────────────────────────────

let compiled: ReturnType<typeof buildGraph> | null = null;

function buildGraph(checkpointer: FileCheckpointer) {
  const g = new StateGraph(StateAnnotation)
    .addNode("surprise_eval", surpriseEvalNode)
    .addNode("generate", generateNode)
    .addNode("execute_tool", executeToolNode)
    .addNode("critic", critiqueNode)
    .addNode("revise", reviseNode)
    .addNode("finalize", finalizeNode)
    .addNode("predict_next", predictNextNode)
    .addConditionalEdges(START, (state: GraphState) =>
      state.activeInference ? "surprise_eval" : "generate",
    )
    .addEdge("surprise_eval", "generate")
    .addConditionalEdges("generate", decideAfterGenerate)
    .addEdge("execute_tool", "generate")
    .addConditionalEdges("critic", decideContinue)
    .addEdge("revise", "critic")
    .addConditionalEdges("finalize", (state: GraphState) =>
      state.activeInference ? "predict_next" : END,
    )
    .addEdge("predict_next", END);
  return g.compile({ checkpointer });
}

export function getAmenoDaemonGraph(checkpointer: FileCheckpointer) {
  if (!compiled) compiled = buildGraph(checkpointer);
  return compiled;
}

export interface InvokeDaemonOptions {
  messages: ChatMessage[];
  maxIterations: number;
  activeInference?: boolean;
  toolsEnabled?: boolean;
  threadId?: string;
  onChunk: (chunk: GraphChunk) => void;
  checkpointer: FileCheckpointer;
}

export async function invokeDaemon(opts: InvokeDaemonOptions): Promise<string> {
  const graph = getAmenoDaemonGraph(opts.checkpointer);
  const stream = await graph.stream(
    {
      messages: opts.messages,
      maxIterations: opts.maxIterations,
      activeInference: opts.activeInference ?? false,
      toolsEnabled: opts.toolsEnabled ?? true,
      iteration: 0,
      draft: "",
      critique: null,
      surprise: null,
      toolHistory: [],
      toolIteration: 0,
    },
    {
      configurable: { thread_id: opts.threadId ?? "default" },
      streamMode: "custom",
    },
  );

  let lastDraft = "";
  for await (const chunk of stream) {
    const payload = chunk as GraphChunk;
    opts.onChunk(payload);
    if (payload.type === "token") {
      lastDraft += payload.token;
    }
  }

  const finalState = (await graph.getState({
    configurable: { thread_id: opts.threadId ?? "default" },
  })) as { values?: GraphState } | undefined;
  return finalState?.values?.draft ?? lastDraft;
}
