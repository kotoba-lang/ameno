/**
 * graph.ts — ameno browser-side LangGraph (Pregel) runtime.
 *
 * Single StateGraph: generate → critique → (revise → critique)* → finalize.
 * Each node calls `mediapipeGenerate` (or `transformersGenerate`) directly,
 * so the LLM stays in the browser. Token streaming is surfaced via
 * LangGraph's `writer()` and consumed by App.svelte through
 * `graph.stream(_, { streamMode: "custom" })`.
 *
 * Authoritative ADR: 90-docs/adr/2605191000-ameno-browser-pregel-reflection.md
 */
import {
  Annotation,
  END,
  START,
  StateGraph,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { LocalCheckpointer } from "./local-checkpointer";
import { mediapipeGenerate } from "./mediapipe-runtime";
import { generate as transformersGenerate, type ChatMessage, type GenerationStats } from "./inference";
import { cosine, embed, isEmbeddingReady } from "./embedding";
import {
  executeToolCall,
  formatToolHistory,
  formatToolsForPrompt,
  parseToolCalls,
  stripToolMarkup,
} from "./tools";

/**
 * Browser-safe writer accessor. LangGraph v1.x ships `writer()` as a free
 * function that reads `AsyncLocalStorageProviderSingleton.getRunnableConfig()`
 * to find the current node's stream writer — but `AsyncLocalStorage` is a
 * Node-only API and is not polyfilled in browser bundles, so the free
 * function throws "Called interrupt() outside the context of a graph."
 * here. We bypass it by pulling the writer straight off the node's
 * `config.configurable.writer` (which LangGraph populates per super-step
 * when `streamMode` includes `"custom"`). When run with no custom-mode
 * consumer the writer is undefined; we fall back to a no-op.
 */
type ChunkWriter = (chunk: GraphChunk) => void;
function getNodeWriter(config: LangGraphRunnableConfig): ChunkWriter {
  const w = (config?.configurable as { writer?: ChunkWriter } | undefined)?.writer;
  return w ?? (() => {});
}

export type GraphPhase =
  | "surprise_eval"
  | "generate"
  | "execute_tool"
  | "critique"
  | "revise"
  | "finalize"
  | "predict_next";

export interface GraphTokenChunk {
  type: "token";
  phase: GraphPhase;
  token: string;
}

export interface GraphPhaseStartChunk {
  type: "phase";
  phase: GraphPhase;
  iteration: number;
}

export interface GraphCritiqueChunk {
  type: "critique";
  score: number;
  feedback: string;
  iteration: number;
}

/**
 * Surprise = distance × 10 between the previous turn's prediction and
 * the actual new user message. Distance flavour is reported via `mode`.
 *
 * - `lexical`: 1 - Jaccard(tokens). ADR-2605191113.
 * - `embedding`: 1 - cosine(MiniLM(predicted), MiniLM(actual)). ADR-2605191120.
 */
export interface GraphSurpriseChunk {
  type: "surprise";
  /** Predicted user utterance from the previous turn (may be empty). */
  prediction: string;
  /** Actual user utterance this turn. */
  actual: string;
  /** 0..10 inclusive. 0 = identical, 10 = no overlap. */
  surprise: number;
  /** Which distance was actually used (lexical when embedding not yet ready). */
  mode: "lexical" | "embedding";
}

/**
 * Committed prediction for the NEXT user turn, written at the tail of the
 * graph after finalize. ADR-2605191113.
 */
export interface GraphPredictionChunk {
  type: "prediction";
  prediction: string;
}

/** A `<tool>{...}</tool>` block successfully parsed out of the model draft. */
export interface GraphToolCallChunk {
  type: "tool_call";
  name: string;
  args: unknown;
  iteration: number;
}

/** Result of executing a tool call. `error` is true when the result string
 *  starts with `error:` (the executor's convention). */
export interface GraphToolResultChunk {
  type: "tool_result";
  name: string;
  result: string;
  error: boolean;
  iteration: number;
}

export interface GraphStatsChunk {
  type: "stats";
  phase: GraphPhase;
  stats: GenerationStats;
}

export type GraphChunk =
  | GraphTokenChunk
  | GraphPhaseStartChunk
  | GraphCritiqueChunk
  | GraphStatsChunk
  | GraphSurpriseChunk
  | GraphPredictionChunk
  | GraphToolCallChunk
  | GraphToolResultChunk;

/** Critique payload parsed out of the critic LLM. */
interface CritiqueResult {
  score: number;
  feedback: string;
}

const StateAnnotation = Annotation.Root({
  /** Conversation history. Reducer concatenates, so nodes return only
   *  the newly-added messages instead of the full list. */
  messages: Annotation<ChatMessage[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  /** Current draft response under review. */
  draft: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),
  /** Last critique result, or null before the critic has run. */
  critique: Annotation<CritiqueResult | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),
  /** Number of revise cycles completed (0 = only initial draft). */
  iteration: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),
  /** Reflection cap — 0 disables critic entirely. */
  maxIterations: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 1,
  }),
  /** Which kernel to dispatch to. */
  kernel: Annotation<"mediapipe" | "transformers">({
    reducer: (_, b) => b,
    default: () => "mediapipe",
  }),
  /** Predicted next user utterance, written by predict_next at the tail
   *  of the previous turn and consumed by surprise_eval at the head of
   *  the next turn. ADR-2605191113. */
  prediction: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),
  /** Latest lexical surprise (0..10), null when no prior prediction. */
  surprise: Annotation<number | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),
  /** Whether the agent runs the active-inference loop (surprise_eval at
   *  head + predict_next at tail) this turn. */
  activeInference: Annotation<boolean>({
    reducer: (_, b) => b,
    default: () => false,
  }),
  /** Which surprise flavour to attempt — embedding falls back to lexical
   *  when the MiniLM pipeline has not loaded yet. ADR-2605191120. */
  surpriseMode: Annotation<"lexical" | "embedding">({
    reducer: (_, b) => b,
    default: () => "lexical",
  }),
  /** Tool calls executed so far this turn (resets between turns). ADR-2605191129. */
  toolHistory: Annotation<Array<{ name: string; args: unknown; result: string }>>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  /** Number of execute_tool → generate loops consumed this turn. */
  toolIteration: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),
  /** Per-turn cap on tool-call loops. */
  maxToolIterations: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 3,
  }),
  /** When false, generate emits no tool affordance and parseToolCalls is
   *  not applied — guarantees previous "no tool use" behaviour bit-exact. */
  toolsEnabled: Annotation<boolean>({
    reducer: (_, b) => b,
    default: () => false,
  }),
});

type GraphState = typeof StateAnnotation.State;

/** Dispatch generate to the right backend without re-importing in nodes. */
async function runtimeGenerate(
  kernel: "mediapipe" | "transformers",
  messages: ChatMessage[],
  onToken: (token: string) => void,
): Promise<GenerationStats> {
  if (kernel === "mediapipe") return mediapipeGenerate(messages, onToken);
  return transformersGenerate(messages, onToken);
}

/**
 * Lexical surprise = 1 - Jaccard(tokens(predicted), tokens(actual)) × 10.
 * Returns 0..10, integer. Treats no-prediction as the neutral score 5
 * so the absence of a prior turn does not bias the agent.
 *
 * ADR-2605191113.
 */
function lexicalSurprise(predicted: string, actual: string): number {
  if (!predicted || !actual) return 5;
  const tok = (s: string): Set<string> =>
    new Set(s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);
  const a = tok(predicted);
  const b = tok(actual);
  if (a.size === 0 && b.size === 0) return 5;
  let interSize = 0;
  for (const x of a) if (b.has(x)) interSize++;
  const unionSize = a.size + b.size - interSize;
  const jaccard = unionSize === 0 ? 1 : interSize / unionSize;
  return Math.round((1 - jaccard) * 10);
}

/**
 * Embedding-cosine surprise. Loads two embeddings in parallel and
 * converts cosine similarity to 0..10. Throws if the pipeline is not
 * ready — callers must check `isEmbeddingReady()` first.
 *
 * ADR-2605191120.
 */
async function embeddingSurprise(predicted: string, actual: string): Promise<number> {
  const [a, b] = await Promise.all([embed(predicted), embed(actual)]);
  const cos = cosine(a, b);
  const dist = 1 - cos;
  return Math.max(0, Math.min(10, Math.round(dist * 10)));
}

/**
 * surpriseEvalNode — compares the prior turn's prediction against the
 * latest user message. Lexical Jaccard by default; if `surpriseMode` is
 * "embedding" AND the MiniLM pipeline is loaded, dispatches to cosine.
 * Falls back to lexical (with mode === "lexical" in the chunk) when
 * embedding is requested but not yet ready.
 *
 * Skipped via conditional edge when state.activeInference is false.
 */
async function surpriseEvalNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<Partial<GraphState>> {
  const write = getNodeWriter(config);
  write({ type: "phase", phase: "surprise_eval", iteration: 0 });

  const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
  const actual = lastUser?.content ?? "";
  const predicted = state.prediction ?? "";

  let surprise: number;
  let mode: "lexical" | "embedding" = "lexical";
  if (state.surpriseMode === "embedding" && predicted && actual && isEmbeddingReady()) {
    try {
      surprise = await embeddingSurprise(predicted, actual);
      mode = "embedding";
    } catch {
      surprise = lexicalSurprise(predicted, actual);
    }
  } else {
    surprise = lexicalSurprise(predicted, actual);
  }

  write({ type: "surprise", prediction: predicted, actual, surprise, mode });
  return { surprise };
}

/**
 * Build the active-inference context block that gets folded into the
 * generate system prompt. Empty string when no prior prediction exists,
 * which keeps the first turn clean.
 */
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

/**
 * generateNode — produce the initial draft. Composes the prompt by
 * prepending optional system blocks for:
 *   - active inference surprise context (ADR-2605191113)
 *   - tool affordance + tool history (ADR-2605191129)
 *
 * Everything else stays out of the prompt to keep token budget tight.
 */
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
  const stats = await runtimeGenerate(state.kernel, prompt, (tok) => {
    draft += tok;
    write({ type: "token", phase: "generate", token: tok });
  });
  write({ type: "stats", phase: "generate", stats });
  return { draft };
}

/**
 * executeToolNode — invoked when the most recent draft contains a tool
 * tag. Runs each tool call sequentially, appends results to toolHistory,
 * and bumps toolIteration. The graph then loops back to generate so the
 * model can use the new context.
 */
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

/**
 * Conditional edge after generate. Tool loop has highest priority; after
 * tools run out, reflection (existing) gets its chance; otherwise we
 * finalize directly.
 */
function decideAfterGenerate(state: GraphState): "execute_tool" | "critic" | "finalize" {
  if (state.toolsEnabled && state.toolIteration < state.maxToolIterations) {
    const calls = parseToolCalls(state.draft);
    const usable = calls.some((c) => c.name && !c.parseError);
    if (usable) return "execute_tool";
  }
  if (state.maxIterations > 0) return "critic";
  return "finalize";
}

/**
 * predictNextNode — assistant predicts the user's likely next short
 * utterance. The prediction is stored in state.prediction and survives
 * across graph invocations via MemorySaver, so surpriseEvalNode at the
 * NEXT turn can score it against the real input.
 *
 * Prompt budget kept tight (≤20 tokens) to bound wall-clock cost.
 */
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
  const stats = await runtimeGenerate(state.kernel, predictPrompt, (tok) => {
    raw += tok;
    write({ type: "token", phase: "predict_next", token: tok });
  });
  write({ type: "stats", phase: "predict_next", stats });

  const prediction = raw.trim().replace(/^["']|["']$/g, "").slice(0, 240);
  write({ type: "prediction", prediction });
  return { prediction };
}

/**
 * critiqueNode — same LLM scores its own draft.
 *
 * We use a tight JSON-only prompt so we can parse a 0-10 score and a single
 * actionable suggestion. Falls back to "accept" (score=7) when parsing fails
 * so a confused critic does not block progress.
 */
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
  const stats = await runtimeGenerate(state.kernel, critiquePrompt, (tok) => {
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

/**
 * reviseNode — regenerate the draft with the critic's feedback in the
 * system prompt. iteration is incremented BEFORE the rewrite so the
 * stream phase chunk and the conditional edge agree on the count.
 */
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
  const stats = await runtimeGenerate(state.kernel, reviseMessages, (tok) => {
    draft += tok;
    write({ type: "token", phase: "revise", token: tok });
  });
  write({ type: "stats", phase: "revise", stats });

  return { draft, iteration: nextIter };
}

/**
 * finalizeNode — commit the final draft to messages as an assistant turn.
 */
function finalizeNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Partial<GraphState> {
  const write = getNodeWriter(config);
  write({ type: "phase", phase: "finalize", iteration: state.iteration });
  // Hide tool markup from the user-visible message. The full draft with
  // markup is still in state.draft / toolHistory for debugging if needed.
  const visible = state.toolsEnabled ? stripToolMarkup(state.draft) : state.draft;
  return {
    messages: [{ role: "assistant", content: visible || state.draft }],
  };
}

/**
 * Conditional edge from critique. Accept early when the score is high
 * enough OR we've spent the iteration budget. iteration counts revisions,
 * so iteration >= maxIterations means we cannot revise any more.
 */
function decideContinue(state: GraphState): "revise" | "finalize" {
  const score = state.critique?.score ?? 10;
  if (score >= 7) return "finalize";
  if (state.iteration >= state.maxIterations) return "finalize";
  return "revise";
}

/**
 * Critic JSON parser. Extracts the first `{...}` block, parses it, clamps
 * the score, and trims feedback to 240 chars. Lenient by design — never
 * throws.
 */
function parseCritique(raw: string): CritiqueResult {
  const match = raw.match(/\{[\s\S]*?\}/);
  if (!match) return { score: 7, feedback: "(critic returned no JSON; accepting)" };
  try {
    const obj = JSON.parse(match[0]) as { score?: unknown; feedback?: unknown };
    const score = typeof obj.score === "number" ? clamp(Math.round(obj.score), 0, 10) : 7;
    const feedback =
      typeof obj.feedback === "string" ? obj.feedback.slice(0, 240) : "(no feedback)";
    return { score, feedback };
  } catch {
    return { score: 7, feedback: "(critic JSON unparseable; accepting)" };
  }
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/**
 * Build the compiled StateGraph. We compile lazily — once on first use —
 * because MediaPipe model load is asynchronous and the graph itself is
 * pure structure.
 */
let compiled: ReturnType<typeof buildGraph> | null = null;

function buildGraph() {
  const graph = new StateGraph(StateAnnotation)
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

  // LocalCheckpointer = MemorySaver + localStorage mirror. State persists
  // across page reloads. ADR-2605191135. Swap to @etzhayyim/sdk/checkpointer
  // (MstCheckpointSaver) to graduate from local-only to substrate-anchored.
  return graph.compile({ checkpointer: new LocalCheckpointer() });
}

export function getAmenoGraph() {
  if (!compiled) compiled = buildGraph();
  return compiled;
}

export interface InvokeAmenoOptions {
  /** Conversation history INCLUDING the new user turn. */
  messages: ChatMessage[];
  /** 0..2. 0 disables reflection entirely. */
  maxIterations: number;
  /** Which inference kernel to use for this turn. */
  kernel: "mediapipe" | "transformers";
  /** When true, runs surprise_eval at head and predict_next at tail. */
  activeInference?: boolean;
  /** lexical (default, no extra model) or embedding (MiniLM, 22 MB lazy). */
  surpriseMode?: "lexical" | "embedding";
  /** Enable browser-local tool use (ReAct loop, max 3 iter). ADR-2605191129. */
  toolsEnabled?: boolean;
  /** thread_id for MemorySaver — defaults to "default" so a single tab keeps state. */
  threadId?: string;
  /** Per-chunk callback. */
  onChunk: (chunk: GraphChunk) => void;
}

/**
 * Convenience wrapper around `graph.stream(_, { streamMode: "custom" })`.
 * Drains the stream and returns the final assistant text.
 */
export async function invokeAmeno(opts: InvokeAmenoOptions): Promise<string> {
  const graph = getAmenoGraph();
  const stream = await graph.stream(
    {
      messages: opts.messages,
      maxIterations: opts.maxIterations,
      kernel: opts.kernel,
      activeInference: opts.activeInference ?? false,
      surpriseMode: opts.surpriseMode ?? "lexical",
      toolsEnabled: opts.toolsEnabled ?? false,
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
      // We don't accumulate here — the graph state owns truth — but we keep
      // the most recent stream of tokens for the UI fallback.
      lastDraft += payload.token;
    }
  }

  // Pull the canonical final draft from the graph state.
  const finalState = (await graph.getState({
    configurable: { thread_id: opts.threadId ?? "default" },
  })) as { values?: GraphState } | undefined;
  return finalState?.values?.draft ?? lastDraft;
}
