/**
 * Shared message + stats types for the ameno daemon graph.
 *
 * Intentionally identical to the svelte appview's `@etzhayyim/ameno/inference`
 * types so that node implementations can be cloned with minimal edits.
 */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface GenerationStats {
  durationMs: number;
  totalTokens: number;
  tokensPerSecond: number;
  /** True when a RAG context block was prepended for this generation. */
  ragActive: boolean;
}
