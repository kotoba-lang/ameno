/**
 * save-result.ts — client helper for com.etzhayyim.apps.ameno.saveResult.
 *
 * Posts to the same-origin ameno.etzhayyim.com Worker which forwards via
 * sdk.pds.xrpc() → atproto.etzhayyim.com → bpmn-dispatcher → ameno-langserver pod
 * → INSERT vertex_ameno_inferenceresult (ADR-2605111200).
 */

export interface SaveResultInput {
  modelId: string;
  prompt: string;
  output: string;
  actorDid?: string;
  loraAdapters?: string[];
  promptTokens?: number;
  outputTokens?: number;
  elapsedMs?: number;
  /** decode tokens/sec × 1000 (integer-only per AT lexicon). */
  tokensPerSec?: number;
  webgpuAdapter?: string;
  ragContextUsed?: boolean;
}

export interface SaveResultOutput {
  status: "queued" | "persisted" | "failed";
  resultId?: string;
  uri?: string;
  error?: string;
}

export async function saveResult(input: SaveResultInput): Promise<SaveResultOutput> {
  const res = await fetch("/xrpc/com.etzhayyim.apps.ameno.saveResult", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    return { status: "failed", error: `HTTP ${res.status}` };
  }
  try {
    return (await res.json()) as SaveResultOutput;
  } catch (e) {
    return { status: "failed", error: e instanceof Error ? e.message : String(e) };
  }
}
