/**
 * list-history.ts — client helper for com.etzhayyim.apps.ameno.listHistory.
 *
 * AT Protocol query: GET /xrpc/{nsid}?actorDid=...&modelId=...&limit=...&offset=...
 * Same-origin to ameno.etzhayyim.com; the worker forwards via sdk.pds.xrpc()
 * to atproto.etzhayyim.com → bpmn-dispatcher → ameno-langserver SELECT.
 */

export interface HistoryItem {
  resultId: string;
  uri: string;
  modelId: string;
  actorDid: string;
  /** Always plaintext (no privacy claim on the prompt for now). */
  prompt: string;
  /** Either plaintext or `signal:v1:{ciphertext}` per Phase 5b. */
  output: string;
  elapsedMs: number;
  tokensPerSec: number;
  createdAt: string;
}

export interface ListHistoryResponse {
  items: HistoryItem[];
  total: number;
  offset: number;
  limit: number;
}

export interface ListHistoryParams {
  actorDid?: string;
  modelId?: string;
  /** 1..100, default 20. */
  limit?: number;
  /** ≥ 0, default 0. */
  offset?: number;
}

export async function listHistory(params: ListHistoryParams = {}): Promise<ListHistoryResponse> {
  const qs = new URLSearchParams();
  if (params.actorDid) qs.set("actorDid", params.actorDid);
  if (params.modelId) qs.set("modelId", params.modelId);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));

  const res = await fetch(`/xrpc/com.etzhayyim.apps.ameno.listHistory?${qs.toString()}`, {
    method: "GET",
    headers: { accept: "application/json" },
  });
  if (!res.ok) {
    return { items: [], total: 0, offset: params.offset ?? 0, limit: params.limit ?? 20 };
  }
  try {
    const body = (await res.json()) as Partial<ListHistoryResponse>;
    return {
      items: Array.isArray(body.items) ? (body.items as HistoryItem[]) : [],
      total: Number(body.total ?? 0),
      offset: Number(body.offset ?? params.offset ?? 0),
      limit: Number(body.limit ?? params.limit ?? 20),
    };
  } catch {
    return { items: [], total: 0, offset: params.offset ?? 0, limit: params.limit ?? 20 };
  }
}
