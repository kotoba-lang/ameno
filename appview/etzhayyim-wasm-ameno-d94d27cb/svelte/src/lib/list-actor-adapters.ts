/**
 * list-actor-adapters.ts — client helper for com.etzhayyim.apps.ameno.listActorAdapters.
 *
 * Same-origin GET to the ameno worker, which forwards via sdk.pds.xrpc()
 * → atproto.etzhayyim.com → bpmn-dispatcher → ameno-langserver SELECT
 * vertex_lora_adapter (Phase 5g).
 */

export interface AdapterRow {
  adapterId: string;
  actorDid: string;
  domain: string;
  status: string;
  /** Engine model id this adapter targets (e.g. gemma-4-e2b-it). */
  baseModel: string;
  /** B2 / R2 URI for the adapter.safetensors blob. */
  weightB2Uri: string;
  weightByteSize: number;
  weightSha256: string;
  adapterRank: number;
  /** α scaling factor × 1000 (lexicon integer-only constraint). */
  adapterAlpha: number;
  adapterFormat: string;
  displayNameYomi: string;
  createdAt: string;
}

export interface ListActorAdaptersParams {
  actorDid: string;
  domain?: string;
  /** 1..100, default 20. */
  limit?: number;
}

export interface ListActorAdaptersResponse {
  items: AdapterRow[];
  total: number;
}

export async function listActorAdapters(
  params: ListActorAdaptersParams,
): Promise<ListActorAdaptersResponse> {
  if (!params.actorDid) return { items: [], total: 0 };
  const qs = new URLSearchParams();
  qs.set("actorDid", params.actorDid);
  if (params.domain) qs.set("domain", params.domain);
  if (params.limit != null) qs.set("limit", String(params.limit));

  const res = await fetch(`/xrpc/com.etzhayyim.apps.ameno.listActorAdapters?${qs.toString()}`, {
    method: "GET",
    headers: { accept: "application/json" },
  });
  if (!res.ok) return { items: [], total: 0 };
  try {
    const body = (await res.json()) as Partial<ListActorAdaptersResponse>;
    return {
      items: Array.isArray(body.items) ? (body.items as AdapterRow[]) : [],
      total: Number(body.total ?? 0),
    };
  } catch {
    return { items: [], total: 0 };
  }
}
