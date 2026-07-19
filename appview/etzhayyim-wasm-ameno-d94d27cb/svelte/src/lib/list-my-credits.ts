/**
 * list-my-credits.ts — client helper for com.etzhayyim.apps.ameno.listMyCredits.
 *
 * Same-origin GET to the ameno worker, which forwards via sdk.pds.xrpc()
 * → atproto.etzhayyim.com → bpmn-dispatcher → ameno-langserver SELECT
 * mv_ameno_credits_balance (Phase 5j).
 */

export interface MyCreditsResponse {
  actorDid: string;
  balance: number;
  eventCount: number;
  lastEventTsMs?: number;
  lastEventCreatedAt?: string;
}

export async function listMyCredits(actorDid: string): Promise<MyCreditsResponse> {
  if (!actorDid) return { actorDid: "", balance: 0, eventCount: 0 };
  const qs = new URLSearchParams({ actorDid });
  const res = await fetch(`/xrpc/com.etzhayyim.apps.ameno.listMyCredits?${qs.toString()}`, {
    method: "GET",
    headers: { accept: "application/json" },
  });
  if (!res.ok) return { actorDid, balance: 0, eventCount: 0 };
  try {
    const body = (await res.json()) as Partial<MyCreditsResponse>;
    return {
      actorDid: body.actorDid ?? actorDid,
      balance: Number(body.balance ?? 0),
      eventCount: Number(body.eventCount ?? 0),
      lastEventTsMs: body.lastEventTsMs ? Number(body.lastEventTsMs) : undefined,
      lastEventCreatedAt: body.lastEventCreatedAt,
    };
  } catch {
    return { actorDid, balance: 0, eventCount: 0 };
  }
}
