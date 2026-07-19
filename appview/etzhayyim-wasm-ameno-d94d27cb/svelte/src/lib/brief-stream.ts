/**
 * brief-stream.ts — EventSource wrapper for com.etzhayyim.apps.ameno.subscribeBriefs.
 *
 * Connects to the AT Protocol PDS (atproto.etzhayyim.com) which routes the SSE
 * stream through bpmn-dispatcher → ameno-langserver → NATS JetStream.
 * Each `event: brief` frame carries one PDS commit's text + provenance.
 */

const PDS_BASE = "https://atproto.etzhayyim.com";

export interface Brief {
  /** at://did/collection/rkey of the source record. */
  uri: string;
  /** Author DID (subject of the original record). */
  authorDid: string;
  /** Source collection NSID. */
  collection: string;
  /** Extracted prompt text from the record (e.g. post.text). */
  text: string;
  /** Server timestamp / firehose seq, in ms. */
  tsMs: number;
}

export interface OpenBriefStreamOptions {
  /** AT collection NSID to subscribe (default: app.bsky.feed.post). */
  collection?: string;
  /** Server closes the stream after N events (1-1000, default 100). */
  maxEvents?: number;
  /** Server closes the stream after this many idle seconds (5-600, default 60). */
  idleTimeoutSec?: number;
  /** Called once for each `brief` event. */
  onBrief: (brief: Brief) => void;
  /** Called when the server closes the stream cleanly. */
  onDone?: (reason: string, delivered: number) => void;
  /** Called on connection / parse / server-reported errors. */
  onError?: (error: Error) => void;
}

/**
 * Open an SSE stream of brief inference prompts.
 *
 * Returns a `close()` function the caller invokes to stop the stream
 * and release the EventSource.
 */
export function openBriefStream(opts: OpenBriefStreamOptions): () => void {
  const params = new URLSearchParams();
  if (opts.collection) params.set("collection", opts.collection);
  if (opts.maxEvents != null) params.set("maxEvents", String(opts.maxEvents));
  if (opts.idleTimeoutSec != null) params.set("idleTimeoutSec", String(opts.idleTimeoutSec));

  const url = `${PDS_BASE}/xrpc/com.etzhayyim.apps.ameno.subscribeBriefs?${params.toString()}`;
  const es = new EventSource(url);
  let closed = false;
  let delivered = 0;

  const close = () => {
    if (closed) return;
    closed = true;
    es.close();
  };

  es.addEventListener("brief", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data) as Brief;
      delivered++;
      opts.onBrief(payload);
    } catch (e) {
      opts.onError?.(e instanceof Error ? e : new Error(String(e)));
    }
  });

  es.addEventListener("done", (ev) => {
    let reason = "server-closed";
    try {
      const payload = JSON.parse((ev as MessageEvent).data) as { reason?: string; delivered?: number };
      reason = payload.reason ?? reason;
    } catch {}
    opts.onDone?.(reason, delivered);
    close();
  });

  es.addEventListener("error", (ev) => {
    let msg = "stream error";
    try {
      const payload = JSON.parse((ev as MessageEvent).data) as { error?: string };
      msg = payload.error ?? msg;
    } catch {
      msg = closed ? "stream closed" : "connection lost";
    }
    opts.onError?.(new Error(msg));
  });

  return close;
}
