/**
 * swarm.ts — Multi-tab peer roster via BroadcastChannel.
 *
 * Each ameno tab in the same browser origin announces itself on a
 * shared channel every 5s. Peers are tracked in an in-memory roster;
 * silent peers fall off after 15s. Work-distribution (lease-based brief
 * fan-out) is intentionally out of v0.1 scope — we ship the *presence*
 * primitive so the UI can show "you're not alone" and so the next ADR
 * can build a lease-based dispatcher on top.
 *
 * Authoritative ADR: 90-docs/adr/2605191524-ameno-multi-tab-swarm.md
 */

const CHANNEL_NAME = "ameno-swarm-v1";
const HELLO_PERIOD_MS = 5_000;
const PEER_TTL_MS = 15_000;

export type SwarmRole = "browser" | "daemon-viewer" | "auto-respond";

export interface SwarmPeer {
  /** Stable per-tab worker DID (ADR-2605191135). */
  did: string;
  /** What kind of ameno surface this peer presents. */
  role: SwarmRole;
  /** Currently selected compute mode (local / daemon-a / …). */
  computeMode: string;
  /** Currently loaded model id, if any. */
  loadedModel: string | null;
  /** Wall-clock of last heartbeat received from this peer (ms). */
  lastSeenMs: number;
}

interface SwarmAnnouncement {
  type: "hello";
  did: string;
  role: SwarmRole;
  computeMode: string;
  loadedModel: string | null;
  /** Sender's local clock — receivers reset to their own clock. */
  sentAtMs: number;
}

interface SwarmGoodbye {
  type: "bye";
  did: string;
}

type SwarmMessage = SwarmAnnouncement | SwarmGoodbye;

/**
 * Open a swarm channel for this tab. Returns a close-handle and a
 * `getPeers` snapshot accessor. The host driver polls `getPeers()`
 * (e.g. via $effect every 1s) to refresh UI — we don't push svelte
 * stores out of the module so this stays UI-framework-agnostic.
 */
export interface SwarmHandle {
  close(): void;
  getPeers(): SwarmPeer[];
  /** True when this tab's DID is lex-smallest among self + active peers.
   *  Deterministic, O(N), no quorum. ADR-2605191603. */
  isLeader(): boolean;
  /** This tab's stable worker DID — exposed for completeness. */
  selfDid(): string;
  /** Update the announcement payload for subsequent heartbeats. */
  update(patch: Partial<{
    role: SwarmRole;
    computeMode: string;
    loadedModel: string | null;
  }>): void;
}

/**
 * Pure helper: lex-smallest DID among {self + peers} = leader.
 * ADR-2605191603. Exposed so callers can compute leader against an
 * arbitrary peer snapshot (e.g. for UI rendering) without going
 * through a SwarmHandle.
 */
export function computeLeader(selfDid: string, peers: SwarmPeer[]): boolean {
  const dids = [selfDid, ...peers.map((p) => p.did)];
  let smallest = selfDid;
  for (const d of dids) if (d < smallest) smallest = d;
  return smallest === selfDid;
}

export interface SwarmOptions {
  did: string;
  role?: SwarmRole;
  computeMode: string;
  loadedModel: string | null;
}

export function openSwarm(opts: SwarmOptions): SwarmHandle {
  // SSR / older browsers may lack BroadcastChannel — return a degenerate
  // handle in that case.
  if (typeof BroadcastChannel === "undefined") {
    return {
      close: () => {},
      getPeers: () => [],
      // No peers => self is trivially leader (single-tab session).
      isLeader: () => true,
      selfDid: () => opts.did,
      update: () => {},
    };
  }

  const ch = new BroadcastChannel(CHANNEL_NAME);
  const peers = new Map<string, SwarmPeer>();
  let role: SwarmRole = opts.role ?? "browser";
  let computeMode = opts.computeMode;
  let loadedModel = opts.loadedModel;

  const sendHello = () => {
    const msg: SwarmAnnouncement = {
      type: "hello",
      did: opts.did,
      role,
      computeMode,
      loadedModel,
      sentAtMs: Date.now(),
    };
    try {
      ch.postMessage(msg);
    } catch {
      /* peer may have closed; ignore */
    }
  };

  const sweep = () => {
    const cutoff = Date.now() - PEER_TTL_MS;
    for (const [did, peer] of peers) {
      if (peer.lastSeenMs < cutoff) peers.delete(did);
    }
  };

  ch.onmessage = (ev: MessageEvent<SwarmMessage>) => {
    const m = ev.data;
    if (!m || typeof m !== "object") return;
    if (m.type === "bye") {
      peers.delete(m.did);
      return;
    }
    if (m.type === "hello") {
      if (m.did === opts.did) return; // ignore self
      peers.set(m.did, {
        did: m.did,
        role: m.role,
        computeMode: m.computeMode,
        loadedModel: m.loadedModel,
        lastSeenMs: Date.now(),
      });
      return;
    }
  };

  // Kick off a hello immediately so other peers learn about us, then
  // settle into the periodic cadence.
  sendHello();
  const helloTimer = window.setInterval(sendHello, HELLO_PERIOD_MS);
  const sweepTimer = window.setInterval(sweep, HELLO_PERIOD_MS / 2);

  const close = () => {
    try {
      const bye: SwarmGoodbye = { type: "bye", did: opts.did };
      ch.postMessage(bye);
    } catch {
      /* ignore */
    }
    window.clearInterval(helloTimer);
    window.clearInterval(sweepTimer);
    try {
      ch.close();
    } catch {
      /* ignore */
    }
    peers.clear();
  };

  // Browser tab close — best-effort goodbye so other tabs drop us
  // before TTL expires.
  if (typeof window !== "undefined") {
    window.addEventListener("beforeunload", close, { once: true });
  }

  const snapshot = () => [...peers.values()].sort((a, b) => a.did.localeCompare(b.did));

  return {
    close,
    getPeers: snapshot,
    isLeader: () => computeLeader(opts.did, snapshot()),
    selfDid: () => opts.did,
    update: (patch) => {
      if (patch.role !== undefined) role = patch.role;
      if (patch.computeMode !== undefined) computeMode = patch.computeMode;
      if (patch.loadedModel !== undefined) loadedModel = patch.loadedModel;
      // Push the update immediately so peers see the new state without
      // waiting up to HELLO_PERIOD_MS.
      sendHello();
    },
  };
}
