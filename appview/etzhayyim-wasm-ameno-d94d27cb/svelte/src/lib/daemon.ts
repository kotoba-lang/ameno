/**
 * daemon.ts — ameno Tier-2 worker identity + heartbeat.
 *
 * Assigns this browser tab a stable `did:web:browser:<uuid>` and tracks
 * uptime / brief throughput so the artificial-organism ecosystem can
 * recognise it as a long-running worker.
 *
 * Authoritative ADR: 90-docs/adr/2605191135-ameno-tier2-daemon-residency.md
 */

const DID_KEY = "ameno.workerDid.v1";
const SESSION_STARTED_AT = Date.now();
const BRIEF_WINDOW_MS = 60_000;

let workerDid: string | null = null;
const recentBriefs: number[] = [];
let totalBriefs = 0;
let totalTokensDecoded = 0;
let lastBriefAt: number | null = null;
let lastError: string | null = null;
let firehoseConnected = false;

/** Try crypto.randomUUID; fallback to a short random suffix if missing. */
function newWorkerId(): string {
  const r = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2, 14);
  return `did:web:browser:${r.replace(/-/g, "")}`;
}

/** Get-or-create the persistent worker DID for this browser origin. */
export function getWorkerDid(): string {
  if (workerDid) return workerDid;
  try {
    const stored = localStorage.getItem(DID_KEY);
    if (stored && stored.startsWith("did:web:browser:")) {
      workerDid = stored;
      return workerDid;
    }
  } catch {
    /* localStorage unavailable; fall through and use ephemeral id */
  }
  const did = newWorkerId();
  try {
    localStorage.setItem(DID_KEY, did);
  } catch {
    /* ignore */
  }
  workerDid = did;
  return did;
}

/** Record one brief processed (used by auto-respond loop). */
export function noteBriefProcessed(tokens = 0): void {
  const now = Date.now();
  recentBriefs.push(now);
  totalBriefs++;
  totalTokensDecoded += tokens;
  lastBriefAt = now;
  // Prune the rolling window in place.
  while (recentBriefs.length > 0 && now - recentBriefs[0] > BRIEF_WINDOW_MS) {
    recentBriefs.shift();
  }
}

export function noteError(msg: string): void {
  lastError = msg;
}

export function setFirehoseConnected(connected: boolean): void {
  firehoseConnected = connected;
  if (connected) lastError = null;
}

export interface DaemonSnapshot {
  did: string;
  uptimeMs: number;
  briefsPerMinute: number;
  totalBriefs: number;
  totalTokensDecoded: number;
  lastBriefAt: number | null;
  lastError: string | null;
  firehoseConnected: boolean;
}

export function getDaemonSnapshot(): DaemonSnapshot {
  return {
    did: getWorkerDid(),
    uptimeMs: Date.now() - SESSION_STARTED_AT,
    briefsPerMinute: recentBriefs.length,
    totalBriefs,
    totalTokensDecoded,
    lastBriefAt,
    lastError,
    firehoseConnected,
  };
}

/** Human-readable uptime, e.g. "2h 14m" / "47s". */
export function formatUptime(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

/** Short DID display: `did:web:browser:01HEY…`. */
export function shortDid(did: string): string {
  const tail = did.slice("did:web:browser:".length);
  return `did:web:browser:${tail.slice(0, 6)}…`;
}
