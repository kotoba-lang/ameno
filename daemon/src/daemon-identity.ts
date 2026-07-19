/**
 * daemon-identity.ts — Persistent worker DID + heartbeat for the daemon.
 *
 * Mirrors the browser's lib/daemon.ts but uses a file under AMENO_HOME
 * instead of localStorage, and uses `did:web:host:<hostname>-<uuid>` to
 * distinguish daemon instances from browser tabs.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { hostname } from "node:os";

const SESSION_STARTED_AT = Date.now();
const BRIEF_WINDOW_MS = 60_000;

let workerDid: string | null = null;
const recentBriefs: number[] = [];
let totalBriefs = 0;
let totalTokensDecoded = 0;
let lastError: string | null = null;
let firehoseConnected = false;

function newUuid(): string {
  const r = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2, 14);
  return r.replace(/-/g, "");
}

function safeHostname(): string {
  const h = hostname();
  return h.replace(/[^a-zA-Z0-9-]/g, "").slice(0, 32) || "anon";
}

export function getWorkerDid(didPath: string): string {
  if (workerDid) return workerDid;
  try {
    if (existsSync(didPath)) {
      const stored = readFileSync(didPath, "utf8").trim();
      if (stored.startsWith("did:web:host:")) {
        workerDid = stored;
        return workerDid;
      }
    }
  } catch {
    /* ignore */
  }
  const did = `did:web:host:${safeHostname()}-${newUuid()}`;
  try {
    mkdirSync(dirname(didPath), { recursive: true });
    writeFileSync(didPath, did, "utf8");
  } catch {
    /* ignore */
  }
  workerDid = did;
  return did;
}

export function noteBriefProcessed(tokens = 0): void {
  const now = Date.now();
  recentBriefs.push(now);
  totalBriefs++;
  totalTokensDecoded += tokens;
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

export function getDaemonSnapshot(currentDid: string): {
  did: string;
  uptimeMs: number;
  briefsPerMinute: number;
  totalBriefs: number;
  totalTokensDecoded: number;
  lastError: string | null;
  firehoseConnected: boolean;
} {
  return {
    did: currentDid,
    uptimeMs: Date.now() - SESSION_STARTED_AT,
    briefsPerMinute: recentBriefs.length,
    totalBriefs,
    totalTokensDecoded,
    lastError,
    firehoseConnected,
  };
}
