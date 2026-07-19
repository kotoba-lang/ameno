/**
 * did-auth.ts — Daemon-side did:key Ed25519 nonce challenge verifier.
 *
 * Companion to svelte/src/lib/did-auth.ts. Issues short-lived single-use
 * nonces from `/auth/nonce` and verifies `Authorization: DIDSig
 * <did:key>:<nonce_id>:<sig_b64url>` headers using Ed25519 over the
 * `${nonce_id}.${nonce}` payload.
 *
 * Authoritative ADR: 90-docs/adr/2605191657-ameno-daemon-did-auth.md
 */
import { ed25519 } from "@noble/curves/ed25519";
import { base58 } from "@scure/base";
import { randomBytes } from "node:crypto";

const NONCE_TTL_MS = 60_000;

/** Allowed did:key list from `AMENO_ALLOWED_DIDS` env (comma-separated).
 *  Empty / unset → no allowlist (any well-formed did:key is accepted).
 *  Set        → only the listed DIDs may authenticate via DIDSig.
 *  ADR-2605191641. */
const ALLOWED_DIDS: ReadonlySet<string> = new Set(
  (process.env.AMENO_ALLOWED_DIDS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.startsWith("did:key:z")),
);

export function isDidAllowed(did: string): boolean {
  if (ALLOWED_DIDS.size === 0) return true;
  return ALLOWED_DIDS.has(did);
}

export function getAllowedDids(): readonly string[] {
  return [...ALLOWED_DIDS];
}

interface NonceEntry {
  nonce: string;
  expiresAtMs: number;
  /** True after first verification — single-use semantics. */
  used: boolean;
}

const nonces = new Map<string, NonceEntry>();

function bytesToB64Url(b: Uint8Array): string {
  return Buffer.from(b).toString("base64url");
}

function b64UrlToBytes(s: string): Uint8Array {
  return new Uint8Array(Buffer.from(s, "base64url"));
}

/** Decode a did:key:z…(Ed25519)to its 32-byte public key. */
function decodeDidKey(did: string): Uint8Array {
  if (!did.startsWith("did:key:z")) {
    throw new Error(`not a did:key: ${did}`);
  }
  const body = base58.decode(did.slice("did:key:z".length));
  if (body.length !== 34 || body[0] !== 0xed || body[1] !== 0x01) {
    throw new Error("did:key is not Ed25519");
  }
  return body.slice(2);
}

/**
 * Issue a new nonce. Caller responsible for cleanup via the sweep
 * timer started at module load.
 */
export function issueNonce(): { nonce_id: string; nonce: string; expires_at_ms: number } {
  const id = randomBytes(8).toString("base64url");
  const nonce = randomBytes(16).toString("base64url");
  const expiresAtMs = Date.now() + NONCE_TTL_MS;
  nonces.set(id, { nonce, expiresAtMs, used: false });
  return { nonce_id: id, nonce, expires_at_ms: expiresAtMs };
}

/** Periodic sweep to bound the in-memory nonce table. */
const sweepTimer = setInterval(() => {
  const now = Date.now();
  for (const [id, entry] of nonces) {
    if (entry.used || entry.expiresAtMs <= now) nonces.delete(id);
  }
}, 5_000);
// Don't keep the event loop alive just for the sweep.
if (typeof sweepTimer.unref === "function") sweepTimer.unref();

export interface VerificationResult {
  ok: boolean;
  did?: string;
  error?: string;
}

/**
 * Verify a `DIDSig <did>:<nonce_id>:<sig_b64url>` Authorization header.
 *
 * Returns `{ ok: true, did }` on success. On any failure path returns
 * `{ ok: false, error }` so the caller can respond 401 with an
 * informative (but not too revealing) reason.
 */
export function verifyDidSig(authHeader: string | undefined): VerificationResult {
  if (!authHeader) return { ok: false, error: "missing Authorization header" };
  if (!authHeader.startsWith("DIDSig ")) {
    return { ok: false, error: "not a DIDSig header" };
  }
  const body = authHeader.slice("DIDSig ".length).trim();
  // did:key:z<base58>:<nonce_id>:<sig> — split from the right so a
  // colon inside the did:key prefix doesn't trip us up (it doesn't,
  // but be defensive).
  const sigIdx = body.lastIndexOf(":");
  if (sigIdx < 0) return { ok: false, error: "malformed DIDSig: no sig separator" };
  const idIdx = body.lastIndexOf(":", sigIdx - 1);
  if (idIdx < 0) return { ok: false, error: "malformed DIDSig: no nonce_id separator" };

  const did = body.slice(0, idIdx);
  const nonceId = body.slice(idIdx + 1, sigIdx);
  const sigB64 = body.slice(sigIdx + 1);

  if (!isDidAllowed(did)) {
    return { ok: false, error: "did not in allowlist" };
  }

  const entry = nonces.get(nonceId);
  if (!entry) return { ok: false, error: "nonce unknown or already consumed" };
  if (entry.used) return { ok: false, error: "nonce already consumed" };
  if (entry.expiresAtMs <= Date.now()) {
    nonces.delete(nonceId);
    return { ok: false, error: "nonce expired" };
  }

  let pub: Uint8Array;
  try {
    pub = decodeDidKey(did);
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  let sig: Uint8Array;
  try {
    sig = b64UrlToBytes(sigB64);
  } catch {
    return { ok: false, error: "signature is not base64url" };
  }
  const payload = new TextEncoder().encode(`${nonceId}.${entry.nonce}`);
  let valid = false;
  try {
    valid = ed25519.verify(sig, payload, pub);
  } catch {
    valid = false;
  }
  if (!valid) return { ok: false, error: "signature verification failed" };

  // Single-use: mark consumed (sweep removes shortly).
  entry.used = true;
  return { ok: true, did };
}

// Re-export the b64 helpers for tests / debug.
export const _internals = { bytesToB64Url, b64UrlToBytes };
