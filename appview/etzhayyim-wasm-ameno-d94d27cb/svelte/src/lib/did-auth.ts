/**
 * did-auth.ts — Browser-side did:key Ed25519 keypair + nonce signer.
 *
 * Replaces the bearer-token auth path (ADR-2605191407 §sec) with a
 * challenge-response flow: the daemon issues a single-use nonce via
 * GET /auth/nonce, we sign it with our persistent did:key private
 * key, and send the result back as
 *   Authorization: DIDSig <did:key>:<nonce_id>:<sig_b64url>
 *
 * Authoritative ADR: 90-docs/adr/2605191657-ameno-daemon-did-auth.md
 */
import { ed25519 } from "@noble/curves/ed25519";
import { base58 } from "@scure/base";

const KEYPAIR_STORAGE = "ameno.did-auth.keypair.v1";

interface StoredKeypair {
  /** Public key as 32 raw bytes, base64url-encoded. */
  pub: string;
  /** Private key as 32 raw bytes, base64url-encoded. localStorage —
   *  no HSM. Acceptable for the dev / single-operator deployments
   *  ADR-2605191657 §scope defines as v0.1. */
  priv: string;
  /** Self-describing did:key form for convenience. */
  did: string;
}

function bytesToB64Url(b: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < b.length; i++) bin += String.fromCharCode(b[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64UrlToBytes(s: string): Uint8Array {
  const pad = s.length % 4 === 0 ? "" : "=".repeat(4 - (s.length % 4));
  const std = (s + pad).replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(std);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Encode an Ed25519 public key as a `did:key:z…` string. The leading
 * `z` marks base58btc multibase; the bytes are
 *   0xed 0x01 || pubkey
 * (multicodec ed25519-pub = 0xed, varint suffix 0x01).
 */
function encodeDidKey(pub: Uint8Array): string {
  if (pub.length !== 32) throw new Error(`Ed25519 pubkey must be 32 bytes, got ${pub.length}`);
  const tagged = new Uint8Array(2 + 32);
  tagged[0] = 0xed;
  tagged[1] = 0x01;
  tagged.set(pub, 2);
  return "did:key:z" + base58.encode(tagged);
}

/** Decode a did:key:z… string back to its 32-byte Ed25519 public key. */
export function decodeDidKey(did: string): Uint8Array {
  if (!did.startsWith("did:key:z")) {
    throw new Error(`not a did:key (must start with did:key:z): ${did}`);
  }
  const body = base58.decode(did.slice("did:key:z".length));
  if (body.length !== 34 || body[0] !== 0xed || body[1] !== 0x01) {
    throw new Error("did:key is not Ed25519 (multicodec prefix mismatch)");
  }
  return body.slice(2);
}

/** Get-or-create the persistent did:key keypair for this browser. */
function ensureKeypair(): StoredKeypair {
  try {
    const raw = localStorage.getItem(KEYPAIR_STORAGE);
    if (raw) {
      const parsed = JSON.parse(raw) as StoredKeypair;
      if (parsed.pub && parsed.priv && parsed.did?.startsWith("did:key:z")) return parsed;
    }
  } catch {
    /* fall through to generate */
  }
  const priv = ed25519.utils.randomPrivateKey();
  const pub = ed25519.getPublicKey(priv);
  const did = encodeDidKey(pub);
  const stored: StoredKeypair = {
    pub: bytesToB64Url(pub),
    priv: bytesToB64Url(priv),
    did,
  };
  try {
    localStorage.setItem(KEYPAIR_STORAGE, JSON.stringify(stored));
  } catch {
    /* private browsing — keep in-memory only */
  }
  return stored;
}

/** Public DID this tab will sign as. */
export function getAuthDid(): string {
  return ensureKeypair().did;
}

/** True when DID auth is initialised (always true after first call). */
export function isDidAuthReady(): boolean {
  try {
    return Boolean(localStorage.getItem(KEYPAIR_STORAGE));
  } catch {
    return false;
  }
}

interface NonceResponse {
  nonce_id: string;
  nonce: string;
  /** Server clock when the nonce expires (ms since epoch). */
  expires_at_ms: number;
}

/** Fetch a fresh nonce from the daemon's /auth/nonce endpoint. */
async function fetchNonce(baseUrl: string, signal?: AbortSignal): Promise<NonceResponse> {
  const r = await fetch(stripTrailingSlash(baseUrl) + "/auth/nonce", {
    method: "GET",
    credentials: "omit",
    signal,
  });
  if (!r.ok) throw new Error(`/auth/nonce HTTP ${r.status}`);
  const body = (await r.json()) as Partial<NonceResponse>;
  if (!body.nonce_id || !body.nonce || !body.expires_at_ms) {
    throw new Error("/auth/nonce payload missing required fields");
  }
  return body as NonceResponse;
}

function stripTrailingSlash(s: string): string {
  return s.endsWith("/") ? s.slice(0, -1) : s;
}

/**
 * Build a DIDSig Authorization header value for the given daemon base URL.
 *
 * Steps:
 *   1. GET /auth/nonce → { nonce_id, nonce, expires_at_ms }
 *   2. Sign the ASCII string `${nonce_id}.${nonce}` with the local
 *      did:key private key.
 *   3. Return `DIDSig <did:key>:<nonce_id>:<sig_b64url>`.
 *
 * Throws on transport or sign failures. Caller should fall back to
 * bearer token if available.
 */
export async function buildDidSigHeader(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<string> {
  const kp = ensureKeypair();
  const { nonce_id, nonce } = await fetchNonce(baseUrl, signal);
  const payload = new TextEncoder().encode(`${nonce_id}.${nonce}`);
  const priv = b64UrlToBytes(kp.priv);
  const sig = ed25519.sign(payload, priv);
  return `DIDSig ${kp.did}:${nonce_id}:${bytesToB64Url(sig)}`;
}

/** For UI display: short fingerprint of the public key. */
export function shortDidKey(): string {
  const did = ensureKeypair().did;
  // did:key:z<base58>... — keep first 12 chars of the base58 suffix.
  const tail = did.slice("did:key:z".length);
  return `did:key:z${tail.slice(0, 12)}…`;
}

/** Wipe the keypair. The next call to `getAuthDid()` regenerates a new
 *  DID — the old one becomes orphaned, daemons that issued nonces to
 *  it will reject the next request. */
export function rotateAuthKey(): string {
  try {
    localStorage.removeItem(KEYPAIR_STORAGE);
  } catch {
    /* ignore */
  }
  return ensureKeypair().did;
}
