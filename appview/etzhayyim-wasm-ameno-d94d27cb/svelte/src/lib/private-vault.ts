/**
 * private-vault.ts — Browser-local AES-256-GCM field encryption.
 *
 * Implements the etzhayyim `signal:v1:{ciphertext}` field convention
 * (10-protocol/atproto/src/signal.ts) for ameno inference outputs that
 * the user marks as private. The server only sees ciphertext; per the
 * Vault zero-knowledge invariant in CLAUDE.md, plaintext / raw key
 * never leave the client.
 *
 * Storage: a 256-bit AES-GCM key per browser origin, persisted in
 * localStorage as base64. Loss of localStorage = lost data; this is the
 * acceptable failure mode for "private mode" without a real Vault wrap.
 * A future phase can swap this for a vault.etzhayyim.com-wrapped device key.
 */

const KEY_STORAGE = "ameno.private-vault.key.v1";
const FIELD_PREFIX = "signal:v1:";
const IV_BYTES = 12;

let cachedKey: CryptoKey | null = null;

function bytesToBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function importKey(rawB64: string): Promise<CryptoKey> {
  const raw = base64ToBytes(rawB64);
  return crypto.subtle.importKey("raw", raw as BufferSource, { name: "AES-GCM" }, false, [
    "encrypt",
    "decrypt",
  ]);
}

/**
 * Get-or-create the per-origin AES-GCM key.
 *
 * First call generates a fresh 256-bit key, persists its base64 form to
 * localStorage, and returns the imported CryptoKey. Subsequent calls
 * return the cached or re-imported key.
 */
export async function ensureKey(): Promise<CryptoKey> {
  if (cachedKey) return cachedKey;
  const stored = localStorage.getItem(KEY_STORAGE);
  if (stored) {
    cachedKey = await importKey(stored);
    return cachedKey;
  }
  const raw = crypto.getRandomValues(new Uint8Array(32));
  localStorage.setItem(KEY_STORAGE, bytesToBase64(raw));
  cachedKey = await crypto.subtle.importKey("raw", raw, { name: "AES-GCM" }, false, [
    "encrypt",
    "decrypt",
  ]);
  return cachedKey;
}

/** True if the field already carries the `signal:v1:` envelope. */
export function isEncrypted(field: string): boolean {
  return typeof field === "string" && field.startsWith(FIELD_PREFIX);
}

/**
 * Encrypt a plaintext string into a `signal:v1:{ciphertext}` envelope.
 *
 * Layout: prefix || base64(iv(12) || ciphertext(N) || tag(16)).
 * Idempotent: returns the input unchanged if it already carries the prefix.
 */
export async function encryptText(plain: string): Promise<string> {
  if (isEncrypted(plain)) return plain;
  const key = await ensureKey();
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const ct = new Uint8Array(
    await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, new TextEncoder().encode(plain)),
  );
  const wrapped = new Uint8Array(iv.length + ct.length);
  wrapped.set(iv, 0);
  wrapped.set(ct, iv.length);
  return FIELD_PREFIX + bytesToBase64(wrapped);
}

/**
 * Decrypt a `signal:v1:{ciphertext}` envelope back to plaintext.
 *
 * Idempotent: returns the input unchanged if it does not carry the prefix
 * (so the caller can apply unconditionally to listHistory items).
 * Throws if the envelope is present but the key is missing or the GCM tag
 * fails (e.g. wrong key after a localStorage clear).
 */
export async function decryptText(ciphered: string): Promise<string> {
  if (!isEncrypted(ciphered)) return ciphered;
  const key = await ensureKey();
  const wrapped = base64ToBytes(ciphered.slice(FIELD_PREFIX.length));
  const iv = wrapped.subarray(0, IV_BYTES);
  const ct = wrapped.subarray(IV_BYTES);
  const pt = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: iv as BufferSource },
    key,
    ct as BufferSource,
  );
  return new TextDecoder().decode(pt);
}
