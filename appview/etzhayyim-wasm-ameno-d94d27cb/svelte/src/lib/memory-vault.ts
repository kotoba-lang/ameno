/**
 * memory-vault.ts — Long-term encrypted memory store for the ameno agent.
 *
 * IndexedDB-backed. Plaintext { content, tags } is AES-GCM encrypted with
 * the per-origin key from private-vault.ts before being persisted. A
 * MiniLM embedding (384-d Float32) is computed at save time and stored
 * alongside (plaintext, by design — it is on the surface of MST-anchored
 * documents too) so cosine search at recall time is O(N) without re-embed.
 *
 * Record schema mirrors `com.etzhayyim.memory.record` so a future swap to
 * `@etzhayyim/sdk/encryptedWrite` (ADR-2605181100) is a one-shot port.
 *
 * Authoritative ADR: 90-docs/adr/2605191206-ameno-long-term-memory-vault.md
 */
import { cosine, embed, isEmbeddingReady } from "./embedding";
import { ensureKey } from "./private-vault";

const DB_NAME = "ameno-memory-v1";
const DB_VERSION = 1;
const STORE = "memories";

/** Stored shape inside IndexedDB. `id` is auto-assigned. */
interface StoredRow {
  id?: number;
  iv: string;          // base64 of 12-byte IV
  ciphertext: string;  // base64 of GCM ciphertext+tag
  embedding: string;   // base64 of Float32Array(384) buffer
  createdAt: number;   // epoch ms
  tags: string[];      // plaintext, used as filter
}

/** Decrypted public shape returned to callers. */
export interface MemoryRecord {
  id: number;
  content: string;
  tags: string[];
  createdAt: number;
}

/** Recall result with similarity. */
export interface MemoryHit extends MemoryRecord {
  similarity: number;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const os = db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        os.createIndex("createdAt", "createdAt");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
  });
  return dbPromise;
}

function u8ToB64(u8: Uint8Array): string {
  const CHUNK = 0x8000;
  let bin = "";
  for (let i = 0; i < u8.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, Array.from(u8.subarray(i, i + CHUNK)));
  }
  return btoa(bin);
}

function b64ToU8(b64: string): Uint8Array {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

function f32ToB64(f: Float32Array): string {
  return u8ToB64(new Uint8Array(f.buffer, f.byteOffset, f.byteLength));
}

function b64ToF32(b64: string): Float32Array {
  const u8 = b64ToU8(b64);
  // Copy into an aligned ArrayBuffer; subarray of larger buffer can mis-align.
  const aligned = new Uint8Array(u8.byteLength);
  aligned.set(u8);
  return new Float32Array(aligned.buffer);
}

async function encryptPayload(content: string, tags: string[]): Promise<{
  iv: Uint8Array;
  ciphertext: Uint8Array;
}> {
  const key = await ensureKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const plaintext = new TextEncoder().encode(JSON.stringify({ content, tags }));
  const ciphertext = new Uint8Array(
    await crypto.subtle.encrypt(
      { name: "AES-GCM", iv: iv as BufferSource },
      key,
      plaintext as BufferSource,
    ),
  );
  return { iv, ciphertext };
}

async function decryptRow(row: StoredRow): Promise<MemoryRecord> {
  const key = await ensureKey();
  const iv = b64ToU8(row.iv);
  const ct = b64ToU8(row.ciphertext);
  const pt = new Uint8Array(
    await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: iv as BufferSource },
      key,
      ct as BufferSource,
    ),
  );
  const obj = JSON.parse(new TextDecoder().decode(pt)) as {
    content?: unknown;
    tags?: unknown;
  };
  return {
    id: row.id ?? 0,
    content: typeof obj.content === "string" ? obj.content : "",
    tags: Array.isArray(obj.tags) ? obj.tags.filter((t): t is string => typeof t === "string") : [],
    createdAt: row.createdAt,
  };
}

/**
 * Save a memory. Embeds content with MiniLM, encrypts, persists.
 * Throws if the embedding pipeline is not ready — callers must check.
 */
export async function saveMemory(content: string, tags: string[] = []): Promise<number> {
  if (!isEmbeddingReady()) {
    throw new Error("MiniLM embedding pipeline must be loaded before saving memory");
  }
  const text = content.trim();
  if (!text) throw new Error("memory content is empty");

  const embedding = await embed(text);
  const { iv, ciphertext } = await encryptPayload(text, tags);
  const row: StoredRow = {
    iv: u8ToB64(iv),
    ciphertext: u8ToB64(ciphertext),
    embedding: f32ToB64(embedding),
    createdAt: Date.now(),
    tags,
  };

  const db = await openDB();
  return new Promise<number>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const req = tx.objectStore(STORE).add(row);
    req.onsuccess = () => resolve(req.result as number);
    req.onerror = () => reject(req.error ?? new Error("memory put failed"));
  });
}

/**
 * Cosine search over all stored memories. Linear scan; fine for the few-K
 * range we expect. Returns top-K rows sorted by similarity descending.
 */
export async function searchMemory(query: string, topK = 3): Promise<MemoryHit[]> {
  if (!isEmbeddingReady()) {
    throw new Error("MiniLM embedding pipeline must be loaded before recall");
  }
  const q = query.trim();
  if (!q) return [];
  const qv = await embed(q);

  const db = await openDB();
  const rows = await new Promise<StoredRow[]>((resolve, reject) => {
    const out: StoredRow[] = [];
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).openCursor();
    req.onsuccess = () => {
      const c = req.result;
      if (!c) return resolve(out);
      out.push(c.value as StoredRow);
      c.continue();
    };
    req.onerror = () => reject(req.error ?? new Error("memory scan failed"));
  });

  const scored = rows.map((row) => ({
    row,
    sim: cosine(qv, b64ToF32(row.embedding)),
  }));
  scored.sort((a, b) => b.sim - a.sim);
  const top = scored.slice(0, Math.max(1, Math.min(20, topK)));
  const out: MemoryHit[] = [];
  for (const { row, sim } of top) {
    try {
      const rec = await decryptRow(row);
      out.push({ ...rec, similarity: sim });
    } catch (e) {
      console.warn("memory decrypt failed for row", row.id, e);
    }
  }
  return out;
}

/** Total stored memory count. */
export async function countMemories(): Promise<number> {
  const db = await openDB();
  return new Promise<number>((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).count();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("memory count failed"));
  });
}

/** Drop every memory record. */
export async function clearMemoryVault(): Promise<void> {
  const db = await openDB();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const req = tx.objectStore(STORE).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error ?? new Error("memory clear failed"));
  });
}
