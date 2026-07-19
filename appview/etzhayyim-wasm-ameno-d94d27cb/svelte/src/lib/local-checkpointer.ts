/**
 * local-checkpointer.ts — Persistent state for the ameno LangGraph runtime.
 *
 * Subclass of `MemorySaver` that mirrors its in-memory `storage` and `writes`
 * maps to localStorage on every super-step. On startup we rehydrate, so a
 * reload (or tab-restart) lands the agent back exactly where it left off:
 * conversation messages, surprise prediction, tool history, reflection
 * iteration — everything that lives in graph state.
 *
 * API shape is intentionally compatible with the eventual MstCheckpointSaver
 * (ADR-2605171800, `@etzhayyim/sdk/checkpointer`); when that lands we swap
 * one line in graph.ts.
 *
 * Authoritative ADR: 90-docs/adr/2605191135-ameno-tier2-daemon-residency.md
 */
import { MemorySaver } from "@langchain/langgraph";
import type { CheckpointMetadata } from "@langchain/langgraph";
import type { PendingWrite } from "@langchain/langgraph-checkpoint";
import type { RunnableConfig } from "@langchain/core/runnables";

const STORAGE_KEY = "ameno.checkpointer.v1";
const FLUSH_DEBOUNCE_MS = 250;
/** localStorage usually has a 5–10 MB quota per origin; cap our slice at 3 MB
 *  and evict the lex-smallest thread (rough proxy for oldest) when over. */
const MAX_PAYLOAD_BYTES = 3 * 1024 * 1024;

interface SerialLeaf {
  c: string;
  m: string;
  p?: string;
}
interface SerialStorage {
  [thread: string]: { [ns: string]: { [id: string]: SerialLeaf } };
}
interface SerialWriteEntry {
  k: string;
  taskId: string;
  channel: string;
  type: string;
  value: string;
}

function u8ToB64(u8: Uint8Array): string {
  const CHUNK = 0x8000;
  let bin = "";
  for (let i = 0; i < u8.length; i += CHUNK) {
    bin += String.fromCharCode.apply(
      null,
      Array.from(u8.subarray(i, i + CHUNK)),
    );
  }
  return btoa(bin);
}

function b64ToU8(b64: string): Uint8Array {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

interface MemorySaverInternal extends MemorySaver {
  storage: Record<
    string,
    Record<string, Record<string, [Uint8Array, Uint8Array, string | undefined]>>
  >;
  writes: Record<string, Record<string, [string, string, Uint8Array]>>;
}

export class LocalCheckpointer extends MemorySaver {
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private suppressFlush = false;

  constructor() {
    super();
    this.suppressFlush = true;
    try {
      this.loadFromStorage();
    } finally {
      this.suppressFlush = false;
    }
  }

  private loadFromStorage(): void {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(STORAGE_KEY);
    } catch {
      return;
    }
    if (!raw) return;
    let parsed: { storage?: SerialStorage; writes?: SerialWriteEntry[] };
    try {
      parsed = JSON.parse(raw) as typeof parsed;
    } catch {
      return;
    }
    const self = this as unknown as MemorySaverInternal;
    // Storage
    if (parsed.storage) {
      for (const [thread, nss] of Object.entries(parsed.storage)) {
        self.storage[thread] = self.storage[thread] ?? {};
        for (const [ns, ids] of Object.entries(nss)) {
          self.storage[thread][ns] = self.storage[thread][ns] ?? {};
          for (const [id, leaf] of Object.entries(ids)) {
            self.storage[thread][ns][id] = [b64ToU8(leaf.c), b64ToU8(leaf.m), leaf.p];
          }
        }
      }
    }
    // Writes
    if (Array.isArray(parsed.writes)) {
      for (const e of parsed.writes) {
        self.writes[e.k] = self.writes[e.k] ?? {};
        self.writes[e.k][e.taskId] = [e.channel, e.type, b64ToU8(e.value)];
      }
    }
  }

  private scheduleFlush(): void {
    if (this.suppressFlush) return;
    if (this.flushTimer !== null) return;
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      this.flushToStorage();
    }, FLUSH_DEBOUNCE_MS);
  }

  /** Force-flush immediately. Useful before `beforeunload`. */
  flushNow(): void {
    if (this.flushTimer !== null) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    this.flushToStorage();
  }

  private flushToStorage(): void {
    const self = this as unknown as MemorySaverInternal;
    const storage: SerialStorage = {};
    for (const [thread, nss] of Object.entries(self.storage)) {
      storage[thread] = {};
      for (const [ns, ids] of Object.entries(nss)) {
        storage[thread][ns] = {};
        for (const [id, leaf] of Object.entries(ids)) {
          storage[thread][ns][id] = {
            c: u8ToB64(leaf[0]),
            m: u8ToB64(leaf[1]),
            p: leaf[2],
          };
        }
      }
    }
    const writes: SerialWriteEntry[] = [];
    for (const [k, taskMap] of Object.entries(self.writes)) {
      for (const [taskId, tuple] of Object.entries(taskMap)) {
        writes.push({
          k,
          taskId,
          channel: tuple[0],
          type: tuple[1],
          value: u8ToB64(tuple[2]),
        });
      }
    }
    let payload = JSON.stringify({ storage, writes });
    while (payload.length > MAX_PAYLOAD_BYTES) {
      if (!this.evictOldestThread(storage)) break;
      payload = JSON.stringify({ storage, writes });
    }
    try {
      localStorage.setItem(STORAGE_KEY, payload);
    } catch (e) {
      console.warn("ameno LocalCheckpointer: flush failed", e);
    }
  }

  private evictOldestThread(storage: SerialStorage): boolean {
    const threads = Object.keys(storage);
    if (threads.length <= 1) return false;
    threads.sort();
    delete storage[threads[0]];
    const self = this as unknown as MemorySaverInternal;
    delete self.storage[threads[0]];
    return true;
  }

  /** Drop all persisted state. */
  async clear(): Promise<void> {
    const self = this as unknown as MemorySaverInternal;
    self.storage = {};
    self.writes = {};
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }

  // ── MemorySaver overrides ────────────────────────────────────────────────

  async put(
    config: RunnableConfig,
    checkpoint: Parameters<MemorySaver["put"]>[1],
    metadata: CheckpointMetadata,
  ): Promise<RunnableConfig> {
    const r = await super.put(config, checkpoint, metadata);
    this.scheduleFlush();
    return r;
  }

  async putWrites(
    config: RunnableConfig,
    writes: PendingWrite[],
    taskId: string,
  ): Promise<void> {
    await super.putWrites(config, writes, taskId);
    this.scheduleFlush();
  }

  async deleteThread(threadId: string): Promise<void> {
    await super.deleteThread(threadId);
    this.scheduleFlush();
  }
}
