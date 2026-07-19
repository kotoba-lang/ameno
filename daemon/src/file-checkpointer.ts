/**
 * file-checkpointer.ts — Disk-persisted MemorySaver for the headless daemon.
 *
 * Mirrors the design of the svelte appview's LocalCheckpointer
 * (ADR-2605191135) but writes to `${AMENO_HOME}/checkpointer.json`
 * instead of localStorage. Same debounce + size cap discipline.
 *
 * Authoritative ADR: 90-docs/adr/2605191229-ameno-daemon-path-a-bun-langgraph.md
 */
import { MemorySaver } from "@langchain/langgraph";
import type { CheckpointMetadata } from "@langchain/langgraph";
import type { PendingWrite } from "@langchain/langgraph-checkpoint";
import type { RunnableConfig } from "@langchain/core/runnables";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";

const FLUSH_DEBOUNCE_MS = 500;
/** 16 MB cap. We have actual disk here, not a localStorage quota. */
const MAX_PAYLOAD_BYTES = 16 * 1024 * 1024;

interface SerialLeaf { c: string; m: string; p?: string }
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
  return Buffer.from(u8).toString("base64");
}
function b64ToU8(b64: string): Uint8Array {
  return new Uint8Array(Buffer.from(b64, "base64"));
}

interface MemorySaverInternal extends MemorySaver {
  storage: Record<
    string,
    Record<string, Record<string, [Uint8Array, Uint8Array, string | undefined]>>
  >;
  writes: Record<string, Record<string, [string, string, Uint8Array]>>;
}

export class FileCheckpointer extends MemorySaver {
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private suppressFlush = false;
  private readonly path: string;

  constructor(filePath: string) {
    super();
    this.path = filePath;
    this.suppressFlush = true;
    try {
      this.loadFromDisk();
    } finally {
      this.suppressFlush = false;
    }
  }

  private loadFromDisk(): void {
    if (!existsSync(this.path)) return;
    let raw: string;
    try {
      raw = readFileSync(this.path, "utf8");
    } catch {
      return;
    }
    let parsed: { storage?: SerialStorage; writes?: SerialWriteEntry[] };
    try {
      parsed = JSON.parse(raw) as typeof parsed;
    } catch {
      return;
    }
    const self = this as unknown as MemorySaverInternal;
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
      this.flushToDisk();
    }, FLUSH_DEBOUNCE_MS);
  }

  flushNow(): void {
    if (this.flushTimer !== null) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    this.flushToDisk();
  }

  private flushToDisk(): void {
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
      mkdirSync(dirname(this.path), { recursive: true });
      writeFileSync(this.path, payload, "utf8");
    } catch (e) {
      console.warn("ameno FileCheckpointer: flush failed", e);
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
