/**
 * embedding.ts — Sentence embedding for active-inference surprise (Tier C).
 *
 * Loads `Xenova/all-MiniLM-L6-v2` via transformers.js the first time
 * `embed()` is called, on the WASM backend (MediaPipe LLM owns WebGPU).
 * Subsequent calls reuse the cached pipeline. Vectors are L2-normalized
 * so cosine similarity reduces to a plain dot product.
 *
 * Authoritative ADR: 90-docs/adr/2605191120-ameno-embedding-surprise-tier-c.md
 */
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

let extractor: FeatureExtractionPipeline | null = null;
let loadPromise: Promise<FeatureExtractionPipeline> | null = null;
/** 0..100. Mirrors transformers.js progress events. */
let loadProgress = 0;
let loadFailed: Error | null = null;

export function getEmbeddingProgress(): number {
  return loadProgress;
}

export function getEmbeddingError(): Error | null {
  return loadFailed;
}

export function isEmbeddingReady(): boolean {
  return extractor !== null;
}

/**
 * Lazy-load the MiniLM pipeline. Idempotent — concurrent callers share
 * the same in-flight promise. Throws if the load eventually fails.
 */
export function ensureEmbeddingLoaded(
  onProgress?: (pct: number) => void,
): Promise<FeatureExtractionPipeline> {
  if (extractor) return Promise.resolve(extractor);
  if (loadPromise) return loadPromise;

  loadProgress = 0;
  loadFailed = null;
  loadPromise = (async () => {
    try {
      // The `pipeline()` overload set is too wide for TS to enumerate
      // here ("union type too complex"), so we cast the call site to
      // `unknown` first and assert the concrete pipeline type after.
      interface PipelineOpts {
        device: string;
        dtype: string;
        progress_callback: (info: unknown) => void;
      }
      const opts: PipelineOpts = {
        device: "wasm",
        dtype: "fp32",
        progress_callback: (info: unknown) => {
          const o = info as { status?: string; progress?: number };
          if (o?.status === "progress" && typeof o.progress === "number") {
            const pct = Math.min(99, Math.max(loadProgress, Math.round(o.progress)));
            loadProgress = pct;
            onProgress?.(pct);
          } else if (o?.status === "ready") {
            loadProgress = 100;
            onProgress?.(100);
          }
        },
      };
      const ext = await (
        pipeline as unknown as (
          task: string,
          model: string,
          opts: PipelineOpts,
        ) => Promise<FeatureExtractionPipeline>
      )("feature-extraction", MODEL_ID, opts);
      extractor = ext;
      loadProgress = 100;
      onProgress?.(100);
      return ext;
    } catch (e) {
      loadFailed = e instanceof Error ? e : new Error(String(e));
      throw loadFailed;
    } finally {
      loadPromise = null;
    }
  })();
  return loadPromise;
}

/**
 * Compute a normalized embedding for `text`. Requires `ensureEmbeddingLoaded()`
 * to have completed; throws otherwise so callers do not silently fall back
 * to a stale or empty vector.
 */
export async function embed(text: string): Promise<Float32Array> {
  if (!extractor) {
    throw new Error("embedding pipeline not loaded; call ensureEmbeddingLoaded() first");
  }
  const t = text.trim();
  if (!t) return new Float32Array(384);
  const out = (await extractor(t, { pooling: "mean", normalize: true })) as {
    data: Float32Array;
  };
  return out.data;
}

/**
 * Cosine similarity over two L2-normalized vectors = dot product. Returns a
 * value in roughly [-1, 1]; in practice near [0, 1] for sentence embeddings.
 */
export function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`cosine: vector dimension mismatch ${a.length} != ${b.length}`);
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}
