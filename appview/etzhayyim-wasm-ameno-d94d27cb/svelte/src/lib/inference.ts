/**
 * Re-export from @etzhayyim/ameno package.
 * @see packages/llm/inference/ameno/src/inference.ts
 */
export {
  MODELS,
  checkWebGPU,
  getGPUInfo,
  loadModel,
  setLoraAdapters,
  getActiveAdapterIds,
  generate,
  type InferenceState,
  type ChatMessage,
  type GenerationStats,
} from "@etzhayyim/ameno/inference";
