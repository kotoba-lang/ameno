import {
  createWorkerExport,
  nsid,
  parseLexiconInput,
  type HostSDK,
  type LexiconOutput,
} from "@etzhayyim/kotodama-host-sdk";

const APP_EMBED_URL = "https://d94d27cb.etzhayyim.com/?embed=1";

type ModelEntry = LexiconOutput<"com.etzhayyim.apps.ameno.listModels">["models"][number];

const MODEL_CATALOG: ReadonlyArray<ModelEntry> = [
  {
    id: "gemma-4-e2b-it",
    displayName: "Gemma 4 E2B (Instruct)",
    huggingfaceModel: "onnx-community/gemma-4-E2B-it-ONNX",
    params: "2.3B effective / 5.1B total",
    context: 128000,
    modalities: ["text", "image", "audio"],
    minVramMb: 4096,
    quantization: "q4f16",
    available: true,
    kernel: "webgpu",
  },
  {
    id: "gemma-4-e4b-it",
    displayName: "Gemma 4 E4B (Instruct)",
    huggingfaceModel: "onnx-community/gemma-4-E4B-it-ONNX",
    params: "4B effective / 9B total",
    context: 128000,
    modalities: ["text", "image", "audio"],
    minVramMb: 6144,
    quantization: "q4f16",
    available: true,
    kernel: "webgpu",
  },
  {
    // Baien — ADR-2605092350. 1.58-bit ternary trunk; runs on CPU/WASM
    // when WebGPU is unavailable. Multimodal grafts (SigLIP image,
    // Whisper audio) are loaded as separate projector heads.
    id: "baien-bitnet-2b",
    displayName: "Baien (BitNet b1.58 2B)",
    huggingfaceModel: "onnx-community/bitnet-b1.58-2B-4T-bf16-ONNX",
    params: "2B (ternary {-1,0,+1})",
    context: 4096,
    modalities: ["text"],
    minVramMb: 0,
    quantization: "ternary-i2s",
    available: true,
    kernel: "wasm-ternary",
  },
  {
    // ADR-2605190824. MediaPipe LLM Inference Web reads the LiteRT `.task`
    // bundle directly (no ONNX hop). Uses the ungated litert-community
    // mirror of Gemma 4 E2B; the original google/* preview repos are
    // HF-gated and require a token proxy (follow-up).
    id: "gemma-4-e2b-mediapipe",
    displayName: "Gemma 4 E2B (MediaPipe LiteRT)",
    huggingfaceModel: "litert-community/gemma-4-E2B-it-litert-lm",
    params: "2B effective",
    context: 32000,
    modalities: ["text"],
    minVramMb: 2048,
    quantization: "q4",
    available: true,
    kernel: "mediapipe-gpu",
  },
  {
    id: "gemma-4-e4b-mediapipe",
    displayName: "Gemma 4 E4B (MediaPipe LiteRT)",
    huggingfaceModel: "litert-community/gemma-4-E4B-it-litert-lm",
    params: "4B effective",
    context: 32000,
    modalities: ["text"],
    minVramMb: 4096,
    quantization: "q4",
    available: true,
    kernel: "mediapipe-gpu",
  },
];

function listModelsHandler(): LexiconOutput<"com.etzhayyim.apps.ameno.listModels"> {
  return { models: [...MODEL_CATALOG] };
}

function cardHomeHandler(): LexiconOutput<"com.etzhayyim.apps.ameno.cardHome"> {
  return {
    title: "Ameno — Browser WebGPU Inference",
    description:
      "Gemma 4 E2B / E4B multimodal LLM, fully in-browser via transformers.js ONNX + WebGPU. Zero server compute.",
    embedUrl: APP_EMBED_URL,
    defaultModelId: "gemma-4-e2b-it",
    availableModels: MODEL_CATALOG.filter((m) => m.available).map((m) => m.id),
    webgpuRequired: false,
    tagline:
      "Per-actor LoRA + RAG. WebGPU (transformers.js) for Gemma 4, MediaPipe LiteRT for Gemma 3n, WASM ternary for Baien.",
  };
}

async function saveResultHandler(
  sdk: HostSDK,
  body: Uint8Array,
): Promise<LexiconOutput<"com.etzhayyim.apps.ameno.saveResult">> {
  const input = parseLexiconInput("com.etzhayyim.apps.ameno.saveResult", body);
  if (!input.modelId) return { status: "failed", error: "modelId required" };
  if (!MODEL_CATALOG.some((m) => m.id === input.modelId)) {
    return { status: "failed", error: `unknown modelId: ${input.modelId}` };
  }
  if (!input.prompt || !input.output) {
    return { status: "failed", error: "prompt and output required" };
  }

  const createdAt = new Date().toISOString();
  const record = {
    $type: "com.etzhayyim.apps.ameno.inferenceResult",
    modelId: input.modelId,
    actorDid: input.actorDid ?? "",
    loraAdapters: input.loraAdapters ?? [],
    prompt: input.prompt,
    output: input.output,
    promptTokens: input.promptTokens ?? 0,
    outputTokens: input.outputTokens ?? 0,
    elapsedMs: input.elapsedMs ?? 0,
    tokensPerSec: input.tokensPerSec ?? 0,
    webgpuAdapter: input.webgpuAdapter ?? "",
    ragContextUsed: Boolean(input.ragContextUsed),
    createdAt,
  };

  // ADR-2605111200: CF Worker is edge proxy only. Persistence path is
  //   XRPC → bpmn-dispatcher → AgentGateway MCP → LangServer pod → INSERT vertex_ameno_inferenceresult.
  // Worker forwards via sdk.pds.xrpc(), which routes to atproto.etzhayyim.com PDS
  // and onward to the server-side dispatcher.
  try {
    const res = (await sdk.pds.xrpc("com.etzhayyim.apps.ameno.saveResult", record)) as
      | LexiconOutput<"com.etzhayyim.apps.ameno.saveResult">
      | undefined;
    if (res?.status) return res;
    return { status: "queued", resultId: res?.resultId, uri: res?.uri };
  } catch (e) {
    return { status: "failed", error: e instanceof Error ? e.message : String(e) };
  }
}

async function listHistoryHandler(
  sdk: HostSDK,
  body: Uint8Array,
): Promise<LexiconOutput<"com.etzhayyim.apps.ameno.listHistory">> {
  const input = parseLexiconInput("com.etzhayyim.apps.ameno.listHistory", body);
  const limit = Math.min(Math.max(Number(input.limit) || 20, 1), 100);
  const offset = Math.max(Number(input.offset) || 0, 0);

  try {
    const res = (await sdk.pds.xrpc("com.etzhayyim.apps.ameno.listHistory", {
      actorDid: input.actorDid ?? "",
      modelId: input.modelId ?? "",
      limit,
      offset,
    })) as LexiconOutput<"com.etzhayyim.apps.ameno.listHistory"> | undefined;
    if (res?.items) return res;
    return { items: [], total: 0, offset, limit };
  } catch {
    return { items: [], total: 0, offset, limit };
  }
}

export default createWorkerExport((sdk) => {
  sdk.app.query(nsid("com.etzhayyim.apps.ameno.listModels"), () => listModelsHandler());
  sdk.app.query(nsid("com.etzhayyim.apps.ameno.cardHome"), () => cardHomeHandler());
  sdk.app.query(nsid("com.etzhayyim.apps.ameno.listHistory"), (_ctx, body) =>
    listHistoryHandler(sdk, body),
  );
  sdk.app.command(nsid("com.etzhayyim.apps.ameno.saveResult"), (_ctx, body) =>
    saveResultHandler(sdk, body),
  );
});
