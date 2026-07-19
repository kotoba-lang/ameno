# etzhayyim-project-ameno — Browser WebGPU Inference Platform

## Overview

**Browser-side WebGPU inference via Gemma 4 E2B multimodal。** transformers.js ONNX + WebGPU compute shaders で推論を完全にブラウザ内で実行。per-actor LoRA adapter merge + RAG context injection。

- **`ameno.etzhayyim.com`** — Browser WebGPU inference
- **nanoid**: `d94d27cb`
- **DID**: `did:web:ameno.etzhayyim.com`

## Architecture

```
ameno.etzhayyim.com (CF Worker — TS Native)
  ├─ Backend (src/app.ts)
  │   ├─ com.etzhayyim.apps.ameno.listModels     — ONNX model registry
  │   ├─ com.etzhayyim.apps.ameno.saveResult     — Persist inference result to the graph query layer
  │   ├─ com.etzhayyim.apps.ameno.listHistory    — Query inference history
  │   └─ com.etzhayyim.apps.ameno.cardHome       — Protocol canvas card
  │
  └─ Frontend (svelte/)
      ├─ inference.ts     — transformers.js pipeline + WebGPU state machine
      ├─ lora-runtime.ts  — WebGPU compute shader LoRA merge (W' = W + α·B·A)
      └─ rag-lora.ts      — RAG context injection + adapter selection (RisingWave)
```

## Model

| Model | Parameters | Context | Capabilities | Min VRAM | Quantization |
|---|---|---|---|---|---|
| gemma-4-E2B-it | 2.3B effective / 5.1B total | 128K | text + image + audio | 4GB | q4f16 |

## Key Features

- **Zero server compute**: 推論は完全にブラウザ内 (WebGPU)
- **Per-actor LoRA**: actor DID ごとの LoRA adapter を B2 CDN から取得、WebGPU compute shader でマージ
- **RAG context injection**: RisingWave embedding search via graph SQL path
- **Crowd-sourced inference**: murakumo InferenceRouterDO の Tier 2 として crowd-sourced browser compute を提供

## Relationship to Murakumo

ameno は murakumo の **Tier 2 inference layer** として機能。murakumo InferenceRouterDO が native MLX (Tier 1) → ameno browser WebGPU (Tier 2) の順で routing。

## Package

**`@etzhayyim/ameno`** (`orgs/etzhayyim/com-etzhayyim-ameno/`) — inference engine, LoRA runtime, RAG-LoRA pipeline の Single Source。project の svelte lib は re-export のみ。

## Key Files

| Purpose | Path |
|---|---|
| **Package (Single Source)** | `orgs/etzhayyim/com-etzhayyim-ameno/src/` |
| Inference engine | `orgs/etzhayyim/com-etzhayyim-ameno/src/inference.ts` |
| WebGPU LoRA merge | `orgs/etzhayyim/com-etzhayyim-ameno/src/lora-runtime.ts` |
| RAG-LoRA pipeline | `orgs/etzhayyim/com-etzhayyim-ameno/src/rag-lora.ts` |
| Backend (TS Native) | `wasm/etzhayyim-wasm-ameno-d94d27cb/src/app.ts` |
| Frontend (re-export) | `wasm/etzhayyim-wasm-ameno-d94d27cb/svelte/src/lib/` |
| Domain WIT | `wit/ameno/package.wit` |

## Hard Constraints

1. **Browser-only inference** — サーバー側推論なし。全推論は WebGPU + transformers.js
2. **ONNX format** — HuggingFace ONNX モデルのみサポート
3. **WebGPU required** — WebGPU 非対応ブラウザでは動作しない (CPU fallback は LoRA merge のみ)
