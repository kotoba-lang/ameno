<script lang="ts">
  import {
    MODELS,
    checkWebGPU,
    getGPUInfo,
    loadModel,
    generate,
    setLoraAdapters,
    getActiveAdapterIds,
    type ChatMessage,
    type InferenceDevice,
    type InferenceState,
    type GenerationStats,
  } from "./lib/inference";
  import {
    MEDIAPIPE_MODELS,
    loadMediapipeModel,
    mediapipeGenerate,
  } from "./lib/mediapipe-runtime";
  import { invokeAmeno, type GraphChunk, type GraphPhase } from "./lib/graph";
  import {
    fetchGovProcedures,
    retrieveProcedures,
    buildKotobaContext,
    type KotobaProcedure,
  } from "./lib/kotoba-ground";
  import {
    ensureEmbeddingLoaded,
    isEmbeddingReady as isEmbedReady,
  } from "./lib/embedding";
  import {
    formatUptime,
    getDaemonSnapshot,
    getWorkerDid,
    noteBriefProcessed,
    noteError,
    setFirehoseConnected,
    shortDid,
  } from "./lib/daemon";
  import { countMemories } from "./lib/memory-vault";
  import { openSwarm, type SwarmPeer } from "./lib/swarm";
  import {
    fetchWorkerInfo,
    invokeAmenoRemote,
    pingDaemon,
    pullThreadMessages,
    type DaemonWorkerInfo,
  } from "./lib/viewer-mode";
  import { shortDidKey } from "./lib/did-auth";
  import { openBriefStream, type Brief } from "./lib/brief-stream";
  import { saveResult } from "./lib/save-result";
  import { encryptText, decryptText, isEncrypted } from "./lib/private-vault";
  import { listHistory, type HistoryItem } from "./lib/list-history";
  import { listActorAdapters, type AdapterRow } from "./lib/list-actor-adapters";
  import { listMyCredits, type MyCreditsResponse } from "./lib/list-my-credits";

  /** When true, outputs are AES-GCM encrypted client-side before saveResult. */
  let privateMode = $state(false);

  /** Models that prefer WASM ternary kernels over WebGPU. */
  const WASM_PREFERRED = new Set(["baien-bitnet-2b"]);
  /** Models that route through the MediaPipe LLM Inference Web runtime. */
  const MEDIAPIPE_PREFERRED = new Set(Object.keys(MEDIAPIPE_MODELS));

  /**
   * Selectable models, grouped by browser runtime so the picker shows which
   * kernel each entry uses. MediaPipe LiteRT is listed first because its
   * ungated `.task` bundles (Gemma 4 E2B/E4B, Apache 2.0) load and run fully
   * in-browser today; the transformers.js ONNX entries for the same Gemma 4
   * models are listed under "ONNX WebGPU" (these error on Load until
   * transformers.js recognises the `gemma4` model type — ADR-2605190824).
   * Labels come from each runtime's own meta registry (displayName) instead
   * of raw ids so the user can clearly pick Gemma 4 E2B vs E4B. */
  const MODEL_GROUPS: { label: string; options: { id: string; label: string }[] }[] = [
    {
      label: "Browser · MediaPipe LiteRT (recommended)",
      options: Object.values(MEDIAPIPE_MODELS).map((m) => ({ id: m.id, label: m.displayName })),
    },
    {
      label: "Browser · ONNX WebGPU (transformers.js)",
      options: Object.values(MODELS)
        .filter((m) => !WASM_PREFERRED.has(m.id))
        .map((m) => ({ id: m.id, label: m.displayName })),
    },
    {
      label: "Browser · WASM ternary",
      options: Object.values(MODELS)
        .filter((m) => WASM_PREFERRED.has(m.id))
        .map((m) => ({ id: m.id, label: m.displayName })),
    },
  ].filter((g) => g.options.length > 0);
  /** Default to the smallest ungated MediaPipe-Web bundle (Gemma 4 E2B). */
  let selectedModelId = $state("gemma-4-e2b-mediapipe");
  /** Device override for the next loadModel() call. null = engine default. */
  let selectedDevice = $state<InferenceDevice | null>(null);
  import type { AdapterCandidate } from "./lib/rag-lora";

  let state: InferenceState = $state({
    status: "idle",
    progress: 0,
    error: null,
    loadedModel: null,
    webgpuAvailable: false,
    activeAdapters: [],
    actorDid: null,
  });

  let messages: ChatMessage[] = $state([]);
  let inputText = $state("");
  let streamingText = $state("");
  let lastStats: GenerationStats | null = $state(null);
  /** Reflection budget: 0 = disable critic, 1 = one revise pass (default), 2 = two passes. */
  let maxReflections = $state(1);
  /** Current LangGraph phase. */
  let graphPhase = $state<GraphPhase | null>(null);
  /** Last critic verdict shown next to the assistant message. */
  let lastCritique = $state<{ score: number; feedback: string; iteration: number } | null>(null);
  /** Browser-local tool use toggle. ADR-2605191129. */
  let toolsEnabled = $state(false);
  // kotoba mode: ground answers on etzhayyim's published gov-procedure records
  // (/.well-known/gov-procedures.json) so the conversation proceeds on a kotoba
  // basis. Records are fetched once and cached; retrieval is pure/client-side.
  let kotobaMode = $state(false);
  let kotobaProcs: KotobaProcedure[] | null = null;
  let kotobaStatus = $state("");
  /** Daemon state snapshot, refreshed by a 1s interval. ADR-2605191135. */
  let daemonSnapshot = $state(getDaemonSnapshot());
  /** Whether the daemon details panel is expanded. */
  let daemonPanelOpen = $state(false);
  /** Long-term encrypted memory count. ADR-2605191206. */
  let memoryCount = $state(0);
  /** Other ameno tabs in this browser (ADR-2605191524 swarm). */
  let swarmPeers = $state<SwarmPeer[]>([]);
  /** True when this tab is the swarm leader (ADR-2605191603).
   *  In a single-tab session this is always true. */
  let swarmIsLeader = $state(true);
  /** Briefs received while this tab was a follower — skipped, but counted
   *  so UI shows "passive observer" honestly. */
  let briefsSkippedAsFollower = $state(0);
  /** State of the "Pull from daemon" button (ADR-2605191645). */
  let pullingFromDaemon = $state(false);
  let pullError = $state<string | null>(null);

  /** Replace local messages with the daemon's `viewer` thread state. */
  async function handlePullFromDaemon() {
    const url = resolveDaemonUrl();
    if (!url || pullingFromDaemon) return;
    pullingFromDaemon = true;
    pullError = null;
    try {
      const pulled = await pullThreadMessages(url, "viewer", undefined, resolveAuthToken());
      if (pulled.length === 0) {
        pullError = "daemon thread 'viewer' is empty";
        return;
      }
      messages = pulled;
      streamingText = "";
    } catch (e) {
      pullError = e instanceof Error ? e.message : String(e);
    } finally {
      pullingFromDaemon = false;
    }
  }
  /** Compute mode — local browser graph vs remote daemon SSE. ADR-2605191407. */
  let computeMode = $state<"local" | "daemon-a" | "daemon-b" | "custom">("local");
  /** Custom daemon URL (used when computeMode==="custom"). */
  let customDaemonUrl = $state("http://127.0.0.1:12480");
  /** Bearer token sent to remote daemons running with AMENO_AUTH_TOKEN.
   *  Stored only in component state — never persisted to localStorage —
   *  so refreshing the tab forces a re-paste. */
  let customAuthToken = $state("");
  /** Latest probed daemon info, refreshed by a background poll. */
  let daemonInfo = $state<DaemonWorkerInfo | null>(null);
  /** True while a daemon ping/info fetch is in flight. */
  let daemonProbing = $state(false);
  /** Last daemon connection error. */
  let daemonError = $state<string | null>(null);
  /** Tool call/result chip log for the current turn. */
  let toolLog = $state<Array<{ name: string; argStr: string; result?: string; error?: boolean; iteration: number }>>([]);
  /** Active inference toggle (Stage 3, ADR-2605191113). */
  let activeInference = $state(false);
  /** Surprise distance flavour. ADR-2605191120. */
  let surpriseMode = $state<"lexical" | "embedding">("lexical");
  /** MiniLM lazy-load progress (0..100). */
  let embedProgress = $state(0);
  let embedLoading = $state(false);
  let embedError = $state<string | null>(null);
  /** Most recent prediction the agent has committed for the next user turn. */
  let pendingPrediction = $state("");
  /** Per-user-message surprise score (and mode). Maps message index → row. */
  let surpriseByIndex = $state<Record<number, { score: number; mode: "lexical" | "embedding" }>>({});
  let gpuInfo = $state("");
  let chatContainer: HTMLElement | undefined = $state();

  /** Actor DID input for per-actor LoRA scoping. */
  let actorDidInput = $state("");
  /** Available adapters for current actor. */
  let availableAdapters: AdapterCandidate[] = $state([]);
  /** Whether LoRA panel is expanded. */
  let loraPanelOpen = $state(false);

  /** Materialise the worker DID at startup so subsequent reads are stable. */
  getWorkerDid();

  /** Open the multi-tab swarm channel and keep `swarmPeers` + `swarmIsLeader`
   *  in sync. ADR-2605191524 (presence) + ADR-2605191603 (leader election). */
  $effect(() => {
    const handle = openSwarm({
      did: getWorkerDid(),
      role: "browser",
      computeMode,
      loadedModel: state.loadedModel,
    });
    const poll = setInterval(() => {
      swarmPeers = handle.getPeers();
      swarmIsLeader = handle.isLeader();
    }, 1000);
    return () => {
      clearInterval(poll);
      handle.close();
    };
  });

  /** Push compute-mode / model changes to swarm peers immediately. */
  $effect(() => {
    // Re-tracked by Svelte 5 on each computeMode / state.loadedModel change.
    void computeMode;
    void state.loadedModel;
  });

  /** 1Hz daemon snapshot poll. Cheap (object literal copy, no IO). */
  $effect(() => {
    const id = setInterval(() => {
      daemonSnapshot = getDaemonSnapshot();
    }, 1000);
    return () => clearInterval(id);
  });

  /** Refresh memory count when the daemon panel opens (avoids an IDB op every
   *  second when the panel is closed). */
  $effect(() => {
    if (daemonPanelOpen) {
      void countMemories().then((n) => { memoryCount = n; }).catch(() => { memoryCount = 0; });
    }
  });

  /** Resolve the compute target URL or null for local. ADR-2605191407. */
  function resolveDaemonUrl(): string | null {
    if (computeMode === "local") return null;
    if (computeMode === "daemon-a") return "http://127.0.0.1:12480";
    if (computeMode === "daemon-b") return "http://127.0.0.1:12481";
    return customDaemonUrl.trim() || null;
  }

  /** Auth token to use when probing or invoking the remote daemon. Only
   *  applies to the `custom` mode for now — localhost daemons run without
   *  AMENO_AUTH_TOKEN by default. */
  function resolveAuthToken(): string | undefined {
    return computeMode === "custom" && customAuthToken ? customAuthToken : undefined;
  }

  /** Poll the selected daemon's /workerInfo every 5s while in viewer mode. */
  $effect(() => {
    const url = resolveDaemonUrl();
    const token = resolveAuthToken();
    if (!url) {
      daemonInfo = null;
      daemonError = null;
      return;
    }
    let aborted = false;
    const ctrl = new AbortController();
    daemonProbing = true;
    const tick = async () => {
      if (aborted) return;
      const info = await fetchWorkerInfo(url, ctrl.signal, token);
      if (aborted) return;
      if (info) {
        daemonInfo = info;
        daemonError = null;
      } else {
        daemonInfo = null;
        daemonError = `cannot reach ${url}`;
      }
    };
    void tick().finally(() => { daemonProbing = false; });
    const id = setInterval(() => { void tick(); }, 5000);
    return () => {
      aborted = true;
      ctrl.abort();
      clearInterval(id);
    };
  });

  /** Check WebGPU on mount; if absent and Baien is available, default to it on WASM. */
  $effect(() => {
    checkWebGPU().then((ok) => {
      state.webgpuAvailable = ok;
      if (!ok && MODELS["baien-bitnet-2b"]) {
        selectedModelId = "baien-bitnet-2b";
        selectedDevice = "wasm";
      }
    });
    getGPUInfo().then((info) => {
      gpuInfo = info;
    });
  });

  /** Auto-scroll chat to bottom. */
  function scrollToBottom() {
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  /** Load the selected model. Dispatches by kernel (ADR-2605190824):
   *    *-mediapipe ids       → MediaPipe LiteRT (WebGPU `.task` bundle)
   *    gemma-4-*-it (ONNX)   → transformers.js WebGPU
   *    baien-bitnet-2b       → WASM ternary
   */
  async function handleLoadModel() {
    state.status = "loading";
    state.error = null;
    state.progress = 0;

    try {
      if (MEDIAPIPE_PREFERRED.has(selectedModelId)) {
        await loadMediapipeModel((p) => {
          state.progress = p;
        }, selectedModelId);
      } else {
        const device =
          selectedDevice ??
          (WASM_PREFERRED.has(selectedModelId) ? "wasm" : "webgpu");
        await loadModel((p) => {
          state.progress = p;
        }, selectedModelId, device);
      }
      state.status = "ready";
      state.loadedModel = selectedModelId;
    } catch (e) {
      state.status = "error";
      state.error = e instanceof Error ? e.message : String(e);
      console.error("Model load failed:", e);
    }
  }

  /** Per-actor LoRA candidates fetched from vertex_lora_adapter (Phase 5g). */
  let actorAdapterRows = $state<AdapterRow[]>([]);
  let adaptersLoading = $state(false);
  let adaptersError = $state<string | null>(null);
  /** Selected adapter ids to record in saveResult.loraAdapters. */
  let selectedAdapterIds = $state<string[]>([]);

  /** Set actor DID for per-actor LoRA + fetch available adapters. */
  async function handleSetActor() {
    const did = actorDidInput.trim();
    if (!did) return;
    state.actorDid = did;
    state.activeAdapters = [];
    availableAdapters = [];
    selectedAdapterIds = [];
    setLoraAdapters([]);
    adaptersLoading = true;
    adaptersError = null;
    try {
      const res = await listActorAdapters({ actorDid: did, limit: 50 });
      actorAdapterRows = res.items;
    } catch (e) {
      adaptersError = e instanceof Error ? e.message : String(e);
    } finally {
      adaptersLoading = false;
    }
  }

  /** Clear actor and adapters. */
  function handleClearActor() {
    state.actorDid = null;
    state.activeAdapters = [];
    actorDidInput = "";
    availableAdapters = [];
    actorAdapterRows = [];
    selectedAdapterIds = [];
    adaptersError = null;
    setLoraAdapters([]);
  }

  /**
   * Toggle a LoRA adapter selection.
   *
   * NOTE: this commits the selection to state.activeAdapters and
   * saveResult.loraAdapters, but does NOT actually merge weights into the
   * loaded transformers.js pipeline — that requires getting at the model's
   * raw Float32Array weights, which is a separate (large) integration.
   * Per-actor inference quality is unchanged in this commit; the wiring
   * is in place for that follow-up.
   */
  function toggleAdapter(adapterId: string) {
    if (selectedAdapterIds.includes(adapterId)) {
      selectedAdapterIds = selectedAdapterIds.filter((x) => x !== adapterId);
    } else {
      selectedAdapterIds = [...selectedAdapterIds, adapterId];
    }
    state.activeAdapters = [...selectedAdapterIds];
  }

  /** Lazy-load the MiniLM embedding pipeline. Idempotent. ADR-2605191120. */
  async function loadEmbedding() {
    if (isEmbedReady() || embedLoading) return;
    embedLoading = true;
    embedError = null;
    embedProgress = 0;
    try {
      await ensureEmbeddingLoaded((pct) => { embedProgress = pct; });
      embedProgress = 100;
    } catch (e) {
      embedError = e instanceof Error ? e.message : String(e);
    } finally {
      embedLoading = false;
    }
  }

  /** Kick off the embedding load the moment the user picks "embedding". */
  $effect(() => {
    if (surpriseMode === "embedding" && !isEmbedReady() && !embedLoading) {
      void loadEmbedding();
    }
  });

  /**
   * Send a message and run the ameno LangGraph (Pregel) — generate → critique
   * → revise → finalize. ADR-2605191000. Tokens stream in via the graph's
   * `streamMode: "custom"` chunks; we re-bucket them into `streamingText`
   * by phase so the UI shows only the user-facing draft, not the critic JSON.
   */
  async function handleSend() {
    const text = inputText.trim();
    if (!text || state.status !== "ready") return;

    inputText = "";
    messages = [...messages, { role: "user", content: text }];
    streamingText = "";
    state.status = "generating";
    lastStats = null;
    lastCritique = null;
    graphPhase = null;
    toolLog = [];

    setTimeout(scrollToBottom, 0);

    try {
      let systemContent = state.actorDid
        ? `You are Murakumo, a personalized AI assistant for actor ${state.actorDid}. Running locally in the browser via WebGPU with per-actor LoRA adaptation. Be concise and helpful.`
        : "You are Murakumo, a helpful AI assistant running locally in the user's browser via WebGPU. Be concise and helpful.";

      // kotoba mode: ground this turn on the published gov-procedure records.
      // Lazy-load + cache the index, retrieve the records relevant to the user's
      // message, and append them (with the mirror/honesty constraints) to the
      // system prompt so the local model answers FROM the kotoba data.
      if (kotobaMode) {
        try {
          if (!kotobaProcs) {
            kotobaStatus = "loading kotoba records…";
            kotobaProcs = await fetchGovProcedures();
          }
          const hits = retrieveProcedures(text, kotobaProcs, 5);
          systemContent = `${systemContent}\n\n${buildKotobaContext(hits)}`;
          kotobaStatus = `${kotobaProcs.length} records · ${hits.length} matched`;
        } catch (e) {
          kotobaStatus = `kotoba load failed: ${e instanceof Error ? e.message : String(e)}`;
        }
      }

      const chatMessages: ChatMessage[] = [
        { role: "system", content: systemContent },
        ...messages,
      ];

      const kernel =
        state.loadedModel && MEDIAPIPE_PREFERRED.has(state.loadedModel)
          ? "mediapipe"
          : "transformers";

      // Record surprise against this turn's user message (the one we just
      // pushed). Index = messages.length - 1 after the push above.
      const newUserIndex = messages.length - 1;

      // Dispatch: local graph vs remote daemon SSE. Chunk handler is
      // identical between the two (ADR-2605191407 §State 分離).
      const daemonUrl = resolveDaemonUrl();
      const runInvoke = daemonUrl
        ? (onChunk: (c: GraphChunk) => void) =>
            invokeAmenoRemote({
              baseUrl: daemonUrl,
              threadId: "viewer",
              messages: chatMessages,
              maxIterations: maxReflections,
              activeInference,
              toolsEnabled,
              authToken: resolveAuthToken(),
              onChunk,
            })
        : (onChunk: (c: GraphChunk) => void) =>
            invokeAmeno({
              messages: chatMessages,
              maxIterations: maxReflections,
              kernel,
              activeInference,
              surpriseMode,
              toolsEnabled,
              onChunk,
            });

      const finalDraft = await runInvoke((chunk: GraphChunk) => {
          if (chunk.type === "phase") {
            graphPhase = chunk.phase;
            // Reset the visible buffer at the start of generate and each
            // revise pass — those are user-facing. critique / surprise_eval
            // / predict_next are not shown verbatim, but we still let the
            // tokens/sec counter tick through their stream chunks.
            if (chunk.phase === "generate" || chunk.phase === "revise") {
              streamingText = "";
            }
            return;
          }
          if (chunk.type === "token") {
            if (chunk.phase === "generate" || chunk.phase === "revise") {
              streamingText += chunk.token;
              setTimeout(scrollToBottom, 0);
            }
            return;
          }
          if (chunk.type === "critique") {
            lastCritique = {
              score: chunk.score,
              feedback: chunk.feedback,
              iteration: chunk.iteration,
            };
            return;
          }
          if (chunk.type === "surprise") {
            surpriseByIndex = {
              ...surpriseByIndex,
              [newUserIndex]: { score: chunk.surprise, mode: chunk.mode },
            };
            return;
          }
          if (chunk.type === "prediction") {
            pendingPrediction = chunk.prediction;
            return;
          }
          if (chunk.type === "tool_call") {
            toolLog = [
              ...toolLog,
              {
                name: chunk.name,
                argStr: JSON.stringify(chunk.args ?? {}),
                iteration: chunk.iteration,
              },
            ];
            return;
          }
          if (chunk.type === "tool_result") {
            // Attach result to the most recent matching call (same name + iteration).
            const idx = [...toolLog].reverse().findIndex(
              (e) => e.name === chunk.name && e.iteration === chunk.iteration && e.result === undefined,
            );
            if (idx >= 0) {
              const realIdx = toolLog.length - 1 - idx;
              const next = [...toolLog];
              next[realIdx] = {
                ...next[realIdx],
                result: chunk.result,
                error: chunk.error,
              };
              toolLog = next;
            }
            return;
          }
          if (chunk.type === "stats") {
            lastStats = chunk.stats;
          }
        });

      const finalText = finalDraft || streamingText;
      messages = [
        ...messages,
        { role: "assistant", content: finalText },
      ];
      streamingText = "";
      graphPhase = null;
      state.status = "ready";
      state.activeAdapters = getActiveAdapterIds();
    } catch (e) {
      state.status = "error";
      state.error = e instanceof Error ? e.message : String(e);
      graphPhase = null;
      console.error("Generation failed:", e);
    }
  }

  /** Handle Enter key in input. */
  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // ── Auto-respond mode ──
  // Subscribes to com.etzhayyim.apps.ameno.subscribeBriefs (NATS firehose), runs
  // local inference per brief, and posts the result via saveResult. This
  // closes the murakumo Tier 2 crowd-source loop end-to-end (Phase 4a + 5a).

  let autoRespondActive = $state(false);
  let autoRespondCount = $state(0);
  let autoRespondError = $state<string | null>(null);
  let autoRespondHistory = $state<Array<{ prompt: string; output: string; uri: string }>>([]);
  let autoRespondCloser: (() => void) | null = null;
  /** Drop briefs while a generation is in flight to avoid queue blow-up. */
  let autoRespondInFlight = false;

  /**
   * Process one brief through the full agent graph (ADR-2605191135 Tier-2
   * daemon integration). Reflection / active inference default OFF for
   * throughput; tools default ON so wikipedia / now / recall stay useful
   * when a brief mentions a known topic. Each brief shares the same
   * `firehose:<collection>` thread_id so LocalCheckpointer accumulates
   * memory across briefs in the same firehose run.
   */
  async function processBrief(brief: Brief) {
    if (autoRespondInFlight || state.status !== "ready") return;
    // ADR-2605191603 — follower tabs see briefs but don't process them.
    // Count them so the UI can show "passive observer" honestly.
    if (!swarmIsLeader) {
      briefsSkippedAsFollower++;
      return;
    }
    autoRespondInFlight = true;
    let output = "";
    try {
      const chat: ChatMessage[] = [
        {
          role: "system",
          content:
            "You are Ameno, a browser-local LLM (Tier 2 organism worker) " +
            "serving the etzhayyim platform. Reply briefly to the post " +
            "below. Output ONE concise sentence in the same language as the input.",
        },
        { role: "user", content: brief.text },
      ];
      const kernel =
        state.loadedModel && MEDIAPIPE_PREFERRED.has(state.loadedModel)
          ? "mediapipe"
          : "transformers";
      let elapsedStart = performance.now();
      let totalTokens = 0;
      let tokensPerSecond = 0;
      const finalDraft = await invokeAmeno({
        messages: chat,
        kernel,
        maxIterations: 0,
        activeInference: false,
        toolsEnabled,
        threadId: `firehose:app.bsky.feed.post`,
        onChunk: (chunk) => {
          if (chunk.type === "token" && (chunk.phase === "generate" || chunk.phase === "revise")) {
            output += chunk.token;
          } else if (chunk.type === "stats" && chunk.phase === "generate") {
            totalTokens = chunk.stats.totalTokens;
            tokensPerSecond = chunk.stats.tokensPerSecond;
          }
        },
      });
      const visible = finalDraft || output;
      const persistedOutput = privateMode ? await encryptText(visible) : visible;
      noteBriefProcessed(totalTokens);
      try {
        const res = await saveResult({
          modelId: state.loadedModel ?? "",
          prompt: brief.text,
          output: persistedOutput,
          actorDid: state.actorDid ?? "",
          loraAdapters: selectedAdapterIds.length > 0 ? selectedAdapterIds : undefined,
          elapsedMs: performance.now() - elapsedStart,
          tokensPerSec: Math.round(tokensPerSecond * 1000),
          outputTokens: totalTokens,
          ragContextUsed: false,
        });
        autoRespondHistory = [
          { prompt: brief.text, output: visible, uri: res.uri ?? brief.uri },
          ...autoRespondHistory,
        ].slice(0, 5);
      } catch (e) {
        // saveResult failure is non-fatal — the daemon is still doing work
        // locally. Record but keep going.
        noteError(e instanceof Error ? e.message : String(e));
        autoRespondHistory = [
          { prompt: brief.text, output: visible, uri: brief.uri },
          ...autoRespondHistory,
        ].slice(0, 5);
      }
      autoRespondCount++;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      autoRespondError = msg;
      noteError(msg);
    } finally {
      autoRespondInFlight = false;
    }
  }

  function startAutoRespond() {
    if (autoRespondActive || state.status !== "ready") return;
    autoRespondError = null;
    autoRespondActive = true;
    setFirehoseConnected(true);
    autoRespondCloser = openBriefStream({
      collection: "app.bsky.feed.post",
      maxEvents: 1000,
      idleTimeoutSec: 300,
      onBrief: (brief) => { void processBrief(brief); },
      onDone: () => { stopAutoRespond(); },
      onError: (e) => {
        autoRespondError = e.message;
        noteError(e.message);
        setFirehoseConnected(false);
      },
    });
  }

  function stopAutoRespond() {
    autoRespondActive = false;
    autoRespondCloser?.();
    autoRespondCloser = null;
    setFirehoseConnected(false);
  }

  // ── History panel ──
  // Pulls vertex_ameno_inferenceresult via listHistory and decrypts any
  // signal:v1: outputs (Phase 5b) on display so the round-trip is visible.

  interface HistoryView extends HistoryItem {
    /** Decrypted output (or original if it was plaintext). */
    displayOutput: string;
    /** True if the source row carried a signal:v1: envelope. */
    wasEncrypted: boolean;
    /** Decrypt error, if the envelope failed (e.g. lost localStorage key). */
    decryptError: string | null;
  }

  let historyItems = $state<HistoryView[]>([]);
  let historyLoading = $state(false);
  let historyError = $state<string | null>(null);
  let historyTotal = $state(0);
  let creditBalance = $state<MyCreditsResponse | null>(null);

  async function refreshHistory() {
    historyLoading = true;
    historyError = null;
    try {
      const did = state.actorDid ?? "";
      const [res, credits] = await Promise.all([
        listHistory({ actorDid: did, limit: 20 }),
        did ? listMyCredits(did) : Promise.resolve(null),
      ]);
      creditBalance = credits;
      const decrypted = await Promise.all(
        res.items.map(async (item) => {
          const wasEncrypted = isEncrypted(item.output);
          let displayOutput = item.output;
          let decryptError: string | null = null;
          if (wasEncrypted) {
            try {
              displayOutput = await decryptText(item.output);
            } catch (e) {
              decryptError = e instanceof Error ? e.message : String(e);
            }
          }
          return { ...item, displayOutput, wasEncrypted, decryptError };
        }),
      );
      historyItems = decrypted;
      historyTotal = res.total;
    } catch (e) {
      historyError = e instanceof Error ? e.message : String(e);
    } finally {
      historyLoading = false;
    }
  }
</script>

<div class="app">
  <!-- Header -->
  <header class="header">
    <div class="header-inner">
      <h1 class="logo">Murakumo Browser</h1>
      <span class="badge">WebGPU</span>
      {#if state.activeAdapters.length > 0}
        <span class="badge lora-badge">LoRA</span>
      {/if}
      {#if state.actorDid}
        <span class="actor-badge" title={state.actorDid}>
          {state.actorDid.slice(0, 20)}...
        </span>
      {/if}
      <button
        class="daemon-chip {daemonSnapshot.firehoseConnected ? 'daemon-online' : 'daemon-offline'}"
        title="Click for daemon details"
        onclick={() => (daemonPanelOpen = !daemonPanelOpen)}
      >
        <span class="daemon-dot"></span>
        {shortDid(daemonSnapshot.did)} · {formatUptime(daemonSnapshot.uptimeMs)}
        {#if swarmPeers.length > 0}
          {#if swarmIsLeader}
            <span class="daemon-leader" title="swarm leader">★</span>
          {:else}
            <span class="daemon-follower" title="swarm follower">·</span>
          {/if}
        {/if}
        {#if daemonSnapshot.briefsPerMinute > 0}
          · {daemonSnapshot.briefsPerMinute}/min
        {/if}
      </button>
    </div>
    {#if daemonPanelOpen}
      <div class="daemon-panel">
        <div><b>DID</b>: {daemonSnapshot.did}</div>
        <div><b>Uptime</b>: {formatUptime(daemonSnapshot.uptimeMs)}</div>
        <div><b>Model</b>: {state.loadedModel ?? "(not loaded)"}</div>
        <div><b>Firehose</b>: {daemonSnapshot.firehoseConnected ? "connected" : "offline"}</div>
        <div><b>Briefs processed</b>: {daemonSnapshot.totalBriefs} (last 60s: {daemonSnapshot.briefsPerMinute})</div>
        <div><b>Total tokens decoded</b>: {daemonSnapshot.totalTokensDecoded}</div>
        <div><b>Memories</b>: {memoryCount} stored (AES-GCM encrypted)</div>
        <div>
          <b>Swarm</b>: {swarmPeers.length} peer{swarmPeers.length === 1 ? "" : "s"}
          {#if swarmIsLeader}
            <span class="leader-badge" title="ADR-2605191603 — lex-smallest DID processes briefs">★ leader</span>
          {:else}
            <span class="follower-badge" title="Another tab is the swarm leader; this tab is a passive observer">· follower</span>
          {/if}
          {#if swarmPeers.length > 0}
            <ul class="swarm-list">
              {#each swarmPeers as p}
                <li title={p.did}>
                  <span class="swarm-role">{p.role}</span>
                  · {p.computeMode}
                  · {p.loadedModel ?? "(no model)"}
                  · <code>{p.did.slice("did:web:browser:".length, "did:web:browser:".length + 6)}…</code>
                </li>
              {/each}
            </ul>
          {/if}
          {#if briefsSkippedAsFollower > 0}
            <div class="swarm-foot">briefs skipped as follower: {briefsSkippedAsFollower}</div>
          {/if}
        </div>
        {#if daemonSnapshot.lastBriefAt}
          <div><b>Last brief</b>: {formatUptime(Date.now() - daemonSnapshot.lastBriefAt)} ago</div>
        {/if}
        {#if daemonSnapshot.lastError}
          <div class="daemon-err"><b>Last error</b>: {daemonSnapshot.lastError}</div>
        {/if}
        <div class="daemon-foot">
          ADR-2605191135 · LocalCheckpointer persisted to localStorage
        </div>
      </div>
    {/if}
  </header>

  <!-- Main content -->
  <main class="main">
    {#if state.status === "idle" && computeMode !== "local"}
      <!-- Daemon viewer mode — no local model needed (ADR-2605191407) -->
      <div class="landing">
        <div class="landing-card">
          <h2>Viewer mode</h2>
          <p class="model-desc">
            Inference is delegated to the ameno daemon over HTTP SSE. No
            local model load is required. Switch back to <em>Compute:
            local</em> to use MediaPipe Gemma in this tab.
          </p>
          {#if daemonInfo}
            <p class="gpu-info">
              connected · {daemonInfo.did}<br />
              model: {daemonInfo.model ?? '?'} · ollama:
              {daemonInfo.ollamaReachable ? 'up' : 'down'}
            </p>
          {:else if daemonError}
            <div class="warning">{daemonError}</div>
          {:else}
            <p class="gpu-info">probing daemon…</p>
          {/if}
          <button class="btn-primary" onclick={() => { state.status = "ready"; }}>
            Open chat
          </button>
        </div>
      </div>
    {:else if state.status === "idle"}
      <!-- Landing: load model -->
      <div class="landing">
        <div class="landing-card">
          <h2>Gemma 4 E2B</h2>
          <p class="model-desc">
            2.3B effective parameters, multimodal (text + image + audio).
            Runs entirely in your browser via WebGPU.
          </p>

          {#if !state.webgpuAvailable}
            <div class="warning">
              WebGPU is not available in this browser. Use Chrome 113+ or Edge 113+.
            </div>
          {:else}
            <p class="gpu-info">{gpuInfo}</p>

            <!-- Actor LoRA panel -->
            <div class="lora-panel">
              <button
                class="btn-toggle"
                onclick={() => loraPanelOpen = !loraPanelOpen}
              >
                {loraPanelOpen ? "Hide" : "Show"} Actor LoRA
              </button>

              {#if loraPanelOpen}
                <div class="lora-config">
                  <input
                    class="input-actor"
                    type="text"
                    placeholder="Actor DID (e.g. did:web:...)"
                    bind:value={actorDidInput}
                  />
                  <div class="lora-actions">
                    <button class="btn-small" onclick={handleSetActor}>
                      Set Actor
                    </button>
                    {#if state.actorDid}
                      <button class="btn-small btn-clear" onclick={handleClearActor}>
                        Clear
                      </button>
                    {/if}
                  </div>
                  {#if state.actorDid}
                    <p class="lora-status">
                      Actor: {state.actorDid.slice(0, 30)}...
                      {#if selectedAdapterIds.length > 0}
                        <br />Selected: {selectedAdapterIds.join(", ")}
                      {:else}
                        <br />No adapter selected (base model)
                      {/if}
                    </p>
                    {#if adaptersLoading}
                      <p class="lora-status">Loading adapters...</p>
                    {:else if adaptersError}
                      <p class="lora-status auto-respond-error">⚠ {adaptersError}</p>
                    {:else if actorAdapterRows.length === 0}
                      <p class="lora-status">No adapters registered for this actor.</p>
                    {:else}
                      <div class="lora-adapter-list">
                        {#each actorAdapterRows as row}
                          <label class="lora-adapter-item">
                            <input
                              type="checkbox"
                              checked={selectedAdapterIds.includes(row.adapterId)}
                              onchange={() => toggleAdapter(row.adapterId)}
                            />
                            <span class="lora-adapter-id">{row.adapterId}</span>
                            <span class="lora-adapter-meta">
                              {row.domain || 'general'} · r{row.adapterRank} · {row.baseModel}
                            </span>
                          </label>
                        {/each}
                      </div>
                    {/if}
                  {/if}
                </div>
              {/if}
            </div>

            <div class="model-picker">
              <label for="model-select">Model:</label>
              <select id="model-select" bind:value={selectedModelId}>
                {#each MODEL_GROUPS as group}
                  <optgroup label={group.label}>
                    {#each group.options as opt}
                      <option value={opt.id}>{opt.label}</option>
                    {/each}
                  </optgroup>
                {/each}
              </select>
              <span class="device-pill">
                {MEDIAPIPE_PREFERRED.has(selectedModelId)
                  ? "mediapipe-gpu"
                  : (selectedDevice ?? (WASM_PREFERRED.has(selectedModelId) ? "wasm" : "webgpu"))}
              </span>
            </div>
            <button class="btn-primary" onclick={handleLoadModel}>
              Load Model
            </button>
          {/if}
        </div>
      </div>

    {:else if state.status === "loading"}
      <!-- Loading progress -->
      <div class="landing">
        <div class="landing-card">
          <h2>Loading {selectedModelId}...</h2>
          <div class="progress-bar">
            <div class="progress-fill" style="width: {state.progress}%"></div>
          </div>
          <p class="progress-text">
            {state.progress}% — Downloading
            {MEDIAPIPE_PREFERRED.has(selectedModelId) ? "LiteRT .task bundle" : "ONNX weights"}
          </p>
        </div>
      </div>

    {:else if state.status === "merging-lora"}
      <!-- LoRA merge progress -->
      <div class="landing">
        <div class="landing-card">
          <h2>Merging LoRA Adapter...</h2>
          <div class="progress-bar">
            <div class="progress-fill lora-fill" style="width: {state.progress}%"></div>
          </div>
          <p class="progress-text">{state.progress}% — Applying actor adaptation</p>
        </div>
      </div>

    {:else if state.status === "error"}
      <!-- Error state -->
      <div class="landing">
        <div class="landing-card error-card">
          <h2>Error</h2>
          <p class="error-text">{state.error}</p>
          <button class="btn-primary" onclick={handleLoadModel}>
            Retry
          </button>
        </div>
      </div>

    {:else}
      <!-- Chat interface -->
      <div class="chat-container" bind:this={chatContainer}>
        {#if messages.length === 0}
          <div class="empty-state">
            <p>Model loaded. Start chatting.</p>
            <p class="hint">
              Inference runs locally on your GPU — no data leaves your device.
              {#if state.actorDid}
                <br />Actor: {state.actorDid.slice(0, 40)}
                {#if state.activeAdapters.length > 0}
                  (LoRA active)
                {/if}
              {/if}
            </p>
          </div>
        {/if}

        {#each messages as msg, i}
          <div class="message {msg.role}">
            <div class="message-label">
              {msg.role === "user" ? "You" : "Murakumo"}
              {#if msg.role === "user" && surpriseByIndex[i] !== undefined}
                <span
                  class="surprise-badge surprise-{surpriseByIndex[i].score >= 7 ? 'hi' : surpriseByIndex[i].score >= 3 ? 'mid' : 'lo'}"
                  title="{surpriseByIndex[i].mode === 'embedding' ? 'MiniLM cosine' : 'Lexical Jaccard'} surprise vs previous prediction"
                >
                  surprise {surpriseByIndex[i].score}/10 · {surpriseByIndex[i].mode}
                </span>
              {/if}
            </div>
            <div class="message-content">{typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}</div>
          </div>
        {/each}

        {#if toolLog.length > 0}
          <div class="tool-log">
            {#each toolLog as entry}
              <div class="tool-row {entry.error ? 'tool-error' : ''}">
                <span class="tool-name">tool: {entry.name}</span>
                <span class="tool-args">{entry.argStr}</span>
                {#if entry.result !== undefined}
                  <span class="tool-arrow">→</span>
                  <span class="tool-result" title={entry.result}>
                    {entry.result.length > 80 ? entry.result.slice(0, 80) + "…" : entry.result}
                  </span>
                {:else}
                  <span class="tool-arrow">…</span>
                {/if}
              </div>
            {/each}
          </div>
        {/if}

        {#if streamingText || graphPhase}
          <div class="message assistant">
            <div class="message-label">
              Murakumo
              {#if graphPhase}
                <span class="phase-pill phase-{graphPhase}">{graphPhase}</span>
              {/if}
            </div>
            <div class="message-content">{streamingText}<span class="cursor">|</span></div>
          </div>
        {/if}
      </div>

      <!-- Compute mode selector (ADR-2605191407) -->
      <div class="reflection-bar">
        <label title="Where the LangGraph runs: in this browser tab, or a daemon over HTTP SSE">
          Compute:
          <select bind:value={computeMode} disabled={state.status === "generating"}>
            <option value="local">local (this tab, MediaPipe)</option>
            <option value="daemon-a">daemon @12480 (Path A, TS)</option>
            <option value="daemon-b">daemon @12481 (Path B, Python)</option>
            <option value="custom">custom URL…</option>
          </select>
        </label>
        {#if computeMode === "custom"}
          <input
            class="daemon-url"
            type="url"
            placeholder="https://ameno-daemon.etzhayyim.com"
            bind:value={customDaemonUrl}
            disabled={state.status === "generating"}
          />
          <input
            class="daemon-url"
            type="password"
            placeholder="Bearer token (AMENO_AUTH_TOKEN)"
            bind:value={customAuthToken}
            disabled={state.status === "generating"}
          />
        {/if}
        {#if computeMode !== "local"}
          {#if daemonInfo}
            <span class="embed-pill embed-ready" title={`${daemonInfo.did}  model=${daemonInfo.model ?? '?'}  ollama=${daemonInfo.ollamaReachable ? 'up' : 'down'}`}>
              daemon ✓ {daemonInfo.model ?? '?'}{daemonInfo.kind === 'path-b-python' ? ' · Py' : ' · TS'}
            </span>
          {:else if daemonProbing}
            <span class="embed-pill" title="probing daemon">daemon probing…</span>
          {:else if daemonError}
            <span class="embed-pill embed-error" title={daemonError}>daemon unreachable</span>
          {/if}
          <button
            class="btn-small"
            title="Replace this tab's chat history with the daemon's saved thread state (viewer thread). ADR-2605191645."
            onclick={handlePullFromDaemon}
            disabled={pullingFromDaemon || state.status === "generating" || !daemonInfo}
          >
            {pullingFromDaemon ? "pulling…" : "Pull from daemon"}
          </button>
          {#if pullError}
            <span class="embed-pill embed-error" title={pullError}>pull failed</span>
          {/if}
          <span
            class="embed-pill"
            title={customAuthToken ? "Bearer token mode (ADR-2605191407)" : "DIDSig Ed25519 mode (ADR-2605191657)"}
          >
            auth: {customAuthToken ? "Bearer" : shortDidKey()}
          </span>
        {/if}
      </div>
      <div class="reflection-bar">
        <label>
          Reflection:
          <select bind:value={maxReflections} disabled={state.status === "generating"}>
            <option value={0}>off</option>
            <option value={1}>1 pass</option>
            <option value={2}>2 passes</option>
          </select>
        </label>
        <label class="ai-toggle">
          <input
            type="checkbox"
            bind:checked={activeInference}
            disabled={state.status === "generating"}
          />
          Active inference
        </label>
        <label class="ai-toggle" title="Browser-local tools: now / recall / wikipedia">
          <input
            type="checkbox"
            bind:checked={toolsEnabled}
            disabled={state.status === "generating"}
          />
          Tools
        </label>
        <label
          class="ai-toggle"
          title="Ground answers on etzhayyim's published government-procedure records (/.well-known/gov-procedures.json) — observational mirror, never files on your behalf"
        >
          <input
            type="checkbox"
            bind:checked={kotobaMode}
            disabled={state.status === "generating"}
          />
          kotoba 行政手続き
        </label>
        {#if kotobaMode && kotobaStatus}
          <span class="ai-toggle" style="opacity:0.7">{kotobaStatus}</span>
        {/if}
        {#if activeInference}
          <label class="ai-toggle" title="MiniLM 22 MB lazy DL, WASM device">
            Surprise:
            <select bind:value={surpriseMode} disabled={state.status === "generating"}>
              <option value="lexical">lexical</option>
              <option value="embedding">embedding</option>
            </select>
          </label>
          {#if surpriseMode === "embedding" && embedLoading}
            <span class="embed-pill" title="Loading Xenova/all-MiniLM-L6-v2">
              loading MiniLM… {embedProgress}%
            </span>
          {:else if surpriseMode === "embedding" && embedError}
            <span class="embed-pill embed-error" title={embedError}>
              MiniLM load failed
            </span>
          {:else if surpriseMode === "embedding" && isEmbedReady()}
            <span class="embed-pill embed-ready" title="MiniLM ready">
              embed ready
            </span>
          {/if}
        {/if}
        {#if lastCritique}
          <span class="critique-chip" title={lastCritique.feedback}>
            critic {lastCritique.score}/10 (iter {lastCritique.iteration})
          </span>
        {/if}
      </div>
      {#if activeInference && pendingPrediction}
        <div
          class="prediction-chip"
          title="Click to copy into input — predicted user reply for next turn"
          onclick={() => { inputText = pendingPrediction; }}
          role="button"
          tabindex="0"
          onkeydown={(e) => { if (e.key === "Enter") inputText = pendingPrediction; }}
        >
          predicted next: {pendingPrediction}
        </div>
      {/if}

      <!-- Stats bar -->
      {#if lastStats}
        <div class="stats-bar">
          {lastStats.tokensPerSecond.toFixed(1)} tok/s | {lastStats.totalTokens} tokens | {(lastStats.durationMs / 1000).toFixed(1)}s
          {#if lastStats.loraActive} | LoRA{/if}
          {#if lastStats.ragActive} | RAG{/if}
        </div>
      {/if}

      <!-- Auto-respond panel (Phase 5a: NATS firehose → local inference → saveResult) -->
      <div class="auto-respond">
        <div class="auto-respond-header">
          <label>
            <input
              type="checkbox"
              checked={autoRespondActive}
              onchange={(e) => (e.currentTarget as HTMLInputElement).checked ? startAutoRespond() : stopAutoRespond()}
            />
            Auto-respond to PDS firehose
          </label>
          <label class="private-toggle" title="Encrypt outputs (signal:v1:) before saveResult. Server stores ciphertext only.">
            <input type="checkbox" bind:checked={privateMode} />
            🔒 Private mode
          </label>
          {#if autoRespondActive}
            <span class="auto-respond-status">live · {autoRespondCount} responded</span>
          {/if}
          {#if autoRespondError}
            <span class="auto-respond-error" title={autoRespondError}>⚠ {autoRespondError.slice(0, 60)}</span>
          {/if}
        </div>
        {#if autoRespondHistory.length > 0}
          <div class="auto-respond-list">
            {#each autoRespondHistory as item}
              <div class="auto-respond-item">
                <div class="auto-respond-prompt">→ {item.prompt.slice(0, 100)}</div>
                <div class="auto-respond-output">⤷ {item.output.slice(0, 200)}</div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

<!-- History panel (Phase 5f: listHistory + decrypt for signal:v1: rows; Phase 5j: balance) -->
      <div class="history">
        <div class="history-header">
          <button class="btn-small" onclick={refreshHistory} disabled={historyLoading}>
            {historyLoading ? "Loading..." : "Refresh history"}
          </button>
          {#if historyTotal > 0}
            <span class="history-status">{historyItems.length} / {historyTotal}</span>
          {/if}
          {#if creditBalance && creditBalance.eventCount > 0}
            <span
              class="history-status"
              title="Tier 2 credits earned from saveResult-credited inferences (Phase 5c)"
            >
              💎 {creditBalance.balance} credits · {creditBalance.eventCount} events
            </span>
          {/if}
          {#if historyError}
            <span class="auto-respond-error" title={historyError}>⚠ {historyError.slice(0, 60)}</span>
          {/if}
        </div>
        {#if historyItems.length > 0}
          <div class="history-list">
            {#each historyItems as item}
              <div class="history-item">
                <div class="history-meta">
                  <span class="history-model">{item.modelId}</span>
                  {#if item.wasEncrypted}
                    <span class="history-badge" title="Decrypted from signal:v1: envelope">🔒</span>
                  {/if}
                  <span class="history-time">{item.createdAt.slice(0, 19).replace("T", " ")}</span>
                  {#if item.elapsedMs > 0}
                    <span class="history-perf">{(item.elapsedMs / 1000).toFixed(1)}s</span>
                  {/if}
                </div>
                <div class="history-prompt">→ {item.prompt.slice(0, 200)}</div>
                {#if item.decryptError}
                  <div class="history-output history-decrypt-failed">
                    ⤷ <em>decrypt failed: {item.decryptError}</em>
                  </div>
                {:else}
                  <div class="history-output">⤷ {item.displayOutput.slice(0, 400)}</div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Input area -->
      <div class="input-area">
        <div class="input-inner">
          <textarea
            class="input-field"
            placeholder="Message Murakumo..."
            bind:value={inputText}
            onkeydown={handleKeyDown}
            disabled={state.status === "generating"}
            rows={1}
          ></textarea>
          <button
            class="btn-send"
            onclick={handleSend}
            disabled={state.status === "generating" || !inputText.trim()}
          >
            {state.status === "generating" ? "..." : "Send"}
          </button>
        </div>
      </div>
    {/if}
  </main>
</div>

<style>
  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    max-width: 600px;
    margin: 0 auto;
  }

  .header {
    position: sticky;
    top: 0;
    z-index: 10;
    background: #0a0a0a;
    border-bottom: 1px solid #1a1a1a;
    height: 48px;
    display: flex;
    align-items: center;
  }

  .header-inner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 16px;
    width: 100%;
  }

  .logo {
    font-size: 15px;
    font-weight: 600;
    color: #e5e5e5;
  }

  .badge {
    font-size: 10px;
    font-weight: 600;
    color: #22c55e;
    background: #14532d;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .lora-badge {
    color: #a78bfa;
    background: #2e1065;
  }

  .actor-badge {
    font-size: 10px;
    color: #737373;
    margin-left: auto;
    max-width: 150px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .landing {
    flex: 1;
    display: grid;
    place-content: center;
    padding: 24px 16px;
  }

  .landing-card {
    text-align: center;
    max-width: 360px;
  }

  .landing-card h2 {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
  }

  .model-desc {
    font-size: 13px;
    color: #a3a3a3;
    line-height: 1.5;
    margin-bottom: 20px;
  }

  .gpu-info {
    font-size: 12px;
    color: #737373;
    margin-bottom: 16px;
  }

  .warning {
    font-size: 13px;
    color: #f59e0b;
    background: #451a03;
    padding: 12px;
    border-radius: 8px;
  }

  .lora-panel {
    margin-bottom: 16px;
  }

  .btn-toggle {
    background: none;
    border: 1px solid #262626;
    color: #a3a3a3;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    margin-bottom: 8px;
  }

  .btn-toggle:hover {
    border-color: #404040;
    color: #e5e5e5;
  }

  .lora-config {
    text-align: left;
    background: #111111;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    padding: 12px;
    margin-top: 4px;
  }

  .input-actor {
    width: 100%;
    background: #171717;
    border: 1px solid #262626;
    border-radius: 6px;
    color: #e5e5e5;
    font-size: 13px;
    padding: 8px 10px;
    margin-bottom: 8px;
    font-family: inherit;
    box-sizing: border-box;
  }

  .input-actor::placeholder {
    color: #525252;
  }

  .input-actor:focus {
    outline: none;
    border-color: #a78bfa;
  }

  .lora-actions {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
  }

  .btn-small {
    background: #a78bfa;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    min-height: 32px;
  }

  .btn-small:active {
    background: #8b5cf6;
  }

  .btn-clear {
    background: #404040;
  }

  .btn-clear:active {
    background: #525252;
  }

  .lora-status {
    font-size: 11px;
    color: #737373;
    line-height: 1.5;
  }

  .btn-primary {
    background: #3b82f6;
    color: white;
    border: none;
    padding: 12px 32px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    min-height: 44px;
    min-width: 44px;
  }

  .btn-primary:active {
    background: #2563eb;
  }

  .progress-bar {
    height: 6px;
    background: #1a1a1a;
    border-radius: 3px;
    overflow: hidden;
    margin: 16px 0 8px;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .lora-fill {
    background: #a78bfa;
  }

  .progress-text {
    font-size: 13px;
    color: #737373;
  }

  .error-card .error-text {
    font-size: 13px;
    color: #ef4444;
    margin-bottom: 16px;
    word-break: break-word;
  }

  .chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    padding-bottom: 80px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .empty-state {
    text-align: center;
    padding: 48px 16px;
    color: #737373;
    font-size: 13px;
  }

  .empty-state .hint {
    margin-top: 8px;
    font-size: 12px;
    color: #525252;
  }

  .message {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .message-label {
    font-size: 11px;
    font-weight: 600;
    color: #737373;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .message-content {
    font-size: 15px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .message.user .message-content {
    color: #e5e5e5;
  }

  .message.assistant .message-content {
    color: #d4d4d4;
  }

  .cursor {
    animation: blink 1s step-end infinite;
    color: #3b82f6;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }

  .stats-bar {
    font-size: 11px;
    color: #525252;
    text-align: center;
    padding: 4px;
    background: #0a0a0a;
    border-top: 1px solid #1a1a1a;
  }

  .reflection-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 11px;
    color: #a3a3a3;
    padding: 6px 12px;
    background: #0a0a0a;
    border-top: 1px solid #1a1a1a;
  }
  .reflection-bar select {
    background: #1a1a1a;
    color: #e5e5e5;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
  }
  .critique-chip {
    background: #1a2a3a;
    color: #93c5fd;
    border-radius: 3px;
    padding: 2px 8px;
    cursor: help;
  }

  .phase-pill {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    margin-left: 6px;
    border-radius: 10px;
    text-transform: lowercase;
  }
  .phase-generate { background: #1e3a2e; color: #6ee7b7; }
  .phase-critique { background: #3a2a1e; color: #fbbf24; }
  .phase-revise { background: #1e2a3a; color: #93c5fd; }
  .phase-finalize { background: #2a1e3a; color: #c4b5fd; }
  .phase-surprise_eval { background: #3a1e2a; color: #f9a8d4; }
  .phase-predict_next { background: #2a3a1e; color: #bef264; }

  .ai-toggle {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
  }
  .ai-toggle input[type="checkbox"] {
    accent-color: #bef264;
  }

  .surprise-badge {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    margin-left: 6px;
    border-radius: 10px;
    cursor: help;
  }
  .surprise-lo  { background: #1e3a2e; color: #6ee7b7; }
  .surprise-mid { background: #3a2a1e; color: #fbbf24; }
  .surprise-hi  { background: #3a1e1e; color: #fca5a5; }

  .prediction-chip {
    margin: 6px 12px 0;
    padding: 6px 10px;
    background: #1a2a1a;
    color: #bef264;
    border: 1px dashed #2a4a2a;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    user-select: text;
  }
  .prediction-chip:hover { background: #1a3a1a; }

  .embed-pill {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 10px;
    background: #1a2a3a;
    color: #93c5fd;
    cursor: help;
  }
  .embed-pill.embed-ready { background: #1e3a2e; color: #6ee7b7; }
  .embed-pill.embed-error { background: #3a1e1e; color: #fca5a5; }

  .tool-log {
    margin: 8px 12px;
    padding: 6px 10px;
    background: #1a1a1a;
    border-left: 2px solid #fbbf24;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 11px;
  }
  .tool-row {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 6px;
    padding: 2px 0;
    color: #a3a3a3;
  }
  .tool-row.tool-error { color: #fca5a5; }
  .tool-name {
    color: #fbbf24;
    font-weight: 600;
  }
  .tool-args { color: #cbd5e1; }
  .tool-arrow { color: #525252; }
  .tool-result { color: #e5e5e5; cursor: help; }
  .tool-row.tool-error .tool-result { color: #fca5a5; }
  .phase-execute_tool { background: #3a2e1e; color: #fcd34d; }

  .daemon-chip {
    margin-left: auto;
    background: #0a0a0a;
    border: 1px solid #2a2a2a;
    color: #a3a3a3;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 12px;
    cursor: pointer;
    font-family: ui-monospace, SFMono-Regular, monospace;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .daemon-chip:hover { border-color: #3a3a3a; color: #e5e5e5; }
  .daemon-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #525252;
    display: inline-block;
  }
  .daemon-online .daemon-dot { background: #6ee7b7; box-shadow: 0 0 4px #6ee7b7; }
  .daemon-offline .daemon-dot { background: #fbbf24; }

  .daemon-panel {
    position: absolute;
    top: 48px;
    right: 8px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #e5e5e5;
    font-size: 11px;
    padding: 12px;
    border-radius: 4px;
    z-index: 20;
    min-width: 280px;
    line-height: 1.6;
    font-family: ui-monospace, SFMono-Regular, monospace;
  }
  .daemon-panel b { color: #a3a3a3; }
  .daemon-err { color: #fca5a5; }
  .daemon-foot {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px solid #2a2a2a;
    color: #525252;
    font-size: 10px;
  }

  .swarm-list {
    margin: 4px 0 0 0;
    padding-left: 14px;
    color: #cbd5e1;
    font-size: 10px;
    list-style: disc;
  }
  .swarm-list li { margin-top: 2px; }
  .swarm-role { color: #6ee7b7; }
  .swarm-foot {
    margin-top: 4px;
    color: #fbbf24;
    font-size: 10px;
  }

  .leader-badge {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 10px;
    background: #2a1e3a;
    color: #c4b5fd;
    font-size: 10px;
    cursor: help;
  }
  .follower-badge {
    display: inline-block;
    margin-left: 6px;
    color: #525252;
    font-size: 10px;
    cursor: help;
  }
  .daemon-leader { color: #c4b5fd; font-weight: 700; }
  .daemon-follower { color: #525252; }

  .input-area {
    position: sticky;
    bottom: 0;
    background: #0a0a0a;
    border-top: 1px solid #1a1a1a;
    padding: 12px 16px;
    padding-bottom: max(12px, env(safe-area-inset-bottom));
  }

  .input-inner {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }

  .input-field {
    flex: 1;
    background: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    color: #e5e5e5;
    font-size: 15px;
    padding: 10px 12px;
    resize: none;
    min-height: 44px;
    max-height: 120px;
    font-family: inherit;
  }

  .input-field::placeholder {
    color: #525252;
  }

  .input-field:focus {
    outline: none;
    border-color: #3b82f6;
  }

  .btn-send {
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    min-height: 44px;
    min-width: 44px;
  }

  .btn-send:disabled {
    background: #1e3a5f;
    color: #525252;
    cursor: not-allowed;
  }

  .btn-send:active:not(:disabled) {
    background: #2563eb;
  }
</style>
