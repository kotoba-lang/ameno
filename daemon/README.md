# @etzhayyim/ameno-daemon

Headless system-resident daemon for ameno (Tier-2 worker host).

Same LangGraph (reflection + active inference + ReAct tools) as the
browser appview, but driven by **Ollama** instead of MediaPipe, with
state persisted to `~/.ameno/checkpointer.json`.

ADR: [`90-docs/adr/2605191229-ameno-daemon-path-a-bun-langgraph.md`](../../../90-docs/adr/2605191229-ameno-daemon-path-a-bun-langgraph.md).

## Why

ADR-2605191135 made ameno *tab-resident* — alive while a browser tab is
open. This daemon is the **真の 常駐化** path A: a headless process that
runs from system startup until you stop it. ADR-2605191229 explains why
it's Bun/Node + Hono + LangGraph instead of Tauri/Wails (Tauri is a UI
shell, not a daemon runtime — we want headless).

## Requirements

- macOS (Linux follow-up via systemd)
- [Bun](https://bun.sh) ≥ 1.0 (recommended) or Node ≥ 22
- [Ollama](https://ollama.com) running on localhost:11434
- An Ollama model pulled, e.g.:

  ```sh
  ollama pull gemma3:4b
  ```

## Install / run (interactive)

```sh
cd 60-apps/etzhayyim-project-ameno/daemon
pnpm install              # via the monorepo workspace
bun run src/server.ts     # or: node --experimental-strip-types src/server.ts
```

You should see:

```
ameno-daemon listening on http://127.0.0.1:12480
  did:        did:web:host:<hostname>-<uuid>
  home:       /Users/you/.ameno
  ollama:     http://localhost:11434 (model: gemma3:4b)
```

Verify:

```sh
curl http://127.0.0.1:12480/healthz
curl http://127.0.0.1:12480/workerInfo
```

Stream a turn:

```sh
curl -N -X POST http://127.0.0.1:12480/threads/demo/stream \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"What time is it?"}],
    "maxIterations": 0,
    "toolsEnabled": true
  }'
```

## Always-on (launchd, macOS)

1. Edit `com.etzhayyim.ameno-daemon.plist` to replace:
   - `YOUR_USERNAME` (your macOS short username)
   - `YOUR_REPO_PATH` (absolute path to this monorepo)
   - `/opt/homebrew/bin/bun` (run `which bun` to confirm)

2. Install + load:

   ```sh
   mkdir -p ~/.ameno
   cp com.etzhayyim.ameno-daemon.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.etzhayyim.ameno-daemon.plist
   ```

3. Verify:

   ```sh
   launchctl list | grep ameno-daemon
   tail -f ~/.ameno/daemon.stdout.log
   ```

4. Stop:

   ```sh
   launchctl unload ~/Library/LaunchAgents/com.etzhayyim.ameno-daemon.plist
   ```

The daemon auto-restarts on crash (`KeepAlive.SuccessfulExit=false`).

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `AMENO_HOME` | `~/.ameno` | state directory (checkpointer + DID) |
| `AMENO_PORT` | `12480` | HTTP listen port |
| `AMENO_HOST` | `127.0.0.1` | listen address. **Do NOT expose to LAN** unless you understand the threat model |
| `AMENO_MODEL` | `gemma3:4b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |

## HTTP API

| method | path | body | returns |
|---|---|---|---|
| GET | `/healthz` | — | `{status, workerDid}` |
| GET | `/workerInfo` | — | `{did, uptimeMs, model, ollamaReachable, ...}` |
| POST | `/threads/:tid/invoke` | `{messages, maxIterations, activeInference, toolsEnabled}` | `{thread_id, draft}` |
| POST | `/threads/:tid/stream` | same | SSE `data: <GraphChunk JSON>\n\n` per super-step |
| GET | `/threads/:tid/state` | — | latest checkpointed state for thread |

## What this daemon is NOT

- **Not a UI shell** — no window, no menu bar item. Use the browser
  appview at `ameno.etzhayyim.com` or `localhost:5173` as the viewer.
- **Not WebGPU** — Bun/Node have no WebGPU. We use Ollama instead.
  If you need browser WebGPU Gemma, use the svelte appview (tab-resident,
  ADR-2605191135).
- **Not Tier 1** — Tier 1 is Murakumo Mac mini fleet (Python
  `kotodama.agent_daemon_main`, ADR-2605182312). This daemon is the
  TypeScript Path A. The Python Path B (port to kotodama) is the
  follow-up ADR.

## Substrate boundary

This daemon respects ADR-2605172000 kotoba rule: no central DB, no
fiat processor. State is local file; future graduation goes to MST via
`@etzhayyim/sdk/checkpointer` (ADR-2605171800).

## License

Apache-2.0.
