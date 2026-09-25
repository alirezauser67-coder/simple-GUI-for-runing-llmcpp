# 🦙 Llama.cpp Runner (`llama_runner`)

A friendly, dark-themed **Tkinter GUI front end for [`llama-server`](https://github.com/ggml-org/llama.cpp)** — no web UI required. Pick a GGUF model, dial in every llama-server flag from a panel, hit Start, and watch your GPU/RAM/CPU in real time. Fully bilingual (English / فارسی) with an instant language toggle.

![windows-badge](https://img.shields.io/badge/platform-Windows%2010%2B-blue) ![python-badge](https://img.shields.io/badge/python-3.9%2B-3776AB)

---

## ✨ Features

- **1-Click llama-server launcher** — every knobs exposed: GGUF layers, context, slots, threads, batch/ubatch, flash attention, KV quantization (`-ctk/-ctv`), load mode, no-mmap, defrag, MoE offload, YaRN, sleeping, request timeout, and more. The **llama-server exe dropdown rescans `C:\llmcap` recursively every time you open it** — any exe/subfolder you drop in (`default` = the root exe, otherwise the folder name) appears without a restart.
- **Speculative decoding panel (Unsloth-style)** — MTP, Ngram, DraftSpark, DraftFlash, or Auto. `Auto` sends no `--spec-type`, which keeps speculation **off** — measured fastest on CPU-limited rigs.
- **Hand-tuned presets** for an RTX 3060 Ti 8 GB / 16 GB / i3-12th rig — including the legendary `🌟 GOD MODE` and the even hotter `🥒 BIG PICKLE: ABSOLUTE MAX`.
- **Smart auto-config** — reads the model's *real* GGUF metadata (layers, quantization, context) without external deps and recommends settings; estimates VRAM/RAM fit visually.
- **📊 GPU Processes tab** — live per-process table of **every** running process (not just GPU ones): VRAM, GPU %, RAM, CPU %, and disk I/O, with sortable columns, filter buttons (All / High GPU / High CPU / High VRAM), status-bar totals, and **Green → kill the selected process tree**.
- **🖥 System monitor** — CPU %, RAM, and disk usage cards with sparkline graphs right on the GPU tab.
- **▶ Run Task tab** — type a prompt (or use the quick presets), hit Run, and the app auto-starts the server on demand, POSTs to `/completion`, and shows a big **tok/s** readout + timing breakdown and the model's answer. Every run is logged (`Run Task: sending prompt…` → `Run Task: finished — N tokens in X.Xs`, or `Run Task FAILED: …`) so nothing dies silently.
- **📈 Task Inspector tab** — one full-window **tok/s chart** of the whole generation: gradient area fill, nice-rounded axis ticks, per-sample dots, **peak marker**, **live latest tok/s badge**, dashed **avg** line, and a stats line (`tokens · avg · peak tok/s · elapsed`). The chart is fed by **both Run Task and the Speed Test**, keeps accumulating while the tab is hidden (run tasks from the Task tab freely), and repaints automatically when you switch back.
- **GGUF metadata parser** — zero-dependency, reads only the metadata header (seeks past the huge tokenizer arrays), so it's fast even on 20+ GB files.
- **Bilingual UI** — English and Persian (Lalezar-aware fonts, RTL-safe labels), flipped instantly with the 🌐 toggle.
- **Modern slate/teal dark theme**, 120+ scenery niceties, log console with filters and live output.

## 🧰 Requirements

| Component | Notes |
|---|---|
| **OS** | Windows 10/11 (monitoring uses `taskkill`, CIM/WMI, `nvidia-smi`) |
| **Python** | 3.9+ with **tkinter** (included with the official Windows installer) |
| **llama-server.exe** | a recent llama.cpp `llama-server.exe` build (e.g. from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)) |
| **A GGUF model** | any `*.gguf` you have, e.g. from Hugging Face |
| *optional* | `psutil` (`pip install psutil`) for more accurate CPU/RAM detection |

No other Python packages are required — everything else is the standard library.

## 🚀 Getting started

```powershell
# 1. (optional) better hardware detection
pip install psutil

# 2. run
python "llama_runner(3).py"
```

1. **Server Setup tab** → browse to `llama-server.exe` and your `.gguf` model.
2. Pick a **preset** (or hit ⚙ Auto-analyze for per-model recommendations), tweak anything.
3. Click **▶ Start Server**, or just go to the **Run Task** tab and let it start on demand.
4. Watch the tune-up on the **GPU Processes** tab; kill runaway processes from the table.

## 🗂 Tabs

| Tab | What it does |
|---|---|
| **Server Setup** | All llama-server settings, command preview, model analysis & fit badge, presets (exe dropdown auto-scans `C:\llmcap`), start/stop/log console. |
| **GPU Processes** | Per-process VRAM / GPU / RAM / CPU / disk table with live refresh, sort, filters, system usage cards, green kill. |
| **Run Task** | One-shot prompt → server, live **tok/s** + prompt/output timings, quick templates (Chat / Summarize / Translate / Code). |
| **Task Inspector** | Full-window live **tok/s chart** (peak/avg/latest badges + stats), header showing server state, fed by Run Task *and* the Speed Test. |

## 🎛 Presets

| Preset | Purpose |
|---|---|
| `🌟 GOD MODE` | Max everything: full offload, 16K ctx, 4K batches, flash + KV quant, locking, no-mmap. |
| `🥒 BIG PICKLE: ABSOLUTE MAX` | Beyond GOD MODE: 32K ctx (q8_0 KV), **MTP speculation**, cache-idle-slots, reasoning off, never sleeps. ⚠️ needs an MTP-head model for the speculation bonus. |
| `🤖 ox-alpha` | Highest tok/s, MTP-tuned draft decoding. |
| `⚡ Max` | Safe full offload default (the startup default). |
| `🏎 MAX SPEED` | Raw tok/s: 4K micro-batches, speculation Auto (= off), thinking off, spec draft auto. |
| `🖥 This PC` | Auto-tuned for the detected RTX 3060 Ti / 16 GB / i3-12th setup. |
| `My Rig` / `Low VRAM / Safe` | Balanced defaults / conservative 20-layer offload. |

## 📈 The GPU tab goes beyond VRAM

Every active process shows: **VRAM (MB), GPU %, RAM, CPU %, disk R/W (MB/s)** and makes the whole control easy to keep an eye on. The per-process probe uses CIM performance counters (`Win32_PerfFormattedData_PerfProc_Process`) so you get real per-second CPU/I/O deltas without extra packages.

## 📝 Notes & tips

- **Spec decoding `Auto` = OFF.** On low-core-count CPUs, CPU-side ngram drafting slows decode; that's why `⚡ Max`/`🏎 MAX SPEED` and `Auto` ship without a `--spec-type` flag. Enable MTP/Ngram only if the model supports/benefits from it.
- **KV quantization** (`q8_0`) roughly halves cache VRAM — combined with flash attention it's usually a strict win on narrow GPUs.
- **`--no-mmap` + `mlock`** put the whole model in RAM; great for speed on Windows, needs enough system memory (13B q4 ≈ 8 GB + overhead).
- **Reasoning models** (R1-style): `MAX SPEED`/`BIG PICKLE` set thinking off to avoid burning tokens; set it back to **On** for full reasoning.
- **The chart shows "No task run yet"** → the first ~50 ms of a generation (including token 1, which would plot at an absurd ~700k tok/s) are skipped on purpose; you need at least 2 valid samples. A prompt the model answers with an immediate EOS (empty output) also earns no samples — rephrase the prompt.
- **Thinking?** → llama.cpp supports reasoning (`--reasoning-format` is in the Server tab), but only reasoning models actually emit thinking tags — e.g. Gemma-4 never does, so don't expect any. The Task Inspector header flips to `● Server RUNNING` once the model is loaded (also works for servers started outside the app, via an HTTP `/health` probe).

## 🔧 Troubleshooting

- **"No GPU found"** → the GPU tab relies on `nvidia-smi`; install/update NVIDIA drivers (NVIDIA-SMI is in `C:\Windows\System32\`).
- **Server won't start** → check the log console; llama-server prints its real error there. Most common: wrong binary path, missing CUDA backend for the model dtype, or VRAM overflow (drop to `Low VRAM / Safe` and reduce context).
- **MTP preset load fails** → your model has no MTP head; switch Spec Decoding to `Auto` or `Ngram` in the panel.

## 📄 Files

```
llama_runner(3).py     # the entire application (single file, ~5.2k lines)
requirements.txt       # optional psutil only
```

## 🆕 Recent changes (this session)

- **Task Inspector rebuilt** — dropped the System/Prompt/Thinking/Output panels and Copy/Clear buttons; the tab is now one big chart (canvas grows to fill the window, ~860×620 at default size).
- **Graph upgraded** — nice-rounded y-axis (no more `18.75`-style labels), 5 evenly spaced x-ticks, gradient area fill, thick smoothed line, dots per sample, peak marker, live latest-tok/s badge, avg reference line, richer empty state.
- **Graph works repeatedly** — fixed the one-shot bug: live draws used to `delete()` the 1×1 unmapped canvas (Task tab active) and blank it permanently; draws now bail *before* wiping, samples keep accumulating, and a `<Map>` binding repaints on tab return.
- **Speed Test feeds the chart** too (was Run Task only), and resets the series at start.
- **Server-state header fixed** — flips to `● Server RUNNING` when the model loads (including externally started servers, via cached HTTP `/health` probe) and back when it stops; used to sit on "Server not running" forever.
- **System prompt no longer injected** — `run_task` sends exactly what you typed (byte-for-byte trace); the old hidden injection was removed.
- **Run Task logging** — `sending prompt…` / `finished — N tokens in X.Xs` / `FAILED: …` / empty-EOS warnings all land in the log pane.
- **Stream robustness** — token-1 bogus sample guard (y-axis used to be wrecked by a 769k tok/s point), throttled 0.3 s live redraws, ASCII-only thinking-tag split (`<thought>`, `<brain>`, `<|think|>`), final force-flush.
- **llama-server exe dropdown** — replaced the `default`/`prims` buttons with a combobox that rescans `C:\llmcap` recursively on every open (root exe = `default`, subfolders by name).
- **Temperature control in Server → Parameters** — new `Temperature:` spinbox (0.0–2.0, default 0.8 = llama.cpp's own default) emitting `--temp` on the command line; live-updates the command preview, keeps its value when applying perf presets, and translates EN/FA. The Run Task tab's own temperature spinbox still overrides it per request.
- **Multimodal panel reworked → "👁 Multimodal & MTP"** — added an **MTP model selector** (e.g. `mtp-gemma-4-12B-it-Q4_0.gguf` next to your main model) emitting `--spec-draft-model` and forcing `--spec-type draft-mtp` while ON (overrides the spec panel's Off/Ngram choice; OFF restores it). Both the **mmproj** and **MTP** rows now have a colored **● ON / ○ OFF** switch (green = flag sent, grey = path remembered but skipped, entry dims when off); picking a file auto-enables its row, auto-detected projectors switch mmproj ON. **The mmproj URL field is gone** — no `--mmproj-url` is ever emitted.
- **Verified end-to-end** — automated harnesses (start server → speed test → 2 tasks from other tabs → return) pass with 0 Tk errors: panels removed, canvas 861×612, series refresh every run, graph redraws on tab switch.
- **⚡ Speed preset list** — new dropdown right of the `Preset:` selector that applies instantly (no Apply click), with three entries: **📊 Baseline (-fa on -t 4)**, **⚡ GGML_CUDA_GRAPH_OPT=1**, **🏆 MiMo Ultra (16K)**. The two small ones are **overlays**: they only *add* their delta on top of the preset currently in use (Baseline forces `-fa on -t 4 -tb 6` + flag off; GGML flips only the env flag) without touching ctx/batches/extra; the main combo keeps pointing at the base preset. MiMo Ultra is a full preset. Also present in the main preset list.
- **🏆 MiMo Ultra (16K)** — full max-speed preset: `-c 16384`, q8_0 KV (16K still fits the 8GB card), `-fa on`, `-t 4 -tb 6`, `-b/-ub 2048`, mlock, defrag 0.1, single slot, plus Extra Args `-bs --samplers top_k;temperature --cache-reuse 256`. `GGML_CUDA_GRAPH_OPT` intentionally **off** — it measured −0.6% on this rig (see below).
- **⚡ CUDA streams checkbox** — new `⚡ CUDA streams` toggle next to Extra Args showing/bumping `GGML_CUDA_GRAPH_OPT=1`. The env var is injected **only into the llama-server child process** (never the whole app) and only takes effect on the next server start; presets without the key reset it to off.
- **Benchmarked on this rig** (llama-bench build 11065, 2026-09-25, Qwen3.5-9B Q4_K_M): baseline **67.0 tok/s tg / 2146 pp** = 84% of the 448 GB/s speed-of-light. `GGML_CUDA_GRAPH_OPT=1` = 66.5 (−0.6%, slightly worse — Ampere doesn't gain the Ada/Blackwell streams win). Real server: 63.3 tok/s, **+1.3% with `-bs --samplers top_k;temperature`**. Threads 8, `-ub 2048`, FA off, KV q8_0 all measured flat. Live P-state during generation: **P2, pegged at the 200W power limit**, mem 6801/7001 MHz → the remaining lever is GPU clocks (power limit / memory OC), not flags.

## ⚖️ License

Use freely — this is a personal tooling project built around [llama.cpp](https://github.com/ggml-org/llama.cpp) (its own license applies to the binary and models).
