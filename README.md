# 🦙 Llama.cpp Runner (`llama_runner`)

A friendly, dark-themed **Tkinter GUI front end for [`llama-server`](https://github.com/ggml-org/llama.cpp)** — no web UI required. Pick a GGUF model, dial in every llama-server flag from a panel, hit Start, and watch your GPU/RAM/CPU in real time. Fully bilingual (English / فارسی) with an instant language toggle.

![windows-badge](https://img.shields.io/badge/platform-Windows%2010%2B-blue) ![python-badge](https://img.shields.io/badge/python-3.9%2B-3776AB)

---

## ✨ Features

- **1-Click llama-server launcher** — every knobs exposed: GGUF layers, context, slots, threads, batch/ubatch, flash attention, KV quantization (`-ctk/-ctv`), load mode, no-mmap, defrag, MoE offload, YaRN, sleeping, request timeout, and more.
- **Speculative decoding panel (Unsloth-style)** — MTP, Ngram, DraftSpark, DraftFlash, or Auto. `Auto` sends no `--spec-type`, which keeps speculation **off** — measured fastest on CPU-limited rigs.
- **Hand-tuned presets** for an RTX 3060 Ti 8 GB / 16 GB / i3-12th rig — including the legendary `🌟 GOD MODE` and the even hotter `🥒 BIG PICKLE: ABSOLUTE MAX`.
- **Smart auto-config** — reads the model's *real* GGUF metadata (layers, quantization, context) without external deps and recommends settings; estimates VRAM/RAM fit visually.
- **📊 GPU Processes tab** — live per-process table of **every** running process (not just GPU ones): VRAM, GPU %, RAM, CPU %, and disk I/O, with sortable columns, filter buttons (All / High GPU / High CPU / High VRAM), status-bar totals, and **Green → kill the selected process tree**.
- **🖥 System monitor** — CPU %, RAM, and disk usage cards with sparkline graphs right on the GPU tab.
- **▶ Run Task tab** — type a prompt (or use the quick presets), hit Run, and the app auto-starts the server on demand, POSTs to `/completion`, and shows a big **tok/s** readout + timing breakdown and the model's answer.
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
| **Server Setup** | All llama-server settings, command preview, model analysis & fit badge, presets, start/stop/log console. |
| **GPU Processes** | Per-process VRAM / GPU / RAM / CPU / disk table with live refresh, sort, filters, system usage cards, green kill. |
| **Run Task** | One-shot prompt → server, live **tok/s** + prompt/output timings, quick templates (Chat / Summarize / Translate / Code). |

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

## 🔧 Troubleshooting

- **"No GPU found"** → the GPU tab relies on `nvidia-smi`; install/update NVIDIA drivers (NVIDIA-SMI is in `C:\Windows\System32\`).
- **Server won't start** → check the log console; llama-server prints its real error there. Most common: wrong binary path, missing CUDA backend for the model dtype, or VRAM overflow (drop to `Low VRAM / Safe` and reduce context).
- **MTP preset load fails** → your model has no MTP head; switch Spec Decoding to `Auto` or `Ngram` in the panel.

## 📄 Files

```
llama_runner(3).py     # the entire application (single file, ~3.7k lines)
requirements.txt       # optional psutil only
```

## ⚖️ License

Use freely — this is a personal tooling project built around [llama.cpp](https://github.com/ggml-org/llama.cpp) (its own license applies to the binary and models).