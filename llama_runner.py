import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import shutil
import re
import atexit
import time
import webbrowser
from collections import deque
import urllib.request
import urllib.error
import json
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tkinter.font as tkfont
    HAS_TKFONT = True
except Exception:
    HAS_TKFONT = False


# ---------------------------------------------------------------------------
# Theme palette — single source of truth for every color in the UI.
# Modern dark slate base with teal/green accents (a calmer, more "app-like"
# look than the old Catppuccin Mocha purple/blue). S(token) resolves a name
# to a hex string so no widget hardcodes a literal.
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":            "#0f172a",  # base canvas
    "surface":       "#1e293b",  # panels, buttons, badges
    "surface_hi":    "#334155",  # hover / selected states
    "border":        "#334155",  # subtle 1px separators
    "fg":            "#e2e8f0",  # primary text
    "fg_muted":      "#94a3b8",  # secondary text, captions
    "accent":        "#2dd4bf",  # teal — primary accent (tabs, titles)
    "ok":            "#22c55e",  # green — good fit / success
    "warn":          "#fbbf24",  # amber — partial / warnings
    "danger":        "#f87171",  # red — risk / errors
    "info":          "#38bdf8",  # sky — info / progress
    "log_bg":        "#0b1120",  # near-black log well
}


def S(name):
    """Resolve a palette token to its hex string, with a safe fallback."""
    return PALETTE.get(name, "#e2e8f0")


# ---------------------------------------------------------------------------
# Minimal GGUF header/metadata reader (no external deps). Reads only the
# metadata block, seeking past large arrays (like the tokenizer vocab)
# instead of loading them, so it stays fast even on huge model files.
# This lets us recommend settings based on the model's *real* layer count
# instead of guessing from the file size alone.
# ---------------------------------------------------------------------------
import struct

_GGUF_MAGIC = b"GGUF"
_GT_UINT8, _GT_INT8, _GT_UINT16, _GT_INT16 = 0, 1, 2, 3
_GT_UINT32, _GT_INT32, _GT_FLOAT32, _GT_BOOL = 4, 5, 6, 7
_GT_STRING, _GT_ARRAY, _GT_UINT64, _GT_INT64, _GT_FLOAT64 = 8, 9, 10, 11, 12

_GGUF_SIMPLE_FORMATS = {
    _GT_UINT8: ("B", 1), _GT_INT8: ("b", 1),
    _GT_UINT16: ("H", 2), _GT_INT16: ("h", 2),
    _GT_UINT32: ("I", 4), _GT_INT32: ("i", 4),
    _GT_FLOAT32: ("f", 4), _GT_BOOL: ("?", 1),
    _GT_UINT64: ("Q", 8), _GT_INT64: ("q", 8),
    _GT_FLOAT64: ("d", 8),
}

# Common quantization codes for general.file_type (not exhaustive, but
# covers the ones people actually download).
_GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 24: "IQ3_XS", 25: "IQ3_XXS", 26: "IQ1_S",
    27: "IQ4_NL", 28: "IQ3_S", 29: "IQ3_M", 30: "IQ2_S", 31: "IQ2_M",
    32: "IQ4_XS", 33: "IQ1_M", 34: "BF16",
}


def _gguf_read_string(f):
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _gguf_skip_array(f):
    (elem_type,) = struct.unpack("<I", f.read(4))
    (count,) = struct.unpack("<Q", f.read(8))
    if elem_type in _GGUF_SIMPLE_FORMATS:
        _, size = _GGUF_SIMPLE_FORMATS[elem_type]
        f.seek(size * count, 1)
    elif elem_type == _GT_STRING:
        for _ in range(count):
            (slen,) = struct.unpack("<Q", f.read(8))
            f.seek(slen, 1)
    elif elem_type == _GT_ARRAY:
        for _ in range(count):
            _gguf_skip_array(f)


def _gguf_read_value(f, vtype):
    if vtype == _GT_STRING:
        return _gguf_read_string(f)
    if vtype in _GGUF_SIMPLE_FORMATS:
        fmt, size = _GGUF_SIMPLE_FORMATS[vtype]
        return struct.unpack("<" + fmt, f.read(size))[0]
    if vtype == _GT_ARRAY:
        _gguf_skip_array(f)
        return None
    return None


def read_gguf_metadata(path, timeout_kv=200000):
    """Read just the GGUF header + metadata key/values, skipping tensor
    data and large arrays. Returns {} on any failure (not a gguf file,
    corrupted, unreadable, etc.) so callers can fall back gracefully."""
    result = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != _GGUF_MAGIC:
                return result
            (version,) = struct.unpack("<I", f.read(4))
            if version >= 2:
                struct.unpack("<Q", f.read(8))  # tensor_count, unused here
                (kv_count,) = struct.unpack("<Q", f.read(8))
            else:
                struct.unpack("<I", f.read(4))
                (kv_count,) = struct.unpack("<I", f.read(4))
            kv_count = min(kv_count, timeout_kv)
            for _ in range(kv_count):
                key = _gguf_read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                result[key] = _gguf_read_value(f, vtype)
    except Exception:
        pass
    return result


def get_model_architecture_info(path):
    """Distills the raw metadata dict down to the fields useful for
    picking good llama-server settings."""
    meta = read_gguf_metadata(path)
    if not meta:
        return None
    arch = meta.get("general.architecture", "")
    file_type_code = meta.get("general.file_type")
    return {
        "architecture": arch or "unknown",
        "name": meta.get("general.name"),
        "block_count": meta.get(f"{arch}.block_count"),
        "context_length": meta.get(f"{arch}.context_length"),
        "embedding_length": meta.get(f"{arch}.embedding_length"),
        "head_count": meta.get(f"{arch}.attention.head_count"),
        "quantization": _GGUF_FILE_TYPES.get(file_type_code, f"type {file_type_code}" if file_type_code is not None else None),
    }


# GGUF architectures that take a CLIP/mmproj projector — i.e. models that
# can actually see images. A base model is a "vision model" only if its
# general.architecture is one of these AND a matching .mmproj is loaded.
VISION_ARCHITECTURES = {
    "llava",       # LLaVA 1.5 / 1.6 / bakllava
    "qwen2vl",     # Qwen2-VL / Qwen2.5-VL
    "qwen3v",      # Qwen3-VL (newer llama.cpp)
    "minicpmv",    # MiniCPM-V
    "internvl",    # InternVL
    "moondream",   # Moondream
    "florence2",   # Florence-2
}

# Well-known pure-text architectures. Anything here can NEVER see images
# no matter what mmproj is attached.
TEXT_ONLY_ARCHS = {
    "llama", "llama2", "llama3",
    "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwq",
    "mistral", "mixtral",
    "gemma", "gemma2", "gemma3",
    "phi2", "phi3", "gpt2", "falcon", "baichuan", "mpt",
    "starcoder", "deepseek2", "deepseek3", "olmo", "granite",
    "arctic", "dbrx", "gptneox", "persimmon", "llava_quant",
}


def detect_mmproj(model_path):
    """Find the best multimodal projector living beside the model.

    Projectors ship in a few shapes: '*.mmproj' files, GGUF files named
    'mmproj-<model>-f16.gguf', or occasionally 'clip-<model>.gguf'. We
    recognize all of them and pick the candidate whose name best matches
    the model's own name (substring first, then shared-name-token ratio),
    so a folder holding several projectors still gets the right one.
    Returns an absolute path, or None."""
    d = os.path.dirname(os.path.abspath(model_path))
    if not os.path.isdir(d):
        return None
    try:
        files = os.listdir(d)
    except OSError:
        return None
    model_basename = os.path.basename(model_path).lower()
    base = os.path.splitext(model_basename)[0]
    cands = []
    for fn in files:
        fnl = fn.lower()
        if fnl == model_basename:
            continue
        if fnl.endswith(".mmproj") or fnl.startswith(("mmproj", "clip")):
            cands.append(fn)
    if not cands:
        return None
    cands.sort()

    def norm(name):
        n = re.sub(r"(?i)(\.(mmproj|gguf|bin))$", "", name)
        n = re.sub(r"(?i)^(mmproj|clip)[-_]?", "", n)
        n = re.sub(r"(?i)[-_ ]?(fp16|bf16|f16|f32|q2_?k|q3_?k|q4_?k|q5_?k|q6_?k|q8_?k|q4_?0|q5_?0|q8_?0)\b", "", n)
        return n.strip("-_ .")

    def toks(s):
        return set(re.split(r"[^a-z0-9]+", s.lower())) - {""}

    def score(fn):
        fnl = fn.lower()
        s = 0.0
        if base and base in fnl:
            s += 3.0
        common = toks(base) & toks(norm(fn))
        denom = max(len(toks(base)), 1)
        s += len(common) / denom * 2.0
        if "clip" in fnl:
            s += 0.5
        if re.search(r"(?i)(f16|bf16|fp16)", fn):
            s += 0.3
        return s

    best = max(cands, key=score)
    pn = norm(best).lower()
    # Only auto-pick a best guess when there's real evidence it belongs to
    # this model: a shared name substring, shared tokens, or it's the only
    # projector in the folder.
    evidence = (base and (base in pn or base in best.lower())) or (toks(base) & toks(pn))
    if not evidence and len(cands) > 1:
        return None
    return os.path.join(d, best)


# ---------------------------------------------------------------------------
# Small helper: hover tooltips so every control can explain itself
# ---------------------------------------------------------------------------
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def set_text(self, text):
        self.text = text

    def show(self, event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 15
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip, text=self.text, justify=tk.RIGHT if _looks_persian(self.text) else tk.LEFT,
            background=S("surface"), foreground=S("fg"),
            relief=tk.SOLID, borderwidth=1,
            font=tip_font(self.text if hasattr(self, "text") else self.text),
            padx=8, pady=5, wraplength=340
        )
        label.pack()

    def hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def _looks_persian(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


# Lalezar is the chosen Persian display font. It lives in the per-user
# Windows font folder; Tkinter resolves user-installed fonts by family name
# on Windows, so "Lalezar" works directly once Tk is up. We verify the family
# is actually registered with Tk and fall back gracefully to Tahoma -> Segoe
# UI if it isn't (e.g. a clean machine or a non-Windows port).
_FA_FONT_FAMILY = "Lalezar"
_FA_FONT_FALLBACK = ("Tahoma", "Segoe UI")
_LATIN_FONT_FAMILY = "Segoe UI"
_LATIN_FONT_FALLBACK = "DejaVu Sans"


def _resolve_fa_family():
    """Return the Persian family name Tk can render, or a fallback."""
    if not HAS_TKFONT:
        return _FA_FONT_FALLBACK[0]
    try:
        avail = set(tkfont.families())
        if _FA_FONT_FAMILY in avail:
            return _FA_FONT_FAMILY
        for f in _FA_FONT_FALLBACK:
            if f in avail:
                return f
    except Exception:
        pass
    return _FA_FONT_FALLBACK[0]


def tip_font(text, size=9, weight="normal"):
    """Font tuple for a piece of text: Lalezar (Persian) or latin otherwise."""
    if _looks_persian(text):
        # +1pt so Persian tooltips visually match Latin ones.
        return (_resolve_fa_family(), size + 1, weight)
    return (_LATIN_FONT_FAMILY, size, weight)


# ---------------------------------------------------------------------------
# Hardware detection - CPU / RAM / GPU (NVIDIA via nvidia-smi, best-effort
# for AMD/Intel), plus a simple "will this model fit" recommendation engine
# similar in spirit to what LM Studio shows on its model cards.
# ---------------------------------------------------------------------------
class HardwareInfo:
    @staticmethod
    def get_cpu_info():
        try:
            logical = os.cpu_count() or 4
            physical = logical
            if HAS_PSUTIL:
                physical = psutil.cpu_count(logical=False) or logical
            return {"physical": physical, "logical": logical}
        except Exception:
            return {"physical": 4, "logical": 4}

    @staticmethod
    def get_ram_info():
        try:
            if HAS_PSUTIL:
                vm = psutil.virtual_memory()
                return {
                    "total_gb": vm.total / (1024 ** 3),
                    "available_gb": vm.available / (1024 ** 3),
                }
            if sys.platform == "win32":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return {
                    "total_gb": stat.ullTotalPhys / (1024 ** 3),
                    "available_gb": stat.ullAvailPhys / (1024 ** 3),
                }
        except Exception:
            pass
        return {"total_gb": 0, "available_gb": 0}

    @staticmethod
    def get_gpu_info():
        """Returns a list of dicts: vendor, name, vram_total_mb, vram_free_mb"""
        gpus = []

        # --- NVIDIA (accurate VRAM numbers) ---
        try:
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi:
                out = subprocess.check_output(
                    [nvidia_smi,
                     "--query-gpu=name,memory.total,memory.free",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=5
                ).decode(errors="ignore").strip()
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        name, total, free = parts
                        gpus.append({
                            "vendor": "NVIDIA",
                            "name": name,
                            "vram_total_mb": float(total),
                            "vram_free_mb": float(free),
                        })
        except Exception:
            pass

        # --- Fallback: at least list adapter names on Windows (no VRAM) ---
        if not gpus and sys.platform == "win32":
            try:
                out = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    stderr=subprocess.DEVNULL, timeout=5
                ).decode(errors="ignore")
                names = [l.strip() for l in out.splitlines()
                         if l.strip() and l.strip().lower() != "name"]
                for n in names:
                    gpus.append({
                        "vendor": "Unknown", "name": n,
                        "vram_total_mb": 0, "vram_free_mb": 0,
                    })
            except Exception:
                pass

        return gpus

    @staticmethod
    def get_gpu_live():
        """Live NVIDIA telemetry for the monitoring tab: real compute
        utilization, actually-used VRAM, and temperature — things the
        static get_gpu_info() snapshot doesn't carry. Returns {} when no
        NVIDIA GPU / nvidia-smi is available."""
        try:
            smi = shutil.which("nvidia-smi")
            if not smi:
                return {}
            out = subprocess.check_output(
                [smi,
                 "--query-gpu=name,utilization.gpu,utilization.memory,"
                 "memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=4
            ).decode(errors="ignore").strip()
            lines = [l for l in out.splitlines() if l.strip()]
            if not lines:
                return {}
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) < 6:
                return {}

            def num(s):
                try:
                    return float(s)
                except ValueError:
                    return None

            return {
                "name": parts[0],
                "util_gpu": num(parts[1]),
                "util_mem": num(parts[2]),
                "mem_used_mb": num(parts[3]),
                "mem_total_mb": num(parts[4]),
                "temp": num(parts[5]),
            }
        except Exception:
            return {}

    @staticmethod
    def get_model_size_gb(path):
        try:
            if path and os.path.exists(path):
                return os.path.getsize(path) / (1024 ** 3)
        except Exception:
            pass
        return 0

    @staticmethod
    def recommend_settings(model_size_gb, gpu_info, ram_info, cpu_info, gguf_info=None):
        """Best-effort recommendation, similar to LM Studio's fit indicator.
        When gguf_info (real architecture metadata) is available, this uses
        the model's actual layer count for a much more precise GPU-layer
        recommendation instead of just guessing from file size."""
        recs = {}
        physical = cpu_info.get("physical", 4)
        threads = max(1, physical - 1) if physical > 1 else 1
        recs["threads"] = threads
        recs["tbatch"] = threads

        block_count = gguf_info.get("block_count") if gguf_info else None
        native_ctx = gguf_info.get("context_length") if gguf_info else None

        if gpu_info:
            gpu = gpu_info[0]
            vram_free_gb = gpu.get("vram_free_mb", 0) / 1024
            if vram_free_gb <= 0:
                recs["ngl"] = 20
                recs["fit"] = "unknown"
                recs["note"] = (f"GPU '{gpu['name']}' detected but VRAM could not be "
                                 f"read — using a conservative offload guess.")
            elif model_size_gb == 0:
                recs["ngl"] = 35
                recs["fit"] = "unknown"
                recs["note"] = "Pick a model file to get a precise recommendation."
            else:
                headroom = vram_free_gb - model_size_gb * 1.15
                if headroom > 0:
                    recs["ngl"] = 99
                    recs["fit"] = "good"
                    recs["note"] = (f"Model should fully offload to GPU "
                                     f"({vram_free_gb:.1f} GB free VRAM).")
                elif block_count:
                    # Precise math: bytes-per-layer from the real layer
                    # count, instead of an arbitrary ratio guess.
                    per_layer_gb = (model_size_gb * 1.05) / block_count
                    fit_layers = max(1, min(block_count, int(vram_free_gb / per_layer_gb)))
                    recs["ngl"] = fit_layers
                    recs["fit"] = "partial"
                    recs["note"] = (f"~{per_layer_gb*1024:.0f} MB/layer across {block_count} layers — "
                                     f"about {fit_layers}/{block_count} layers fit in "
                                     f"{vram_free_gb:.1f} GB free VRAM. The rest run on CPU.")
                else:
                    ratio = max(0.05, min(1.0, vram_free_gb / (model_size_gb * 1.15)))
                    recs["ngl"] = max(1, int(ratio * 40))
                    recs["fit"] = "partial"
                    recs["note"] = (f"Only partial GPU offload fits "
                                     f"({vram_free_gb:.1f} GB free vs ~{model_size_gb:.1f} GB model). "
                                     f"Remaining layers run on CPU and it will be slower.")
        else:
            recs["ngl"] = 0
            recs["fit"] = "cpu"
            recs["note"] = "No GPU detected — this will run on CPU only, which is slow for large models."

        # Cap the recommended context at whatever the model was actually
        # trained with — asking for more than that doesn't help and just
        # burns extra RAM/VRAM on the KV cache.
        if native_ctx:
            recs["ctx"] = min(16384, int(native_ctx))
            if native_ctx < 8192:
                recs["ctx"] = int(native_ctx)

        avail_ram = ram_info.get("available_gb", 0)
        if model_size_gb and avail_ram and model_size_gb > avail_ram * 0.9:
            recs["fit"] = "risk"
            recs["note"] = (recs.get("note", "") +
                             f" Warning: the model (~{model_size_gb:.1f} GB) is close to "
                             f"or larger than your available RAM ({avail_ram:.1f} GB) — "
                             f"it may fail to load or cause heavy swapping.").strip()

        return recs


# ---------------------------------------------------------------------------
# GPU Processes tab helpers (per-process VRAM/engine usage via Windows
# performance counters, detailed card stats via nvidia-smi, taskkill).
# ---------------------------------------------------------------------------
def _ps(command, timeout=5):
    """Run a PowerShell snippet and return stdout ('' on any failure)."""
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True, text=True, timeout=timeout
        )
        return p.stdout.strip()
    except Exception:
        return ""


def get_gpu_detailed_stats():
    """Full NVIDIA snapshot: name, VRAM, util, temp, power, fan, clocks."""
    empty = {"name": "NVIDIA GPU", "used": 0.0, "total": 0.0, "gpu": 0.0,
             "temp": 0.0, "power": 0.0, "power_limit": 0.0, "fan": 0.0,
             "clock": 0.0, "mem_clock": 0.0}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return empty
    try:
        out = subprocess.check_output(
            [smi,
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
             "temperature.gpu,power.draw,power.limit,fan.speed,clocks.gr,clocks.mem",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=4
        ).decode(errors="ignore").strip()
        parts = [x.strip() for x in out.splitlines()[0].split(",")]
        if len(parts) < 10:
            return empty

        def num(s):
            try:
                return float(s)
            except ValueError:
                return 0.0

        return {
            "name": parts[0], "used": num(parts[1]), "total": num(parts[2]),
            "gpu": num(parts[3]), "temp": num(parts[4]), "power": num(parts[5]),
            "power_limit": num(parts[6]), "fan": num(parts[7]),
            "clock": num(parts[8]), "mem_clock": num(parts[9]),
        }
    except Exception:
        return empty


def get_system_stats():
    """CPU %, RAM used/total (MB), and disk used/total (bytes) for all fixed drives.

    One fast CIM pass (no perf-counter 1s sampling delay) so the 1s monitor
    loop stays responsive. Returns {} on failure.
    """
    out = _ps(r"""
$cpu = 0.0
$proc = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue
if ($proc) { $cpu = [double]($proc | Measure-Object -Property LoadPercentage -Sum).Sum }
$trk = 0.0; $frk = 0.0
$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
if ($os) { $trk = [double]$os.TotalVisibleMemorySize; $frk = [double]$os.FreePhysicalMemory }
$tsz = 0.0; $tfr = 0.0
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Size) { $tsz += [double]$_.Size; $tfr += [double]$_.FreeSpace }
}
"$cpu|$trk|$frk|$tsz|$tfr"
""")
    if "|" not in out:
        return {}
    try:
        cpu, trk, frk, tsz, tfr = (float(x) for x in out.split("|"))
    except ValueError:
        return {}
    return {"cpu": min(cpu, 100.0),
            "ram_used": trk - frk, "ram_total": trk,
            "disk_used": tsz - tfr, "disk_total": tsz}


def get_proc_vram_mb():
    """pid -> VRAM MB, from the 'GPU Process Memory' perf counters."""
    out = _ps(r"""
Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue |
Select-Object -ExpandProperty CounterSamples |
Where-Object {$_.CookedValue -gt 1048576} |
ForEach-Object {
    if ($_.InstanceName -match 'pid_(\d+)_') {
        "$($Matches[1])|$([math]::Round($_.CookedValue / 1MB, 0))"
    }
}""")
    result = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        try:
            pid, value = line.split("|")
            pid, value = int(pid), float(value)
            result[pid] = result.get(pid, 0.0) + value
        except ValueError:
            pass
    return result


def get_proc_gpu_pct():
    """pid -> GPU engine utilization %, from the 'GPU Engine' perf counters."""
    out = _ps(r"""
Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue |
Select-Object -ExpandProperty CounterSamples |
Where-Object {$_.CookedValue -gt 0.5} |
ForEach-Object {
    if ($_.InstanceName -match 'pid_(\d+)_') {
        "$($Matches[1])|$($_.CookedValue)"
    }
}""")
    result = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        try:
            pid, value = line.split("|")
            pid, value = int(pid), float(value)
            result[pid] = result.get(pid, 0.0) + value
        except ValueError:
            pass
    return {pid: min(v, 100.0) for pid, v in result.items()}


def get_proc_rsc():
    """pid -> {cpu %, ram bytes, io bytes/s} via the PerfProc perf class.

    PercentProcessorTime is relative to ONE core, so it is divided by the CPU
    count to read as % of total system capacity (Task-Manager style). One fast
    CIM call, no sampling delay. Returns {} on failure.
    """
    nproc = max(1, os.cpu_count() or 1)
    out = _ps(r"""
Get-CimInstance Win32_PerfFormattedData_PerfProc_Process -ErrorAction SilentlyContinue |
Where-Object { $_.IDProcess -gt 0 -and $_.Name -notmatch '^_Total$|^Idle$|^_pct|^_zomb' } |
ForEach-Object {
    "$($_.IDProcess)|$($_.PercentProcessorTime)|$($_.WorkingSet)|$($_.IOReadBytesPerSec)|$($_.IOWriteBytesPerSec)"
}""")
    result = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        try:
            pid, cpu, ws, rd, wr = line.split("|")
            pid, cpu, ws, rd, wr = int(pid), float(cpu), int(ws), float(rd), float(wr)
        except ValueError:
            continue
        result[pid] = {"cpu": min(cpu / nproc, 100.0), "ram": ws, "io": rd + wr}
    return result


def get_proc_names(pids):
    if not pids:
        return {}
    ids = ",".join(str(p) for p in pids)
    out = _ps(f"Get-Process -Id {ids} -ErrorAction SilentlyContinue | "
              "ForEach-Object { \"$($_.Id)|$($_.ProcessName)\" }")
    names = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        try:
            pid, name = line.split("|", 1)
            names[int(pid)] = name
        except ValueError:
            pass
    return names


def taskkill_tree(pid, force=False):
    args = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        args.append("/F")
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hand-tuned presets. "My Rig" is dialed in for RTX 3060 Ti (8GB VRAM) +
# 16GB DDR4 + Intel i3 12th gen (4C/8T) to push close to full utilization
# without running out of VRAM/RAM on typical 7B-13B GGUF models.
# ---------------------------------------------------------------------------
PRESETS = {
    "🌟 GOD MODE (RTX 3060 Ti / 16GB / i3-12th — MAX everything)": {
        "ngl": 99, "ctx": 16384, "slots": 1, "threads": 8, "tbatch": 8,
        "batch": 4096, "ubatch": 4096, "flash": True, "mlock": True, "unified": False, "kv_quant": True,
        "no_mmap": True, "no_warmup": True, "defrag": 0.1,
        "cont_batching": True, "metrics_endpoint": True, "yarn": False, "moe_cpu": False, "moe_cpu_layers": 10,
    },
    "🥒 BIG PICKLE: ABSOLUTE MAX (RTX 3060 Ti / 16GB / i3-12th — beyond GOD MODE)": {
        # Everything GOD MODE pushes, plus the new-generation speed knobs on
        # top: 32K context (made affordable by q8_0 KV cache), MTP speculative
        # decoding for the highest possible decode rate, non-MMAP + mlock so
        # the full model lives in fast RAM, idle-slot caching so repeat chats
        # skip reprocessing, and the server never sleeping. Requires an MTP
        # model for the speculation bonus — if yours doesn't ship an MTP head,
        # switch Spec Decoding to Auto/Ngram in the panel and it runs the same.
        "ngl": 99, "ctx": 32768, "slots": 1, "threads": 8, "tbatch": 8,
        "batch": 4096, "ubatch": 4096, "flash": True, "load_mode": "mlock", "unified": False,
        "kv_quant": True, "ctk": "q8_0", "ctv": "q8_0",
        "no_mmap": True, "no_warmup": True, "defrag": 0.1,
        "cont_batching": True, "metrics_endpoint": True, "yarn": False,
        "moe_cpu": False, "moe_cpu_layers": 13,
        "cache_idle_slots": True, "sleep_idle": False, "sleep_idle_seconds": 300,
        "reasoning_format": "auto", "reasoning": "off",
        "spec_mode": "MTP", "spec_draft_auto": True,
    },
    "🤖 ox-alpha (MAX tok/s — MTP spec-decode tuned)": {
        # Built for one goal: highest tokens/sec on RTX 3060 Ti 8GB with
        # an MTP model. Full offload + KV q8_0 frees VRAM; MTP speculation
        # multiplies decode speed; draft count left on Auto so the server
        # picks 2 per step for GPU (best accepted-length/latency balance).
        "ngl": 99, "ctx": 8192, "slots": 1, "threads": 4, "tbatch": 8,
        "batch": 4096, "ubatch": 4096, "flash": True, "mlock": True, "unified": False,
        "kv_quant": True, "no_mmap": True, "no_warmup": False, "defrag": 0.1,
        "cont_batching": True, "metrics_endpoint": False, "yarn": False,
        "moe_cpu": False, "moe_cpu_layers": 0,
        "spec_mode": "MTP", "spec_draft_auto": True,
    },
    "⚡ Max (RTX 3060 Ti / 16GB — safe full offload)": {
        "ngl": 99, "ctx": 8192, "slots": 1, "threads": 4, "tbatch": 6,
        "batch": 2048, "ubatch": 2048, "flash": True, "mlock": True, "unified": False, "kv_quant": True,
        "cont_batching": True, "metrics_endpoint": False, "yarn": False, "moe_cpu": False, "moe_cpu_layers": 10,
    },
    "My Rig (RTX 3060 Ti / 16GB / i3-12th)": {
        "ngl": 99, "ctx": 8192, "slots": 1, "threads": 4, "tbatch": 6,
        "batch": 1024, "ubatch": 1024, "flash": True, "mlock": True, "unified": False,
        "cont_batching": False, "metrics_endpoint": False, "yarn": False, "moe_cpu": False, "moe_cpu_layers": 10,
    },
    "🏎 MAX SPEED (raw tok/s over everything)": {
        # Every knob pointed at throughput for an RTX 3060 Ti 8GB:
        # full GPU offload, 4K micro-batches (fast prompt processing),
        # speculative decoding on Auto (switch Spec Decoding to MTP if your
        # model ships an MTP head, or Ngram for chat/code), q8_0 KV cache
        # to free VRAM, thinking OFF so reasoning models don't burn tokens,
        # sleep-idle off so the model never has to wake/reload.
        "ngl": 99, "ctx": 8192, "slots": 1, "threads": 4, "tbatch": 8,
        "batch": 4096, "ubatch": 4096, "defrag": 0.1,
        "flash": True, "load_mode": "mlock", "unified": False,
        "kv_quant": True, "ctk": "q8_0", "ctv": "q8_0",
        "no_mmap": False, "no_warmup": True,
        "cont_batching": True, "metrics_endpoint": False, "yarn": False,
        "moe_cpu": False, "moe_cpu_layers": 13,
        "cache_idle_slots": False, "sleep_idle": False, "sleep_idle_seconds": 600,
        "reasoning_format": "auto", "reasoning": "off",
        # Spec on "Auto" = no flag, which is Spec Decoding OFF in llama-server
        # and measured fastest on CPU-limited rigs. If your model ships an MTP
        # head, switch it to MTP for a real speculative speedup (MTP doesn't
        # change output); Ngram is also zero-download but trained on context.
        "spec_mode": "Auto",
    },
    "🖥 This PC (RTX 3060 Ti 8GB / 16GB — detected & tuned)": {
        # Auto-detected hardware: RTX 3060 Ti (8GB VRAM) + 16GB RAM, i3-12th class CPU.
        # Full GPU offload (99 layers) + q8_0 KV cache fits 7B-13B quants with an 8K context.
        "ngl": 99, "ctx": 8192, "slots": 1, "threads": 4, "tbatch": 6,
        "batch": 2048, "ubatch": 2048, "defrag": 0.1,
        "flash": True, "load_mode": "mlock", "unified": False,
        "kv_quant": True, "ctk": "q8_0", "ctv": "q8_0",
        "no_mmap": False, "no_warmup": False,
        "cont_batching": True, "metrics_endpoint": False, "yarn": False,
        # MoE: off by default (dense 7B-13B fits fully); 13 CPU layers is the
        # sweet spot if you switch to an MoE model like Gemma4 later.
        "moe_cpu": False, "moe_cpu_layers": 13,
        # New llama-server abilities: cache idle slots so returning chats skip
        # the prompt reprocess, and put the server to sleep after 10 min idle.
        "cache_idle_slots": True, "sleep_idle": True, "sleep_idle_seconds": 600,
        "reasoning_format": "auto",
    },
    "Low VRAM / Safe": {
        "ngl": 20, "ctx": 4096, "slots": 1, "threads": 4, "tbatch": 4,
        "batch": 512, "flash": True, "mlock": False, "unified": False,
        "cont_batching": False, "metrics_endpoint": False, "yarn": False, "moe_cpu": False, "moe_cpu_layers": 10,
    },
}
CONTEXT_QUICK_VALUES = [2048, 4096, 8192, 16384, 32768, 65536]

FIT_COLORS = {
    "good": S("ok"),
    "partial": S("warn"),
    "cpu": S("warn"),
    "risk": S("danger"),
    "unknown": S("info"),
}
FIT_LABELS = {
    "good": "✓ Full GPU offload should fit",
    "partial": "⚠ Partial GPU offload only",
    "cpu": "⚠ CPU-only (no GPU found)",
    "risk": "✗ May not fit in memory",
    "unknown": "? Not enough info yet",
}
FIT_LABELS_FA = {
    "good": "✓ کل مدل روی GPU جا می‌شود",
    "partial": "⚠ فقط بخشی روی GPU جا می‌شود",
    "cpu": "⚠ فقط CPU (کارت گرافیک پیدا نشد)",
    "risk": "✗ ممکن است در حافظه جا نشود",
    "unknown": "؟ اطلاعات کافی نیست",
}

# ---------------------------------------------------------------------------
# Bilingual UI text (English / Persian). Every visible label, button, and
# tooltip goes through TR[key][lang] so the whole UI can flip languages
# instantly with the 🌐 toggle in the header.
# ---------------------------------------------------------------------------
TR = {
    "app_title": {"en": "Llama.cpp Runner", "fa": "اجراکننده Llama.cpp"},
    "app_subtitle": {"en": "a friendlier front end for llama-server",
                      "fa": "یک رابط کاربری ساده‌تر برای llama-server"},
    "lang_toggle": {"en": "🌐 فارسی", "fa": "🌐 English"},

    "tab_server": {"en": "  Server Setup  ", "fa": "  تنظیمات سرور  "},
    "tab_gpu_procs": {"en": "  GPU Processes  ", "fa": "  پردازش‌های GPU  "},
    "tab_tasks": {"en": "  Run Task  ", "fa": "  اجرای کار  "},
    "task_running": {"en": "● Server RUNNING (port {})", "fa": "● سرور در حال اجرا (پورت {})"},
    "task_not_running": {"en": "● Server not running — starts on demand", "fa": "● سرور اجرا نیست — در صورت نیاز شروع می‌شود"},
    "task_presets": {"en": "Quick:", "fa": "سریع:"},
    "task_pres_chat": {"en": "💬 Chat", "fa": "💬 چت"},
    "task_pres_summarize": {"en": "📝 Summarize", "fa": "📝 خلاصه"},
    "task_pres_translate": {"en": "🌍 Translate", "fa": "🌍 ترجمه"},
    "task_pres_code": {"en": "🐍 Python Code", "fa": "🐍 کد پایتون"},
    "task_pres_chat_text": {"en": "Hello!", "fa": "سلام!"},
    "task_pres_summarize_text": {"en": "Summarize the following text in a few clear sentences:\n\n",
                                 "fa": "متن زیر را در چند جمله کوتاه خلاصه کنید:\n\n"},
    "task_pres_translate_text": {"en": "Translate the following to Persian:\n\n",
                                 "fa": "متن زیر را به انگلیسی ترجمه کنید:\n\n"},
    "task_pres_code_text": {"en": "Write Python code that prints the first 10 Fibonacci numbers.",
                            "fa": "کد پایتون بنویس که ۱۰ عدد اول فیبوناچی را چاپ کند."},
    "task_prompt": {"en": "Prompt / Task", "fa": "پرسش / کار"},
    "task_n_predict": {"en": "Max tokens", "fa": "حداکثر توکن"},
    "task_temp": {"en": "Temp", "fa": "دما"},
    "task_run": {"en": "▶ Run Task", "fa": "▶ اجرا"},
    "task_output": {"en": "Model Output", "fa": "خروجی مدل"},
    "task_empty_prompt": {"en": "Type a prompt first.", "fa": "ابتدا یک پرسش بنویسید."},
    "task_no_model": {"en": "Select a model in the Server Setup tab first.", "fa": "ابتدا در تب تنظیمات سرور یک مدل انتخاب کنید."},
    "task_running_task": {"en": "Generating…", "fa": "در حال تولید..."},
    "task_done": {"en": "Done", "fa": "تمام"},
    "gpu_mon_title": {"en": "GPU Monitor", "fa": "مانیتور GPU"},
    "mon_system": {"en": "System", "fa": "سیستم"},
    "mon_cpu": {"en": "CPU", "fa": "پردازنده"},
    "mon_ram": {"en": "RAM", "fa": "حافظه"},
    "mon_disk": {"en": "DISK", "fa": "دیسک"},
    "gpu_mon_search_ph": {"en": "Search process...", "fa": "جستجوی پردازش..."},
    "gpu_filter_all": {"en": "All", "fa": "همه"},
    "gpu_filter_high_gpu": {"en": "High GPU", "fa": "GPU بالا"},
    "gpu_filter_high_cpu": {"en": "High CPU", "fa": "CPU بالا"},
    "gpu_filter_high_vram": {"en": "High VRAM", "fa": "VRAM بالا"},
    "gpu_pause_btn": {"en": "⏸ Pause", "fa": "⏸ توقف"},
    "gpu_resume_btn": {"en": "▶ Resume", "fa": "▶ ادامه"},
    "gpu_end_btn": {"en": "✕ End Process", "fa": "✕ بستن پردازش"},
    "gpu_force_btn": {"en": "☠ Force Kill", "fa": "☠ بستن اجباری"},
    "gpu_col_pid": {"en": "PID", "fa": "PID"},
    "gpu_col_process": {"en": "Process", "fa": "پردازش"},
    "gpu_col_vram": {"en": "VRAM", "fa": "VRAM"},
    "gpu_col_gpu": {"en": "GPU", "fa": "GPU"},
    "gpu_col_cpu": {"en": "CPU", "fa": "پردازنده"},
    "gpu_col_ram": {"en": "RAM", "fa": "حافظه"},
    "gpu_col_disk": {"en": "DISK", "fa": "دیسک"},
    "gpu_col_status": {"en": "Status", "fa": "وضعیت"},
    "gpu_no_proc": {"en": "Select a process first.", "fa": "اول یک پردازش را انتخاب کنید."},
    "gpu_confirm_end": {"en": "Really close these processes?\n\n", "fa": "آیا واقعاً این پردازش‌ها بسته شوند؟\n\n"},
    "tab_monitoring": {"en": "  Live Monitoring  ", "fa": "  مانیتورینگ زنده  "},
    "tab_scanner": {"en": "  Model Scanner  ", "fa": "  اسکنر مدل‌ها  "},

    # Monitoring tab translations
    "monitor_title": {"en": "Live System Monitoring", "fa": "مانیتورینگ زنده سیستم"},
    "monitor_desc": {"en": "Real-time CPU, RAM, GPU, VRAM, Disk & Network usage",
                     "fa": "مصرف لحظه‌ای CPU، RAM، GPU، VRAM، دیسک و شبکه"},
    "monitor_cpu": {"en": "CPU", "fa": "پردازنده"},
    "monitor_ram": {"en": "RAM", "fa": "حافظه"},
    "monitor_gpu": {"en": "GPU", "fa": "کارت گرافیک"},
    "monitor_vram": {"en": "VRAM", "fa": "حافظه گرافیک"},
    "monitor_disk": {"en": "Disk I/O", "fa": "دیسک I/O"},
    "monitor_net": {"en": "Network I/O", "fa": "شبکه I/O"},
    "monitor_read": {"en": "Read", "fa": "خواندن"},
    "monitor_write": {"en": "Write", "fa": "نوشتن"},
    "monitor_recv": {"en": "Recv", "fa": "دریافت"},
    "monitor_sent": {"en": "Sent", "fa": "ارسال"},
    "monitor_auto_refresh": {"en": "Auto-refresh", "fa": "تازه‌سازی خودکار"},
    "monitor_interval": {"en": "Interval:", "fa": "فاصله:"},
    "monitor_refresh_btn": {"en": "Refresh Now", "fa": "تازه‌سازی الان"},
    "monitor_interval_vals": {"en": "0.5s,1s,2s,5s", "fa": "۰.۵ث,۱ث,۲ث,۵ث"},

    # Scanner tab translations
    "scan_title": {"en": "GGUF Model Scanner", "fa": "اسکنر مدل‌های GGUF"},
    "scan_desc": {"en": "Scan a folder for .gguf models and load them directly",
                  "fa": "یک پوشه را برای مدل‌های .gguf اسکن کنید و مستقیماً بارگذاری کنید"},
    "scan_folder": {"en": "Folder to scan:", "fa": "پوشه برای اسکن:"},
    "scan_browse": {"en": "Browse", "fa": "مرور"},
    "scan_recursive": {"en": "Recursive", "fa": "بازگشتی"},
    "scan_btn": {"en": "🔍 Scan", "fa": "🔍 اسکن"},
    "scan_columns": {"en": "Name,Size,Quant,Arch,Ctx,Modified,Path",
                     "fa": "نام,اندازه,کوانت,معماری,کانتکست,تغییر,مسیر"},
    "load_selected_model": {"en": "Load Selected", "fa": "بارگذاری انتخاب"},
    "copy_model_path": {"en": "Copy Path", "fa": "کپی مسیر"},
    "invalid_folder": {"en": "Please select a valid folder", "fa": "لطفاً یک پوشه معتبر انتخاب کنید"},
    "error": {"en": "Error", "fa": "خطا"},

    "files_section": {"en": "🗂 Files", "fa": "🗂 فایل‌ها"},
    "model_label": {"en": "Model (.gguf):", "fa": "مدل (.gguf):"},
    "model_tip": {"en": "The GGUF model file llama-server should load.",
                  "fa": "فایل مدل GGUF که llama-server باید بارگذاری کند."},
    "server_label": {"en": "Server exe:", "fa": "فایل سرور:"},
    "server_tip": {"en": "Path to llama-server.exe (or llama-server on Linux/Mac).",
                   "fa": "مسیر فایل llama-server.exe (یا llama-server در لینوکس/مک)."},
    "analyze_btn": {"en": "🔍 Recommend Settings For This Model",
                     "fa": "🔍 پیشنهاد تنظیمات برای این مدل"},

    "params_section": {"en": "🎛 Parameters", "fa": "🎛 پارامترها"},
    "preset_label": {"en": "Preset:", "fa": "پیش‌تنظیم:"},
    "preset_tip": {"en": "Ready-made setting bundles for common hardware.",
                   "fa": "مجموعه تنظیمات آماده برای سخت‌افزارهای رایج."},
    "apply_preset_btn": {"en": "Apply Preset", "fa": "اعمال پیش‌تنظیم"},

    "gpu_layers_label": {"en": "GPU Layers:", "fa": "لایه‌های GPU:"},
    "gpu_layers_tip": {
        "en": "How many model layers to offload to the GPU (-ngl). Higher = faster but uses "
              "more VRAM. Set to 0 for CPU-only, or a high number (e.g. 99) to try offloading "
              "every layer.",
        "fa": "چند لایه از مدل به GPU منتقل شود (-ngl). عدد بیشتر = سریع‌تر ولی حافظه گرافیک "
              "(VRAM) بیشتری مصرف می‌کند. برای اجرا فقط با CPU مقدار 0، و برای انتقال همه لایه‌ها "
              "عددی بزرگ مثل 99 را انتخاب کنید.",
    },
    "slots_label": {"en": "Slots:", "fa": "اسلات‌ها:"},
    "slots_tip": {
        "en": "Number of parallel request slots (-np). Increase only if you need to serve "
              "multiple concurrent requests.",
        "fa": "تعداد اسلات‌های همزمان برای درخواست‌ها (-np). فقط زمانی افزایش دهید که نیاز به "
              "پاسخ‌گویی همزمان چند درخواست دارید.",
    },
    "context_label": {"en": "Context:", "fa": "کانتکست:"},
    "context_tip": {
        "en": "Max context window size in tokens (-c). Larger context uses more RAM/VRAM for "
              "the KV cache. Use the quick buttons for common sizes.",
        "fa": "حداکثر اندازه پنجره کانتکست بر حسب توکن (-c). کانتکست بزرگ‌تر، حافظه RAM/VRAM "
              "بیشتری برای KV cache مصرف می‌کند. از دکمه‌های سریع برای مقادیر رایج استفاده کنید.",
    },
    "threads_label": {"en": "Threads:", "fa": "ترد (Thread):"},
    "threads_tip": {
        "en": "CPU threads used for generation (-t). Physical core count (4 on an i3-12th gen) "
              "is usually fastest per-token; logical/hyperthread count (8) pushes more total "
              "CPU usage, which helps when layers spill to CPU.",
        "fa": "تعداد تردهای CPU برای تولید متن (-t). تعداد هسته فیزیکی (۴ در i3 نسل ۱۲) معمولاً "
              "برای هر توکن سریع‌تر است؛ تعداد ترد منطقی/هایپرترد (۸) مصرف کل CPU را بیشتر می‌کند "
              "که وقتی بخشی از لایه‌ها روی CPU اجرا می‌شوند کمک‌کننده است.",
    },
    "tbatch_label": {"en": "Batch Threads:", "fa": "ترد پردازش دسته‌ای:"},
    "tbatch_tip": {
        "en": "CPU threads used for prompt batch processing (-tb). Safe to max out at your "
              "logical thread count (8 on an i3-12th gen).",
        "fa": "تعداد تردهای CPU برای پردازش دسته‌ای پرامپت (-tb). می‌توانید آن را تا حداکثر تعداد "
              "ترد منطقی سیستم (۸ در i3 نسل ۱۲) افزایش دهید.",
    },
    "batch_label": {"en": "Batch:", "fa": "دسته (Batch):"},
    "batch_tip": {
        "en": "Prompt batch size in tokens (-b). Bigger batches push more work through the GPU "
              "at once (higher throughput) but use more VRAM/RAM.",
        "fa": "اندازه دسته پرامپت بر حسب توکن (-b). دسته بزرگ‌تر باعث می‌شود کار بیشتری همزمان "
              "روی GPU پردازش شود (توان عملیاتی بیشتر) اما VRAM/RAM بیشتری مصرف می‌کند.",
    },
    "ubatch_label": {"en": "Physical Batch (ubatch):", "fa": "دسته فیزیکی (ubatch):"},
    "ubatch_tip": {
        "en": "Micro-batch size actually processed per GPU step (-ub / --ubatch-size). This is "
              "the real optimization lever: llama.cpp splits the logical batch (-b) into chunks "
              "of this size. Setting it equal to -b (e.g. both 2048-4096) maximizes GPU "
              "throughput on prompt processing; smaller values use less VRAM but process "
              "prompts slower.",
        "fa": "اندازه دسته کوچک (micro-batch) که واقعاً در هر مرحله روی GPU پردازش می‌شود "
              "(-ub / --ubatch-size). این پارامتر واقعی برای بهینه‌سازی است: llama.cpp دسته "
              "منطقی (-b) را به قطعاتی به این اندازه تقسیم می‌کند. برابر قرار دادن آن با -b "
              "(مثلاً هر دو ۲۰۴۸ تا ۴۰۹۶) بیشترین توان پردازش پرامپت روی GPU را می‌دهد؛ مقادیر "
              "کوچک‌تر VRAM کمتری مصرف می‌کنند اما پردازش پرامپت را کندتر می‌کنند.",
    },
    "defrag_label": {"en": "KV Defrag Threshold:", "fa": "آستانه Defrag کش KV:"},
    "defrag_tip": {
        "en": "--defrag-thold. Fraction of KV cache fragmentation (0.0-1.0) that triggers an "
              "automatic cache defragmentation pass. Keeps long-running servers with many "
              "requests from slowly losing speed as the KV cache fragments. 0.1 is a good "
              "default; set to -1 to disable.",
        "fa": "--defrag-thold. درصد قطعه‌قطعه‌شدن (fragmentation) کش KV (بین 0.0 تا 1.0) که "
              "باعث اجرای خودکار یکپارچه‌سازی کش می‌شود. از افت تدریجی سرعت در سرورهایی که "
              "مدت طولانی و با درخواست‌های زیاد کار می‌کنند جلوگیری می‌کند. مقدار پیش‌فرض خوب "
              "0.1 است؛ برای غیرفعال‌کردن، -1 بگذارید.",
    },
    "port_label": {"en": "Port:", "fa": "پورت:"},
    "port_tip": {"en": "Local port the HTTP server will listen on.",
                 "fa": "پورت محلی که سرور HTTP روی آن گوش می‌دهد."},

    "flash_cb": {"en": "Flash Attention", "fa": "Flash Attention"},
    "flash_tip": {
        "en": "Faster, more memory-efficient attention kernel (-fa on). Recommended on "
              "virtually every modern GPU, and required for KV cache quantization to actually "
              "be fast instead of slower.",
        "fa": "کرنل توجه (attention) سریع‌تر و کم‌مصرف‌تر از نظر حافظه (-fa on). روی تقریباً هر "
              "GPU مدرنی توصیه می‌شود، و برای اینکه کوانتیزه‌کردن کش KV واقعاً سریع باشد (نه کندتر) "
              "لازم است.",
    },
    "mlock_cb": {"en": "mlock", "fa": "قفل حافظه (mlock)"},
    "mlock_tip": {
        "en": "Locks the model in RAM so the OS can't swap it out. Needs enough free RAM to "
              "hold the whole model.",
        "fa": "مدل را در RAM قفل می‌کند تا سیستم‌عامل آن را به فایل صفحه‌بندی (swap) منتقل نکند. "
              "به RAM آزاد کافی برای نگه‌داشتن کل مدل نیاز دارد.",
    },
    "unified_cb": {"en": "Unified KV", "fa": "کش KV یکپارچه"},
    "unified_tip": {
        "en": "Share one KV cache across all slots instead of one per slot (--kv-unified).",
        "fa": "به‌جای یک کش KV جداگانه برای هر اسلات، یک کش مشترک بین همه اسلات‌ها استفاده "
              "می‌شود (--kv-unified).",
    },
    "kvquant_tip": {
        "en": "Enables the K/V cache type selectors (-ctk / -ctv). Roughly halves the VRAM used by "
              "the context/KV cache with near-zero quality loss, letting you run a bigger "
              "context or fit more GPU layers in the same VRAM. Needs Flash Attention on to "
              "actually be fast (auto-enabled if you check this).",
        "fa": "انتخاب‌گرهای نوع کش K/V را فعال می‌کند (-ctk / -ctv). مصرف VRAM کش "
              "KV/کانتکست را تقریباً نصف می‌کند با افت کیفیت تقریباً صفر، و اجازه می‌دهد کانتکست "
              "بزرگ‌تر یا لایه‌های بیشتری در همان VRAM جا شود. برای سریع‌بودن واقعی به روشن‌بودن "
              "Flash Attention نیاز دارد (با فعال‌کردن این گزینه خودکار روشن می‌شود).",
    },
    "nommap_cb": {"en": "Disable mmap (load fully into RAM)", "fa": "غیرفعال‌کردن mmap (بارگذاری کامل در RAM)"},
    "nommap_tip": {
        "en": "--no-mmap. Normally llama.cpp memory-maps the model file and pages it in as "
              "needed, which makes startup fast but can cause first-token stutter while pages "
              "load. Disabling mmap forces the whole file to be read into RAM up front - slower "
              "to start, but avoids page-fault stalls during generation once loaded. Best "
              "combined with mlock.",
        "fa": "--no-mmap. معمولاً llama.cpp فایل مدل را با mmap نگاشت می‌کند و صفحات آن را در "
              "صورت نیاز بارگذاری می‌کند؛ این کار شروع را سریع می‌کند اما ممکن است در اولین "
              "توکن‌ها با تأخیر همراه باشد. غیرفعال‌کردن mmap کل فایل را از ابتدا در RAM می‌خواند - "
              "شروع کندتر است، اما بعد از بارگذاری، توقف‌های ناشی از page-fault در حین تولید متن "
              "را از بین می‌برد. بهتر است همراه با mlock استفاده شود.",
    },
    "nowarmup_cb": {"en": "Skip Warmup (faster start)", "fa": "رد کردن Warmup (شروع سریع‌تر)"},
    "nowarmup_tip": {
        "en": "--no-warmup. Skips the dummy warmup inference llama-server normally runs right "
              "after loading. Shaves a few seconds off startup time, at the cost of the very "
              "first real request being a bit slower (it does the warmup work instead).",
        "fa": "--no-warmup. اجرای آزمایشی (warmup) که llama-server معمولاً بلافاصله بعد از "
              "بارگذاری انجام می‌دهد را رد می‌کند. چند ثانیه از زمان شروع کم می‌کند، اما اولین "
              "درخواست واقعی کمی کندتر خواهد بود (چون به‌جای warmup آن کار انجام می‌شود).",
    },

    # --- Advanced Performance ---
    "advanced_perf": {"en": "🚀 Advanced Performance", "fa": "🚀 عملکرد پیشرفته"},
    "advanced_desc": {
        "en": "Extra llama-server flags for batching, MoE CPU offload, idle power-saving, "
              "multi-GPU device selection, and per-tensor buffer overrides. Leave everything "
              "off/empty for llama-server's own defaults.",
        "fa": "پرچم‌های اضافه llama-server برای بچینگ، انتقال لایه‌های MoE به CPU، صرفه‌جویی انرژی "
              "در بیکاری، انتخاب چند کارت گرافیک و تغییر بافر تک‌تک تنسورها. برای استفاده از "
              "پیش‌فرض‌های خود سرور، همه را خاموش/خالی بگذارید.",
    },
    "cont_batching_cb": {"en": "Continuous Batching", "fa": "بچینگ پیوسته"},
    "cont_batching_tip": {
        "en": "Enable continuous batching (--cont-batching). Allows new requests to be batched "
              "with ongoing ones, significantly improving throughput for concurrent requests.",
        "fa": "بچینگ پیوسته را فعال می‌کند (--cont-batching). اجازه می‌دهد درخواست‌های جدید با "
              "درخواست‌های در حال اجرا بچ شوند، توان عملیاتی را برای درخواست‌های همزمان به شدت بالا می‌برد.",
    },
    "metrics_endpoint_cb": {"en": "Enable Metrics Endpoint", "fa": "نقطه‌ پایانی متریک‌ها"},
    "metrics_endpoint_tip": {
        "en": "Expose /metrics endpoint for Prometheus/Grafana monitoring (--metrics).",
        "fa": "نقطه‌پایانی /metrics را برای مانیتورینگ Prometheus/Grafana فعال می‌کند (--metrics).",
    },
    "yarn_cb": {"en": "Extended Context (YaRN)", "fa": "کانتکست گسترش‌یافته (YaRN)"},
    "yarn_tip": {
        "en": "Enable YaRN scaling for extended context beyond native (--rope-scaling yarn). "
              "Allows using context larger than model was trained for with minimal perplexity increase.",
        "fa": "مقیاس‌دهی YaRN برای کانتکست فراتر از اصلی را فعال می‌کند (--rope-scaling yarn). "
              "اجازه می‌دهد از کانتکست بزرگ‌تر از آنکه مدل برایش آموزش دیده با افزایش کمینه_PERPLEXITY_ استفاده کنید.",
    },
    "moe_cpu_cb": {"en": "Offload MoE Layers to CPU", "fa": "انتقال لایه‌های MoE به CPU"},
    "moe_cpu_tip": {
        "en": "Offload Mixture-of-Experts layers to CPU (-ncmoe / --n-cpu-moe). Useful for large "
              "MoE models that don't fit entirely in VRAM. On Gemma4 with a 12GB card, values "
              "around 13-15 tend to work well.",
        "fa": "لایه‌های Mixture-of-Experts را به CPU منتقل می‌کند (-ncmoe / --n-cpu-moe). برای "
              "مدل‌های MoE بزرگ که کاملاً در VRAM جا نمی‌شوند مفید است. روی Gemma4 با کارت ۱۲گیگابایتی، "
              "مقادیر حدود ۱۳ تا ۱۵ معمولاً خوب کار می‌کنند.",
    },
    "moe_cpu_layers_label": {"en": "MoE layers on CPU:", "fa": "لایه‌های MoE روی CPU:"},
    "moe_cpu_layers_tip": {
        "en": "Number of MoE expert layers to keep on CPU (N in --n-cpu-moe N). Higher = more VRAM saved.",
        "fa": "تعداد لایه‌های کارشناس MoE که روی CPU بمانند (N در --n-cpu-moe N). بیشتر = ذخیره VRAM بیشتر.",
    },
    "cache_idle_cb": {"en": "Cache Idle Slots", "fa": "کش اسلات‌های بیکار"},
    "cache_idle_tip": {
        "en": "--cache-idle-slots. Idle slots are saved to the prompt cache so they don't need "
              "to be reprocessed when the user comes back.",
        "fa": "--cache-idle-slots. اسلات‌های بیکار در کش پرامپت ذخیره می‌شوند تا وقتی کاربر برمی‌گردد "
              "نیازی به پردازش دوباره نباشد.",
    },
    "sleep_idle_cb": {"en": "Sleep When Idle", "fa": "خواب در بیکاری"},
    "sleep_idle_tip": {
        "en": "--sleep-idle-seconds N. Puts the server to sleep after N seconds of inactivity "
              "(works in both single- and multi-model setups).",
        "fa": "--sleep-idle-seconds N. سرور را پس از N ثانیه بیکاری به حالت خواب می‌برد "
              "(هم در حالت تک‌مدل و هم چندمدله کار می‌کند).",
    },
    "sleep_idle_seconds_label": {"en": "Idle seconds:", "fa": "ثانیه بیکاری:"},
    "sleep_idle_seconds_tip": {
        "en": "Seconds of inactivity before the server sleeps (--sleep-idle-seconds N).",
        "fa": "ثانیه‌های بیکاری قبل از خوابیدن سرور (--sleep-idle-seconds N).",
    },
    "kvquant_cb": {"en": "Quantize KV Cache", "fa": "کوانتیزه‌کردن کش KV"},
    "ctk_label": {"en": "K:", "fa": "K:"},
    "ctv_label": {"en": "V:", "fa": "V:"},
    "ctk_tip": {
        "en": "KV cache data type for K (-ctk / --cache-type-k). Quantizing roughly halves KV VRAM "
              "use; needs Flash Attention to be fast.",
        "fa": "نوع داده کش KV برای K ‏(-ctk / --cache-type-k). کوانتیزه‌کردن مصرف VRAM کش KV را تقریباً "
              "نصف می‌کند؛ برای سریع‌بودن به Flash Attention نیاز دارد.",
    },
    "ctv_tip": {
        "en": "KV cache data type for V (-ctv / --cache-type-v). Quantizing roughly halves KV VRAM "
              "use; needs Flash Attention to be fast.",
        "fa": "نوع داده کش KV برای V ‏(-ctv / --cache-type-v). کوانتیزه‌کردن مصرف VRAM کش KV را تقریباً "
              "نصف می‌کند؛ برای سریع‌بودن به Flash Attention نیاز دارد.",
    },
    "load_mode_label": {"en": "Load Mode:", "fa": "حالت بارگذاری:"},
    "load_mode_tip": {
        "en": "-lm / --load-mode: how the model file is loaded. auto = mmap when supported "
              "(server default), mmap = memory-map, mlock = lock in RAM so the OS can't swap it, "
              "dio = DirectIO, none = plain load.",
        "fa": "‏-lm / --load-mode: نحوه بارگذاری فایل مدل. auto یعنی mmap در صورت پشتیبانی (پیش‌فرض سرور)، "
              "mmap یعنی نگاشت حافظه، mlock مدل را در RAM قفل می‌کند تا swap نشود، dio یعنی DirectIO، "
              "none یعنی بارگذاری ساده.",
    },
    "reasoning_format_label": {"en": "Reasoning format:", "fa": "فرمت استدلال:"},
    "reasoning_format_tip": {
        "en": "--reasoning-format (auto/none/deepseek). Controls how thinking tokens are parsed and "
              "displayed — it does NOT turn reasoning on/off (that's reasoning-budget / the buttons "
              "on the left).",
        "fa": "--reasoning-format ‏(auto/none/deepseek). نحوه تجزیه و نمایش توکن‌های تفکر را کنترل "
              "می‌کند - روشن/خاموش‌کردن استدلال با این نیست (آن کار reasoning-budget و دکمه‌های سمت "
              "چپ است).",
    },
    "devices_label": {"en": "Devices (-dev):", "fa": "دستگاه‌ها (-dev):"},
    "devices_tip": {
        "en": "Comma-separated list of devices to use for offloading, e.g. CUDA0,CUDA1. "
              "Use llama-server --list-devices to see available devices. Empty = default device.",
        "fa": "فهرست دستگاه‌ها برای انتقال، جدا شده با ویرگول، مثل CUDA0,CUDA1. برای دیدن دستگاه‌های "
              "موجود از llama-server --list-devices استفاده کنید. خالی = دستگاه پیش‌فرض.",
    },
    "ot_label": {"en": "Override Tensor (-ot):", "fa": "تغییر تنسور (-ot):"},
    "ot_tip": {
        "en": "Override tensor buffer types (-ot). Comma-separated <pattern>=<buffer> pairs; each "
              "is sent as its own -ot flag. Example: blk\\.(0|1)\\.ffn_.*=CPU",
        "fa": "تغییر نوع بافر تنسورها (-ot). جفت‌های <الگو>=<بافر> جدا شده با ویرگول؛ هر کدام به‌صورت "
              "یک پرچم -ot جدا ارسال می‌شود. مثال: blk\\.(0|1)\\.ffn_.*=CPU",
    },
    "mmproj_section": {"en": "👁 Multimodal (mmproj)", "fa": "👁 چندرسانه‌ای (mmproj)"},
    "mmproj_desc": {
        "en": "For vision models only: point llama-server at an mmproj projector file so it can "
              "understand images. Use either a local file OR a URL — never both (the local file "
              "wins if both are set). Leave empty for text-only models.",
        "fa": "فقط برای مدل‌های بینایی: فایل پروجکتور mmproj را به llama-server بدهید تا تصاویر را "
              "بفهمد. از فایل محلی یا آدرس اینترنتی استفاده کنید - هرگز هر دو را با هم نه (در صورت "
              "تنظیم هر دو، فایل محلی استفاده می‌شود). برای مدل‌های متنی خالی بگذارید.",
    },
    "mmproj_file_label": {"en": "mmproj file:", "fa": "فایل mmproj:"},
    "mmproj_file_tip": {
        "en": "Local .mmproj multimodal projector file (--mmproj). Needed for vision models.",
        "fa": "فایل پروجکتور چندرسانه‌ای محلی .mmproj ‏(--mmproj). برای مدل‌های بینایی لازم است.",
    },
    "mmproj_url_label": {"en": "mmproj URL:", "fa": "آدرس mmproj:"},
    "mmproj_url_tip": {
        "en": "Remote URL of a multimodal projector (-mmu / --mmproj-url). Only used when no local "
              "mmproj file is set (the two are never sent together).",
        "fa": "آدرس راه‌دور پروجکتور چندرسانه‌ای (-mmu / --mmproj-url). فقط وقتی استفاده می‌شود که فایل "
              "محلی mmproj تنظیم نشده باشد (هر دو هرگز با هم ارسال نمی‌شوند).",
    },
    "vision_check_btn": {"en": "🔍 Check Vision", "fa": "🔍 بررسی بینایی"},
    "vision_check_tip": {
        "en": "Inspect the selected model: does it support images? Reads the GGUF architecture, "
              "auto-detects a matching .mmproj next to the model, and clears up why a text-only "
              "model can't see images.",
        "fa": "مدل انتخاب‌شده را بررسی می‌کند: آیا تصویر را می‌فهمد؟ معماری GGUF را می‌خواند، فایل .mmproj "
              "هم‌پوشه را خودکار پیدا می‌کند و مشخص می‌کند چرا مدل متنی نمی‌تواند تصویر ببیند.",
    },
    "vision_yes": {"en": "👁 Vision model ({arch})", "fa": "👁 مدل بینایی ({arch})"},
    "vision_no": {"en": "✗ Text-only model — no vision capability", "fa": "✗ مدل متنی — قابلیت دیدن تصویر ندارد"},
    "vision_unknown": {"en": "? Couldn't read model metadata", "fa": "؟ خواندن متادیتای مدل ممکن نشد"},
    "vision_mmproj_auto": {"en": "auto mmproj: {name}", "fa": "mmproj خودکار: {name}"},
    "vision_no_mmproj": {"en": "no .mmproj in model folder", "fa": "فایل .mmproj در پوشه مدل نیست"},
    "vision_need_model": {
        "en": "Select a valid GGUF model in the Server Setup tab first — the vision check reads "
              "the model's own metadata.",
        "fa": "ابتدا در تب تنظیمات سرور یک مدل GGUF معتبر انتخاب کنید — بررسی بینایی، متادیتای خود مدل "
              "را می‌خواند.",
    },
    "vision_msg_text_title": {"en": "Not a vision model", "fa": "مدل بینایی نیست"},
    "vision_msg_text_body": {
        "en": "This model ({arch}) is text-only — it physically cannot understand images, so no "
              "mmproj will ever fix it.\n\nVision is a property of the MODEL, not of this app:\n"
              "• You need a multimodal GGUF model PLUS its matching .mmproj projector file.\n"
              "• Vision-ready llama.cpp models: Qwen2.5-VL, LLaVA 1.6, MiniCPM-V, InternVL, "
              "Moondream, Florence-2.\n"
              "• On Hugging Face search e.g. \"Qwen2.5-VL GGUF\": download the model .gguf AND the "
              ".mmproj attached to the same page, and put both in one folder.",
        "fa": "این مدل ({arch}) فقط متنی است — از نظر فنی امکان دیدن تصویر را ندارد، پس هیچ فایل mmproj "
              "هرگز آن را درست نمی‌کند.\n\nدید تصویری ویژگیِ خودِ مدل است، نه این برنامه:\n"
              "• به یک مدل GGUF چندرسانه‌ای به‌همراه فایل پروجکتور .mmproj مخصوص آن نیاز دارید.\n"
              "• مدل‌های آماده‌ی بینایی برای llama.cpp: Qwen2.5-VL، LLaVA 1.6، MiniCPM-V، InternVL، "
              "Moondream، Florence-2.\n"
              "• در Hugging Face مثلاً «Qwen2.5-VL GGUF» را جستجو کنید: هم فایل .gguf مدل و هم فایل .mmproj "
              "همان صفحه را دانلود و هر دو را در یک پوشه بگذارید.",
    },
    "vision_msg_vision_title": {"en": "Vision model detected", "fa": "مدل بینایی شناسایی شد"},
    "vision_msg_vision_ok": {
        "en": "✓ {arch} supports images.\nAuto-loaded mmproj: {name}\n\nThe server will now understand "
              "images from any OpenAI-compatible client (/v1/chat/completions with base64 images).",
        "fa": "✓ معماری {arch} از تصویر پشتیبانی می‌کند.\nmmproj به‌صورت خودکار بارگذاری شد: {name}\n\n"
              "سرور حالا تصاویر را می‌فهمد (هر کلاینتی سازگار با OpenAI مثل /v1/chat/completions با "
              "تصویر base64).",
    },
    "vision_msg_vision_noproj": {
        "en": "{arch} is a vision model, but no .mmproj projector was found next to it.\n\n"
              "Download the matching projector from the same Hugging Face page (it's usually named "
              "like the model, e.g. \"{stem}.mmproj\" or \"mmproj-f16.gguf\"), place it in the "
              "model's folder, and press Check Vision again.\n\n"
              "Without an mmproj, llama-server will start but cannot process images.",
        "fa": "{arch} یک مدل بینایی است، اما هیچ فایل پروجکتور .mmproj کنار آن پیدا نشد.\n\n"
              "پروجکتور مخصوص همان مدل را از همان صفحه Hugging Face دانلود کنید (معمولاً همنام مدل است، "
              "مثلاً «{stem}.mmproj» یا «mmproj-f16.gguf»)، آن را در پوشه مدل بگذارید و دوباره «بررسی "
              "بینایی» را بزنید.\n\nبدون فایل mmproj سرور شروع می‌شود ولی نمی‌تواند تصویر را پردازش کند.",
    },
    "vision_possible": {"en": "~ possibly multimodal ({arch})", "fa": "~ احتمالاً چندرسانه‌ای ({arch})"},
    "vision_msg_possible_title": {"en": "Looks multimodal", "fa": "به‌نظر چندرسانه‌ای می‌رسد"},
    "vision_msg_possible_body": {
        "en": "{arch} isn't on my known-architectures list either way, but a matching projector "
              "({name}) was found right next to it, which usually means this is a vision model.\n\n"
              "The mmproj is loaded, so llama-server will use it. If the server logs a CLIP/clip "
              "error or rejects images, the model is actually text-only.",
        "fa": "معماری {arch} نه در لیست شناخته‌شده من است و نه به‌عنوان متنی علامت خورده، اما یک پروجکتور "
              "همنام ({name}) دقیقاً کنار آن پیدا شد که معمولاً یعنی این یک مدل بینایی است.\n\n"
              "mmproj بارگذاری شده و سرور از آن استفاده می‌کند. اگر سرور خطای CLIP/clip داد یا تصویر را "
              "رد کرد، یعنی مدل در واقع فقط متنی است.",
    },
    "vision_msg_unknown_title": {"en": "Can't tell from metadata", "fa": "از روی متادیتا مشخص نیست"},
    "vision_msg_unknown_body": {
        "en": "Couldn't read this model's architecture (arch = \"{arch}\"), so I can't say whether it "
              "supports vision. Make sure it's a valid GGUF model; if it is a vision model, download "
              "its .mmproj and place it in the same folder.",
        "fa": "معماری این مدل خوانده نشد (arch = \"{arch}\")، پس نمی‌توانم بگویم آیا بینایی دارد یا نه. "
              "مطمئن شوید فایل GGUF معتبر است؛ اگر مدل بینایی است، فایل .mmproj آن را دانلود و در همان "
              "پوشه بگذارید.",
    },

    "reasoning_section": {"en": "🧠 Reasoning / Thinking Mode", "fa": "🧠 حالت استدلال / تفکر"},
    "reasoning_desc": {
        "en": "For reasoning models (Qwen3, DeepSeek-R1, QwQ...) — controls whether the model "
              "shows its chain-of-thought before answering.",
        "fa": "برای مدل‌های استدلال‌گر (Qwen3، DeepSeek-R1، QwQ و...) — کنترل می‌کند که آیا مدل "
              "قبل از پاسخ، زنجیره فکری (chain-of-thought) خود را نشان دهد یا نه.",
    },
    "think_on_btn": {"en": "🧠 Thinking ON", "fa": "🧠 تفکر روشن"},
    "think_on_tip": {
        "en": "Default. Lets reasoning-capable models think step-by-step before answering - "
              "usually better on math/logic/coding, but slower and uses more tokens per reply.",
        "fa": "حالت پیش‌فرض. به مدل‌های استدلال‌گر اجازه می‌دهد قبل از پاسخ، مرحله‌به‌مرحله فکر "
              "کنند - معمولاً در ریاضی/منطق/برنامه‌نویسی بهتر است، اما کندتر است و توکن بیشتری "
              "در هر پاسخ مصرف می‌کند.",
    },
    "think_off_btn": {"en": "🚫 No Thinking (faster)", "fa": "🚫 بدون تفکر (سریع‌تر)"},
    "think_off_tip": {
        "en": 'Adds --reasoning-budget 0, --chat-template-kwargs {"enable_thinking": false}, '
              "and --jinja. Skips the reasoning/chain-of-thought step for much faster replies. "
              "Only affects reasoning-capable models - does nothing on regular (non-reasoning) "
              "models.",
        "fa": 'پرچم‌های --reasoning-budget 0 و --chat-template-kwargs {"enable_thinking": false} '
              "و --jinja را اضافه می‌کند. مرحله استدلال/زنجیره فکری را رد می‌کند تا پاسخ‌ها خیلی "
              "سریع‌تر شوند. فقط روی مدل‌های استدلال‌گر تأثیر دارد - روی مدل‌های عادی (غیر استدلال‌گر) "
              "هیچ تأثیری ندارد.",
    },

    # --- Speculative Decoding (mirrors Unsloth Studio's hub panel) ---
    "spec_section": {"en": "⚡ Speculative Decoding", "fa": "⚡ رمزگشایی گمانه‌زن"},
    "spec_mode_label": {"en": "Spec Decoding:", "fa": "رمزگشایی گمانه‌زن:"},
    "spec_mode_tip": {
        "en": "Speculative decoding. Auto sends NO flag (speculation off — llama-server's own "
              "default; this measured fastest on CPU-limited rigs). Pick Ngram for chat/code "
              "with zero extra downloads, or MTP/DSpark/DFlash when your model ships a drafter "
              "sidecar. DSpark downloads a sidecar of about 11 GB and DFlash one of about 1.5 GB, "
              "both trading VRAM for speed; on quantized targets their greedy output can differ "
              "from a non speculative run. MTP and ngram do not change output.",
        "fa": "رمزگشایی گمانه‌زن. حالت Auto هیچ پرچمی نمی‌فرستد (گمانه‌زنی خاموش — پیش‌فرض خود "
              "سرور؛ این حالت روی سیستم‌های کم‌ترد CPU سریع‌ترین اندازه‌گیری شده است). برای چت/کد "
              "گزینه Ngram را بزنید (بدون دانلود اضافه)، یا اگر مدل شما درافر جانبی (sidecar) دارد "
              "MTP/DSpark/DFlash را انتخاب کنید. DSpark سایدکاری حدود ۱۱ گیگابایت و DFlash حدود "
              "۱.۵ گیگابایت دانلود می‌کند که هر دو VRAM را در ازای سرعت مصرف می‌کنند؛ روی مدل‌های "
              "کوانتیزه‌شده خروجی آن‌ها ممکن است با اجرای بدون گمانه‌زنی متفاوت باشد. MTP و ngram خروجی را تغییر نمی‌دهند.",
    },
    "spec_draft_tokens_label": {"en": "Draft Tokens:", "fa": "توکن‌های پیش‌نویس:"},
    "spec_draft_tokens_tip": {
        "en": "Max draft tokens per step (--spec-draft-n-max, 1-16). Leave at default: "
              "MTP and DFlash use 2 on GPU, 3 on CPU/Mac; DSpark uses 3.",
        "fa": "حداکثر توکن‌های پیش‌نویس در هر مرحله (--spec-draft-n-max، بین ۱ تا ۱۶). مقدار پیش‌فرض: "
              "MTP و DFlash روی GPU عدد ۲ و روی CPU/Mac عدد ۳؛ DSpark عدد ۳.",
    },
    "spec_draft_auto_cb": {"en": "Auto", "fa": "خودکار"},
    "spec_draft_auto_tip": {
        "en": "Checked (default): llama-server picks the draft count per device - "
              "MTP and DFlash draft 2 tokens on GPU, 3 on CPU/Mac; DSpark drafts 3. "
              "Uncheck to type your own Draft Tokens value.",
        "fa": "تیک‌دار (پیش‌فرض): llama-server تعداد پیش‌نویس را بر اساس سخت‌افزار انتخاب می‌کند - "
              "MTP و DFlash روی GPU عدد ۲ و روی CPU/Mac عدد ۳؛ DSpark عدد ۳. "
              "برای وارد‌کردن مقدار دلخواه، تیک را بردارید.",
    },
    "spec_cache_label": {"en": "Draft KV Type-K:", "fa": "نوع کش KV دراف:"},
    "spec_cache_tip": {
        "en": "Spec Decoding KV Cache Dtype (--spec-draft-type-k). Only used by the DSpark/DFlash "
              "strategies; f16 is the default.",
        "fa": "نوع داده کش KV برای رمزگشایی گمانه‌زن (--spec-draft-type-k). فقط در استراتژی‌های "
              "DSpark/DFlash استفاده می‌شود؛ مقدار پیش‌فرض f16 است.",
    },
    "spec_apply_btn": {"en": "✓ Apply", "fa": "✓ اعمال"},
    "spec_hint_auto": {"en": "no flag — speculation off", "fa": "بدون پرچم — گمانه‌زنی خاموش"},
    "spec_hint_off": {"en": "speculative decoding off", "fa": "رمزگشایی گمانه‌زن خاموش است"},
    "spec_hint_ngram": {"en": "ngram needs no draft count", "fa": "ngram به تعداد پیش‌نویس نیاز ندارد"},
    "spec_hint_def": {"en": "drafts: server default (2 on GPU)", "fa": "پیش‌نویس: پیش‌فرض سرور (۲ روی GPU)"},
    "spec_hint_manual": {"en": "drafts: your value", "fa": "پیش‌نویس: مقدار دلخواه شما"},

    "extra_args_label": {"en": "Extra Args:", "fa": "آرگومان‌های اضافه:"},
    "extra_args_tip": {
        "en": "Any additional llama-server command-line flags, space separated, appended as-is.",
        "fa": "هر پرچم اضافه‌ی دیگر برای خط فرمان llama-server، جدا شده با فاصله، همان‌طور که "
              "نوشته شده اضافه می‌شود.",
    },
    "preview_label": {"en": "Command preview:", "fa": "پیش‌نمایش دستور:"},

    "start_btn": {"en": "  START SERVER  ", "fa": "  شروع سرور  "},
    "stop_btn": {"en": "  STOP SERVER  ", "fa": "  توقف سرور  "},
    "copy_btn": {"en": "Copy Command", "fa": "کپی دستور"},
    "clear_btn": {"en": "Clear Log", "fa": "پاک‌کردن لاگ"},
    "copy_log_btn": {"en": "Copy Log", "fa": "کپی لاگ"},
    "jump_latest_btn": {"en": "Latest", "fa": "آخرین"},
    "filter_log_btn": {"en": "Filter", "fa": "فیلتر"},
    "filter_log_placeholder": {"en": "Search log...", "fa": "جستجوی لاگ..."},
    "open_browser_btn": {"en": "Open in Browser", "fa": "باز کردن در مرورگر"},
    "log_label": {"en": "Log:", "fa": "لاگ:"},
    "auto_scroll": {"en": "Auto-scroll", "fa": "اسکرول خودکار"},

    "status_ready": {"en": "Ready", "fa": "آماده"},
    "load_status_idle": {"en": "Model not loaded", "fa": "مدلی بارگذاری نشده"},

    # --- System tab ---
    "sys_title": {"en": "Detected Hardware", "fa": "سخت‌افزار شناسایی‌شده"},
    "sys_desc": {
        "en": "This mirrors the kind of hardware check tools like LM Studio show before you "
              "run a model - it estimates whether your model will fit in VRAM/RAM.",
        "fa": "این بخش شبیه بررسی سخت‌افزاری است که ابزارهایی مثل LM Studio قبل از اجرای مدل "
              "نشان می‌دهند - تخمین می‌زند که آیا مدل شما در VRAM/RAM جا می‌شود یا نه.",
    },
    "refresh_btn": {"en": "Refresh", "fa": "تازه‌سازی"},
    "cpu_section": {"en": "⚙️ CPU", "fa": "⚙️ پردازنده (CPU)"},
    "ram_section": {"en": "🧠 RAM", "fa": "🧠 حافظه (RAM)"},
    "gpu_section": {"en": "🎮 GPU", "fa": "🎮 کارت گرافیک (GPU)"},
    "model_fit_section": {"en": "🎯 Selected Model Fit", "fa": "🎯 سازگاری مدل انتخاب‌شده"},
    "model_fit_default": {
        "en": "Select a model on the Server Setup tab to see a compatibility estimate here.",
        "fa": "یک مدل را در تب «تنظیمات سرور» انتخاب کنید تا تخمین سازگاری اینجا نمایش داده شود.",
    },
    "notes_section": {"en": "📝 Notes", "fa": "📝 نکات"},
    "notes_text": {
        "en": "• VRAM detection is currently accurate for NVIDIA GPUs (via nvidia-smi).\n"
              "• AMD/Intel GPUs are listed by name only; VRAM-based recommendations won't be exact.\n"
              "• 'Fit' estimates are approximate - actual usage also depends on context length, "
              "quantization, and KV cache settings.",
        "fa": "• تشخیص VRAM در حال حاضر فقط برای کارت‌های NVIDIA دقیق است (از طریق nvidia-smi).\n"
              "• کارت‌های AMD/Intel فقط با نام نمایش داده می‌شوند؛ پیشنهادهای مبتنی بر VRAM دقیق نخواهند بود.\n"
              "• تخمین «سازگاری» تقریبی است - مصرف واقعی به طول کانتکست، نوع کوانتیزاسیون و "
              "تنظیمات کش KV هم بستگی دارد.",
    },
    "psutil_tip": {
        "en": "\nTip: install 'psutil' (pip install psutil) for more accurate RAM/CPU detection.",
        "fa": "\nنکته: برای تشخیص دقیق‌تر RAM/CPU، پکیج «psutil» را نصب کنید (pip install psutil).",
    },
    "loading_model": {"en": "Loading model...", "fa": "در حال بارگذاری مدل..."},
    "model_loaded_prefix": {"en": "✓ Model loaded in", "fa": "✓ مدل بارگذاری شد در"},
    "model_loaded_suffix": {"en": "s — ready for requests", "fa": " ثانیه — آماده دریافت درخواست"},
    "speed_prefix": {"en": "Speed:", "fa": "سرعت:"},
    "speed_avg_label": {"en": "avg", "fa": "میانگین"},
    "speed_gen": {"en": "gen", "fa": "تولید"},
    "speed_prm": {"en": "prompt", "fa": "پرامپت"},
    "speed_tip": {
        "en": "Live token stats from the server log: current decode speed (tok/s), "
              "rolling average over the last 5 measurements, and session totals for "
              "generated and prompt tokens. Prompt processing speed shows here before "
              "the first token is generated.",
        "fa": "آمار زنده توکن از لاگ سرور: سرعت لحظه‌ای تولید (tok/s)، میانگین متحرک ۵ "
              "اندازه‌گیری آخر، و مجموع کل توکن‌های تولیدشده و پرامپت در این نشست. تا قبل از "
              "اولین توکن خروجی، سرعت پردازش پرامپت اینجا نمایش داده می‌شود.",
    },
    "uptime_prefix": {"en": "Uptime:", "fa": "زمان اجرا:"},
    "checking_hardware": {"en": "? Checking hardware...", "fa": "؟ در حال بررسی سخت‌افزار..."},
    "browse_btn": {"en": "Browse", "fa": "انتخاب فایل"},

    # Scanner tab translations
    "tab_scanner": {"en": "  Model Scanner  ", "fa": "  اسکنر مدل‌ها  "},
    "scan_folder": {"en": "Folder:", "fa": "پوشه:"},
    "browse_btn": {"en": "Browse", "fa": "مرور"},
    "scan_models_btn": {"en": "Scan", "fa": "اسکن"},
    "recursive_scan": {"en": "Recursive", "fa": "بازگشتی"},
    "skip_mmproj": {"en": "Skip .mmproj", "fa": "رد .mmproj"},
    "skip_mmproj_tip": {"en": "Skip .mmproj files (multimodal projectors) - they don't have context/quantization info.",
                        "fa": "فایل‌های .mmproj را رد کنید (پروجکتورهای چندرسانه‌ای) - آن‌ها اطلاعات کانتکست/کوانتیزاسیون ندارند."},
    "show_full_path": {"en": "Show full path", "fa": "نمایش مسیر کامل"},
    "scan_ready": {"en": "Ready to scan", "fa": "آماده برای اسکن"},
    "scan_progress": {"en": "Scanning... {current}/{total}", "fa": "در حال اسکن... {current}/{total}"},
    "scan_complete": {"en": "Scan complete: {count} models found", "fa": "اسکن کامل: {count} مدل یافت شد"},
    "discovered_models": {"en": "Discovered Models", "fa": "مدل‌های کشف‌شده"},
    "model_name": {"en": "Name", "fa": "نام"},
    "model_size": {"en": "Size", "fa": "اندازه"},
    "model_quant": {"en": "Quant", "fa": "کوانت"},
    "model_arch": {"en": "Arch", "fa": "معماری"},
    "model_ctx": {"en": "Ctx", "fa": "کانتکست"},
    "model_modified": {"en": "Modified", "fa": "تغییر"},
    "model_path": {"en": "Path", "fa": "مسیر"},
    "load_selected_model": {"en": "Load Selected", "fa": "بارگذاری انتخاب"},
    "copy_model_path": {"en": "Copy Path", "fa": "کپی مسیر"},
    "invalid_folder": {"en": "Please select a valid folder", "fa": "لطفاً یک پوشه معتبر انتخاب کنید"},
    "error": {"en": "Error", "fa": "خطا"},
}


class LlamaRunner:
    def __init__(self, root):
        self.root = root
        self.lang = "en"
        self._i18n_registry = []  # list of (widget_or_tooltip, key, kind)
        self.root.title("Llama.cpp Runner")
        self.root.geometry("1180x900")
        self.root.minsize(1000, 700)
        self.process = None
        self.load_start_time = None
        self.model_ready = False
        self.speed_samples = deque(maxlen=5)
        # Session token accounting, fed by the server log's eval-time lines.
        self.token_stats = {"gen": 0, "prompt": 0, "prompt_tps": None}
        self._gguf_cache_path = None
        self._gguf_cache_info = None

        # GPU Processes tab state
        self._gpu_mon_active = True
        self._gpu_refresh_enabled = True
        self._gpu_filter_mode = "all"
        self._gpu_sort_column = "vram"
        self._gpu_sort_reverse = True
        self._gpu_rows = []
        self._gpu_vram_hist = deque(maxlen=60)
        self._gpu_load_hist = deque(maxlen=60)
        self._gpu_power_hist = deque(maxlen=60)
        self._cpu_hist = deque(maxlen=60)
        self._ram_hist = deque(maxlen=60)
        self._disk_hist = deque(maxlen=60)

        self.root.configure(bg=S("bg"))
        self._setup_style()

        # Resolve the Persian/display family once Tk is up so every
        # font helper can reuse it without re-querying tkfont.families().
        self.fa_family = _resolve_fa_family()

        # Cached hardware info, refreshed on demand
        self.cpu_info = {}
        self.ram_info = {}
        self.gpu_info = []

        # ---- Main layout: horizontal paned window (left = controls, right = log) ----
        self.main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # LEFT PANE: notebook with Server Setup / System tabs
        left_pane = ttk.Frame(self.main_paned)
        self.main_paned.add(left_pane, weight=1)

        # RIGHT PANE: Log (always visible, larger)
        right_pane = ttk.Frame(self.main_paned)
        self.main_paned.add(right_pane, weight=1)  # equal weight, user can drag divider

        self._build_left_notebook(left_pane)
        self._build_log_panel(right_pane)
        # Give the panes a usable default split once geometry is known,
        # so neither side ever starts collapsed.
        self.root.after(80, self._init_sash_position)

        # Auto-detect hardware once at startup so the compatibility badge
        # is populated immediately.
        self.root.after(200, self.refresh_hardware)

        # Safety net: if the app exits in any way other than the normal
        # close button (crash, killed from taskbar, etc.) still try to
        # take the llama-server process down with it.
        atexit.register(self._kill_process_tree)

    # ------------------------------------------------------------------
    # Bilingual (EN/FA) support
    # ------------------------------------------------------------------
    def t(self, key):
        return TR.get(key, {}).get(self.lang, key)

    def _reg(self, widget_or_tip, key, kind):
        """Registers a widget (or Tooltip) so its text updates automatically
        when the language is toggled. kind controls how it's applied:
        'label'/'button'/'checkbutton' -> widget.config(text=...)
        'label_prefixed' -> widget.config(text='  ' + translation)
        'labelframe' -> widget.config(text=...) (ttk.LabelFrame)
        'tooltip' -> widget_or_tip.set_text(...)
        'tab' -> notebook.tab(tab, text=...)
        Anything registered also gets its font swapped to the Persian face
        (Lalezar) when the rendered text contains Persian glyphs, and back to
        the latin face otherwise — this is what makes the whole UI render in
        Lalezar in Persian mode without touching every widget by hand."""
        self._i18n_registry.append((widget_or_tip, key, kind))

    # --- Font helpers -------------------------------------------------
    # font_for(text, size, weight) picks the family from the rendered text,
    # so Persian strings always render in Lalezar and latin strings stay on
    # the readable sans-serif face. fa_font/en_font return fixed tuples for
    # places (badges, buttons) where we want one face unconditionally.
    def font_for(self, text, size=10, weight="normal"):
        if _looks_persian(text):
            # Persian glyphs read ~1pt smaller than Latin at the same point
            # size — bump so both scripts *feel* the same size.
            return (self.fa_family, size + 1, weight)
        return (_LATIN_FONT_FAMILY, size, weight)

    def fa_font(self, size=10, weight="normal"):
        return (self.fa_family, size + 1, weight)

    def en_font(self, size=10, weight="normal"):
        return (_LATIN_FONT_FAMILY, size, weight)

    def _apply_text(self, widget, text, kind):
        """Set a widget's translated text AND fit its font to the script.
        Centralizes the ‘text + matching face’ rule so the language toggle
        and initial build both go through one path."""
        try:
            if kind == "label_prefixed":
                widget.config(text="  " + text)
            elif kind == "tooltip":
                widget.set_text(text)
            else:
                widget.config(text=text)
            # Tooltips manage their own font at show-time; only configure
            # the face on real widgets that expose a `font` option.
            if kind != "tooltip":
                try:
                    current = widget.cget("font")
                    weight = "bold"
                    if isinstance(current, str):
                        weight = "normal"
                    elif isinstance(current, tuple) and len(current) >= 3 and current[2]:
                        weight = current[2]
                    widget.config(font=self.font_for(text,
                                                     size=self._font_size_for(widget, current),
                                                     weight=weight if isinstance(weight, str) and weight else "normal"))
                except (tk.TclError, AttributeError):
                    pass
        except tk.TclError:
            pass

    def _font_size_for(self, widget, current):
        """Keep the widget's existing point size when we swap font families,
        so e.g. the 18pt title and 9pt captions stay their relative sizes."""
        try:
            if isinstance(current, tuple) and len(current) >= 2:
                return current[1]
            f = widget.cget("font")
            if isinstance(f, tuple) and len(f) >= 2:
                return f[1]
        except Exception:
            pass
        return 10

    def toggle_language(self):
        self.lang = "fa" if self.lang == "en" else "en"
        self.refresh_language()

    def refresh_language(self):
        for widget, key, kind in self._i18n_registry:
            text = self.t(key)
            self._apply_text(widget, text, kind)
        for tab, key in getattr(self, "_i18n_tabs", []):
            tab_text = self.t(key)
            self.notebook.tab(tab, text=tab_text)
        self.lang_btn.config(text=self.t("lang_toggle"),
                             font=self.font_for(self.t("lang_toggle"), 10, "bold"))
        # A few header labels and badges set their face explicitly (bigger
        # sizes) — keep them in sync with the active language too.
        self._restyle_header()
        # Fit badge / model detail text depend on live state, not just a
        # static key, so recompute them instead of blindly re-translating.
        self.update_compatibility()
        if getattr(self, "spec_hint_lbl", None):
            self._update_spec_controls()
        if not self.process:
            self.load_status_var.set(self.t("load_status_idle"))
            self.load_status_label.config(font=self.fa_font(10, "bold") if self.lang == "fa" else self.en_font(10, "bold"))
            self.status_var.set(self.t("status_ready"))
        else:
            self.load_status_label.config(font=self.fa_font(10, "bold") if self.lang == "fa" else self.en_font(10, "bold"))

    def _restyle_header(self):
        """Re-face the title/subtitle/lang button to the active language.
        Sizes are preserved per-widget so the 18pt title stays 18pt."""
        try:
            self.title_label.config(font=self.font_for(self.t("app_title"), 18, "bold"))
            self.subtitle_label.config(font=self.font_for(self.t("app_subtitle"), 9))
            self.lang_btn.config(font=self.font_for(self.t("lang_toggle"), 10, "bold"))
        except (tk.TclError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Small UX helpers: pane sizing, wheel scrolling, hover, thresholds
    # ------------------------------------------------------------------
    def _init_sash_position(self):
        """Split the paned window ~54/46 once the real width is known so
        both panes start readable and the divider is never at zero."""
        try:
            w = self.main_paned.winfo_width()
            if w > 1:
                self.main_paned.sashpos(0, max(430, int(w * 0.54)))
        except (tk.TclError, Exception):
            pass

    def _bind_mousewheel_recursive(self, widget):
        """Bind the mouse wheel over every Server Setup child so scrolling
        works anywhere over the tab. Widgets that consume the wheel
        themselves are skipped so their native behavior stays intact."""
        if isinstance(widget, (ttk.Combobox, ttk.Spinbox, ttk.Scrollbar,
                               ttk.Treeview, scrolledtext.ScrolledText)):
            return
        widget.bind("<MouseWheel>", self._on_server_wheel, add="+")
        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child)

    def _on_server_wheel(self, event):
        canvas = getattr(self, "_server_canvas", None)
        if canvas is not None and canvas.winfo_ismapped():
            step = -2 if event.delta > 0 else 2
            canvas.yview_scroll(step, "units")

    def _on_server_canvas_resize(self, event):
        """Keep the scrolled form as wide as the visible area so widgets
        stretch instead of clipping horizontally."""
        self._server_canvas.itemconfigure(self._server_win, width=event.width)

    def _attach_hover(self, widget, normal_bg, hover_bg):
        """Give a plain tk.Button the same hover feedback the ttk styles
        provide, using only existing palette colors."""
        widget.bind("<Enter>", lambda e: widget.config(bg=hover_bg))
        widget.bind("<Leave>", lambda e: widget.config(bg=normal_bg))

    def _progress_style_name(self, pct):
        """Shared color-threshold rule for every usage meter in the app."""
        if pct >= 90:
            return "Danger.Horizontal.TProgressbar"
        if pct >= 70:
            return "Warn.Horizontal.TProgressbar"
        return "Ok.Horizontal.TProgressbar"

    def _set_progress(self, bar, pct):
        """Clamp + set a progress bar's value and apply the matching
        threshold style (green under 70%, amber 70-90%, red above 90%)."""
        pct = max(0.0, min(100.0, float(pct or 0)))
        try:
            bar["value"] = pct
            bar.config(style=self._progress_style_name(pct))
        except (tk.TclError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _setup_style(self):
        bg, surf, surf_hi = S("bg"), S("surface"), S("surface_hi")
        fg, muted = S("fg"), S("fg_muted")
        accent, ok, danger, info = S("accent"), S("ok"), S("danger"), S("info")

        style = ttk.Style()
        style.theme_use("clam")

        L = (_LATIN_FONT_FAMILY, 10)          # body latin face
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=L)
        style.configure("TButton", background=surf, foreground=fg,
                        font=(_LATIN_FONT_FAMILY, 10, "bold"), padding=7,
                        borderwidth=0, relief=tk.FLAT)
        style.map("TButton",
                  background=[("pressed", S("border")), ("active", surf_hi)],
                  bordercolor=[("active", surf_hi)], relief=[("pressed", tk.FLAT)])
        style.configure("TEntry", fieldbackground=surf, foreground=fg,
                        insertcolor=fg, font=L, borderwidth=1, relief=tk.SOLID)
        style.map("TEntry", bordercolor=[("focus", accent)],
                  lightcolor=[("focus", accent)], darkcolor=[("focus", accent)])
        # Dark terminal-style variant used for the command preview readout.
        style.configure("Mono.TEntry", fieldbackground=S("log_bg"),
                        foreground=S("ok"), insertcolor=S("fg"),
                        borderwidth=1, relief=tk.SOLID)
        style.configure("TSpinbox", fieldbackground=surf, foreground=fg,
                        arrowcolor=fg, insertcolor=fg, font=L, borderwidth=1, relief=tk.SOLID)
        style.map("TSpinbox", bordercolor=[("focus", accent)])
        style.configure("TCheckbutton", background=bg, foreground=fg, font=L)
        style.map("TCheckbutton",
                  background=[("active", bg)], foreground=[("active", accent)])
        style.configure("TRadiobutton", background=bg, foreground=fg, font=L)
        style.configure("TCombobox", fieldbackground=surf, foreground=fg,
                        background=surf, arrowcolor=fg, font=L, borderwidth=1, relief=tk.SOLID)
        style.map("TCombobox", fieldbackground=[("readonly", surf)],
                  foreground=[("readonly", fg)], bordercolor=[("focus", accent)])
        style.configure("TProgressbar", background=accent, troughcolor=surf,
                        bordercolor=surf, lightcolor=accent, darkcolor=accent, borderwidth=0)
        # Threshold variants so every usage meter reads green/amber/red
        # with the same rule (<70 ok, 70-90 warn, 90+ danger).
        for _name, _col in (("Ok", S("ok")), ("Warn", S("warn")), ("Danger", S("danger"))):
            style.configure(f"{_name}.Horizontal.TProgressbar",
                            background=_col, troughcolor=surf,
                            bordercolor=surf, lightcolor=_col, darkcolor=_col,
                            borderwidth=0)

        style.configure("TLabelframe", background=bg, foreground=accent,
                        font=(_LATIN_FONT_FAMILY, 11, "bold"),
                        borderwidth=1, relief=tk.SOLID, bordercolor=S("border"))
        style.configure("TLabelframe.Label", background=bg, foreground=accent,
                        font=(_LATIN_FONT_FAMILY, 11, "bold"))

        style.configure("TNotebook", background=bg, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure("TNotebook.Tab", background=surf, foreground=muted,
                        font=(_LATIN_FONT_FAMILY, 10, "bold"), padding=(16, 7),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", bg), ("active", surf_hi)],
                  foreground=[("selected", accent), ("active", fg)],
                  expand=[("selected", (0, 0, 0, 0))])
        # Flat thin underline under the selected tab via a light top border.
        style.configure("Tab.TFrame", background=accent)
        style.configure("TPanedWindow", background=bg, sashthickness=6)

        # Native chrome that clam leaves light: scrollbars, treeviews and
        # combobox popdowns get the dark palette so no white boxes remain.
        style.configure("TScrollbar", background=surf, troughcolor=bg,
                        bordercolor=bg, arrowcolor=muted, relief=tk.FLAT)
        style.map("TScrollbar",
                  background=[("pressed", S("border")), ("active", surf_hi)],
                  arrowcolor=[("active", fg)])
        style.configure("Treeview", background=S("log_bg"), fieldbackground=S("log_bg"),
                        foreground=fg, borderwidth=0, relief=tk.FLAT)
        style.map("Treeview",
                  background=[("selected", accent)],
                  foreground=[("selected", bg)])
        style.configure("Treeview.Heading", background=surf, foreground=fg,
                        font=(_LATIN_FONT_FAMILY, 9, "bold"), borderwidth=0,
                        relief=tk.FLAT)
        style.map("Treeview.Heading", background=[("active", surf_hi)])
        # Card look: raised surface panels for the Monitoring tab meters.
        style.configure("Card.TLabelframe", background=S("surface"),
                        foreground=accent, borderwidth=1, relief=tk.SOLID,
                        bordercolor=S("border"))
        style.configure("Card.TLabelframe.Label", background=S("surface"),
                        foreground=accent, font=(_LATIN_FONT_FAMILY, 11, "bold"))
        style.configure("Card.TFrame", background=S("surface"))
        style.configure("Card.TLabel", background=S("surface"), foreground=fg)
        style.configure("CardMuted.TLabel", background=S("surface"), foreground=muted)
        # Visible separators (clam leaves them near-invisible on dark bg).
        style.configure("TSeparator", background=S("border"))
        for opt, val in (("*TCombobox*Listbox.background", surf),
                         ("*TCombobox*Listbox.foreground", fg),
                         ("*TCombobox*Listbox.selectBackground", accent),
                         ("*TCombobox*Listbox.selectForeground", bg),
                         ("*TCombobox*Listbox.font", (_LATIN_FONT_FAMILY, 10))):
            self.root.option_add(opt, val)

        style.configure("Accent.TButton", background=ok, foreground=bg,
                        font=(_LATIN_FONT_FAMILY, 11, "bold"), padding=(20, 11),
                        borderwidth=0, relief=tk.FLAT)
        style.map("Accent.TButton",
                  background=[("pressed", S("border")), ("active", accent)],
                  foreground=[("pressed", S("fg")), ("active", bg)])
        style.configure("Stop.TButton", background=danger, foreground=bg,
                        font=(_LATIN_FONT_FAMILY, 11, "bold"), padding=10,
                        borderwidth=0, relief=tk.FLAT)
        style.map("Stop.TButton",
                  background=[("pressed", S("border")), ("active", S("danger"))],
                  foreground=[("pressed", S("fg"))])
        style.configure("Info.TButton", background=info, foreground=bg,
                        font=(_LATIN_FONT_FAMILY, 10, "bold"), padding=7,
                        borderwidth=0, relief=tk.FLAT)
        style.map("Info.TButton",
                  background=[("pressed", S("border")), ("active", accent)],
                  foreground=[("pressed", S("fg")), ("active", bg)])

        self.bg, self.surf, self.surf_hi = bg, surf, surf_hi
        self.fg, self.muted = fg, muted
        self.accent, self.ok, self.danger, self.info = accent, ok, danger, info

    # ------------------------------------------------------------------
    # Small helpers to build translated widgets in one call
    # ------------------------------------------------------------------
    def _labeled(self, parent, key, width=14):
        lbl = ttk.Label(parent, text=self.t(key), width=width, anchor=tk.W)
        lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(lbl, key, "label")
        tip_key = key[:-6] + "_tip" if key.endswith("_label") else key + "_tip"
        if tip_key in TR:
            tip = Tooltip(lbl, self.t(tip_key))
            self._reg(tip, tip_key, "tooltip")
        return lbl

    def _mkcheck(self, parent, key, variable):
        cb = ttk.Checkbutton(parent, text=self.t(key), variable=variable)
        self._reg(cb, key, "checkbutton")
        tip_key = key[:-3] + "_tip" if key.endswith("_cb") else key + "_tip"
        if tip_key in TR:
            tip = Tooltip(cb, self.t(tip_key))
            self._reg(tip, tip_key, "tooltip")
        return cb

    # ------------------------------------------------------------------
    # Tab 1: Server setup (files, params, start/stop, log)
    # ------------------------------------------------------------------
    def _build_server_tab(self, parent):
        # --- Files ---
        files_frame = ttk.LabelFrame(parent, text=self.t("files_section"), padding=8)
        files_frame.pack(fill=tk.X, pady=(0, 8))
        self._reg(files_frame, "files_section", "labelframe")

        file_frame = ttk.Frame(files_frame)
        file_frame.pack(fill=tk.X, pady=4)
        self._labeled(file_frame, "model_label")
        self.model_var = tk.StringVar()
        self.model_var.trace_add("write", lambda *a: self.on_model_changed())
        self.model_entry = ttk.Entry(file_frame, textvariable=self.model_var)
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.browse_model_btn = ttk.Button(file_frame, text=self.t("browse_btn"), command=self.browse_model)
        self.browse_model_btn.pack(side=tk.LEFT)
        self._reg(self.browse_model_btn, "browse_btn", "button")

        self.model_size_var = tk.StringVar(value="No model selected")
        ttk.Label(files_frame, textvariable=self.model_size_var, foreground=S("fg_muted"),
                  font=self.en_font(9)).pack(anchor=tk.W, padx=(120, 0))

        server_frame = ttk.Frame(files_frame)
        server_frame.pack(fill=tk.X, pady=4)
        self._labeled(server_frame, "server_label")
        self.server_var = tk.StringVar(value=r"C:\llmcap\llama-server.exe")
        self.server_entry = ttk.Entry(server_frame, textvariable=self.server_var)
        self.server_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.browse_server_btn = ttk.Button(server_frame, text=self.t("browse_btn"), command=self.browse_server)
        self.browse_server_btn.pack(side=tk.LEFT)
        self._reg(self.browse_server_btn, "browse_btn", "button")

        # --- Compatibility badge ---
        self.badge_frame = ttk.Frame(files_frame)
        self.badge_frame.pack(fill=tk.X, pady=(6, 0))
        self.fit_badge = tk.Label(self.badge_frame, text=self.t("checking_hardware"),
                                   bg=S("surface"), fg=S("fg"),
                                   font=self.en_font(10, "bold"),
                                   padx=10, pady=5, bd=0)
        self.fit_badge.pack(side=tk.LEFT)
        self.analyze_btn = ttk.Button(self.badge_frame, text=self.t("analyze_btn"), style="Info.TButton",
                                       command=self.analyze_and_recommend)
        self.analyze_btn.pack(side=tk.LEFT, padx=8)
        self._reg(self.analyze_btn, "analyze_btn", "button")

        self.model_detail_var = tk.StringVar(value="")
        ttk.Label(files_frame, textvariable=self.model_detail_var, foreground=S("fg_muted"),
                  font=self.en_font(9), justify=tk.LEFT, wraplength=850).pack(anchor=tk.W, pady=(4, 0))

        # --- Parameters ---
        params_frame = ttk.LabelFrame(parent, text=self.t("params_section"), padding=8)
        params_frame.pack(fill=tk.X, pady=8)
        self._reg(params_frame, "params_section", "labelframe")

        # --- Quick presets, so you don't have to hand-tune every field ---
        preset_frame = ttk.Frame(params_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 8))
        self.preset_label_widget = ttk.Label(preset_frame, text=self.t("preset_label"), font=self.en_font(9, "bold"))
        self.preset_label_widget.pack(side=tk.LEFT, padx=(0, 6))
        self._reg(self.preset_label_widget, "preset_label", "label")
        self.preset_var = tk.StringVar(value="⚡ Max (RTX 3060 Ti / 16GB — safe full offload)")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, state="readonly",
                                     values=list(PRESETS.keys()), width=34)
        preset_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.apply_preset_btn = ttk.Button(preset_frame, text=self.t("apply_preset_btn"), style="Info.TButton",
                                            command=self.apply_preset)
        self.apply_preset_btn.pack(side=tk.LEFT)
        self._reg(self.apply_preset_btn, "apply_preset_btn", "button")
        preset_tip = Tooltip(preset_combo, self.t("preset_tip"))
        self._reg(preset_tip, "preset_tip", "tooltip")

        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=4)
        self._labeled(row1, "gpu_layers_label", width=13)
        self.ngl_var = tk.IntVar(value=99)
        ttk.Spinbox(row1, from_=0, to=200, textvariable=self.ngl_var, width=6).pack(side=tk.LEFT, padx=(0, 16))

        self._labeled(row1, "slots_label", width=7)
        self.slots_var = tk.IntVar(value=1)
        ttk.Spinbox(row1, from_=1, to=16, textvariable=self.slots_var, width=4).pack(side=tk.LEFT)

        ctx_row = ttk.Frame(params_frame)
        ctx_row.pack(fill=tk.X, pady=4)
        self._labeled(ctx_row, "context_label", width=13)
        self.ctx_var = tk.IntVar(value=8192)
        ttk.Spinbox(ctx_row, from_=512, to=131072, increment=512, textvariable=self.ctx_var, width=8).pack(side=tk.LEFT, padx=(0, 10))
        for val in CONTEXT_QUICK_VALUES:
            label = f"{val // 1024}K"
            ttk.Button(ctx_row, text=label, width=5,
                       command=lambda v=val: self.ctx_var.set(v)).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=4)
        self._labeled(row2, "threads_label", width=13)
        self.threads_var = tk.IntVar(value=4)
        ttk.Spinbox(row2, from_=1, to=64, textvariable=self.threads_var, width=6).pack(side=tk.LEFT, padx=(0, 16))

        self._labeled(row2, "tbatch_label", width=13)
        self.tbatch_var = tk.IntVar(value=6)
        ttk.Spinbox(row2, from_=1, to=64, textvariable=self.tbatch_var, width=6).pack(side=tk.LEFT, padx=(0, 16))

        self._labeled(row2, "batch_label", width=7)
        self.batch_var = tk.IntVar(value=2048)
        ttk.Spinbox(row2, from_=128, to=16384, increment=128, textvariable=self.batch_var, width=8).pack(side=tk.LEFT)

        # --- More optimization knobs: physical batch (ubatch) and KV
        # cache defrag threshold - both real llama-server perf flags. ---
        row2b = ttk.Frame(params_frame)
        row2b.pack(fill=tk.X, pady=4)
        self._labeled(row2b, "ubatch_label", width=18)
        self.ubatch_var = tk.IntVar(value=2048)
        ttk.Spinbox(row2b, from_=32, to=16384, increment=32, textvariable=self.ubatch_var, width=8).pack(side=tk.LEFT, padx=(0, 16))

        self._labeled(row2b, "defrag_label", width=18)
        self.defrag_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(row2b, from_=-1.0, to=1.0, increment=0.05, textvariable=self.defrag_var, width=6, format="%.2f").pack(side=tk.LEFT)

        row3 = ttk.Frame(params_frame)
        row3.pack(fill=tk.X, pady=4)
        self._labeled(row3, "port_label", width=13)
        self.port_var = tk.IntVar(value=8080)
        ttk.Spinbox(row3, from_=1024, to=65535, textvariable=self.port_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Separator(row3, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self.flash_var = tk.BooleanVar(value=True)  # RTX 3060 Ti supports it, big VRAM saver
        cb1 = self._mkcheck(row3, "flash_cb", self.flash_var)
        cb1.pack(side=tk.LEFT, padx=(0, 8))

        # KV quantization requires flash attention — keep the dependent
        # pair visually tighter than unrelated neighbors.
        self.kv_quant_var = tk.BooleanVar(value=True)
        self.kv_quant_var.trace_add("write", self.on_kv_quant_toggled)
        cb4 = self._mkcheck(row3, "kvquant_cb", self.kv_quant_var)
        cb4.pack(side=tk.LEFT, padx=(0, 4))

        kv_types = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        self.ctk_lbl = ttk.Label(row3, text=self.t("ctk_label"))
        self.ctk_lbl.pack(side=tk.LEFT)
        self._reg(self.ctk_lbl, "ctk_label", "label")
        self.ctk_var = tk.StringVar(value="q8_0")
        self.ctk_combo = ttk.Combobox(row3, textvariable=self.ctk_var, state="readonly",
                                       values=kv_types, width=7)
        self.ctk_combo.pack(side=tk.LEFT, padx=(0, 8))
        ctk_tip = Tooltip(self.ctk_combo, self.t("ctk_tip"))
        self._reg(ctk_tip, "ctk_tip", "tooltip")
        self.ctv_lbl = ttk.Label(row3, text=self.t("ctv_label"))
        self.ctv_lbl.pack(side=tk.LEFT)
        self._reg(self.ctv_lbl, "ctv_label", "label")
        self.ctv_var = tk.StringVar(value="q8_0")
        self.ctv_combo = ttk.Combobox(row3, textvariable=self.ctv_var, state="readonly",
                                       values=kv_types, width=7)
        self.ctv_combo.pack(side=tk.LEFT, padx=(0, 16))
        ctv_tip = Tooltip(self.ctv_combo, self.t("ctv_tip"))
        self._reg(ctv_tip, "ctv_tip", "tooltip")
        self._update_kv_widgets()

        row4 = ttk.Frame(params_frame)
        row4.pack(fill=tk.X, pady=4)

        lm_lbl = ttk.Label(row4, text=self.t("load_mode_label"))
        lm_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(lm_lbl, "load_mode_label", "label")
        self.load_mode_var = tk.StringVar(value="mlock")
        lm_combo = ttk.Combobox(row4, textvariable=self.load_mode_var, state="readonly",
                                 values=["auto", "none", "mmap", "mlock", "dio"], width=7)
        lm_combo.pack(side=tk.LEFT)
        lm_tip = Tooltip(lm_combo, self.t("load_mode_tip"))
        self._reg(lm_tip, "load_mode_tip", "tooltip")
        ttk.Separator(row4, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        self.unified_var = tk.BooleanVar(value=False)
        cb3 = self._mkcheck(row4, "unified_cb", self.unified_var)
        cb3.pack(side=tk.LEFT, padx=(0, 16))

        self.cache_idle_slots_var = tk.BooleanVar(value=False)
        cb_idle = self._mkcheck(row4, "cache_idle_cb", self.cache_idle_slots_var)
        cb_idle.pack(side=tk.LEFT, padx=(0, 16))

        self.no_mmap_var = tk.BooleanVar(value=False)
        cb5 = self._mkcheck(row4, "nommap_cb", self.no_mmap_var)
        cb5.pack(side=tk.LEFT, padx=(0, 16))

        self.no_warmup_var = tk.BooleanVar(value=False)
        cb6 = self._mkcheck(row4, "nowarmup_cb", self.no_warmup_var)
        cb6.pack(side=tk.LEFT)

        # --- Advanced Performance Settings ---
        adv_frame = ttk.LabelFrame(parent, text="🚀 Advanced Performance", padding=8)
        adv_frame.pack(fill=tk.X, pady=(8, 0))
        self._reg(adv_frame, "advanced_perf", "labelframe")

        adv_desc = ttk.Label(adv_frame, text=self.t("advanced_desc"),
                             foreground=S("fg_muted"), font=self.en_font(9),
                             wraplength=850, justify=tk.LEFT)
        adv_desc.pack(anchor=tk.W, pady=(0, 2))
        self._reg(adv_desc, "advanced_desc", "label")

        # Row 1: Continuous Batching & Metrics
        adv_row1 = ttk.Frame(adv_frame)
        adv_row1.pack(fill=tk.X, pady=4)

        self.cont_batching_var = tk.BooleanVar(value=True)
        cb_cb = self._mkcheck(adv_row1, "cont_batching_cb", self.cont_batching_var)
        cb_cb.pack(side=tk.LEFT, padx=(0, 16))

        self.metrics_endpoint_var = tk.BooleanVar(value=False)
        cb_me = self._mkcheck(adv_row1, "metrics_endpoint_cb", self.metrics_endpoint_var)
        cb_me.pack(side=tk.LEFT, padx=(0, 16))

        self.yarn_var = tk.BooleanVar(value=False)
        cb_yarn = self._mkcheck(adv_row1, "yarn_cb", self.yarn_var)
        cb_yarn.pack(side=tk.LEFT, padx=(0, 16))

        # Row 2: MoE offloading
        adv_row2 = ttk.Frame(adv_frame)
        adv_row2.pack(fill=tk.X, pady=4)

        self.moe_cpu_var = tk.BooleanVar(value=False)
        cb_moe = self._mkcheck(adv_row2, "moe_cpu_cb", self.moe_cpu_var)
        cb_moe.pack(side=tk.LEFT, padx=(0, 16))

        moe_lbl = ttk.Label(adv_row2, text=self.t("moe_cpu_layers_label"), font=self.en_font(9))
        moe_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(moe_lbl, "moe_cpu_layers_label", "label")
        moe_lbl_tip = Tooltip(moe_lbl, self.t("moe_cpu_layers_tip"))
        self._reg(moe_lbl_tip, "moe_cpu_layers_tip", "tooltip")
        self.moe_cpu_layers_var = tk.IntVar(value=10)
        moe_spin = ttk.Spinbox(adv_row2, from_=0, to=100, textvariable=self.moe_cpu_layers_var, width=5)
        moe_spin.pack(side=tk.LEFT)
        moe_spin_tip = Tooltip(moe_spin, self.t("moe_cpu_layers_tip"))
        self._reg(moe_spin_tip, "moe_cpu_layers_tip", "tooltip")

        # Row 3: sleep-when-idle
        adv_row3 = ttk.Frame(adv_frame)
        adv_row3.pack(fill=tk.X, pady=4)

        self.sleep_idle_var = tk.BooleanVar(value=False)
        cb_sleep = self._mkcheck(adv_row3, "sleep_idle_cb", self.sleep_idle_var)
        cb_sleep.pack(side=tk.LEFT, padx=(0, 16))

        sleep_lbl = ttk.Label(adv_row3, text=self.t("sleep_idle_seconds_label"), font=self.en_font(9))
        sleep_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(sleep_lbl, "sleep_idle_seconds_label", "label")
        self.sleep_idle_seconds_var = tk.IntVar(value=600)
        sleep_spin = ttk.Spinbox(adv_row3, from_=1, to=86400, textvariable=self.sleep_idle_seconds_var, width=7)
        sleep_spin.pack(side=tk.LEFT)
        sleep_tip = Tooltip(sleep_spin, self.t("sleep_idle_seconds_tip"))
        self._reg(sleep_tip, "sleep_idle_seconds_tip", "tooltip")

        # Row 4: multi-GPU device list (-dev)
        adv_row4 = ttk.Frame(adv_frame)
        adv_row4.pack(fill=tk.X, pady=4)

        dev_lbl = ttk.Label(adv_row4, text=self.t("devices_label"), font=self.en_font(9), width=22, anchor=tk.W)
        dev_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(dev_lbl, "devices_label", "label")
        self.devices_var = tk.StringVar(value="")
        dev_entry = ttk.Entry(adv_row4, textvariable=self.devices_var, width=24)
        dev_entry.pack(side=tk.LEFT)
        dev_tip = Tooltip(dev_entry, self.t("devices_tip"))
        self._reg(dev_tip, "devices_tip", "tooltip")

        # Row 5: tensor buffer overrides (-ot, repeatable)
        adv_row5 = ttk.Frame(adv_frame)
        adv_row5.pack(fill=tk.X, pady=4)

        ot_lbl = ttk.Label(adv_row5, text=self.t("ot_label"), font=self.en_font(9), width=22, anchor=tk.W)
        ot_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(ot_lbl, "ot_label", "label")
        self.override_tensor_var = tk.StringVar(value="")
        ot_entry = ttk.Entry(adv_row5, textvariable=self.override_tensor_var)
        ot_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ot_tip = Tooltip(ot_entry, self.t("ot_tip"))
        self._reg(ot_tip, "ot_tip", "tooltip")

        # --- Reasoning / thinking toggle, for hybrid reasoning models like
        # Qwen3, DeepSeek-R1, QwQ. No effect on regular non-reasoning models. ---
        reasoning_frame = ttk.LabelFrame(parent, text=self.t("reasoning_section"), padding=8)
        reasoning_frame.pack(fill=tk.X, pady=(0, 8))
        self._reg(reasoning_frame, "reasoning_section", "labelframe")
        self.reasoning_desc_label = ttk.Label(reasoning_frame, text=self.t("reasoning_desc"),
                                               foreground=S("fg_muted"), font=self.en_font(9),
                                               wraplength=850, justify=tk.LEFT)
        self.reasoning_desc_label.pack(anchor=tk.W)
        self._reg(self.reasoning_desc_label, "reasoning_desc", "label")
        reasoning_btns = ttk.Frame(reasoning_frame)
        reasoning_btns.pack(fill=tk.X, pady=(6, 0))
        self.reasoning_mode = tk.StringVar(value="on")
        self.think_on_btn = tk.Button(reasoning_btns, text=self.t("think_on_btn"), bd=0,
                                       font=self.en_font(10, "bold"), padx=16, pady=7,
                                       cursor="hand2", activebackground=S("surface_hi"),
                                       command=lambda: self.set_reasoning_mode("on"))
        self.think_on_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.think_off_btn = tk.Button(reasoning_btns, text=self.t("think_off_btn"), bd=0,
                                        font=self.en_font(10, "bold"), padx=16, pady=7,
                                        cursor="hand2", activebackground=S("surface_hi"),
                                        command=lambda: self.set_reasoning_mode("off"))
        self.think_off_btn.pack(side=tk.LEFT)
        rf_lbl = ttk.Label(reasoning_btns, text=self.t("reasoning_format_label"))
        rf_lbl.pack(side=tk.LEFT, padx=(16, 5))
        self._reg(rf_lbl, "reasoning_format_label", "label")
        self.reasoning_format_var = tk.StringVar(value="auto")
        rf_combo = ttk.Combobox(reasoning_btns, textvariable=self.reasoning_format_var,
                                 state="readonly", values=["auto", "none", "deepseek"], width=9)
        rf_combo.pack(side=tk.LEFT)
        rf_tip = Tooltip(rf_combo, self.t("reasoning_format_tip"))
        self._reg(rf_tip, "reasoning_format_tip", "tooltip")
        self._reg(self.think_on_btn, "think_on_btn", "button")
        self._reg(self.think_off_btn, "think_off_btn", "button")
        think_on_tip = Tooltip(self.think_on_btn, self.t("think_on_tip"))
        think_off_tip = Tooltip(self.think_off_btn, self.t("think_off_tip"))
        self._reg(think_on_tip, "think_on_tip", "tooltip")
        self._reg(think_off_tip, "think_off_tip", "tooltip")
        self.set_reasoning_mode("on")

        # --- Speculative Decoding — mirrors Unsloth Studio's hub panel:
        # one compact strategy select, and the Draft Tokens / draft-KV
        # dtype controls materialize inline ONLY when the chosen strategy
        # uses them (conditional rendering, like their React panel). ---
        spec_frame = ttk.LabelFrame(parent, text=self.t("spec_section"), padding=8)
        spec_frame.pack(fill=tk.X, pady=(0, 8))
        self._reg(spec_frame, "spec_section", "labelframe")

        spec_grid = ttk.Frame(spec_frame)
        spec_grid.pack(fill=tk.X)
        spec_grid.columnconfigure(7, weight=1)

        def _spec_label(col, key, tip_key):
            lbl = ttk.Label(spec_grid, text=self.t(key))
            lbl.grid(row=0, column=col, sticky="w", padx=(12, 6))
            self._reg(lbl, key, "label")
            tip = Tooltip(lbl, self.t(tip_key))
            self._reg(tip, tip_key, "tooltip")
            return lbl

        self.spec_mode_lbl = _spec_label(0, "spec_mode_label", "spec_mode_tip")
        self.spec_mode_var = tk.StringVar(value="Auto")
        spec_combo = ttk.Combobox(spec_grid, textvariable=self.spec_mode_var, state="readonly",
                                   values=["Auto", "MTP", "DSpark", "DFlash", "Ngram",
                                           "MTP+Ngram", "Off"], width=10)
        spec_combo.grid(row=0, column=1, sticky="w")
        spec_mode_tip = Tooltip(spec_combo, self.t("spec_mode_tip"))
        self._reg(spec_mode_tip, "spec_mode_tip", "tooltip")

        # Draft Tokens group: label + Auto tick + value, Unsloth's "blank
        # field = default" expressed as a toggle.
        self.spec_draft_lbl = _spec_label(2, "spec_draft_tokens_label", "spec_draft_tokens_tip")
        self.spec_draft_auto_var = tk.BooleanVar(value=True)
        self.spec_auto_cb = ttk.Checkbutton(spec_grid, text=self.t("spec_draft_auto_cb"),
                                             variable=self.spec_draft_auto_var)
        self.spec_auto_cb.grid(row=0, column=3, sticky="w", padx=(0, 4))
        self._reg(self.spec_auto_cb, "spec_draft_auto_cb", "checkbutton")
        auto_tip = Tooltip(self.spec_auto_cb, self.t("spec_draft_auto_tip"))
        self._reg(auto_tip, "spec_draft_auto_tip", "tooltip")
        self.spec_draft_nmax_var = tk.IntVar(value=2)
        self.spec_nmax_spin = ttk.Spinbox(spec_grid, from_=1, to=16,
                                           textvariable=self.spec_draft_nmax_var, width=4)
        self.spec_nmax_spin.grid(row=0, column=4, sticky="w", padx=(0, 2))
        spec_nmax_tip = Tooltip(self.spec_nmax_spin, self.t("spec_draft_tokens_tip"))
        self._reg(spec_nmax_tip, "spec_draft_tokens_tip", "tooltip")
        self.spec_draft_auto_var.trace_add("write", lambda *a: self._update_spec_controls())

        # Draft KV dtype group (DSpark/DFlash only).
        self.spec_cache_lbl = _spec_label(5, "spec_cache_label", "spec_cache_tip")
        self.spec_draft_cache_var = tk.StringVar(value="f16")
        self.spec_cache_combo = ttk.Combobox(spec_grid, textvariable=self.spec_draft_cache_var,
                                              state="readonly",
                                              values=["f16", "bf16", "q8_0", "q4_0", "q4_1",
                                                      "q5_0", "q5_1", "iq4_nl", "f32"], width=7)
        self.spec_cache_combo.grid(row=0, column=6, sticky="w")
        spec_cache_tip = Tooltip(self.spec_cache_combo, self.t("spec_cache_tip"))
        self._reg(spec_cache_tip, "spec_cache_tip", "tooltip")

        self.spec_apply_btn = ttk.Button(spec_grid, text="  " + self.t("spec_apply_btn"),
                                          style="Info.TButton",
                                          command=self.apply_spec_settings)
        self.spec_apply_btn.grid(row=0, column=8, sticky="e")
        self._reg(self.spec_apply_btn, "spec_apply_btn", "button")

        # Live one-line status of what the panel will actually send.
        self.spec_hint_lbl = ttk.Label(spec_frame, text="", font=self.en_font(9),
                                        foreground=S("fg_muted"))
        self.spec_hint_lbl.pack(anchor=tk.W, pady=(6, 0))
        self.on_spec_mode_changed()

        # --- Multimodal projector (vision models). Local file wins over
        # URL; llama-server only accepts one of the two flags. ---
        mm_frame = ttk.LabelFrame(parent, text=self.t("mmproj_section"), padding=8)
        mm_frame.pack(fill=tk.X, pady=(0, 8))
        self._reg(mm_frame, "mmproj_section", "labelframe")

        mm_desc = ttk.Label(mm_frame, text=self.t("mmproj_desc"),
                            foreground=S("fg_muted"), font=self.en_font(9),
                            wraplength=850, justify=tk.LEFT)
        mm_desc.pack(anchor=tk.W, pady=(0, 2))
        self._reg(mm_desc, "mmproj_desc", "label")

        mm_file_row = ttk.Frame(mm_frame)
        mm_file_row.pack(fill=tk.X, pady=4)
        mmf_lbl = ttk.Label(mm_file_row, text=self.t("mmproj_file_label"), width=14, anchor=tk.W)
        mmf_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(mmf_lbl, "mmproj_file_label", "label")
        self.mmproj_path_var = tk.StringVar(value="")
        mmf_entry = ttk.Entry(mm_file_row, textvariable=self.mmproj_path_var)
        mmf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        mmf_tip = Tooltip(mmf_entry, self.t("mmproj_file_tip"))
        self._reg(mmf_tip, "mmproj_file_tip", "tooltip")
        self.browse_mmproj_btn = ttk.Button(mm_file_row, text=self.t("browse_btn"),
                                             command=self.browse_mmproj)
        self.browse_mmproj_btn.pack(side=tk.LEFT)
        self._reg(self.browse_mmproj_btn, "browse_btn", "button")

        mm_url_row = ttk.Frame(mm_frame)
        mm_url_row.pack(fill=tk.X, pady=4)
        mmu_lbl = ttk.Label(mm_url_row, text=self.t("mmproj_url_label"), width=14, anchor=tk.W)
        mmu_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(mmu_lbl, "mmproj_url_label", "label")
        self.mmproj_url_var = tk.StringVar(value="")
        mmu_entry = ttk.Entry(mm_url_row, textvariable=self.mmproj_url_var)
        mmu_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        mmu_tip = Tooltip(mmu_entry, self.t("mmproj_url_tip"))
        self._reg(mmu_tip, "mmproj_url_tip", "tooltip")

        mm_vis_row = ttk.Frame(mm_frame)
        mm_vis_row.pack(fill=tk.X, pady=(6, 0))
        self.vision_check_btn = ttk.Button(mm_vis_row, text=self.t("vision_check_btn"),
                                            command=self._check_vision)
        self.vision_check_btn.pack(side=tk.LEFT)
        self._reg(self.vision_check_btn, "vision_check_btn", "button")
        vis_tip = Tooltip(self.vision_check_btn, self.t("vision_check_tip"))
        self._reg(vis_tip, "vision_check_tip", "tooltip")
        self.vision_status_lbl = ttk.Label(mm_vis_row, text="", font=self.en_font(9))
        self.vision_status_lbl.pack(side=tk.LEFT, padx=10)
        self._refresh_vision_status()

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(4, 2))

        extra_frame = ttk.Frame(parent)
        extra_frame.pack(fill=tk.X, pady=5)
        self._labeled(extra_frame, "extra_args_label", width=13)
        self.extra_var = tk.StringVar()
        ttk.Entry(extra_frame, textvariable=self.extra_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # --- Command preview ---
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill=tk.X, pady=(0, 5))
        self.preview_label_widget = ttk.Label(preview_frame, text=self.t("preview_label"), font=self.en_font(9, "bold"),
                                               foreground=S("fg_muted"))
        self.preview_label_widget.pack(anchor=tk.W)
        self._reg(self.preview_label_widget, "preview_label", "label")
        self.preview_var = tk.StringVar(value="")
        preview_entry = ttk.Entry(preview_frame, textvariable=self.preview_var, state="readonly")
        preview_entry.pack(fill=tk.X, pady=(2, 0))
        # Terminal-style readout for the generated command line.
        preview_entry.configure(style="Mono.TEntry", font=("Consolas", 9))
        for var in (self.ngl_var, self.ctx_var, self.slots_var, self.threads_var,
                    self.tbatch_var, self.batch_var, self.ubatch_var, self.defrag_var,
                    self.port_var, self.flash_var, self.load_mode_var, self.unified_var,
                    self.kv_quant_var, self.ctk_var, self.ctv_var,
                    self.no_mmap_var, self.no_warmup_var,
                    self.cache_idle_slots_var, self.sleep_idle_var,
                    self.sleep_idle_seconds_var, self.moe_cpu_var,
                    self.moe_cpu_layers_var, self.reasoning_format_var,
                    self.devices_var, self.override_tensor_var,
                    self.mmproj_path_var, self.mmproj_url_var,
                    self.spec_mode_var, self.spec_draft_nmax_var, self.spec_draft_cache_var,
                    self.extra_var, self.model_var, self.server_var):
            var.trace_add("write", lambda *a: self.update_preview())

        # --- Buttons ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=8)
        self.start_btn = ttk.Button(btn_frame, text=self.t("start_btn"), style="Accent.TButton", command=self.start_server)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text=self.t("stop_btn"), style="Stop.TButton", command=self.stop_server, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self._reg(self.start_btn, "start_btn", "button")
        self._reg(self.stop_btn, "stop_btn", "button")
        self.copy_btn = ttk.Button(btn_frame, text=self.t("copy_btn"), command=self.copy_command)
        self.copy_btn.pack(side=tk.LEFT, padx=5)
        self._reg(self.copy_btn, "copy_btn", "button")
        self.clear_btn = ttk.Button(btn_frame, text=self.t("clear_btn"), command=self.clear_log)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)
        self._reg(self.clear_btn, "clear_btn", "button")

        self.status_var = tk.StringVar(value=self.t("status_ready"))
        ttk.Label(btn_frame, textvariable=self.status_var, foreground=S("ok")).pack(side=tk.LEFT, padx=20)

        status_bar = ttk.Frame(parent)
        status_bar.pack(fill=tk.X, pady=(0, 8))
        self.load_status_var = tk.StringVar(value=self.t("load_status_idle"))
        self.load_status_label = tk.Label(status_bar, textvariable=self.load_status_var,
                                           bg=S("surface"), fg=S("fg"),
                                           font=self.en_font(10, "bold"),
                                           padx=10, pady=5, bd=0, relief=tk.FLAT)
        self.load_status_label.pack(side=tk.LEFT, padx=(0, 8))

        self.speed_var = tk.StringVar(value="⚡ --")
        self.speed_label = tk.Label(status_bar, textvariable=self.speed_var,
                                     bg=S("surface"), fg=S("info"),
                                     font=self.en_font(10, "bold"),
                                     padx=10, pady=5, bd=0, relief=tk.FLAT,
                                     cursor="sb_h_double_arrow")
        self.speed_label.pack(side=tk.LEFT, padx=(0, 8))
        speed_tip = Tooltip(self.speed_label, self.t("speed_tip"))
        self._reg(speed_tip, "speed_tip", "tooltip")

        # Dedicated session-totals pill so token counts are always visible,
        # independent of the speed readout.
        self.tok_var = tk.StringVar(value="Σ --")
        self.tok_label = tk.Label(status_bar, textvariable=self.tok_var,
                                   bg=S("surface"), fg=S("accent"),
                                   font=self.en_font(10, "bold"),
                                   padx=10, pady=5, bd=0, relief=tk.FLAT)
        self.tok_label.pack(side=tk.LEFT, padx=(0, 8))

        self.open_browser_btn = ttk.Button(status_bar, text=self.t("open_browser_btn"),
                                            command=self.open_in_browser, state=tk.DISABLED)
        self.open_browser_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._reg(self.open_browser_btn, "open_browser_btn", "button")

        self.uptime_var = tk.StringVar(value="")
        ttk.Label(status_bar, textvariable=self.uptime_var, foreground=S("fg_muted"),
                  font=self.en_font(9)).pack(side=tk.LEFT)

        self.update_preview()

    # ------------------------------------------------------------------
    # Tab 2: System info & compatibility
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # New layout: left pane (notebook) + right pane (log)
    # ------------------------------------------------------------------
    def _build_left_notebook(self, parent):
        """Build the left pane with Server Setup and System tabs."""
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 10))
        self.title_label = ttk.Label(header, text=self.t("app_title"),
                                      font=self.font_for(self.t("app_title"), 20, "bold"),
                                      foreground=S("accent"))
        self.title_label.pack(side=tk.LEFT)
        self.subtitle_label = ttk.Label(header, text="  " + self.t("app_subtitle"),
                                         font=self.font_for(self.t("app_subtitle"), 10),
                                         foreground=S("fg_muted"))
        self.subtitle_label.pack(side=tk.LEFT, pady=(8, 0))
        self._reg(self.title_label, "app_title", "label")
        self._reg(self.subtitle_label, "app_subtitle", "label_prefixed")

        # Right side of header: version badge + lang toggle
        header_right = ttk.Frame(header)
        header_right.pack(side=tk.RIGHT)
        self.version_label = tk.Label(header_right, text="v2.0", font=self.en_font(9, "bold"),
                                       bg=S("surface_hi"), fg=S("accent"), padx=8, pady=2, bd=0)
        self.version_label.pack(side=tk.RIGHT, padx=(8, 0))
        self.lang_btn = tk.Button(header_right, text=self.t("lang_toggle"), bd=0,
                                   font=self.font_for(self.t("lang_toggle"), 10, "bold"),
                                   bg=S("surface"), fg=S("fg"),
                                   activebackground=S("surface_hi"), padx=12, pady=4,
                                   borderwidth=0, relief=tk.FLAT, cursor="hand2",
                                   command=self.toggle_language)
        self.lang_btn.pack(side=tk.RIGHT)
        # Same hover feedback the ttk buttons get from their style maps.
        self._attach_hover(self.lang_btn, S("surface"), S("surface_hi"))

        # Accent rule under the header separates chrome from content.
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Server Setup is a long form, so it scrolls: a canvas + inner
        # frame + vertical scrollbar. The other tabs stay plain frames.
        self.server_tab = tk.Frame(self.notebook, bg=S("bg"))
        self._server_canvas = tk.Canvas(self.server_tab, bg=S("bg"),
                                         highlightthickness=0, bd=0)
        self._server_vsb = ttk.Scrollbar(self.server_tab, orient="vertical",
                                          command=self._server_canvas.yview)
        self._server_canvas.configure(yscrollcommand=self._server_vsb.set)
        self._server_win = self._server_canvas.create_window((0, 0), anchor="nw")
        self.server_inner = ttk.Frame(self._server_canvas, padding=10)
        self._server_canvas.itemconfigure(self._server_win, window=self.server_inner)
        self.server_inner.bind(
            "<Configure>",
            lambda e: self._server_canvas.configure(
                scrollregion=self._server_canvas.bbox("all")))
        self._server_canvas.bind("<Configure>", self._on_server_canvas_resize)
        self._server_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._server_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.gpu_procs_tab = ttk.Frame(self.notebook, padding=10)
        self.tasks_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.server_tab, text=self.t("tab_server"))
        self.notebook.add(self.gpu_procs_tab, text=self.t("tab_gpu_procs"))
        self.notebook.add(self.tasks_tab, text=self.t("tab_tasks"))
        self._i18n_tabs = [(self.server_tab, "tab_server"), (self.gpu_procs_tab, "tab_gpu_procs"),
                           (self.tasks_tab, "tab_tasks")]

        self._build_server_tab(self.server_inner)
        self._build_gpu_procs_tab(self.gpu_procs_tab)
        self._build_tasks_tab(self.tasks_tab)

        # Wheel scrolling works over any part of the Server Setup tab.
        self._bind_mousewheel_recursive(self.server_inner)

    def _build_log_panel(self, parent):
        """Build the right pane - large log area always visible."""
        # Header
        log_header = ttk.Frame(parent)
        log_header.pack(fill=tk.X, pady=(0, 4))
        self.log_label_widget = ttk.Label(log_header, text=self.t("log_label"),
                                          font=self.en_font(10, "bold"), foreground=S("info"))
        self.log_label_widget.pack(side=tk.LEFT)
        self._reg(self.log_label_widget, "log_label", "label")

        # Log control buttons
        log_ctrl = ttk.Frame(log_header)
        log_ctrl.pack(side=tk.RIGHT)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        self.auto_scroll_cb = ttk.Checkbutton(log_ctrl, text=self.t("auto_scroll"),
                                               variable=self.auto_scroll_var)
        self.auto_scroll_cb.pack(side=tk.LEFT, padx=4)
        self._reg(self.auto_scroll_cb, "auto_scroll", "checkbutton")

        # Jump-to-latest is handy when auto-scroll is off.
        self.jump_latest_btn = ttk.Button(log_ctrl, text="⬇ " + self.t("jump_latest_btn"),
                                           command=self.jump_to_latest, style="Info.TButton")
        self.jump_latest_btn.pack(side=tk.LEFT, padx=2)
        self._reg(self.jump_latest_btn, "jump_latest_btn", "button")

        self.clear_log_btn = ttk.Button(log_ctrl, text="🗑 Clear", command=self.clear_log, style="Info.TButton")
        self.clear_log_btn.pack(side=tk.LEFT, padx=2)
        self._reg(self.clear_log_btn, "clear_btn", "button")
        self.copy_log_btn = ttk.Button(log_ctrl, text="📋 Copy", command=self.copy_log, style="Info.TButton")
        self.copy_log_btn.pack(side=tk.LEFT, padx=2)
        self._reg(self.copy_log_btn, "copy_log_btn", "button")

        # Search/filter box
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(search_frame, text="🔍 Filter:", font=self.en_font(9)).pack(side=tk.LEFT, padx=(0, 4))
        self.log_filter_var = tk.StringVar()
        self.log_filter_entry = ttk.Entry(search_frame, textvariable=self.log_filter_var, width=30)
        self.log_filter_entry.pack(side=tk.LEFT, padx=2)
        self.log_filter_var.trace_add("write", lambda *a: self.filter_log())
        self._reg(self.log_filter_entry, "log_filter", "entry")

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(2, 6))

        # Large log area, outlined with a 1px border frame so the well
        # reads as a distinct panel against the dark background.
        log_holder = tk.Frame(parent, bg=S("border"), bd=0,
                               highlightthickness=0)
        log_holder.pack(fill=tk.BOTH, expand=True)
        self.log = scrolledtext.ScrolledText(
            log_holder, bg=S("log_bg"), fg=S("fg"), insertbackground=S("fg"),
            font=("Consolas", 9), wrap=tk.WORD, state=tk.DISABLED,
            borderwidth=1, relief=tk.SOLID
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.log.tag_config("info", foreground=S("fg_muted"))
        self.log.tag_config("warn", foreground=S("warn"))
        self.log.tag_config("error", foreground=S("danger"))
        self.log.tag_config("success", foreground=S("ok"))

    # ------------------------------------------------------------------
    # Tab: Run Task — quick prompt against the running server, simple tok/s
    # ------------------------------------------------------------------
    def _build_tasks_tab(self, parent):
        bg, surf, surf_hi = S("bg"), S("surface"), S("surface_hi")
        fg, muted = S("fg"), S("fg_muted")
        ok, warn, danger, info = S("ok"), S("warn"), S("danger"), S("info")

        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 8))
        title = ttk.Label(header, text=self.t("tab_tasks"),
                          font=self.font_for(self.t("tab_tasks"), 16, "bold"),
                          foreground=S("accent"))
        title.pack(side=tk.LEFT)
        self._reg(title, "tab_tasks", "label")
        self.task_server_state = tk.Label(header, text=self.t("task_not_running"), bg=surf,
                                          fg=warn, font=self.en_font(9), padx=10, pady=3, bd=0)
        self.task_server_state.pack(side=tk.RIGHT)
        self._reg(self.task_server_state, "task_not_running", "label")

        presets = ttk.Frame(parent)
        presets.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(presets, text=self.t("task_presets"), font=self.en_font(9, "bold"),
                  foreground=muted).pack(side=tk.LEFT)
        for key in ("task_pres_chat", "task_pres_summarize", "task_pres_translate", "task_pres_code"):
            b = tk.Button(presets, text=self.t(key), bd=0, font=self.en_font(9),
                          bg=surf, fg=info, activebackground=surf_hi,
                          activeforeground=info, padx=10, pady=4, cursor="hand2",
                          command=lambda k=key: self._task_load_preset(k))
            b.pack(side=tk.LEFT, padx=3)
            self._reg(b, key, "button")

        prompt_lbl = ttk.Label(parent, text=self.t("task_prompt"), font=self.en_font(10, "bold"))
        prompt_lbl.pack(anchor=tk.W)
        self._reg(prompt_lbl, "task_prompt", "label")
        self.task_prompt = scrolledtext.ScrolledText(parent, height=6, wrap=tk.WORD,
                                                     bg=surf, fg=fg, bd=0,
                                                     highlightthickness=1,
                                                     highlightbackground=S("border"),
                                                     insertbackground=fg,
                                                     font=self.font_for("", 10))
        self.task_prompt.pack(fill=tk.X, pady=(2, 8))

        opts = ttk.Frame(parent)
        opts.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(opts, text=self.t("task_n_predict"), font=self.en_font(9)).pack(side=tk.LEFT)
        self.task_npredict_var = tk.StringVar(value="256")
        sp = ttk.Spinbox(opts, from_=8, to=4096, increment=8, width=6,
                         textvariable=self.task_npredict_var)
        sp.pack(side=tk.LEFT, padx=6)
        ttk.Label(opts, text=self.t("task_temp"), font=self.en_font(9)).pack(side=tk.LEFT, padx=(12, 0))
        self.task_temp_var = tk.StringVar(value="0.7")
        sp2 = ttk.Spinbox(opts, from_=0.0, to=2.0, increment=0.1, width=5, format="%.1f",
                          textvariable=self.task_temp_var)
        sp2.pack(side=tk.LEFT, padx=6)
        run_btn = tk.Button(opts, text=self.t("task_run"), command=self.run_task, bd=0,
                            cursor="hand2", bg=surf_hi, fg=ok,
                            activebackground=surf, padx=18, pady=6,
                            font=self.en_font(10, "bold"))
        run_btn.pack(side=tk.RIGHT)
        self._reg(run_btn, "task_run", "button")

        result = tk.Frame(parent, bg=surf, highlightthickness=1, highlightbackground=S("border"))
        result.pack(fill=tk.X, pady=(0, 8))
        self.task_tps = tk.Label(result, text="-- tok/s", font=("Consolas", 26, "bold"),
                                 fg=info, bg=surf, padx=12, pady=8)
        self.task_tps.pack(anchor=tk.W)
        self.task_metrics = tk.Label(result, text="", font=self.en_font(9), fg=muted, bg=surf,
                                     anchor=tk.W, padx=14)
        self.task_metrics.pack(anchor=tk.W, pady=(0, 4))
        self.task_status = tk.Label(result, text="", font=self.en_font(9), fg=S("accent"), bg=surf,
                                    anchor=tk.W, padx=14)
        self.task_status.pack(anchor=tk.W, pady=(0, 8))

        out_lbl = ttk.Label(parent, text=self.t("task_output"), font=self.en_font(10, "bold"))
        out_lbl.pack(anchor=tk.W)
        self._reg(out_lbl, "task_output", "label")
        self.task_output = scrolledtext.ScrolledText(parent, height=8, wrap=tk.WORD,
                                                     bg=surf, fg=fg, bd=0,
                                                     highlightthickness=1,
                                                     highlightbackground=S("border"),
                                                     insertbackground=fg,
                                                     font=self.font_for("", 10))
        self.task_output.pack(fill=tk.BOTH, expand=True)

        self._task_busy = False
        self._update_task_server_state()
        self.task_prompt.bind("<FocusIn>", lambda e: self._update_task_server_state())

    def _server_healthy(self):
        return bool(self.process and self.process.poll() is None and self.model_ready)

    def _update_task_server_state(self):
        if self._server_healthy():
            self.task_server_state.config(
                text=self.t("task_running").format(self.port_var.get()), fg=S("ok"))
        else:
            self.task_server_state.config(text=self.t("task_not_running"), fg=S("warn"))

    def _task_load_preset(self, key):
        text = self.t(key + "_text")
        self.task_prompt.delete("1.0", tk.END)
        self.task_prompt.insert("1.0", text)

    def run_task(self):
        if self._task_busy:
            return
        prompt = self.task_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning(self.t("task_run"), self.t("task_empty_prompt"))
            return
        if not self.model_var.get().strip():
            messagebox.showwarning(self.t("task_run"), self.t("task_no_model"))
            return
        self._task_busy = True
        self.task_tps.config(text="--", fg=S("info"))
        self.task_status.config(text=self.t("task_running_task"), fg=S("info"))
        self._update_task_server_state()

        def worker():
            try:
                port = self.port_var.get()
                if not self._server_healthy():
                    self.root.after(0, lambda: (self.log_write("Starting server for task...", "info"),
                                                self.start_server()))
                deadline = time.time() + 240
                while time.time() < deadline:
                    if self._server_healthy():
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("Server did not become ready in time")
                body = json.dumps({
                    "prompt": prompt,
                    "n_predict": int(self.task_npredict_var.get()),
                    "temperature": float(self.task_temp_var.get()),
                    "stream": False,
                    "cache_prompt": False,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/completion", data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=900) as resp:
                    out = json.loads(resp.read().decode("utf-8", errors="replace"))
                timings = out.get("timings", {}) or {}
                self.root.after(0, lambda: self._task_finish(
                    out.get("content", ""),
                    timings.get("predicted_per_second"),
                    timings.get("prompt_per_second"),
                    timings.get("predicted_n", 0),
                    timings.get("prompt_n", 0),
                    timings.get("total_ms", 0)))
            except Exception as e:
                self.root.after(0, lambda e=e: self._task_fail(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _task_finish(self, content, tps, pts, pred_n, prompt_n, total_ms):
        self._task_busy = False
        if tps:
            self.task_tps.config(text=f"{tps:.1f} tok/s", fg=S("ok"))
        else:
            self.task_tps.config(text="tok/s n/a", fg=S("warn"))
        metrics = []
        if prompt_n:
            metrics.append(f"{prompt_n} prompt tokens ({pts:.1f} tok/s)" if pts else f"{prompt_n} prompt tokens")
        if pred_n:
            metrics.append(f"{pred_n} generated tokens")
        if total_ms:
            metrics.append(f"{total_ms / 1000:.1f}s total")
        self.task_metrics.config(text="   •   ".join(metrics))
        self.task_status.config(text=self.t("task_done"), fg=S("accent"))
        self.task_output.delete("1.0", tk.END)
        self.task_output.insert("1.0", content.strip() or "(empty response)")
        self._update_task_server_state()

    def _task_fail(self, error):
        self._task_busy = False
        self.task_tps.config(text="ERR", fg=S("danger"))
        self.task_status.config(text=error, fg=S("danger"))
        self._update_task_server_state()

    # ------------------------------------------------------------------
    # Tab: GPU Processes — per-process VRAM/GPU usage with kill controls
    # ------------------------------------------------------------------
    def _build_gpu_procs_tab(self, parent):
        bg, surf, surf_hi = S("bg"), S("surface"), S("surface_hi")
        fg, muted = S("fg"), S("fg_muted")
        ok, warn, danger, info = S("ok"), S("warn"), S("danger"), S("info")

        # --- Header: title + GPU name + live dot ---
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 8))
        gpu_title = ttk.Label(header, text=self.t("gpu_mon_title"),
                              font=self.font_for(self.t("gpu_mon_title"), 16, "bold"),
                              foreground=S("accent"))
        gpu_title.pack(side=tk.LEFT)
        self._reg(gpu_title, "gpu_mon_title", "label")
        self.gpu_mon_name = tk.Label(header, text="Detecting GPU...", bg=surf, fg=muted,
                                     font=self.en_font(9), padx=10, pady=3, bd=0)
        self.gpu_mon_name.pack(side=tk.LEFT, padx=12)
        self.gpu_mon_live = tk.Label(header, text="● LIVE", bg=surf, fg=ok,
                                     font=self.en_font(9, "bold"), padx=10, pady=3, bd=0)
        self.gpu_mon_live.pack(side=tk.RIGHT)

        # --- Summary cards ---
        cards = ttk.Frame(parent)
        cards.pack(fill=tk.X, pady=(0, 8))
        self.gpu_card_vram = self._gpu_card(cards, "VRAM", "0 / 0 GB", S("accent"))
        self.gpu_card_load = self._gpu_card(cards, "GPU LOAD", "0%", info)
        self.gpu_card_temp = self._gpu_card(cards, "TEMPERATURE", "0°C", warn)
        self.gpu_card_power = self._gpu_card(cards, "POWER", "0 / 0 W", S("info"))
        for c in (self.gpu_card_vram, self.gpu_card_load, self.gpu_card_temp, self.gpu_card_power):
            c.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        # --- Sparkline graphs ---
        graphs = ttk.Frame(parent)
        graphs.pack(fill=tk.X, pady=(0, 8))
        self.gpu_graph_vram = self._gpu_graph(graphs, "VRAM (MB)", S("accent"))
        self.gpu_graph_load = self._gpu_graph(graphs, "LOAD (%)", info)
        self.gpu_graph_power = self._gpu_graph(graphs, "POWER (W)", S("accent"))

        # --- System (CPU / RAM / DISK) ---
        sys_head = ttk.Frame(parent)
        sys_head.pack(fill=tk.X, pady=(0, 2))
        sys_title = ttk.Label(sys_head, text=self.t("mon_system"), font=self.en_font(8, "bold"),
                              foreground=muted)
        sys_title.pack(side=tk.LEFT)
        self._reg(sys_title, "mon_system", "label")

        sys_cards = ttk.Frame(parent)
        sys_cards.pack(fill=tk.X, pady=(0, 8))
        self.gpu_card_cpu = self._gpu_sys_card(sys_cards, "mon_cpu", "0%", info, 0.0)
        self.gpu_card_ram = self._gpu_sys_card(sys_cards, "mon_ram", "0 / 0 GB", S("accent"), 0.0)
        self.gpu_card_disk = self._gpu_sys_card(sys_cards, "mon_disk", "0 / 0 GB", warn, 0.0)
        for c in (self.gpu_card_cpu, self.gpu_card_ram, self.gpu_card_disk):
            c.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        graphs_sys = ttk.Frame(parent)
        graphs_sys.pack(fill=tk.X, pady=(0, 8))
        self.gpu_graph_cpu = self._gpu_graph(graphs_sys, "CPU (%)", info)
        self.gpu_graph_ram = self._gpu_graph(graphs_sys, "RAM (%)", S("accent"))
        self.gpu_graph_disk = self._gpu_graph(graphs_sys, "DISK (%)", warn)

        # --- Controls: search + filters ---
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(0, 8))
        search_wrap = tk.Frame(controls, bg=surf, highlightthickness=1, highlightbackground=S("border"))
        search_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.gpu_search_var = tk.StringVar()
        self.gpu_search_entry = ttk.Entry(search_wrap, textvariable=self.gpu_search_var)
        self.gpu_search_entry.pack(fill=tk.X, expand=True, ipady=5, padx=6, pady=4)
        self.gpu_search_var.trace_add("write", lambda *a: self._gpu_apply_rows())
        self.gpu_search_entry.insert(0, self.t("gpu_mon_search_ph"))
        self.gpu_search_entry.bind("<FocusIn>", self._gpu_clear_search)
        self.gpu_search_entry.bind("<FocusOut>", self._gpu_restore_search)

        filters = ttk.Frame(parent)
        filters.pack(fill=tk.X, pady=(0, 8))
        self.gpu_filter_btns = {}
        for mode, key, col in (("all", "gpu_filter_all", fg),
                               ("high_gpu", "gpu_filter_high_gpu", danger),
                               ("high_cpu", "gpu_filter_high_cpu", info),
                               ("high_vram", "gpu_filter_high_vram", warn)):
            btn = tk.Button(filters, text=self.t(key), bd=0, font=self.en_font(9, "bold"),
                            bg=surf, fg=muted, activebackground=surf_hi,
                            activeforeground=f"{col}", padx=12, pady=5, cursor="hand2",
                            command=lambda m=mode: self._gpu_set_filter(m))
            btn.pack(side=tk.LEFT, padx=2)
            self._reg(btn, key, "button")
            self.gpu_filter_btns[mode] = (btn, col)
        self._gpu_set_filter("all")

        # --- Actions ---
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(0, 8))
        self.gpu_pause_btn = self._gpu_button(actions, self.t("gpu_resume_btn"), self._gpu_toggle_refresh, info)
        self.gpu_pause_btn.pack(side=tk.LEFT)
        self._reg(self.gpu_pause_btn, "gpu_resume_btn", "button")
        end_btn = self._gpu_button(actions, self.t("gpu_end_btn"), self._gpu_end_process, ok)
        end_btn.pack(side=tk.LEFT, padx=6)
        self._reg(end_btn, "gpu_end_btn", "button")
        force_btn = self._gpu_button(actions, self.t("gpu_force_btn"), self._gpu_force_process, danger)
        force_btn.pack(side=tk.LEFT)
        self._reg(force_btn, "gpu_force_btn", "button")
        self.gpu_filter_status = ttk.Label(actions, text="", font=self.en_font(9), foreground=muted)
        self.gpu_filter_status.pack(side=tk.LEFT, padx=16)

        # --- Process table ---
        table_box = tk.Frame(parent, bg=surf, highlightthickness=1, highlightbackground=S("border"))
        table_box.pack(fill=tk.BOTH, expand=True)
        columns = ("pid", "process", "vram", "gpu", "ram", "cpu", "disk")
        self.gpu_tree = ttk.Treeview(table_box, columns=columns, show="headings", selectmode="extended")
        hdrs = {"pid": "PID", "process": "Process", "vram": "VRAM", "gpu": "GPU",
                "ram": "RAM", "cpu": "CPU", "disk": "DISK"}
        self._gpu_hdr_keys = {c: k for c, k in zip(columns, ("gpu_col_pid", "gpu_col_process",
                                                             "gpu_col_vram", "gpu_col_gpu",
                                                             "gpu_col_ram", "gpu_col_cpu",
                                                             "gpu_col_disk"))}
        for c in columns:
            self.gpu_tree.heading(c, text=self.t(self._gpu_hdr_keys[c]),
                                  command=lambda cc=c: self._gpu_sort_by(cc))
        self.gpu_tree.column("pid", width=80, anchor=tk.CENTER)
        self.gpu_tree.column("process", width=260, anchor=tk.W)
        self.gpu_tree.column("vram", width=100, anchor=tk.CENTER)
        self.gpu_tree.column("gpu", width=80, anchor=tk.CENTER)
        self.gpu_tree.column("ram", width=110, anchor=tk.CENTER)
        self.gpu_tree.column("cpu", width=80, anchor=tk.CENTER)
        self.gpu_tree.column("disk", width=110, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(table_box, orient=tk.VERTICAL, command=self.gpu_tree.yview)
        hsb = ttk.Scrollbar(table_box, orient=tk.HORIZONTAL, command=self.gpu_tree.xview)
        self.gpu_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.gpu_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.gpu_tree.tag_configure("hot", foreground=S("danger"))

        # --- Status bar ---
        bar = tk.Frame(parent, bg=surf)
        bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(8, 0))
        self.gpu_status_left = tk.Label(bar, text="Ready", bg=surf, fg=muted,
                                        font=self.en_font(9), anchor=tk.W)
        self.gpu_status_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=4)
        self.gpu_status_right = tk.Label(bar, text="", bg=surf, fg=S("accent"), font=self.en_font(9, "bold"))
        self.gpu_status_right.pack(side=tk.RIGHT, padx=10)

        self._gpu_mon_loop()

    def _gpu_card(self, parent, title, value, accent):
        frame = tk.Frame(parent, bg=S("surface"), padx=12, pady=10,
                         highlightthickness=1, highlightbackground=S("border"))
        tk.Label(frame, text=title, font=self.en_font(8, "bold"),
                 fg=S("fg_muted"), bg=S("surface"), anchor=tk.W).pack(anchor=tk.W)
        v = tk.Label(frame, text=value, font=("Consolas", 16, "bold"), fg=accent, bg=S("surface"))
        v.pack(anchor=tk.W, pady=(4, 0))
        frame.v_label = v
        return frame

    def _gpu_sys_card(self, parent, title_key, value, accent, frac=0.0):
        frame = tk.Frame(parent, bg=S("surface"), padx=12, pady=10,
                         highlightthickness=1, highlightbackground=S("border"))
        t = tk.Label(frame, text=self.t(title_key), font=self.en_font(8, "bold"),
                     fg=S("fg_muted"), bg=S("surface"), anchor=tk.W)
        t.pack(anchor=tk.W)
        self._reg(t, title_key, "label")
        v = tk.Label(frame, text=value, font=("Consolas", 15, "bold"), fg=accent, bg=S("surface"))
        v.pack(anchor=tk.W, pady=(4, 2))
        bar = tk.Canvas(frame, bg=S("surface_hi"), height=5, highlightthickness=0)
        bar.pack(fill=tk.X)
        frame.v_label = v
        frame.bar = bar
        return frame

    def _gpu_bar(self, canvas, frac):
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 10:
            return
        fill_w = max(2, int(w * max(0.0, min(frac, 1.0))))
        canvas.create_rectangle(0, 0, fill_w, 5, fill=S("accent"), outline="")

    def _gpu_graph(self, parent, title, accent):
        frame = tk.Frame(parent, bg=S("surface"), highlightthickness=1, highlightbackground=S("border"))
        frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(frame, text=title, font=self.en_font(8, "bold"),
                 fg=S("fg_muted"), bg=S("surface"), anchor=tk.W).pack(fill=tk.X, padx=8, pady=(4, 0))
        cv = tk.Canvas(frame, bg=S("surface"), height=55, highlightthickness=0)
        cv.pack(fill=tk.X, padx=6, pady=(2, 4))
        frame.cv = cv
        frame.accent = accent
        return frame

    def _gpu_button(self, parent, text, command, accent_fg):
        return tk.Button(parent, text=text, command=command, bd=0, cursor="hand2",
                         bg=S("surface_hi"), fg=accent_fg,
                         activebackground=S("surface"), padx=14, pady=6,
                         font=self.en_font(9, "bold"))

    def _gpu_clear_search(self, e=None):
        if self.gpu_search_entry.get() == self.t("gpu_mon_search_ph"):
            self.gpu_search_entry.delete(0, tk.END)

    def _gpu_restore_search(self, e=None):
        if not self.gpu_search_entry.get().strip():
            self.gpu_search_entry.insert(0, self.t("gpu_mon_search_ph"))

    def _gpu_query(self):
        q = self.gpu_search_entry.get().strip().lower()
        return "" if q in ("", self.t("gpu_mon_search_ph").lower()) else q

    def _gpu_set_filter(self, mode):
        self._gpu_filter_mode = mode
        for m, (btn, col) in self.gpu_filter_btns.items():
            active = m == mode
            btn.config(bg=S("surface_hi") if active else S("surface"),
                       fg=col if active else S("fg_muted"))
        self._gpu_apply_rows()

    def _gpu_toggle_refresh(self):
        self._gpu_refresh_enabled = not self._gpu_refresh_enabled
        self.gpu_pause_btn.config(text=self.t("gpu_resume_btn") if self._gpu_refresh_enabled
                                  else self.t("gpu_pause_btn"))
        self._reg(self.gpu_pause_btn, "gpu_resume_btn" if self._gpu_refresh_enabled
                  else "gpu_pause_btn", "button")
        self.gpu_mon_live.config(text="● LIVE" if self._gpu_refresh_enabled else "● PAUSED",
                                 fg=S("ok") if self._gpu_refresh_enabled else S("warn"))
        if self._gpu_refresh_enabled:
            self._gpu_collect()

    def _gpu_collect(self):
        def worker():
            vram, gpu = get_proc_vram_mb(), get_proc_gpu_pct()
            rsc = get_proc_rsc()
            pids = set(vram) | set(gpu)
            for p, s in rsc.items():
                if s["ram"] > 50 * 1048576 or s["cpu"] >= 0.5 or s["io"] >= 0.5 * 1048576:
                    pids.add(p)
            names = get_proc_names(pids)
            rows = [{"pid": p, "name": names.get(p, f"PID {p}"),
                     "vram": round(vram.get(p, 0)), "gpu": round(gpu.get(p, 0), 1),
                     "ram": rsc.get(p, {}).get("ram", 0),
                     "cpu": round(rsc.get(p, {}).get("cpu", 0.0), 1),
                     "io": rsc.get(p, {}).get("io", 0.0)}
                    for p in pids]
            stats = get_gpu_detailed_stats()
            sys_stats = get_system_stats()
            self.root.after(0, lambda: self._gpu_apply_data(rows, stats, sys_stats))
        threading.Thread(target=worker, daemon=True).start()

    def _gpu_apply_data(self, rows, stats, sys_stats=None):
        if not self._gpu_mon_active:
            return
        sys_stats = sys_stats or {}
        self._gpu_rows = rows
        self.gpu_mon_name.config(text=stats["name"])
        self.gpu_card_vram.v_label.config(text=f'{stats["used"] / 1024:.2f} / {stats["total"] / 1024:.2f} GB')
        self.gpu_card_load.v_label.config(text=f'{stats["gpu"]:.0f}%')
        tcol = S("ok") if stats["temp"] < 65 else (S("warn") if stats["temp"] < 80 else S("danger"))
        self.gpu_card_temp.v_label.config(text=f'{stats["temp"]:.0f}°C', fg=tcol)
        self.gpu_card_power.v_label.config(text=f'{stats["power"]:.0f} / {stats["power_limit"]:.0f} W')
        self._gpu_vram_hist.append(stats["used"])
        self._gpu_load_hist.append(stats["gpu"])
        self._gpu_power_hist.append(stats["power"])
        self._gpu_draw(self.gpu_graph_vram, self._gpu_vram_hist, stats["total"])
        self._gpu_draw(self.gpu_graph_load, self._gpu_load_hist, 100)
        # Power graph y-scale falls back to 1 if limit unknown.
        self._gpu_draw(self.gpu_graph_power, self._gpu_power_hist, stats["power_limit"] or 1)
        self._gpu_apply_rows()

        # --- System cards + sparklines ---
        if sys_stats and sys_stats.get("ram_total"):
            cpu_pct = sys_stats.get("cpu", 0.0)
            ram_total = sys_stats["ram_total"]
            ram_pct = sys_stats.get("ram_used", 0.0) / ram_total * 100.0
            disk_total = sys_stats.get("disk_total", 0.0)
            disk_pct = (sys_stats.get("disk_used", 0.0) / disk_total * 100.0) if disk_total else 0.0
            self.gpu_card_cpu.v_label.config(text=f'{cpu_pct:.0f}%')
            self.gpu_card_ram.v_label.config(
                text=f'{sys_stats["ram_used"] / (1024 ** 2):.1f} / {ram_total / (1024 ** 2):.1f} GB')
            self.gpu_card_disk.v_label.config(
                text=f'{sys_stats["disk_used"] / (1024 ** 3):.0f} / {disk_total / (1024 ** 3):.0f} GB')
            self._gpu_bar(self.gpu_card_cpu.bar, cpu_pct / 100.0)
            self._gpu_bar(self.gpu_card_ram.bar, ram_pct / 100.0)
            self._gpu_bar(self.gpu_card_disk.bar, disk_pct / 100.0)
            self._cpu_hist.append(cpu_pct)
            self._ram_hist.append(ram_pct)
            self._disk_hist.append(disk_pct)
            self._gpu_draw(self.gpu_graph_cpu, self._cpu_hist, 100)
            self._gpu_draw(self.gpu_graph_ram, self._ram_hist, 100)
            self._gpu_draw(self.gpu_graph_disk, self._disk_hist, 100)

    def _gpu_draw(self, graph, data, scale_max):
        cv = graph.cv
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 20 or h < 20 or len(data) < 2:
            # Draw a flat baseline so an empty chart still has a frame.
            cv.create_line(0, h - 2, w, h - 2, fill=S("border"))
            return
        mx = max(max(data), scale_max, 1) * 1.1
        pts = []
        n = len(data)
        for i, v in enumerate(data):
            x = (i / (n - 1)) * (w - 4) + 2
            y = h - 2 - (v / mx) * (h - 6)
            pts += [x, y]
        cv.create_line(pts, fill=graph.accent, width=2, smooth=True)

    def _gpu_apply_rows(self):
        if not getattr(self, "gpu_tree", None):
            return
        q = self._gpu_query()
        rows = [r for r in self._gpu_rows
                if q in r["name"].lower() or q in str(r["pid"])]
        if self._gpu_filter_mode == "high_gpu":
            rows = [r for r in rows if r["gpu"] >= 50]
        elif self._gpu_filter_mode == "high_cpu":
            rows = [r for r in rows if r["cpu"] >= 50]
        elif self._gpu_filter_mode == "high_vram":
            rows = [r for r in rows if r["vram"] >= 1024]
        col = self._gpu_sort_column
        fn = {"pid": lambda r: r["pid"], "process": lambda r: r["name"].lower(),
              "vram": lambda r: r["vram"], "gpu": lambda r: r["gpu"],
              "ram": lambda r: r["ram"], "cpu": lambda r: r["cpu"],
              "disk": lambda r: r["io"]}[col]
        rows.sort(key=fn, reverse=self._gpu_sort_reverse)
        selected = set()
        for it in self.gpu_tree.selection():
            vals = self.gpu_tree.item(it, "values")
            if vals:
                try:
                    selected.add(int(vals[0]))
                except ValueError:
                    pass
        self.gpu_tree.delete(*self.gpu_tree.get_children())
        fm = self._gpu_filter_mode
        for r in rows:
            # Red = a different metric is pegged too (ther filter's own metric
            # is already guaranteed, so it is not repainted onto every row).
            g_hot = (r["gpu"] >= 50) if fm != "high_gpu" else False
            c_hot = (r["cpu"] >= 90) if fm != "high_cpu" else False
            tag = "hot" if (g_hot or c_hot) else ""
            it = self.gpu_tree.insert("", "end", values=(
                r["pid"], r["name"], f'{r["vram"]:,} MB', f'{r["gpu"]:.1f}%',
                f'{r["ram"] / 1048576:,.0f} MB', f'{r["cpu"]:.1f}%',
                f'{r["io"] / 1048576:,.1f} MB/s'), tags=(tag,))
            if r["pid"] in selected:
                self.gpu_tree.selection_add(it)
        total = sum(r["vram"] for r in rows)
        total_ram = sum(r["ram"] for r in rows) / 1048576
        self.gpu_status_left.config(
            text=f'{len(rows)} shown (of {len(self._gpu_rows)}) • VRAM: {total:,} MB • RAM: {total_ram:,.0f} MB')
        sel = self.gpu_tree.selection()
        if sel:
            vals = self.gpu_tree.item(sel[0], "values")
            self.gpu_status_right.config(text=f"{vals[1]} — PID {vals[0]}"
                                              f'{f" (+{len(sel)-1})" if len(sel) > 1 else ""}')
        else:
            self.gpu_status_right.config(text="")

    def _gpu_sort_by(self, column):
        if self._gpu_sort_column == column:
            self._gpu_sort_reverse = not self._gpu_sort_reverse
        else:
            self._gpu_sort_column = column
            self._gpu_sort_reverse = True
        self._gpu_apply_rows()

    def _gpu_selected_pids(self):
        pids = []
        for it in self.gpu_tree.selection():
            vals = self.gpu_tree.item(it, "values")
            if vals:
                try:
                    pids.append(int(vals[0]))
                except ValueError:
                    pass
        return pids

    def _gpu_confirmed_kill(self, force):
        pids = self._gpu_selected_pids()
        if not pids:
            messagebox.showwarning(self.t("gpu_col_pid"), self.t("gpu_no_proc"))
            return
        names = [self.gpu_tree.item(it, "values")[1] for it in self.gpu_tree.selection()]
        label = "\n".join(f"• {n} (PID {p})" for n, p in zip(names, pids))
        if not messagebox.askyesno("Force Kill" if force else "End Process",
                                   self.t("gpu_confirm_end") + label):
            return
        ok_all = True
        for pid in pids:
            ok = taskkill_tree(pid, force=force)
            if not ok:
                ok = taskkill_tree(pid, force=True)
            if not ok:
                ok_all = False
        self.gpu_status_left.config(
            text="✔ Closed." if ok_all else "Some processes could not be closed (access denied / already exited).")

    def _gpu_end_process(self):
        self._gpu_confirmed_kill(force=False)

    def _gpu_force_process(self):
        self._gpu_confirmed_kill(force=True)

    def _gpu_mon_loop(self):
        if not self._gpu_mon_active:
            return
        # Skip the (slow) PowerShell/nvidia-smi work when the tab is hidden.
        tab_visible = str(self.gpu_procs_tab) == str(self.notebook.select())
        if self._gpu_refresh_enabled and tab_visible:
            self._gpu_collect()
        self.root.after(1000, self._gpu_mon_loop)

    # ------------------------------------------------------------------
    # New: Model Scanner tab
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Hardware detection / recommendations
    # ------------------------------------------------------------------
    def refresh_hardware(self):
        for lbl in ("cpu_label", "ram_label", "gpu_label"):
            if hasattr(self, lbl):
                getattr(self, lbl).config(text="Detecting...")
        thread = threading.Thread(target=self._detect_hardware_worker, daemon=True)
        thread.start()

    def _detect_hardware_worker(self):
        cpu = HardwareInfo.get_cpu_info()
        ram = HardwareInfo.get_ram_info()
        gpu = HardwareInfo.get_gpu_info()
        self.root.after(0, lambda: self._apply_hardware_results(cpu, ram, gpu))

    def _apply_hardware_results(self, cpu, ram, gpu):
        self.cpu_info, self.ram_info, self.gpu_info = cpu, ram, gpu

        # The hardware labels lived on the now-removed System tab; the info
        # itself is still kept in cpu_info/ram_info/gpu_info for the fit badge.
        if hasattr(self, "cpu_label"):
            self.cpu_label.config(
                text=f"{cpu.get('physical', '?')} physical cores / {cpu.get('logical', '?')} logical threads"
            )

        total = ram.get("total_gb", 0)
        avail = ram.get("available_gb", 0)
        used_pct = ((total - avail) / total * 100) if total else 0
        if hasattr(self, "ram_label"):
            self.ram_label.config(text=f"{avail:.1f} GB available of {total:.1f} GB total ({used_pct:.0f}% in use)")
            self._set_progress(self.ram_bar, used_pct)

        if gpu and hasattr(self, "gpu_label"):
            lines = []
            for g in gpu:
                if g.get("vram_total_mb", 0) > 0:
                    lines.append(f"{g['vendor']}: {g['name']} — "
                                 f"{g['vram_free_mb']/1024:.1f} GB free of {g['vram_total_mb']/1024:.1f} GB VRAM")
                else:
                    lines.append(f"{g['vendor']}: {g['name']} (VRAM not readable)")
            self.gpu_label.config(text="\n".join(lines))
            g0 = gpu[0]
            if g0.get("vram_total_mb", 0) > 0:
                used_pct_vram = (1 - g0["vram_free_mb"] / g0["vram_total_mb"]) * 100
                self._set_progress(self.vram_bar, used_pct_vram)
            else:
                self._set_progress(self.vram_bar, 0)
        elif hasattr(self, "gpu_label"):
            self.gpu_label.config(text="No dedicated GPU detected (CPU-only mode).")
            self._set_progress(self.vram_bar, 0)

        self.update_compatibility()
        self.log_write("Hardware detection refreshed.", "info")

    def update_compatibility(self, force_reanalyze=False):
        model = self.model_var.get().strip()
        size_gb = HardwareInfo.get_model_size_gb(model)
        if size_gb:
            size_prefix = "Model size on disk" if self.lang == "en" else "حجم مدل روی دیسک"
            self.model_size_var.set(f"{size_prefix}: {size_gb:.2f} GB")
        else:
            self.model_size_var.set("No model selected" if self.lang == "en" else "مدلی انتخاب نشده")

        gguf_info = None
        if model and os.path.exists(model):
            if force_reanalyze or getattr(self, "_gguf_cache_path", None) != model:
                gguf_info = get_model_architecture_info(model)
                self._gguf_cache_path = model
                self._gguf_cache_info = gguf_info
            else:
                gguf_info = self._gguf_cache_info

        if gguf_info:
            if self.lang == "en":
                self.model_detail_var.set(
                    f"Architecture: {gguf_info.get('architecture')}  |  "
                    f"Layers: {gguf_info.get('block_count', '?')}  |  "
                    f"Native context: {gguf_info.get('context_length', '?')}  |  "
                    f"Quantization: {gguf_info.get('quantization', '?')}"
                )
            else:
                self.model_detail_var.set(
                    f"معماری: {gguf_info.get('architecture')}  |  "
                    f"لایه‌ها: {gguf_info.get('block_count', '?')}  |  "
                    f"کانتکست اصلی: {gguf_info.get('context_length', '?')}  |  "
                    f"کوانتیزاسیون: {gguf_info.get('quantization', '?')}"
                )
        elif model and os.path.exists(model):
            self.model_detail_var.set(
                "Could not read GGUF metadata for this file (may not be a valid GGUF model)."
                if self.lang == "en" else
                "نتوانستیم متادیتای GGUF این فایل را بخوانیم (ممکن است فایل GGUF معتبر نباشد)."
            )
        else:
            self.model_detail_var.set("")

        recs = HardwareInfo.recommend_settings(size_gb, self.gpu_info, self.ram_info, self.cpu_info, gguf_info)
        self._last_recs = recs

        fit = recs.get("fit", "unknown")
        color = FIT_COLORS.get(fit, S("info"))
        fit_labels = FIT_LABELS if self.lang == "en" else FIT_LABELS_FA
        label = fit_labels.get(fit, "? Unknown")
        self.fit_badge.config(text=label, bg=color, fg=S("bg"))

        note = recs.get("note", "")
        if hasattr(self, "model_fit_label"):
            self.model_fit_label.config(text=note or self.t("model_fit_default"))

    def on_model_changed(self):
        self.update_compatibility()
        self._vision_auto()

    # ------------------------------------------------------------------
    # Vision / mmproj helpers
    # ------------------------------------------------------------------
    def _auto_detect_mmproj(self, silent=True):
        """Auto-fill the mmproj field when exactly one projector lives in
        the model's folder. Returns the chosen path or None. Silent mode
        only logs when a projector is actually auto-selected."""
        model = self.model_var.get().strip()
        if not model or not os.path.exists(model):
            return None
        found = detect_mmproj(model)
        if not found:
            return None
        cur = self.mmproj_path_var.get().strip()
        if cur and os.path.abspath(cur) == os.path.abspath(found):
            return found
        self.mmproj_path_var.set(found)
        self.mmproj_url_var.set("")
        self.log_write(self.t("vision_mmproj_auto").format(name=os.path.basename(found)), "success")
        return found

    def _set_vision_status_text(self, arch, status, mmproj):
        if not hasattr(self, "vision_status_lbl"):
            return
        if status == "vision":
            txt = self.t("vision_yes").format(arch=arch)
            col = S("ok")
        elif status == "possible":
            txt = self.t("vision_possible").format(arch=arch)
            col = S("info")
        elif status == "text":
            txt = self.t("vision_no")
            col = S("warn")
        else:
            txt = self.t("vision_unknown")
            col = S("info")
        if mmproj:
            txt += "  ·  " + self.t("vision_mmproj_auto").format(name=os.path.basename(mmproj))
        else:
            txt += "  ·  " + self.t("vision_no_mmproj")
        self.vision_status_lbl.configure(text=txt, foreground=col)

    def _classify_vision(self, arch, mmproj):
        """Four-way verdict: 'vision' | 'possible' | 'text' | 'unknown'.
        A matching projector next to an *unrecognized* architecture counts
        as 'possible' multimodal — better than wrongly calling it text."""
        if arch in VISION_ARCHITECTURES:
            return "vision"
        if arch in TEXT_ONLY_ARCHS:
            return "text"
        return "possible" if mmproj else "unknown"

    def _vision_auto(self):
        """Recompute the vision status line quietly on model change.
        Uses the metadata cache already filled by update_compatibility."""
        model = self.model_var.get().strip()
        if not hasattr(self, "vision_status_lbl"):
            return
        status, arch, mmproj = "unknown", "unknown", None
        if model and os.path.exists(model):
            mmproj = self._auto_detect_mmproj(silent=True)
            info = getattr(self, "_gguf_cache_info", None) or {}
            arch = str(info.get("architecture") or "unknown").lower()
            status = self._classify_vision(arch, bool(mmproj))
        self._set_vision_status_text(arch, status, mmproj)

    def _refresh_vision_status(self):
        return self._vision_auto()

    def _check_vision(self):
        """🔍 Check Vision button: confirm whether the selected model can
        see images, auto-hook its .mmproj, and explain the situation —
        including the 'why can't ALL models have vision' question."""
        model = self.model_var.get().strip()
        if not model or not os.path.exists(model):
            messagebox.showwarning(self.t("vision_msg_text_title"), self.t("vision_need_model"))
            return
        try:
            info = get_model_architecture_info(model)
            arch = str((info or {}).get("architecture") or "unknown").lower()
        except Exception:
            arch = "unknown"
        mmproj = self._auto_detect_mmproj(silent=False)
        status = self._classify_vision(arch, bool(mmproj))
        self._set_vision_status_text(arch, status, mmproj)
        if status == "vision":
            if mmproj:
                messagebox.showinfo(self.t("vision_msg_vision_title"),
                                    self.t("vision_msg_vision_ok").format(
                                        arch=arch, name=os.path.basename(mmproj)))
            else:
                stem = os.path.splitext(os.path.basename(model))[0]
                messagebox.showwarning(self.t("vision_msg_vision_title"),
                                       self.t("vision_msg_vision_noproj").format(arch=arch, stem=stem))
        elif status == "possible":
            messagebox.showinfo(self.t("vision_msg_possible_title"),
                                self.t("vision_msg_possible_body").format(
                                    arch=arch, name=os.path.basename(mmproj)))
        elif status == "text":
            messagebox.showinfo(self.t("vision_msg_text_title"),
                                self.t("vision_msg_text_body").format(arch=arch))
        else:
            messagebox.showinfo(self.t("vision_msg_unknown_title"),
                                self.t("vision_msg_unknown_body").format(arch=arch))

    def set_reasoning_mode(self, mode):
        self.reasoning_mode.set(mode)
        if mode == "on":
            self.think_on_btn.config(bg=S("ok"), fg=S("bg"), relief=tk.SUNKEN)
            self.think_off_btn.config(bg=S("surface"), fg=S("fg"), relief=tk.RAISED)
        else:
            self.think_off_btn.config(bg=S("danger"), fg=S("bg"), relief=tk.SUNKEN)
            self.think_on_btn.config(bg=S("surface"), fg=S("fg"), relief=tk.RAISED)
        self.update_preview()

    def on_spec_mode_changed(self):
        self._update_spec_controls()

    def _update_spec_controls(self):
        """Unsloth-style conditional layout: the Draft Tokens group and the
        draft KV dtype selector appear inline only when the selected
        strategy actually uses them; otherwise they vanish from the row.
        Within an applicable strategy, Auto keeps llama-server's per-device
        default (MTP/DFlash: 2 on GPU, 3 on CPU/Mac; DSpark: 3), unchecking
        it hands the value to the spinbox. The hint line states exactly
        what will be sent so no control feels dead."""
        mode = self.spec_mode_var.get()
        spec_active = mode not in ("Auto", "Off")
        nmax_applies = spec_active and mode != "Ngram"
        manual_nmax = nmax_applies and not self.spec_draft_auto_var.get()
        has_cache = mode in ("DSpark", "DFlash")

        for w in (self.spec_draft_lbl, self.spec_auto_cb, self.spec_nmax_spin):
            w.grid() if nmax_applies else w.grid_remove()
        for w in (self.spec_cache_lbl, self.spec_cache_combo):
            w.grid() if has_cache else w.grid_remove()

        self.spec_nmax_spin.config(state=tk.NORMAL if manual_nmax else tk.DISABLED)

        if manual_nmax:
            hint_key = "spec_hint_manual"
        elif mode == "Auto":
            hint_key = "spec_hint_auto"
        elif mode == "Off":
            hint_key = "spec_hint_off"
        elif mode == "Ngram":
            hint_key = "spec_hint_ngram"
        else:
            hint_key = "spec_hint_def"
        self.spec_hint_lbl.config(text=self.t(hint_key))
        fg, dim = S("fg"), S("fg_muted")
        self.spec_draft_lbl.config(foreground=fg if nmax_applies else dim)
        self.spec_cache_lbl.config(foreground=fg if has_cache else dim)
        self.spec_mode_lbl.config(foreground=fg if spec_active else dim)
        self.update_preview()

    def apply_spec_settings(self):
        """Explicit Apply for the Speculative Decoding panel: with Auto on
        it defers the draft count to the server default; otherwise clamps
        the value into the valid 1-16 range. Refreshes the command preview
        and confirms the effective flags in the log."""
        mode = self.spec_mode_var.get()
        detail = mode
        if mode in ("MTP", "MTP+Ngram", "DSpark", "DFlash"):
            if self.spec_draft_auto_var.get():
                detail += " — draft tokens: server default (auto)"
            else:
                try:
                    nmax = int(self.spec_draft_nmax_var.get())
                except (ValueError, TypeError, tk.TclError):
                    nmax = 2
                nmax = max(1, min(16, nmax))
                self.spec_draft_nmax_var.set(nmax)
                detail += f" — {nmax} draft tokens"
        if mode in ("DSpark", "DFlash"):
            detail += f", draft KV {self.spec_draft_cache_var.get()}"
        self.update_preview()
        self.log_write(f"Speculative decoding applied: {detail}", "success")

    def on_kv_quant_toggled(self, *args):
        # Quantized KV cache is slower than FP16 without flash attention,
        # so keep the two in sync automatically.
        if self.kv_quant_var.get() and not self.flash_var.get():
            self.flash_var.set(True)
            self.log_write("Enabled Flash Attention automatically (required for fast KV cache quantization).", "info")
        self._update_kv_widgets()

    def _update_kv_widgets(self):
        state = "readonly" if self.kv_quant_var.get() else tk.DISABLED
        for w in (self.ctk_combo, self.ctv_combo):
            w.config(state=state)

    def apply_preset(self):
        name = self.preset_var.get()
        p = PRESETS.get(name)
        if not p:
            return
        self.ngl_var.set(p["ngl"])
        self.ctx_var.set(p["ctx"])
        self.slots_var.set(p["slots"])
        self.threads_var.set(p["threads"])
        self.tbatch_var.set(p["tbatch"])
        self.batch_var.set(p["batch"])
        self.flash_var.set(p["flash"])
        # Old presets carry a boolean "mlock"; map True -> "mlock",
        # False -> "auto". New presets may set "load_mode" directly.
        if "load_mode" in p:
            self.load_mode_var.set(p["load_mode"])
        else:
            self.load_mode_var.set("mlock" if p.get("mlock") else "auto")
        self.unified_var.set(p["unified"])
        self.kv_quant_var.set(p.get("kv_quant", False))
        self.ctk_var.set(p.get("ctk", "q8_0"))
        self.ctv_var.set(p.get("ctv", "q8_0"))
        self.ubatch_var.set(p.get("ubatch", p["batch"]))
        self.defrag_var.set(p.get("defrag", 0.1))
        self.no_mmap_var.set(p.get("no_mmap", False))
        self.no_warmup_var.set(p.get("no_warmup", False))
        # Advanced-performance toggles (optional keys, older presets skip).
        for key, var in (("cont_batching", self.cont_batching_var),
                         ("metrics_endpoint", self.metrics_endpoint_var),
                         ("yarn", self.yarn_var),
                         ("moe_cpu", self.moe_cpu_var),
                         ("moe_cpu_layers", self.moe_cpu_layers_var),
                         ("cache_idle_slots", self.cache_idle_slots_var),
                         ("sleep_idle", self.sleep_idle_var),
                         ("sleep_idle_seconds", self.sleep_idle_seconds_var)):
            if key in p:
                var.set(p[key])
        if "reasoning_format" in p:
            self.reasoning_format_var.set(p["reasoning_format"])
        if p.get("reasoning") in ("on", "off"):
            self.set_reasoning_mode(p["reasoning"])
        if "devices" in p:
            self.devices_var.set(p["devices"])
        if "override_tensor" in p:
            self.override_tensor_var.set(p["override_tensor"])
        # Speculative decoding (optional keys).
        if "spec_mode" in p:
            self.spec_draft_auto_var.set(bool(p.get("spec_draft_auto", True)))
            self.spec_mode_var.set(p["spec_mode"])
            if not self.spec_draft_auto_var.get() and "spec_draft_nmax" in p:
                self.spec_draft_nmax_var.set(max(1, min(16, int(p["spec_draft_nmax"]))))
            if p.get("spec_draft_cache"):
                self.spec_draft_cache_var.set(p["spec_draft_cache"])
            self.on_spec_mode_changed()
        self.log_write(f"Applied preset: {name}", "success")

    def analyze_and_recommend(self):
        """Reads the model's real GGUF metadata (layer count, native
        context, quantization) and fills in GPU layers / threads / context
        with settings tailored to this exact model + your detected
        hardware, instead of generic guesses."""
        model = self.model_var.get().strip()
        if not model or not os.path.exists(model):
            messagebox.showwarning("No model selected", "Please choose a valid GGUF model file first.")
            return
        self.log_write("Analyzing model file...", "info")
        self.update_compatibility(force_reanalyze=True)
        recs = getattr(self, "_last_recs", {})
        if "ngl" in recs:
            self.ngl_var.set(recs["ngl"])
        if "threads" in recs:
            self.threads_var.set(recs["threads"])
        if "tbatch" in recs:
            self.tbatch_var.set(recs["tbatch"])
        if "ctx" in recs:
            self.ctx_var.set(recs["ctx"])
        self.log_write(f"Recommended settings applied: {recs.get('note', '')}", "success")

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------
    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select GGUF Model",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir=r"D:\model"
        )
        if path:
            self.model_var.set(path)

    def browse_mmproj(self):
        path = filedialog.askopenfilename(
            title="Select mmproj projector",
            filetypes=[("Projector files", "*.mmproj"),
                       ("Projector GGUF", "*mmproj*.gguf"),
                       ("GGUF files", "*.gguf"),
                       ("All files", "*.*")],
            initialdir=r"D:\model"
        )
        if path:
            self.mmproj_path_var.set(path)

    def browse_server(self):
        path = filedialog.askopenfilename(
            title="Select llama-server.exe",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            initialdir=r"C:\llmcap"
        )
        if path:
            self.server_var.set(path)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------
    def log_write(self, text, tag="info"):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n", tag)
        if self.auto_scroll_var.get():
            self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def jump_to_latest(self):
        """Scroll the log to the end (used by the ⬇ Latest button)."""
        self.log.see(tk.END)

    def clear_log(self):
        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)

    def filter_log(self):
        """Filter log display based on search text."""
        search = self.log_filter_var.get().lower()
        self.log.config(state=tk.NORMAL)
        if search:
            pos = self.log.search(search, "1.0", tk.END, nocase=True)
            if pos:
                self.log.see(pos)
        self.log.config(state=tk.DISABLED)

    def build_command(self):
        """Build the llama-server command line from current UI settings."""
        cmd = [self.server_var.get(), "-m", self.model_var.get()]
        cmd += ["-ngl", str(self.ngl_var.get())]
        cmd += ["-c", str(self.ctx_var.get())]
        cmd += ["-np", str(self.slots_var.get())]
        cmd += ["-t", str(self.threads_var.get())]
        cmd += ["-tb", str(self.tbatch_var.get())]
        cmd += ["-b", str(self.batch_var.get())]
        cmd += ["-ub", str(self.ubatch_var.get())]
        cmd += ["--defrag-thold", str(self.defrag_var.get())]
        cmd += ["--port", str(self.port_var.get())]
        if self.flash_var.get():
            cmd += ["-fa", "on"]
        load_mode = self.load_mode_var.get().strip()
        if load_mode and load_mode != "auto":
            cmd += ["-lm", load_mode]
        if self.unified_var.get():
            cmd += ["--kv-unified"]
        if self.cache_idle_slots_var.get():
            cmd += ["--cache-idle-slots"]
        if self.kv_quant_var.get():
            # Roughly halves KV cache VRAM use; needs flash attention to be
            # fast rather than a net slowdown (auto-enabled via the checkbox).
            ctk = self.ctk_var.get().strip()
            ctv = self.ctv_var.get().strip()
            if ctk and ctk != "off":
                cmd += ["-ctk", ctk]
            if ctv and ctv != "off":
                cmd += ["-ctv", ctv]
        if self.no_mmap_var.get():
            cmd += ["--no-mmap"]
        if self.no_warmup_var.get():
            cmd += ["--no-warmup"]
        if self.reasoning_mode.get() == "off":
            cmd += ["--reasoning-budget", "0"]
            cmd += ["--chat-template-kwargs", '{"enable_thinking": false}']
            if "--jinja" not in cmd:
                cmd += ["--jinja"]
        reasoning_format = self.reasoning_format_var.get().strip()
        if reasoning_format and reasoning_format != "auto":
            cmd += ["--reasoning-format", reasoning_format]
        # Advanced performance flags
        if self.cont_batching_var.get():
            cmd += ["--cont-batching"]
        if self.metrics_endpoint_var.get():
            cmd += ["--metrics"]
        if self.yarn_var.get():
            cmd += ["--rope-scaling", "yarn"]
        if self.moe_cpu_var.get():
            cmd += ["--n-cpu-moe", str(self.moe_cpu_layers_var.get())]
        if self.sleep_idle_var.get():
            cmd += ["--sleep-idle-seconds", str(self.sleep_idle_seconds_var.get())]
        devices = self.devices_var.get().strip()
        if devices:
            cmd += ["-dev", devices]
        for ot in (t.strip() for t in self.override_tensor_var.get().split(",")):
            if ot:
                cmd += ["-ot", ot]
        # --- Speculative decoding (Unsloth-style panel). --spec-type takes
        # a comma-separated strategy list; Auto = server's smart default. ---
        mode = self.spec_mode_var.get()
        if mode == "Auto":
            # No flag: speculation stays OFF (llama-server default). Emitting
            # --spec-default here added CPU-side ngram drafting that slowed
            # decode on low-core-count rigs — deliberate, matching the plain
            # FASTWORKER command that measured faster.
            pass
        elif mode == "Off":
            cmd += ["--spec-type", "none"]
        else:
            spec_types = {
                "MTP": "draft-mtp",
                "DSpark": "draft-dspark",
                "DFlash": "draft-dflash",
                "Ngram": "ngram-mod",
                "MTP+Ngram": "draft-mtp,ngram-mod",
            }
            stype = spec_types.get(mode)
            if stype:
                cmd += ["--spec-type", stype]
                # Auto keeps the flag off so the server uses its per-device
                # default draft count; manual mode emits the user's value.
                if mode != "Ngram" and not self.spec_draft_auto_var.get():
                    cmd += ["--spec-draft-n-max", str(self.spec_draft_nmax_var.get())]
                if mode in ("DSpark", "DFlash"):
                    cmd += ["--spec-draft-type-k", self.spec_draft_cache_var.get()]
        mmproj_path = self.mmproj_path_var.get().strip()
        mmproj_url = self.mmproj_url_var.get().strip()
        if mmproj_path:
            if mmproj_url:
                self.log_write("Warning: both mmproj file and URL set — using the local file only.", "warn")
            cmd += ["--mmproj", mmproj_path]
        elif mmproj_url:
            cmd += ["--mmproj-url", mmproj_url]
        extra = self.extra_var.get().strip()
        if extra:
            cmd += extra.split()
        return cmd

    def update_preview(self):
        try:
            cmd = self.build_command()
            self.preview_var.set(" ".join(cmd))
        except Exception:
            pass

    def copy_command(self):
        cmd = self.build_command()
        text = " ".join(cmd)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.log_write("Command copied to clipboard.", "success")

    def copy_log(self):
        """Copy all log content to clipboard."""
        self.log.config(state=tk.NORMAL)
        content = self.log.get("1.0", tk.END)
        self.log.config(state=tk.DISABLED)
        if content.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.log_write("Log copied to clipboard.", "success")
        else:
            self.log_write("Log is empty.", "warn")

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------
    def start_server(self):
        if self.process and self.process.poll() is None:
            self.log_write("Server already running!", "warn")
            return
        model = self.model_var.get().strip()
        server = self.server_var.get().strip()
        if not model:
            self.log_write("ERROR: No model selected!", "error")
            messagebox.showerror("Missing model", "Please select a GGUF model file first.")
            return
        if not os.path.exists(model):
            self.log_write(f"ERROR: Model not found: {model}", "error")
            return
        if not os.path.exists(server):
            self.log_write(f"ERROR: Server not found: {server}", "error")
            return

        recs = getattr(self, "_last_recs", None)
        if recs and recs.get("fit") == "risk":
            proceed = messagebox.askyesno(
                "Possible memory issue",
                recs.get("note", "This model may not fit in available memory.") +
                "\n\nStart the server anyway?"
            )
            if not proceed:
                self.log_write("Start cancelled by user due to memory warning.", "warn")
                return

        cmd = self.build_command()
        self.log_write(f"Command: {' '.join(cmd)}", "info")
        self.log_write("Starting server...", "info")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Starting...")

        self.load_start_time = time.time()
        self.model_ready = False
        self.load_status_var.set(self.t("loading_model"))
        self.load_status_label.config(bg=S("warn"), fg=S("bg"))
        self.speed_samples.clear()
        self.token_stats = {"gen": 0, "prompt": 0, "prompt_tps": None}
        self.speed_var.set("⚡ --")
        self.tok_var.set("Σ --")
        self.speed_label.config(bg=S("surface"), fg=S("info"))
        self.open_browser_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self.run_process, args=(cmd,), daemon=True)
        thread.start()
        self.root.after(1000, self._tick_uptime)

    def run_process(self, cmd):
        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True
            )
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                popen_kwargs["startupinfo"] = startupinfo
                # Its own process group lets us taskkill /T the whole tree
                # (llama-server + any helper processes) instead of just
                # the top process, which is what left it running before.
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                popen_kwargs["preexec_fn"] = os.setsid
            self.process = subprocess.Popen(cmd, **popen_kwargs)
            self.root.after(0, lambda: self.status_var.set("Running"))
            self.root.after(0, lambda: self.log_write("Server started!", "success"))

            # Log parsing for "ready" text varies a lot between llama.cpp
            # builds and is easy to miss, so the real signal is polling
            # the server's own /health endpoint - that's the same check
            # llama-server uses internally, so it can't get out of sync.
            health_thread = threading.Thread(
                target=self._poll_health, args=(self.port_var.get(),), daemon=True
            )
            health_thread.start()

            for line in self.process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tag = "info"
                lower = line.lower()
                if "error" in lower or "fail" in lower:
                    tag = "error"
                elif "warn" in lower or "deprecated" in lower:
                    tag = "warn"
                elif "loaded" in lower or "listening" in lower:
                    tag = "success"
                self.root.after(0, lambda l=line, t=tag: self.log_write(l, t))

                # --- Detect "model finished loading / server ready" ---
                if not self.model_ready and re.search(
                    r"(server is listening|all slots are idle|starting the main loop|http server listening)",
                    lower
                ):
                    self.model_ready = True
                    elapsed = time.time() - self.load_start_time if self.load_start_time else 0
                    self.root.after(0, lambda e=elapsed: self._mark_model_loaded(e))

                # --- Token accounting: llama.cpp prints one eval-time line
                #     for prompt processing ("prompt eval time ...") and a
                #     separate one for generation ("eval time ..."). Track
                #     them apart so the counter shows true generated-token
                #     totals and the speed badge never mixes prompt tok/s
                #     with decode tok/s. ---
                if "eval time" in lower:
                    is_prompt = "prompt" in lower
                    # Newer builds count generation in "runs", older ones
                    # in "tokens" — accept both.
                    m_tok = re.search(r"/\s*(\d+)\s*(?:tokens|runs)", lower)
                    m_tps = re.search(r"([\d.]+)\s*tokens per second", lower)
                    if m_tok:
                        n = int(m_tok.group(1))
                        self.root.after(0, lambda n=n, ip=is_prompt: self._account_tokens(n, ip))
                    if m_tps:
                        tps = float(m_tps.group(1))
                        if is_prompt:
                            self.root.after(0, lambda t=tps: self._update_prompt_speed(t))
                        else:
                            self.root.after(0, lambda t=tps: self._update_speed(t))
            self.process.wait()
            code = self.process.returncode
            self.root.after(0, lambda: self.log_write(f"Server exited (code {code})", "warn" if code == 0 else "error"))
        except Exception as e:
            self.root.after(0, lambda: self.log_write(f"ERROR: {e}", "error"))
        finally:
            self.root.after(0, self.on_process_end)

    def _poll_health(self, port):
        """Polls llama-server's /health endpoint until it responds OK, the
        process dies, or the user stops it. This is the reliable fallback
        (and usually the first to fire) for detecting 'model loaded'."""
        url = f"http://127.0.0.1:{port}/health"
        while True:
            if self.process is None or self.process.poll() is not None:
                return
            if self.model_ready:
                return
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if resp.status == 200:
                        self.model_ready = True
                        elapsed = time.time() - self.load_start_time if self.load_start_time else 0
                        self.root.after(0, lambda e=elapsed: self._mark_model_loaded(e))
                        return
            except Exception:
                pass
            time.sleep(1)

    def _mark_model_loaded(self, elapsed_seconds):
        self.load_status_var.set(f"{self.t('model_loaded_prefix')} {elapsed_seconds:.1f}{self.t('model_loaded_suffix')}")
        self.load_status_label.config(bg=S("ok"), fg=S("bg"))
        self.log_write(f"Model finished loading in {elapsed_seconds:.1f}s.", "success")
        self.open_browser_btn.config(state=tk.NORMAL)

    def _fmt_tok(self, n):
        """1234 -> '1.2K', 356 -> '356'."""
        return f"{n/1000:.1f}K" if n >= 10000 else str(n)

    def _account_tokens(self, n, is_prompt):
        if is_prompt:
            self.token_stats["prompt"] += n
        else:
            self.token_stats["gen"] += n
        self._render_speed()

    def _update_prompt_speed(self, tps):
        self.token_stats["prompt_tps"] = tps
        self._render_speed()

    def _update_speed(self, tps):
        self.speed_samples.append(tps)
        self._render_speed()

    def _render_speed(self):
        """Compose the speed badge: current decode speed, rolling average,
        and session totals for generated vs prompt tokens."""
        st = self.token_stats
        seg = []
        if self.speed_samples:
            last = self.speed_samples[-1]
            avg = sum(self.speed_samples) / len(self.speed_samples)
            seg.append(f"{last:.1f} tok/s")
            seg.append(f"{self.t('speed_avg_label')} {len(self.speed_samples)}: {avg:.1f}")
        elif st.get("prompt_tps"):
            seg.append(f"{self.t('speed_prm')} {st['prompt_tps']:.0f} tok/s")
        if not (self.speed_samples or st["gen"] or st["prompt"] or st.get("prompt_tps")):
            # Nothing measured yet this session.
            self.speed_var.set("⚡ --")
            self.tok_var.set("Σ --")
            return
        seg.append(f"{self._fmt_tok(st['gen'])} {self.t('speed_gen')}")
        seg.append(f"{self._fmt_tok(st['prompt'])} {self.t('speed_prm')}")
        self.speed_var.set("⚡ " + "  ·  ".join(seg))
        self.tok_var.set(f"Σ {self._fmt_tok(st['gen'])} {self.t('speed_gen')} · "
                          f"{self._fmt_tok(st['prompt'])} {self.t('speed_prm')}")
        self.speed_label.config(bg=S("info"), fg=S("bg"))

    def open_in_browser(self):
        url = f"http://localhost:{self.port_var.get()}"
        webbrowser.open(url)
        self.log_write(f"Opening {url} in your browser...", "info")

    def _tick_uptime(self):
        if self.process and self.process.poll() is None and self.load_start_time:
            elapsed = int(time.time() - self.load_start_time)
            mm, ss = divmod(elapsed, 60)
            hh, mm = divmod(mm, 60)
            self.uptime_var.set(f"{self.t('uptime_prefix')} {hh:02d}:{mm:02d}:{ss:02d}")
            self.root.after(1000, self._tick_uptime)
        else:
            self.uptime_var.set("")

    def on_process_end(self):
        self.process = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self.model_ready = False
        self.load_status_var.set("Model not loaded")
        self.load_status_label.config(bg=S("surface"), fg=S("fg"))
        self.open_browser_btn.config(state=tk.DISABLED)
        self.uptime_var.set("")

    def _kill_process_tree(self):
        """Forcefully kill the server and any child processes it spawned.
        Plain .terminate() only signals the top process, which is why the
        server could keep running after the app closed - this kills the
        whole tree instead."""
        proc = self.process
        if not proc or proc.poll() is not None:
            return
        pid = proc.pid
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
                )
            else:
                import signal
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def stop_server(self):
        if self.process and self.process.poll() is None:
            self.log_write("Stopping server (killing full process tree)...", "warn")
            self._kill_process_tree()
        else:
            self.log_write("No server running", "warn")

    def on_close(self):
        self._gpu_mon_active = False
        self._kill_process_tree()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LlamaRunner(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
