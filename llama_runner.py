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
import http.client
import socket
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
    "violet":        "#a78bfa",  # violet — gradient end / accents
    "ok_dim":        "#15803d",  # dim green — pulse "off" phase
    "accent_dim":    "#0f766e",  # dim teal — pulse "off" phase
    "info_dim":      "#0369a1",  # dim sky — pulse "off" phase
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
# Hand-tuned presets, dialed in for RTX 3060 Ti (8GB VRAM) +
# 16GB DDR4 + Intel i3 12th gen (4C/8T) to push close to full utilization
# without running out of VRAM/RAM on typical 7B-13B GGUF models.
# ---------------------------------------------------------------------------
PRESETS = {
    "🔴 THE GOD (Unsloth × GOD MODE — all layers)": {
        # Unsloth Studio + GOD MODE fused into one: Unsloth's 32K ctx,
        # 4 slots, jinja/reasoning extras + GOD MODE's 4K/4K batches,
        # mlock, q8_0 KV, defrag, no-warmup speed knobs — and ALL layers
        # on the GPU (ngl 99) for the full offload.
        "ngl": 99, "ctx": 32640, "slots": 4, "port": 8080,
        "threads": 8, "tbatch": 8, "batch": 4096, "ubatch": 4096,
        "flash": True, "load_mode": "mlock", "unified": True,
        "kv_quant": True, "ctk": "q8_0", "ctv": "q8_0",
        "defrag": 0.1, "no_warmup": True,
        "cont_batching": True, "metrics_endpoint": True, "yarn": False,
        "moe_cpu": False, "moe_cpu_layers": 13,
        "cache_idle_slots": False, "sleep_idle": False, "sleep_idle_seconds": 600,
        "reasoning_format": "auto", "reasoning": "on",
        "spec_mode": "Auto",
        "extra": ("--no-context-shift --fit off "
                  "--slot-save-path C:\\Users\\asus\\.unsloth\\studio\\cache\\llama-slots "
                  "--jinja --spec-default"),
    },
    "🚀 MAX TPS+ (peg GPU & CPU)": {
        # Built for one number: highest sustained tok/s on the RTX 3060 Ti
        # / 16GB / i3-12th rig with every core and the whole GPU busy.
        # Decode is GPU-bound → full offload + mlock (no page-in hitches)
        # + q8_0 KV with flash attention (smaller attention footprint).
        # Prompt eval is the other half of "speed" → 4K/4K batches and all
        # 8 logical threads on -t/-tb so the CPU side never waits. 4K
        # context keeps the attention span per token short (attention cost
        # grows with sequence length). Reasoning OFF so thinking models
        # don't burn the measured window on hidden CoT; sleep/idle caching
        # off so nothing ever pauses mid-benchmark. Spec Auto = no flag
        # (measured fastest on this CPU; flip to MTP/Ngram in the panel
        # if your model likes it).
        "ngl": 99, "ctx": 4096, "slots": 1, "threads": 8, "tbatch": 8,
        "batch": 4096, "ubatch": 4096, "defrag": 0.1,
        "flash": True, "load_mode": "mlock", "unified": True,
        "kv_quant": True, "ctk": "q8_0", "ctv": "q8_0",
        "no_warmup": True,
        "cont_batching": True, "metrics_endpoint": False, "yarn": False,
        "moe_cpu": False, "moe_cpu_layers": 13,
        "cache_idle_slots": False, "sleep_idle": False, "sleep_idle_seconds": 600,
        "reasoning_format": "auto", "reasoning": "off",
        "spec_mode": "Auto",
    },
    "🧪 Unsloth Studio (exact launch flags)": {
        # Mirrors the live Unsloth llama.cpp command, minus -m / --alias
        # (those come from the selected model). --slot-save-path, --jinja and
        # --spec-default ride in "extra": build_command() emits no flag for
        # Spec "Auto" or reasoning-ON, so extra supplies them verbatim.
        # defrag -1 = disabled (the launch command carries no --defrag-thold),
        # load_mode auto = no --load-mode flag, kv_quant off = no -ctk/-ctv.
        "ngl": 48, "ctx": 32640, "slots": 4, "port": 8080,
        "threads": 8, "tbatch": 8, "batch": 2048, "ubatch": 512,
        "flash": True, "load_mode": "auto", "unified": True, "kv_quant": False,
        "defrag": -1.0, "no_warmup": False,
        "cont_batching": False, "metrics_endpoint": True, "yarn": False,
        "moe_cpu": False, "moe_cpu_layers": 13,
        "cache_idle_slots": False, "sleep_idle": False, "sleep_idle_seconds": 600,
        "reasoning_format": "auto", "reasoning": "on",
        "spec_mode": "Auto",
        "extra": ("--no-context-shift --video-fps 1 --fit off "
                  "--slot-save-path C:\\Users\\asus\\.unsloth\\studio\\cache\\llama-slots "
                  "--jinja --spec-default"),
    },
    "🌟 GOD MODE (RTX 3060 Ti / 16GB / i3-12th — MAX everything)": {
        "ngl": 99, "ctx": 16384, "slots": 1, "threads": 8, "tbatch": 8,
        "batch": 4096, "ubatch": 4096, "flash": True, "load_mode": "mlock", "unified": False, "kv_quant": True,
        "no_warmup": True, "defrag": 0.1,
        "cont_batching": True, "metrics_endpoint": True, "yarn": False, "moe_cpu": False, "moe_cpu_layers": 10,
    },
    "⚡ Max (RTX 3060 Ti / 16GB — safe full offload)": {
        "ngl": 99, "ctx": 8192, "slots": 1, "threads": 4, "tbatch": 6,
        "batch": 2048, "ubatch": 2048, "flash": True, "mlock": True, "unified": False, "kv_quant": True,
        "cont_batching": True, "metrics_endpoint": False, "yarn": False, "moe_cpu": False, "moe_cpu_layers": 10,
    },
    "Low VRAM / Safe": {
        "ngl": 20, "ctx": 4096, "slots": 1, "threads": 4, "tbatch": 4,
        "batch": 512, "flash": True, "mlock": False, "unified": False,
        "cont_batching": False, "metrics_endpoint": False, "yarn": False, "moe_cpu": False, "moe_cpu_layers": 10,
    },
    # --- Benchmarked speed presets (llama-bench build 11065, this rig,
    #     2026-09-25, Qwen3.5-9B Q4_K_M): Baseline tg128 = 67.0 tok/s /
    #     pp512 = 2146. GGML_CUDA_GRAPH_OPT=1 = 66.5 (-0.6%, re-test any
    #     time). -bs --samplers on the real server = 64.2 vs 63.3 (+1.3%).
    #     The first two are SMALL/overlay presets ("partial": True): they
    #     only add their delta on top of the preset currently in use.
    #     These three also live in the ⚡ Speed list right of the preset
    #     selector.
    "📊 Baseline (-fa on -t 4)": {
        # Small overlay: force the tested baseline flags (-fa on, -t 4,
        # -tb 6) and turn the CUDA-stream env flag OFF, while keeping the
        # current preset's ctx / batches / extras untouched.
        "partial": True,
        "only": ["flash", "threads", "tbatch", "graph_opt"],
        "flash": True, "threads": 4, "tbatch": 6, "graph_opt": False,
    },
    "⚡ GGML_CUDA_GRAPH_OPT=1": {
        # Small overlay: only the concurrent Q/K/V CUDA-stream env flag,
        # added on top of whatever preset is in use — clean A/B against
        # the current settings. Applied to the server on next start.
        "partial": True,
        "only": ["graph_opt"],
        "graph_opt": True,
    },
    "🏆 MiMo Ultra (16K)": {
        # Full preset: everything measured FASTEST on this rig, at 16K
        # context — -fa on, -t 4/-tb 6, -ub 2048 (PP), q8_0 KV so 16K
        # still fits the 8GB card, mlock, defrag 0.1, single slot (CUDA
        # graphs stay on), -bs + backend sampler chain (+1.3% measured),
        # --cache-reuse 256. GGML_CUDA_GRAPH_OPT stays OFF because it
        # measured -0.6% slower here — flip it on via the ⚡ list to A/B.
        "ngl": 99, "ctx": 16384, "slots": 1, "port": 8080,
        "threads": 4, "tbatch": 6, "batch": 2048, "ubatch": 2048,
        "flash": True, "load_mode": "mlock", "unified": False,
        "kv_quant": True, "ctk": "q8_0", "ctv": "q8_0",
        "defrag": 0.1, "no_warmup": False,
        "cont_batching": True, "metrics_endpoint": False, "yarn": False,
        "moe_cpu": False, "moe_cpu_layers": 10,
        "cache_idle_slots": False, "sleep_idle": False, "sleep_idle_seconds": 600,
        "spec_mode": "Auto", "graph_opt": False,
        "extra": "-bs --samplers top_k;temperature --cache-reuse 256",
    },
}
# The three presets surfaced in their own ⚡ Speed list, packed right of
# the main preset selector and applied instantly on selection.
SPEED_PRESETS = [
    "📊 Baseline (-fa on -t 4)",
    "⚡ GGML_CUDA_GRAPH_OPT=1",
    "🏆 MiMo Ultra (16K)",
]
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
    "tab_inspector": {"en": "  Task Inspector  ", "fa": "  بازرس کار  "},

    # --- Task Inspector tab ---
    "insp_graph_title": {"en": "⚡ Decode speed across the whole generation",
                         "fa": "⚡ سرعت تولید در کل پاسخ"},
    "insp_graph_hint": {"en": "tok/s vs token index — one point per streamed chunk",
                        "fa": "tok/s بر اساس شماره توکن — یک نقطه به ازای هر قطعه"},
    "insp_no_data": {"en": "No task run yet — run a task to see its trace here.",
                     "fa": "هنوز کاری اجرا نشده — برای دیدن ردپا، یک کار اجرا کنید."},
    "insp_stats": {"en": "tokens {} · avg {:.1f} · peak {:.1f} tok/s · {:.1f}s",
                   "fa": "توکن {} · میانگین {:.1f} · اوج {:.1f} tok/s · {:.1f}ث"},
    "insp_system": {"en": "System Prompt (notes only — never sent to the model)",
                    "fa": "پرامپت سیستم (قبل از پرامپت شما درج می‌شود)"},
    "insp_system_ph": {"en": "Optional. e.g. You are a helpful assistant.",
                       "fa": "اختیاری. مثلاً: تو یک دستیار مفید هستی."},
    "insp_prompt": {"en": "Prompt (what you sent)", "fa": "پرامپت (آنچه فرستادید)"},
    "insp_thinking": {"en": "Thinking / Reasoning", "fa": "تفکر / استدلال"},
    "insp_output": {"en": "Output (final answer)", "fa": "خروجی (پاسخ نهایی)"},
    "insp_empty": {"en": "(empty)", "fa": "(خالی)"},
    "insp_clear": {"en": "🧹 Clear trace", "fa": "🧹 پاک کردن ردپا"},
    "insp_copied": {"en": "Trace copied to clipboard.", "fa": "ردپا در کلیپ‌بورد کپی شد."},
    "insp_copy": {"en": "📋 Copy trace", "fa": "📋 کپی ردپا"},
    "insp_system_tip": {
        "en": "Optional system prompt. When set, Run Task prepends it to the "
              "user prompt (llama-server /completion takes a single prompt, so "
              "the system text is merged in). Shown here for transparency.",
        "fa": "پرامپت سیستم اختیاری. اگر پر شود، «اجرای کار» آن را قبل از پرامپت "
              "شما می‌چسباند (چون /completion فقط یک prompt می‌گیرد).",
    },
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
    "task_kill": {"en": "⏹ Kill Task", "fa": "⏹ لغو کار"},
    "task_kill_tip": {
        "en": "Cancel ONLY the generation currently running — the server "
              "process stays up and keeps serving. Works by dropping this "
              "request's connection, which llama-server treats as a cancel "
              "and frees the slot immediately (verified live).",
        "fa": "فقط همین تولید در حال اجرا را لغو کن — خود پردازش سرور بالا "
              "می‌ماند و به کار ادامه می‌دهد. با قطع اتصال همین درخواست "
              "انجام می‌شود که سرور آن را لغو تلقی کرده و اسلات را فوراً "
              "آزاد می‌کند (روی سرور واقعی آزمایش شده).",
    },
    "task_cancelled": {"en": "Cancelled — server still running",
                       "fa": "لغو شد — سرور همچنان در حال اجرا"},
    "task_no_idle": {"en": "No task running — nothing to cancel.",
                     "fa": "کاری در حال اجرا نیست — چیزی لغو نشد."},
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
    # Auto-scanning server-location dropdown (rescans C:\llmcap on open).
    "srv_preset_def": {"en": "default", "fa": "پیش‌فرض"},
    "srv_preset_tip": {"en": "Server locations found under C:\\llmcap — rescanned every time the list opens, so a new folder shows up automatically.",
                       "fa": "مسیرهای سرور در C:\\llmcap — با هر باز شدن لیست دوباره جستجو می‌شود؛ پوشهٔ جدید خودکار اضافه می‌شود."},
    "server_tip": {"en": "Path to llama-server.exe (or llama-server on Linux/Mac).",
                   "fa": "مسیر فایل llama-server.exe (یا llama-server در لینوکس/مک)."},
    "analyze_btn": {"en": "🔍 Recommend Settings For This Model",
                     "fa": "🔍 پیشنهاد تنظیمات برای این مدل"},

    "params_section": {"en": "🎛 Parameters", "fa": "🎛 پارامترها"},
    "preset_label": {"en": "Preset:", "fa": "پیش‌تنظیم:"},
    "preset_tip": {"en": "Ready-made setting bundles for common hardware.",
                   "fa": "مجموعه تنظیمات آماده برای سخت‌افزارهای رایج."},
    "apply_preset_btn": {"en": "Apply Preset", "fa": "اعمال پیش‌تنظیم"},
    "speed_preset_label": {"en": "⚡ Speed:", "fa": "⚡ سرعت:"},
    "speed_preset_tip": {
        "en": "Applies instantly. Small presets ADD to the preset currently "
              "in use (Baseline = -fa on -t 4, GGML_CUDA_GRAPH_OPT=1 = "
              "env flag only); MiMo Ultra is a full 16K max-speed preset.",
        "fa": "بلافاصله اعمال می‌شود. پیش‌تنظیم‌های کوچک به پیش‌تنظیم فعلی "
              "اضافه می‌شوند (پایه = -fa on -t 4، GGML_CUDA_GRAPH_OPT=1 = "
              "فقط کلید env)؛ MiMo Ultra یک پیش‌تنظیم کامل ۱۶K است.",
    },
    "graph_opt_label": {"en": "⚡ CUDA streams", "fa": "⚡ جریان‌های CUDA"},
    "graph_opt_tip": {
        "en": "GGML_CUDA_GRAPH_OPT=1 — concurrent Q/K/V CUDA streams, applied "
              "only when the server next starts. Measured on this rig: 66.5 vs "
              "67.0 tok/s (-0.6%), so leave it off unless a new build/driver "
              "changes that.",
        "fa": "GGML_CUDA_GRAPH_OPT=1 — جریان‌های هم‌زمان Q/K/V؛ فقط در اجرای "
              "بعدی سرور اعمال می‌شود. نتیجه تست روی همین سیستم: ۶۶٫۵ در مقابل "
              "۶۷٫۰ (۰٫۶-٪)، پس خاموش بماند مگر بیلد/درایور جدید چیز دیگری "
              "نشان دهد.",
    },

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

    "temp_label": {"en": "Temperature:", "fa": "دما:"},
    "temp_tip": {
        "en": "--temp. Server-side sampling temperature (0.0-2.0); llama.cpp's "
              "default is 0.8. Lower = more deterministic, higher = more "
              "creative. This is the server-wide default — the Run Task tab "
              "has its own temperature spinbox which overrides it per request.",
        "fa": "--temp. دمای نمونه‌گیری سمت سرور (۰.۰ تا ۲.۰)؛ پیش‌فرض llama.cpp "
              "برابر 0.8 است. مقادیر پایین‌تر = پاسخ قطعی‌تر، بالاتر = "
              "خلاقانه‌تر. این مقدار پیش‌فرض سراسری سرور است — تب Run Task "
              "دکمهٔ دمای خودش را دارد که روی هر درخواست بازنویسی می‌کند.",
    },

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
    # Mini dih.py speed test, lives inside Advanced Performance.
    "bench_btn": {"en": "⏱ Speed Test", "fa": "⏱ سرعت‌سنج"},
    "bench_tip": {
        "en": "Mini benchmark of the currently loaded model (mini dih.py): streams a fixed "
              "prompt and reports TTFT + tok/s + prompt speed. Server must be running.",
        "fa": "بنچمارک کوچک مدل بارگذاری‌شده (نسخه کوچک dih.py): یک پرامپت ثابت را استریم می‌کند "
              "و زمان اولین توکن + سرعت tok/s را نشان می‌دهد. سرور باید در حال اجرا باشد.",
    },
    "bench_idle": {"en": "no test yet", "fa": "هنوز تستی نشده"},
    "bench_running": {"en": "testing the current model…", "fa": "در حال تست مدل فعلی..."},
    "bench_not_ready": {"en": "server not ready — start it first", "fa": "سرور آماده نیست — اول اجرا کنید"},
    "bench_task_busy": {"en": "a Run Task is in progress — wait for it", "fa": "یک کار در حال اجراست — صبر کنید"},
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
    "mmproj_section": {"en": "👁 Multimodal & MTP", "fa": "👁 چندرسانه‌ای و MTP"},
    "mmproj_desc": {
        "en": "Two model add-ons, each with its own ON/OFF switch: an mmproj "
              "projector so vision models can see images, and a separate MTP "
              "draft model (e.g. mtp-gemma-4-…-Q4_0.gguf) for multi-token "
              "speculation. Paths are remembered when switched off — only the "
              "flag stops being sent.",
        "fa": "دو افزونهٔ مدل، هر کدام با کلید روشن/خاموش خودش: پروجکتور mmproj "
              "برای دیدن تصاویر، و یک مدل MTP جداگانه (مثلاً mtp-gemma-4-…-Q4_0.gguf) "
              "برای پیش‌بینی چندتوکنی. مسیرها موقع خاموش‌شدن حفظ می‌شوند — فقط "
              "پرچم ارسال نمی‌شود.",
    },
    "mmproj_file_label": {"en": "mmproj file:", "fa": "فایل mmproj:"},
    "mmproj_file_tip": {
        "en": "Local .mmproj multimodal projector file (--mmproj). Needed for vision models.",
        "fa": "فایل پروجکتور چندرسانه‌ای محلی .mmproj ‏(--mmproj). برای مدل‌های بینایی لازم است.",
    },
    "mtp_file_label": {"en": "MTP model:", "fa": "مدل MTP:"},
    "mtp_file_tip": {
        "en": "Separate MTP draft GGUF (--spec-draft-model), e.g. "
              "mtp-gemma-4-12B-it-Q4_0.gguf — keep it next to the main model. "
              "When the toggle is ON the server loads it and forces "
              "--spec-type draft-mtp for multi-token prediction speculation.",
        "fa": "فایل MTP جداگانه ‏(--spec-draft-model)، مثلاً "
              "mtp-gemma-4-12B-it-Q4_0.gguf — کنار مدل اصلی بگذارید. وقتی کلید "
              "روشن باشد سرور آن را لود کرده و ‏--spec-type draft-mtp را برای "
              "پیش‌بینی چندتوکنی اجباری می‌کند.",
    },
    "mmproj_toggle_tip": {
        "en": "ON = send --mmproj at startup. OFF = keep the path but send no flag.",
        "fa": "روشن = ‏--mmproj هنگام شروع ارسال شود. خاموش = مسیر حفظ شود ولی پرچمی ارسال نشود.",
    },
    "mtp_toggle_tip": {
        "en": "ON = load this MTP draft model (--spec-draft-model + --spec-type draft-mtp). "
              "OFF = no speculation, server runs plain.",
        "fa": "روشن = این مدل MTP لود شود ‏(--spec-draft-model + --spec-type draft-mtp). "
              "خاموش = بدون speculation، سرور ساده اجرا شود.",
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
    "ctx_pill_idle": {"en": "Ctx --", "fa": "کانتکست --"},
    "ctx_pill_na": {"en": "Ctx n/a", "fa": "کانتکست در دسترس نیست"},
    "ctx_pill_free": {"en": "free", "fa": "آزاد"},
    "ctx_pill_tip": {
        "en": "Accurate live context usage: each slot's occupancy "
              "(n_prompt_tokens from llama-server's /slots — prompt tokens "
              "plus generation, summed across all slots) against the -c "
              "total captured at launch. 'free' is how many context tokens "
              "remain before the server must start truncating. Falls back "
              "to the server log (last prompt + generation) if /slots "
              "doesn't report occupancy.",
        "fa": "مصرف دقیق و لحظه‌ای کانتکست: اشغال‌شدگی هر اسلات "
              "(n_prompt_tokens از ‏/slots سرور — توکن‌های پرامپت به‌علاوه "
              "تولید، مجموع‌شده روی همه اسلات‌ها) در برابر مقدار کل ‑c "
              "زمان اجرا. «آزاد» یعنی چند توکن تا پر شدن کانتکست مانده است. "
              "اگر /slots اطلاعی نداد، از لاگ سرور (آخرین پرامپت + تولید) "
              "استفاده می‌شود.",
    },
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
        # Live context pill: -c captured at launch + last /slots reading.
        self._run_ctx_total = 0
        self._last_ctx = None
        self._ctx_slots_fresh = False   # /slots has reported real occupancy
        self._ctx_log_est = 0           # stdout eval-line fallback estimate
        self._pending_prompt_n = 0
        # Run Task cancellation: event + the live HTTP connection whose
        # socket shutdown makes llama-server cancel just that generation.
        self._task_cancel_evt = None
        self._task_conn = None
        # Task Inspector: last-run trace + streamed decode-speed series.
        # series = [(token_index, tok_per_s), ...]; reset per run.
        self.insp_series = []
        self.insp_t0 = None
        self._insp_last_flush = 0.0
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
        # Re-registration above may have reset the server-state labels to
        # the static "not running" text — recompute from live health.
        self._update_task_server_state()
        if not self.process:
            self.load_status_var.set(self.t("load_status_idle"))
            self.load_status_label.config(font=self.fa_font(10, "bold") if self.lang == "fa" else self.en_font(10, "bold"))
            self.status_var.set(self.t("status_ready"))
            if getattr(self, "ctx_pill_var", None):
                self.ctx_pill_var.set(self.t("ctx_pill_idle"))
                self._ctx_pill_color(S("fg_muted"))
        else:
            self.load_status_label.config(font=self.fa_font(10, "bold") if self.lang == "fa" else self.en_font(10, "bold"))
            if getattr(self, "ctx_pill_var", None) and getattr(self, "_last_ctx", None):
                self._render_ctx(*self._last_ctx)

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

    # ------------------------------------------------------------------
    # Lightweight UI motion: one shared pulse heartbeat + gradient strip.
    # Everything here is event-driven — when the app is idle nothing is
    # scheduled, so CPU/GPU cost is effectively zero at rest.
    # ------------------------------------------------------------------
    @staticmethod
    def _hex(color):
        color = str(color).lstrip("#")
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def _draw_gradient(self, event=None):
        """Paint the teal→sky→violet accent strip. Runs only at build
        time and on window resize — never on a timer."""
        c = getattr(self, "_grad_canvas", None)
        if c is None:
            return
        w = c.winfo_width()
        if w < 8:
            return
        c.delete("g")
        stops = [self._hex(S("accent")), self._hex(S("info")),
                 self._hex(S("violet"))]
        step = max(2, w // 60)
        for x in range(0, w, step):
            t = x / max(1, w - 1)
            if t < 0.5:
                a, b, k = stops[0], stops[1], t / 0.5
            else:
                a, b, k = stops[1], stops[2], (t - 0.5) / 0.5
            col = "#%02x%02x%02x" % tuple(
                int(a[i] + (b[i] - a[i]) * k) for i in range(3))
            c.create_rectangle(x, 0, min(x + step, w), 4,
                               fill=col, outline=col, tags="g")

    def _ensure_pulse(self):
        """Arm the shared 700ms pulse if it isn't already pending."""
        if not getattr(self, "_pulse_pending", False):
            self._pulse_pending = True
            self.root.after(700, self._pulse_tick)

    def _pulse_tick(self):
        """Single heartbeat for every live indicator: server status pill,
        Run Task server badge, generating status + big tok/s number, and
        the GPU LIVE badge — they breathe between bright and dim palette
        colors. Re-arms ONLY while something is active; when all are idle
        the loop ends and the UI costs nothing."""
        self._pulse_pending = False
        phase = not getattr(self, "_pulse_phase", False)
        self._pulse_phase = phase
        running = bool((self.process and self.process.poll() is None)
                       or getattr(self, "_health_probe_ok", False))
        task_busy = bool(getattr(self, "_task_busy", False))
        gpu_live = bool(getattr(self, "_gpu_refresh_enabled", False)
                        and getattr(self, "_gpu_mon_active", False))
        if running:
            try:
                self.status_lbl.config(
                    fg=S("ok") if phase else S("ok_dim"))
            except (tk.TclError, AttributeError):
                pass
            try:
                self.task_server_state.config(
                    fg=S("ok") if phase else S("ok_dim"))
            except (tk.TclError, AttributeError):
                pass
        if task_busy:
            try:
                self.task_status.config(
                    fg=S("accent") if phase else S("info"))
                self.task_tps.config(
                    fg=S("info") if phase else S("accent"))
            except (tk.TclError, AttributeError):
                pass
        if gpu_live:
            try:
                self.gpu_mon_live.config(
                    fg=S("ok") if phase else S("ok_dim"))
            except (tk.TclError, AttributeError):
                pass
        if running or task_busy or gpu_live:
            self._pulse_pending = True
            self.root.after(700, self._pulse_tick)

    def _set_busy_ui(self, busy):
        """Show/hide the indeterminate spinner next to the load pill.
        Packed + animating ONLY while the server is starting/loading;
        fully removed when idle so no timer keeps running at rest."""
        try:
            if busy:
                if not self.busy_pb.winfo_ismapped():
                    self.busy_pb.pack(side=tk.LEFT, padx=(0, 8),
                                      after=self.load_status_label)
                self.busy_pb.start(60)
            else:
                self.busy_pb.stop()
                self.busy_pb.pack_forget()
        except (tk.TclError, AttributeError):
            pass

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
        # Kill Task (server-tab twin): red text like a destructive action
        # but on the neutral chip, with a readable disabled state so it is
        # always visible even before a task is running.
        style.configure("Kill.TButton", background=surf, foreground=danger,
                        font=(_LATIN_FONT_FAMILY, 10, "bold"), padding=7,
                        borderwidth=0, relief=tk.FLAT)
        style.map("Kill.TButton",
                  background=[("pressed", S("border")), ("active", surf_hi)],
                  foreground=[("disabled", muted), ("pressed", S("fg")),
                              ("active", danger)])

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

    def _mk_toggle(self, parent, var, tip_key=None, dim=None):
        """Flat ON/OFF switch button bound to a BooleanVar.

        Green '● ON' while the flag will be sent, grey '○ OFF' when the
        value is remembered but skipped. `dim` (usually the path Entry)
        greys out alongside OFF so the state reads at a glance."""
        btn = tk.Button(parent, bd=0, cursor="hand2", padx=10, pady=1,
                        font=self.en_font(9, "bold"))

        def paint(*_):
            on = bool(var.get())
            btn.config(text="● ON" if on else "○ OFF",
                       bg=S("ok") if on else S("surface_hi"),
                       fg=S("bg") if on else S("fg_muted"),
                       activebackground=S("surface"),
                       activeforeground=S("fg"))
            if dim is not None:
                try:
                    dim.config(fg=S("fg") if on else S("fg_muted"))
                except tk.TclError:
                    pass

        btn.config(command=lambda: var.set(not var.get()))
        var.trace_add("write", paint)
        paint()
        btn.pack(side=tk.LEFT, padx=(6, 0))
        if tip_key:
            tip = Tooltip(btn, self.t(tip_key))
            self._reg(tip, tip_key, "tooltip")
        return btn

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
        # Auto-scanning preset dropdown: every llama-server.exe found under
        # C:\llmcap (root exe = "default", others named by their folder).
        # Rescanned each time the dropdown opens, so dropping a new folder
        # into C:\llmcap makes it appear automatically — no restart needed.
        self.srv_preset_combo = ttk.Combobox(server_frame, state="readonly",
                                             width=14,
                                             postcommand=self._scan_server_presets)
        self.srv_preset_combo.pack(side=tk.LEFT, padx=(6, 0))
        self.srv_preset_combo.bind("<<ComboboxSelected>>", self._on_server_preset)
        self._srv_preset_paths = {}
        self._scan_server_presets()
        srv_combo_tip = Tooltip(self.srv_preset_combo, self.t("srv_preset_tip"))
        self._reg(srv_combo_tip, "srv_preset_tip", "tooltip")

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
                                     values=list(PRESETS.keys()), width=26)
        preset_combo.pack(side=tk.LEFT, padx=(0, 8))
        # Benchmarked speed presets in their own list right of the main
        # selector — selecting one applies it immediately (no Apply click).
        self.speed_preset_label = ttk.Label(preset_frame, text=self.t("speed_preset_label"), font=self.en_font(9, "bold"))
        self.speed_preset_label.pack(side=tk.LEFT, padx=(0, 6))
        self._reg(self.speed_preset_label, "speed_preset_label", "label")
        self.speed_preset_var = tk.StringVar(value="")
        self.speed_preset_combo = ttk.Combobox(preset_frame, textvariable=self.speed_preset_var, state="readonly",
                                               values=SPEED_PRESETS, width=22)
        self.speed_preset_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.speed_preset_combo.bind("<<ComboboxSelected>>", self._on_speed_preset)
        speed_tip = Tooltip(self.speed_preset_combo, self.t("speed_preset_tip"))
        self._reg(speed_tip, "speed_preset_tip", "tooltip")
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

        # Server-wide sampling temperature (--temp). Defaults to llama.cpp's
        # own 0.8; Run Task's spinbox overrides it per request.
        self._labeled(row2b, "temp_label", width=11)
        self.temp_server_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(row2b, from_=0.0, to=2.0, increment=0.1,
                    textvariable=self.temp_server_var, width=5,
                    format="%.1f").pack(side=tk.LEFT, padx=(0, 16))

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

        # Row 6: mini speed test of the CURRENTLY LOADED model (mini dih.py)
        # — one button + one result line, nothing else (kept tiny on purpose).
        adv_row6 = ttk.Frame(adv_frame)
        adv_row6.pack(fill=tk.X, pady=(6, 2))
        self.bench_btn = tk.Button(adv_row6, text=self.t("bench_btn"), bd=0,
                                   cursor="hand2", bg=S("surface_hi"), fg=S("info"),
                                   activebackground=S("surface"),
                                   activeforeground=S("info"),
                                   padx=12, pady=5,
                                   font=self.en_font(10, "bold"),
                                   command=self.run_speed_test)
        self.bench_btn.pack(side=tk.LEFT, padx=(0, 10))
        self._reg(self.bench_btn, "bench_btn", "button")
        # Hover glow: fills solid sky-blue (text flips to dark).
        self.bench_btn.bind("<Enter>",
                            lambda e: self.bench_btn.config(bg=S("info"),
                                                            fg=S("bg")))
        self.bench_btn.bind("<Leave>",
                            lambda e: self.bench_btn.config(bg=S("surface_hi"),
                                                            fg=S("info")))
        bench_tip = Tooltip(self.bench_btn, self.t("bench_tip"))
        self._reg(bench_tip, "bench_tip", "tooltip")
        self.bench_result_lbl = ttk.Label(adv_row6, text=self.t("bench_idle"),
                                          font=("Consolas", 9),
                                          foreground=S("fg_muted"))
        self.bench_result_lbl.pack(side=tk.LEFT)
        self._reg(self.bench_result_lbl, "bench_idle", "label")
        self._bench_busy = False
        self._bench_conn = None

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
        self.mmproj_enabled_var = tk.BooleanVar(value=True)
        mmf_entry = ttk.Entry(mm_file_row, textvariable=self.mmproj_path_var)
        mmf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        mmf_tip = Tooltip(mmf_entry, self.t("mmproj_file_tip"))
        self._reg(mmf_tip, "mmproj_file_tip", "tooltip")
        self.browse_mmproj_btn = ttk.Button(mm_file_row, text=self.t("browse_btn"),
                                             command=self.browse_mmproj)
        self.browse_mmproj_btn.pack(side=tk.LEFT)
        self._reg(self.browse_mmproj_btn, "browse_btn", "button")
        self.mmproj_toggle_btn = self._mk_toggle(
            mm_file_row, self.mmproj_enabled_var, "mmproj_toggle_tip",
            dim=mmf_entry)

        # --- MTP draft model (separate GGUF, e.g. mtp-gemma-4-…-Q4_0.gguf).
        # ON emits --spec-draft-model + forces --spec-type draft-mtp. ---
        mtp_file_row = ttk.Frame(mm_frame)
        mtp_file_row.pack(fill=tk.X, pady=4)
        mtp_lbl = ttk.Label(mtp_file_row, text=self.t("mtp_file_label"), width=14, anchor=tk.W)
        mtp_lbl.pack(side=tk.LEFT, padx=(0, 5))
        self._reg(mtp_lbl, "mtp_file_label", "label")
        self.mtp_model_var = tk.StringVar(value="")
        self.mtp_enabled_var = tk.BooleanVar(value=False)
        mtp_entry = ttk.Entry(mtp_file_row, textvariable=self.mtp_model_var)
        mtp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        mtp_tip = Tooltip(mtp_entry, self.t("mtp_file_tip"))
        self._reg(mtp_tip, "mtp_file_tip", "tooltip")
        self.browse_mtp_btn = ttk.Button(mtp_file_row, text=self.t("browse_btn"),
                                          command=self.browse_mtp)
        self.browse_mtp_btn.pack(side=tk.LEFT)
        self._reg(self.browse_mtp_btn, "browse_btn", "button")
        self.mtp_toggle_btn = self._mk_toggle(
            mtp_file_row, self.mtp_enabled_var, "mtp_toggle_tip",
            dim=mtp_entry)

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
        # Env flag surfaced as a checkbox: presets flip it, and it only
        # affects the llama-server child process (never the whole app).
        self.graph_opt_var = tk.BooleanVar(value=False)
        self.graph_opt_btn = ttk.Checkbutton(extra_frame, text=self.t("graph_opt_label"),
                                             variable=self.graph_opt_var)
        self.graph_opt_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._reg(self.graph_opt_btn, "graph_opt_label", "button")
        graph_tip = Tooltip(self.graph_opt_btn, self.t("graph_opt_tip"))
        self._reg(graph_tip, "graph_opt_tip", "tooltip")

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
                    self.temp_server_var,
                    self.port_var, self.flash_var, self.load_mode_var, self.unified_var,
                    self.kv_quant_var, self.ctk_var, self.ctv_var,
                    self.no_warmup_var,
                    self.cache_idle_slots_var, self.sleep_idle_var,
                    self.sleep_idle_seconds_var, self.moe_cpu_var,
                    self.moe_cpu_layers_var, self.reasoning_format_var,
                    self.devices_var, self.override_tensor_var,
                    self.mmproj_path_var, self.mmproj_enabled_var,
                    self.mtp_model_var, self.mtp_enabled_var,
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
        # Kill Task twin right beside Copy Command — always clickable (red
        # on the neutral chip); clicking with nothing running just reports
        # "no task" instead of sitting there gray and dead.
        self.kill_task_btn_srv = ttk.Button(btn_frame, text=self.t("task_kill"),
                                            style="Kill.TButton",
                                            command=self.kill_task)
        self.kill_task_btn_srv.pack(side=tk.LEFT, padx=5)
        self._reg(self.kill_task_btn_srv, "task_kill", "button")
        kill_srv_tip = Tooltip(self.kill_task_btn_srv, self.t("task_kill_tip"))
        self._reg(kill_srv_tip, "task_kill_tip", "tooltip")
        self.clear_btn = ttk.Button(btn_frame, text=self.t("clear_btn"), command=self.clear_log)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)
        self._reg(self.clear_btn, "clear_btn", "button")

        self.status_var = tk.StringVar(value=self.t("status_ready"))
        # Pill-style status chip (colored face + breathing pulse while live).
        self.status_lbl = tk.Label(btn_frame, textvariable=self.status_var,
                                   bg=S("surface_hi"), fg=S("ok"),
                                   font=self.en_font(9, "bold"),
                                   padx=10, pady=3, bd=0)
        self.status_lbl.pack(side=tk.LEFT, padx=20)

        status_bar = ttk.Frame(parent)
        status_bar.pack(fill=tk.X, pady=(0, 8))
        self.load_status_var = tk.StringVar(value=self.t("load_status_idle"))
        self.load_status_label = tk.Label(status_bar, textvariable=self.load_status_var,
                                           bg=S("surface"), fg=S("fg"),
                                           font=self.en_font(10, "bold"),
                                           padx=10, pady=5, bd=0, relief=tk.FLAT)
        self.load_status_label.pack(side=tk.LEFT, padx=(0, 8))
        # Indeterminate spinner — visible only while starting/loading.
        self.busy_pb = ttk.Progressbar(status_bar, orient=tk.HORIZONTAL,
                                       mode="indeterminate", length=160)

        self.speed_var = tk.StringVar(value="⚡ --")
        self.speed_label = tk.Label(status_bar, textvariable=self.speed_var,
                                     bg=S("surface"), fg=S("info"),
                                     font=self.en_font(10, "bold"),
                                     padx=10, pady=5, bd=0, relief=tk.FLAT,
                                     cursor="sb_h_double_arrow")
        self.speed_label.pack(side=tk.LEFT, padx=(0, 8))
        speed_tip = Tooltip(self.speed_label, self.t("speed_tip"))
        self._reg(speed_tip, "speed_tip", "tooltip")

        # Live context counter (used / total / free) from /slots. This
        # replaces the old Σ totals pill — the speed pill already carries
        # the same gen/prompt session numbers, so the bar stays uncrowded.
        self.ctx_pill_var = tk.StringVar(value=self.t("ctx_pill_idle"))
        self.ctx_pill_lbl = tk.Label(status_bar, textvariable=self.ctx_pill_var,
                                     bg=S("surface"), fg=S("fg_muted"),
                                     font=self.en_font(10, "bold"),
                                     padx=10, pady=5, bd=0, relief=tk.FLAT)
        self.ctx_pill_lbl.pack(side=tk.LEFT, padx=(0, 8))
        ctx_tip = Tooltip(self.ctx_pill_lbl, self.t("ctx_pill_tip"))
        self._reg(ctx_tip, "ctx_pill_tip", "tooltip")

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

        # Teal→sky→violet gradient strip: richer color for a few cents of
        # build/resize paint, zero cost while idle.
        self._grad_canvas = tk.Canvas(parent, height=4, highlightthickness=0,
                                      bg=S("bg"), bd=0)
        self._grad_canvas.pack(fill=tk.X, pady=(0, 6))
        self._grad_canvas.bind("<Configure>", self._draw_gradient)
        self.root.after(60, self._draw_gradient)

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
        self.inspector_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.server_tab, text=self.t("tab_server"))
        self.notebook.add(self.gpu_procs_tab, text=self.t("tab_gpu_procs"))
        self.notebook.add(self.tasks_tab, text=self.t("tab_tasks"))
        self.notebook.add(self.inspector_tab, text=self.t("tab_inspector"))
        self._i18n_tabs = [(self.server_tab, "tab_server"), (self.gpu_procs_tab, "tab_gpu_procs"),
                           (self.tasks_tab, "tab_tasks"), (self.inspector_tab, "tab_inspector")]

        self._build_server_tab(self.server_inner)
        self._build_gpu_procs_tab(self.gpu_procs_tab)
        self._build_tasks_tab(self.tasks_tab)
        self._build_inspector_tab(self.inspector_tab)

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
        # Same live context counter as the Server Setup status bar.
        self.task_ctx_lbl = tk.Label(header, textvariable=self.ctx_pill_var,
                                     bg=surf, fg=S("fg_muted"),
                                     font=self.en_font(9, "bold"), padx=10, pady=3, bd=0)
        self.task_ctx_lbl.pack(side=tk.RIGHT, padx=(0, 8))
        task_ctx_tip = Tooltip(self.task_ctx_lbl, self.t("ctx_pill_tip"))
        self._reg(task_ctx_tip, "ctx_pill_tip", "tooltip")

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
        self.task_run_btn = run_btn
        self._reg(run_btn, "task_run", "button")
        # Hover glow: chip brightens to a solid green "go" face.
        run_btn.bind("<Enter>",
                     lambda e: run_btn.config(bg=S("ok"), fg=S("bg")))
        run_btn.bind("<Leave>",
                     lambda e: run_btn.config(bg=S("surface_hi"), fg=ok))

        # Kill Task: cancels only the in-flight generation; server keeps
        # running. Always clickable — with no task running it reports that
        # instead of staying gray/dead.
        self.kill_task_btn = tk.Button(opts, text=self.t("task_kill"),
                                       command=self.kill_task, bd=0,
                                       cursor="hand2", bg=surf_hi, fg=danger,
                                       activebackground=surf,
                                       activeforeground=danger,
                                       padx=14, pady=6,
                                       font=self.en_font(10, "bold"))
        self.kill_task_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self._reg(self.kill_task_btn, "task_kill", "button")
        kill_tip = Tooltip(self.kill_task_btn, self.t("task_kill_tip"))
        self._reg(kill_tip, "task_kill_tip", "tooltip")
        # Hover glow: fills solid red (text flips to dark for contrast).
        self.kill_task_btn.bind(
            "<Enter>",
            lambda e: self.kill_task_btn.config(bg=S("danger"), fg=S("bg")))
        self.kill_task_btn.bind(
            "<Leave>",
            lambda e: self.kill_task_btn.config(bg=S("surface_hi"),
                                                 fg=S("danger")))

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

    # ------------------------------------------------------------------
    # Tab: Task Inspector — system prompt / prompt / thinking / output
    # plus a decode-speed chart across the whole generation.
    # ------------------------------------------------------------------
    def _build_inspector_tab(self, parent):
        bg, surf, surf_hi = S("bg"), S("surface"), S("surface_hi")
        fg, muted = S("fg"), S("fg_muted")
        ok, warn, danger, info = S("ok"), S("warn"), S("danger"), S("info")

        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 8))
        title = ttk.Label(header, text=self.t("tab_inspector"),
                          font=self.font_for(self.t("tab_inspector"), 16, "bold"),
                          foreground=S("accent"))
        title.pack(side=tk.LEFT)
        self._reg(title, "tab_inspector", "label")
        self.insp_server_state = tk.Label(header, text=self.t("task_not_running"),
                                          bg=surf, fg=warn, font=self.en_font(9),
                                          padx=10, pady=3, bd=0)
        self.insp_server_state.pack(side=tk.RIGHT)
        self._reg(self.insp_server_state, "task_not_running", "label")

        # --- Chart: the tab IS the chart — tok/s across the whole
        # generation, sized to fill every remaining pixel ---
        chart_card = tk.Frame(parent, bg=surf, highlightthickness=1,
                              highlightbackground=S("border"))
        chart_card.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
        chart_head = tk.Frame(chart_card, bg=surf)
        chart_head.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(chart_head, text=self.t("insp_graph_title"), bg=surf,
                 fg=info, font=self.en_font(11, "bold")).pack(side=tk.LEFT)
        self.insp_stats_lbl = tk.Label(chart_head, text="", bg=surf, fg=muted,
                                       font=self.en_font(9))
        self.insp_stats_lbl.pack(side=tk.RIGHT)
        self.insp_cv = tk.Canvas(chart_card, bg=S("log_bg"), height=430,
                                 highlightthickness=0, bd=0)
        self.insp_cv.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))
        self.insp_cv.bind("<Configure>", lambda e: self.insp_draw_graph())
        # Fires every time the tab becomes visible — the reliable
        # redraw when returning from Task/Server tabs (Configure is
        # not guaranteed on notebook remap).
        self.insp_cv.bind("<Map>", lambda e: self.insp_draw_graph())
        tk.Label(chart_card, text=self.t("insp_graph_hint"), bg=surf,
                 fg=muted, font=self.en_font(8)).pack(anchor=tk.W,
                                                      padx=12, pady=(2, 10))

        self.insp_draw_graph()

    def insp_draw_graph(self):
        """Render the streamed tok/s series as a full-size area chart:
        gradient fill, nice axis ticks, per-point dots, peak marker,
        live latest-value badge and an average reference line."""
        cv = getattr(self, "insp_cv", None)
        if cv is None:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 40 or h < 40:
            # Canvas is on an unselected notebook tab (unmapped reports
            # 1x1). Never delete() here: wiping-then-bailing used to
            # erase the graph while the user ran a task from another
            # tab. Samples keep accumulating; <Map> redraws on return.
            return
        cv.delete("all")
        muted, info = S("fg_muted"), S("info")
        bg_c = S("log_bg")

        def _mix(c1, c2, t):
            try:
                a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
                b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
                return "#%02x%02x%02x" % tuple(
                    int(a[k] + (b[k] - a[k]) * t) for k in range(3))
            except Exception:
                return c1

        pad_l, pad_r, pad_t, pad_b = 52, 16, 14, 24
        cw, ch = w - pad_l - pad_r, h - pad_t - pad_b
        series = getattr(self, "insp_series", [])
        if len(series) < 2:
            cv.create_text(w / 2, h / 2 - 8, text=self.t("insp_no_data"),
                           fill=muted, font=self.en_font(10))
            cv.create_text(w / 2, h / 2 + 14, text=self.t("insp_graph_hint"),
                           fill=S("border"), font=self.en_font(8))
            return
        ys = [v for _, v in series]
        xs = [i for i, _ in series]
        peak = max(ys)
        # nice-rounded y max: smallest 1/2/2.5/5x10^n step that covers data
        raw_max = peak * 1.15 if peak > 0 else 1.0
        mag = 1.0
        while mag * 10 <= raw_max:
            mag *= 10
        ymax = mag * 10
        for mult in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8):
            if mag * mult >= raw_max:
                ymax = mag * mult
                break
        xmin, xmax = min(xs), max(xs)
        if xmax <= xmin:
            xmax = xmin + 1
        avg = sum(ys) / len(ys)

        def px(i):
            return pad_l + (i - xmin) / (xmax - xmin) * cw

        def py(v):
            return pad_t + ch - (v / ymax) * ch

        yfmt = "{:.0f}" if ymax >= 20 else "{:.1f}"
        # gridlines + y labels at nice values (5 rows)
        for k in range(5):
            y = pad_t + ch * k / 4
            cv.create_line(pad_l, y, w - pad_r, y, fill=S("border"),
                           dash=(2, 4))
            cv.create_text(pad_l - 8, y, text=yfmt.format(ymax * (1 - k / 4)),
                           anchor=tk.E, fill=muted, font=self.en_font(8))
        # x labels: 5 evenly spaced token indices
        for k in range(5):
            i = xmin + (xmax - xmin) * k / 4
            cv.create_text(px(i), h - 8, text=str(int(i)), anchor=tk.N,
                           fill=muted, font=self.en_font(8))
        # area fill: base layer + stronger clipped bottom band (gradient)
        area = [(pad_l, pad_t + ch)] + [(px(i), py(v)) for i, v in series] \
            + [(px(xmax), pad_t + ch)]
        flat = [c for p in area for c in p]
        cv.create_polygon(flat, fill=_mix(info, bg_c, 0.55), outline="",
                          stipple="gray25")
        y_split = pad_t + ch * 0.6
        inside = []
        for k in range(len(area)):
            x1, y1 = area[k]
            x2, y2 = area[(k + 1) % len(area)]
            i1, i2 = y1 >= y_split, y2 >= y_split
            if i1:
                inside.append((x1, y1))
            if i1 != i2 and y2 != y1:
                t = (y_split - y1) / (y2 - y1)
                inside.append((x1 + (x2 - x1) * t, y_split))
        if len(inside) >= 3:
            cv.create_polygon([c for p in inside for c in p],
                              fill=_mix(info, bg_c, 0.25), outline="")
        # line (dark casing under bright stroke for contrast)
        line = [c for p in [(px(i), py(v)) for i, v in series] for c in p]
        cv.create_line(line, fill=_mix(info, bg_c, 0.4), width=5, smooth=True,
                       capstyle=tk.ROUND)
        cv.create_line(line, fill=info, width=3, smooth=True,
                       capstyle=tk.ROUND)
        # point dots (skip when dense)
        if len(series) <= 80:
            for i, v in series:
                x, y = px(i), py(v)
                cv.create_oval(x - 3, y - 3, x + 3, y + 3, fill=info,
                               outline=bg_c, width=1)
        # average reference line
        cv.create_line(pad_l, py(avg), w - pad_r, py(avg),
                       fill=S("violet"), dash=(5, 3), width=1.5)
        cv.create_text(w - pad_r - 6, max(pad_t + 9, py(avg) - 9),
                       text=f"avg {avg:.1f}", anchor=tk.E,
                       fill=S("violet"), font=self.en_font(8, "bold"))
        # peak marker + label
        pi = max(range(len(series)), key=lambda k: series[k][1])
        if 0 < pi < len(series) - 1:
            xp, yp = px(series[pi][0]), py(series[pi][1])
            cv.create_oval(xp - 5, yp - 5, xp + 5, yp + 5,
                           outline=S("violet"), width=2)
            ly = yp - 13 if yp - 13 >= pad_t + 8 else yp + 15
            cv.create_text(xp, ly, text=f"peak {series[pi][1]:.0f}",
                           anchor=tk.CENTER, fill=S("violet"),
                           font=self.en_font(8, "bold"))
        # latest-value badge
        li, lv = series[-1]
        xl, yl = px(li), py(lv)
        anchor = tk.E if xl > w - pad_r - 46 else tk.W
        ox = -8 if anchor == tk.E else 8
        cv.create_text(xl + ox, yl, text=f"{lv:.0f} tok/s", anchor=anchor,
                       fill=S("fg"), font=self.en_font(9, "bold"))
        # axis frame
        cv.create_rectangle(pad_l, pad_t, w - pad_r, pad_t + ch,
                            outline=S("border"))
        cv.create_text(12, pad_t + ch / 2, text="tok/s", angle=90,
                       anchor=tk.CENTER, fill=muted, font=self.en_font(8))
        # stats line
        elapsed = 0.0
        if self.insp_t0:
            elapsed = time.time() - self.insp_t0
        try:
            self.insp_stats_lbl.config(
                text=self.t("insp_stats").format(len(ys), avg, peak, elapsed))
        except tk.TclError:
            pass

    def insp_reset_trace(self):
        """Called at the start of every Run Task: fresh chart for THIS
        run (the text panels were removed — the chart is the tab now)."""
        self.insp_series = []
        self.insp_t0 = time.time()
        self._insp_last_flush = 0.0
        self.insp_stats_lbl.config(text="")
        self.insp_draw_graph()

    def _insp_add_sample(self, sample):
        """UI thread: one streamed tok/s point + chart redraw (used by
        both Run Task and the speed test)."""
        self.insp_series.append(sample)
        self.insp_draw_graph()

    def insp_flush_live(self, prompt_text=None, thinking_text=None,
                        output_text=None, sample=None):
        """UI-thread refresh while a task streams. Text panels are gone,
        so this only records the sample and redraws; the text args are
        kept for call-site compatibility."""
        if sample is not None:
            self.insp_series.append(sample)
        self.insp_draw_graph()

    def _server_healthy(self):
        """True when llama-server is up with the model loaded.

        Works for a server this app spawned AND one started outside the
        app (self.process is None): the HTTP /health probe is the source
        of truth. Trusting only the child-process handle made externally
        started servers look 'not running' forever, so Run Task sat in
        its wait loop and the Inspector never saw a single token."""
        if self.process and self.process.poll() is None and self.model_ready:
            return True
        return self._probe_health()

    def _probe_health(self, timeout=0.8):
        """One cached HTTP GET /health against the configured port."""
        now = time.time()
        if now - getattr(self, "_health_probe_at", 0.0) < 1.0:
            return getattr(self, "_health_probe_ok", False)
        ok = False
        try:
            port = int(self.port_var.get())
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
                ok = resp.status == 200
        except Exception:
            ok = False
        if ok:
            self.model_ready = True
        self._health_probe_at = now
        self._health_probe_ok = ok
        return ok

    def _update_task_server_state(self):
        running = self._server_healthy()
        if running:
            text = self.t("task_running").format(self.port_var.get())
            self.task_server_state.config(text=text, fg=S("ok"))
            try:
                self.insp_server_state.config(text=text, fg=S("ok"))
            except (tk.TclError, AttributeError):
                pass
        else:
            self.task_server_state.config(text=self.t("task_not_running"),
                                          fg=S("warn"))
            try:
                self.insp_server_state.config(text=self.t("task_not_running"),
                                              fg=S("warn"))
            except (tk.TclError, AttributeError):
                pass

    def _task_load_preset(self, key):
        text = self.t(key + "_text")
        self.task_prompt.delete("1.0", tk.END)
        self.task_prompt.insert("1.0", text)

    @staticmethod
    def _split_thinking(text):
        """Pull a reasoning/thinking block out of generated text.
        Returns (thinking, answer). Handles the common ASCII tag pairs
        (Qwen <|think|>, DeepSeek <thought>, brain tags); if no marker,
        thinking is '' and everything is the answer."""
        if not text:
            return "", ""
        pairs = (("<|think|>", "<|/think|>"),
                 ("<thought>", "</thought>"),
                 ("<brain>", "</brain>"))
        for open_t, close_t in pairs:
            lo, lc = open_t.lower(), close_t.lower()
            i = text.lower().find(lo)
            if i < 0:
                continue
            body_start = i + len(open_t)
            j = text.lower().find(lc, body_start)
            if j < 0:
                # Stream cut before the close tag: rest is thinking.
                return text[body_start:].strip(), text[:i].strip()
            return (text[body_start:j].strip(),
                    (text[:i] + text[j + len(close_t):]).strip())
        return "", text.strip()

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
        # System prompt is DISPLAY-ONLY in the Inspector (never injected).
        # /completion takes a single prompt string; we send exactly what
        # you typed so the trace matches the request byte-for-byte.

        self._task_busy = True
        self._task_cancel_evt = threading.Event()
        self._task_conn = None
        self._ensure_pulse()
        self.task_tps.config(text="--", fg=S("info"))
        self.task_status.config(text=self.t("task_running_task"), fg=S("info"))
        self.task_run_btn.config(state=tk.DISABLED)
        self._update_task_server_state()
        self.insp_reset_trace()
        # Show live state in the Inspector header while this run streams.
        try:
            self.insp_server_state.config(text=self.t("task_running_task"),
                                          fg=S("info"))
        except (tk.TclError, AttributeError):
            pass
        # Stash what the Inspector should show once this run settles.
        self._insp_prompt_shown = prompt
        self._insp_think_shown = ""
        self._insp_out_shown = ""
        # Log every step: if anything dies before the first token, the
        # reason shows up in the log pane instead of vanishing silently.
        self.log_write(
            f"Run Task: sending prompt ({len(prompt)} chars, "
            f"n_predict={self.task_npredict_var.get()}, "
            f"temp={self.task_temp_var.get()})...", "info")

        def worker():
            conn = None
            try:
                port = self.port_var.get()
                if not self._server_healthy():
                    self.root.after(0, lambda: (self.log_write("Starting server for task...", "info"),
                                                self.start_server()))
                deadline = time.time() + 240
                while time.time() < deadline:
                    if self._task_cancel_evt.is_set():
                        self.root.after(0, self._task_on_cancelled)
                        return
                    if self._server_healthy():
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("Server did not become ready in time")
                if self._task_cancel_evt.is_set():
                    self.root.after(0, self._task_on_cancelled)
                    return

                n_predict = int(self.task_npredict_var.get())
                temperature = float(self.task_temp_var.get())
                body = json.dumps({
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "stream": True,
                    "cache_prompt": False,
                }).encode("utf-8")
                # http.client (not urllib) so kill_task can reach the live
                # socket and shutdown() it — verified to make llama-server
                # cancel this generation and free the slot at once.
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
                self._task_conn = conn
                conn.request("POST", "/completion", body=body,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                if resp.status != 200:
                    resp.read()
                    raise RuntimeError(f"HTTP {resp.status}")

                # --- Streamed read: accumulate answer + thinking and a
                # decode-speed sample per chunk for the Inspector chart. ---
                t_start = time.perf_counter()
                t_first = None
                out_parts, think_parts = [], []
                token_idx = 0
                timings = {}

                def chunk_obj(raw):
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        return None
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if not line:
                            return None
                    if line == "[DONE]":
                        return False
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        return None
                    return o if isinstance(o, dict) else None

                def flush(sample, force=False):
                    now = time.time()
                    if not force and now - self._insp_last_flush < 0.3:
                        return
                    self._insp_last_flush = now
                    snap_out = "".join(out_parts)
                    snap_think = "".join(think_parts)
                    # Split live so the panels always match the final view.
                    t_now, a_now = self._split_thinking(snap_out)
                    think_now = (snap_think + ("\n" + t_now if t_now else "")).strip()
                    self.root.after(0, lambda s=sample, a=a_now, th=think_now: self.insp_flush_live(
                        self._insp_prompt_shown, th, a, sample=s))

                while True:
                    if self._task_cancel_evt.is_set():
                        break
                    raw = resp.readline()
                    if not raw:
                        break
                    o = chunk_obj(raw)
                    if o is False:
                        break
                    if o is None:
                        continue
                    if isinstance(o.get("timings"), dict):
                        timings = o["timings"]
                    # thinking may arrive as its own field...
                    r = (o.get("reasoning_content") or o.get("reasoning")
                         or "")
                    if r:
                        think_parts.append(r)
                    piece = o.get("content", "")
                    if not piece:
                        ch = o.get("choices")
                        if isinstance(ch, list) and ch:
                            d = ch[0].get("delta") or {}
                            piece = (d.get("content") or ch[0].get("text")
                                     or "")
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        out_parts.append(piece)
                        token_idx += 1
                        # cumulative decode tok/s at this token index;
                        # token 1 has a near-zero elapsed time (would plot
                        # a million tok/s), and anything under 50ms is
                        # still start-up jitter — skip those samples.
                        el = time.perf_counter() - t_first
                        if token_idx > 1 and el >= 0.05:
                            flush((token_idx, token_idx / el))
                        else:
                            flush(None)
                    else:
                        flush(None)
                    if o.get("stop"):
                        break
                try:
                    resp.read()
                except Exception:
                    pass

                if self._task_cancel_evt.is_set():
                    self.root.after(0, self._task_on_cancelled)
                    return

                answer = "".join(out_parts)
                tagged_think, answer = self._split_thinking(answer)
                thinking = "".join(think_parts).strip()
                if tagged_think:
                    thinking = (thinking + "\n" + tagged_think).strip()
                self._insp_think_shown = thinking
                self._insp_out_shown = answer
                done_in = time.perf_counter() - t_start
                self.root.after(
                    0, lambda n=token_idx, s=done_in: self.log_write(
                        f"Run Task: finished — {n} tokens in {s:.1f}s"
                        + (" (EMPTY — model emitted EOS immediately; "
                           "rephrase the prompt)" if n <= 1 else ""),
                        "success" if n > 1 else "warn"))
                # final flush (always, regardless of throttle)
                self.root.after(0, lambda: self.insp_flush_live(
                    self._insp_prompt_shown, thinking, answer))
                self.root.after(0, lambda: self._task_finish(
                    answer,
                    timings.get("predicted_per_second"),
                    timings.get("prompt_per_second"),
                    timings.get("predicted_n", token_idx),
                    timings.get("prompt_n", 0),
                    timings.get("total_ms", 0)))
            except Exception as e:
                if self._task_cancel_evt is not None and self._task_cancel_evt.is_set():
                    self.root.after(0, self._task_on_cancelled)
                else:
                    self.root.after(0, lambda e=e: self._task_fail(str(e)))
            finally:
                self._task_conn = None
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        threading.Thread(target=worker, daemon=True).start()

    def kill_task(self):
        """Cancel only the in-flight Run Task generation. Drops this
        request's HTTP socket — llama-server treats the disconnect as a
        cancel, frees the slot, and the server process keeps running
        (verified live: slot idle within ~1s of shutdown)."""
        if not self._task_busy:
            try:
                self.task_status.config(text=self.t("task_no_idle"), fg=S("warn"))
            except Exception:
                pass
            self.log_write("No task running — nothing to cancel (server untouched).", "info")
            return
        if self._task_cancel_evt is None:
            self._task_cancel_evt = threading.Event()
        self._task_cancel_evt.set()
        conn = self._task_conn
        if conn is not None:
            try:
                sock = conn.sock
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        self.log_write("Cancelling current task (server keeps running)...", "warn")
        # If the worker is still in its server-ready wait (no socket yet),
        # its event check will notice — this watchdog just guarantees the
        # UI resets even if that worker died without reporting.
        self.root.after(2500, self._task_cancel_watchdog)

    def _task_cancel_watchdog(self):
        if (self._task_busy and self._task_cancel_evt is not None
                and self._task_cancel_evt.is_set()):
            self._task_on_cancelled()

    def _task_on_cancelled(self):
        """Idempotent: worker and watchdog may both call this."""
        if not self._task_busy:
            return
        self._task_busy = False
        self._task_conn = None
        self.task_tps.config(text="⏹", fg=S("warn"))
        self.task_status.config(text=self.t("task_cancelled"), fg=S("warn"))
        self.task_run_btn.config(state=tk.NORMAL)
        self._update_task_server_state()
        self.log_write("Task cancelled — server still running.", "warn")

    def _task_finish(self, content, tps, pts, pred_n, prompt_n, total_ms):
        self._task_busy = False
        self.task_run_btn.config(state=tk.NORMAL)
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
        self.task_run_btn.config(state=tk.NORMAL)
        self.task_tps.config(text="ERR", fg=S("danger"))
        self.task_status.config(text=error, fg=S("danger"))
        self._update_task_server_state()
        self.log_write(f"Run Task FAILED: {error}", "error")

    # ------------------------------------------------------------------
    # Mini speed test (compact dih.py) — benchmarks the model that is
    # currently LOADED in llama-server via a streamed /completion, and
    # reports TTFT + decode tok/s + prompt tok/s on one small label in
    # the Advanced Performance panel.
    # ------------------------------------------------------------------
    def run_speed_test(self):
        if self._bench_busy:
            return
        if self._task_busy:
            self.bench_result_lbl.config(text=self.t("bench_task_busy"),
                                         foreground=S("warn"))
            return
        if not self._server_healthy():
            self.bench_result_lbl.config(text=self.t("bench_not_ready"),
                                         foreground=S("warn"))
            return
        self._bench_busy = True
        self.bench_btn.config(state=tk.DISABLED)
        self.bench_result_lbl.config(text=self.t("bench_running"),
                                     foreground=S("info"))
        # A speed test IS a streamed generation, so it earns the chart:
        # give it a fresh tok/s series (text panels keep the last task).
        self.insp_series = []
        self.insp_t0 = time.time()
        self._insp_last_flush = 0.0
        self.insp_stats_lbl.config(text="")
        self.insp_draw_graph()
        self._update_task_server_state()

        def worker():
            conn = None
            try:
                port = self.port_var.get()
                prompt = ("Explain how a GPU accelerates large language models. "
                          "Give a concise technical answer in about 150 words.")
                start = time.perf_counter()
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
                self._bench_conn = conn

                def chunk_obj(raw_line):
                    """Accept plain JSON lines AND SSE `data:`-prefixed
                    lines (this server's stream shape was neither purely
                    one nor the other — unparseable lines used to drop
                    every token and end with 'no text')."""
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        return None
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if not line:
                            return None
                    if line == "[DONE]":
                        return False
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        return None
                    return obj if isinstance(obj, dict) else None

                # --- Pass 1: streamed (gives a real TTFT) ---
                body = json.dumps({
                    "prompt": prompt, "n_predict": 256,
                    "temperature": 0.7, "stream": True,
                    "cache_prompt": False,
                }).encode("utf-8")
                conn.request("POST", "/completion", body=body,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                first_token = None
                pieces = []
                timings = {}
                if resp.status == 200:
                    b_idx = 0
                    b_last = 0.0
                    while True:
                        raw = resp.readline()
                        if not raw:
                            break
                        obj = chunk_obj(raw)
                        if obj is False:
                            break
                        if obj is None:
                            continue
                        if isinstance(obj.get("timings"), dict):
                            timings = obj["timings"]
                        piece = obj.get("content", "")
                        if not piece:
                            ch = obj.get("choices")
                            if isinstance(ch, list) and ch:
                                delta = ch[0].get("delta") or {}
                                piece = (delta.get("content")
                                         or ch[0].get("text") or "")
                        if piece:
                            if first_token is None:
                                first_token = time.perf_counter()
                            pieces.append(piece)
                            # Feed the Task Inspector's tok/s chart the
                            # same way Run Task does (cumulative decode
                            # speed, throttled, no bogus token-1 sample).
                            b_idx += 1
                            bel = time.perf_counter() - first_token
                            now = time.time()
                            if (b_idx > 1 and bel >= 0.05
                                    and now - b_last >= 0.3):
                                b_last = now
                                s = (b_idx, b_idx / bel)
                                self.root.after(
                                    0, lambda s=s: self._insp_add_sample(s))
                        if obj.get("stop"):
                            break
                    try:
                        resp.read()  # drain so the keep-alive conn reuses
                    except Exception:
                        pass
                else:
                    try:
                        resp.read()
                    except Exception:
                        pass

                text = "".join(pieces)

                # --- Pass 2: non-stream fallback — the exact request Run
                # Task uses (proven on this server) when the stream yields
                # neither content nor timings. ---
                if not text and not timings:
                    body = json.dumps({
                        "prompt": prompt, "n_predict": 256,
                        "temperature": 0.7, "stream": False,
                        "cache_prompt": False,
                    }).encode("utf-8")
                    conn.request("POST", "/completion", body=body,
                                 headers={"Content-Type": "application/json"})
                    resp2 = conn.getresponse()
                    if resp2.status != 200:
                        resp2.read()
                        raise RuntimeError(f"HTTP {resp2.status}")
                    out = json.loads(
                        resp2.read().decode("utf-8", errors="replace"))
                    text = out.get("content") or ""
                    if isinstance(out.get("timings"), dict):
                        timings = out["timings"]
                    if not text and not timings:
                        raise RuntimeError("model returned no text")

                end = time.perf_counter()

                # --- Metrics ---
                if first_token:
                    ttft = first_token - start
                else:
                    # non-stream: first token is ready after prompt eval
                    ttft = float(timings.get("prompt_ms") or 0) / 1000.0
                    if ttft <= 0:
                        ttft = end - start
                if timings.get("predicted_per_second"):
                    speed = float(timings["predicted_per_second"])
                elif timings.get("predicted_n"):
                    gen_s = (timings.get("predicted_ms") or 0) / 1000.0
                    speed = (int(timings["predicted_n"]) / gen_s
                             if gen_s > 0 else 0.0)
                else:
                    gen_s = max(end - (first_token or start), 1e-6)
                    speed = max(1, round(len(text) / 4)) / gen_s
                if timings.get("predicted_n"):
                    n = int(timings["predicted_n"])
                else:
                    n = max(1, round(len(text) / 4))
                parts_disp = [f"TTFT {ttft:.2f}s", f"{speed:.1f} tok/s"]
                pp = timings.get("prompt_per_second")
                if pp:
                    parts_disp.append(f"prompt {float(pp):.0f}/s")
                parts_disp.append(f"{n} tok")
                parts_disp.append(f"{(end - start):.1f}s")
                msg = " • ".join(parts_disp)
                self.root.after(0, lambda m=msg: self._bench_done(m, None))
            except Exception as e:
                err = str(e) or e.__class__.__name__
                self.root.after(0, lambda err=err: self._bench_done(None, err))
            finally:
                self._bench_conn = None
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        threading.Thread(target=worker, daemon=True).start()

    def _bench_done(self, msg, err):
        self._bench_busy = False
        self._bench_conn = None
        try:
            self.bench_btn.config(state=tk.NORMAL)
        except tk.TclError:
            return
        if err:
            self.bench_result_lbl.config(text=err, foreground=S("danger"))
            self.log_write(f"Speed test failed: {err}", "warn")
        else:
            self.bench_result_lbl.config(text=msg, foreground=S("ok"))
            self.log_write(f"Speed test (current model): {msg}", "info")

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
        if self._gpu_refresh_enabled:
            self._ensure_pulse()
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
        # A freshly matched projector means this model wants vision —
        # switch the row ON (the path is new, so nothing is clobbered).
        self.mmproj_enabled_var.set(True)
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

    def _on_speed_preset(self, _event=None):
        """⚡ Speed list: applies instantly. Full presets (MiMo Ultra)
        become the preset in use; small partial presets only ADD their
        delta on top of the preset in use, so the main combo keeps
        pointing at the base preset."""
        name = self.speed_preset_var.get()
        p = PRESETS.get(name)
        if not p:
            return
        if not p.get("partial"):
            self.preset_var.set(name)
        self.apply_preset(name)

    def apply_preset(self, name=None):
        if name is None:
            name = self.preset_var.get()
        p = PRESETS.get(name)
        if not p:
            return
        # Small ⚡ presets ("partial"): add only their delta on top of the
        # preset currently in use — never reset ctx/batches/extras.
        if p.get("partial"):
            overlay_vars = {
                "ngl": self.ngl_var, "ctx": self.ctx_var,
                "slots": self.slots_var, "threads": self.threads_var,
                "tbatch": self.tbatch_var, "batch": self.batch_var,
                "ubatch": self.ubatch_var, "flash": self.flash_var,
                "defrag": self.defrag_var, "port": self.port_var,
                "extra": self.extra_var, "graph_opt": self.graph_opt_var,
            }
            applied = []
            for key in p.get("only", []):
                if key in p and key in overlay_vars:
                    overlay_vars[key].set(p[key])
                    applied.append(key)
            if self.graph_opt_var.get():
                self.log_write("GGML_CUDA_GRAPH_OPT=1 → applied on next server start.", "info")
            self.log_write(f"Added on top of current preset: {name} ({', '.join(applied)})",
                           "success")
            self.update_preview()
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
                self.spec_draft_cache_var.set(p.get("spec_draft_cache"))
            self.on_spec_mode_changed()
        if "port" in p:
            self.port_var.set(p["port"])
        if "extra" in p:
            self.extra_var.set(p["extra"])
        # Env flag: presets without the key reset it to off, so a speed
        # preset never leaks into the next one.
        self.graph_opt_var.set(bool(p.get("graph_opt", False)))
        if self.graph_opt_var.get():
            self.log_write("GGML_CUDA_GRAPH_OPT=1 → applied on next server start.", "info")
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

    def browse_mtp(self):
        """Pick the MTP draft GGUF. Defaults to the main model's folder —
        these files normally ship next to the model they accelerate."""
        model_dir = os.path.dirname(self.model_var.get().strip())
        path = filedialog.askopenfilename(
            title="Select MTP draft model",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir=model_dir if os.path.isdir(model_dir) else r"D:\model"
        )
        if path:
            self.mtp_model_var.set(path)
            self.mtp_enabled_var.set(True)  # picking a file means you want it

    def browse_server(self):
        path = filedialog.askopenfilename(
            title="Select llama-server.exe",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            initialdir=r"C:\llmcap"
        )
        if path:
            self.server_var.set(path)

    def set_server_preset(self, folder):
        """Jump the Server exe field to <folder>\\llama-server.exe."""
        self.server_var.set(os.path.join(folder, "llama-server.exe"))
        self.log_write(f"Server exe preset: {self.server_var.get()}", "info")

    def _scan_server_presets(self):
        """Rebuild the Server-preset dropdown from C:\\llmcap.

        Walks the tree for every llama-server.exe; the root one is shown
        as 'default', every other one gets its folder name. Runs on each
        dropdown open (Combobox postcommand), so new folders added to
        C:\\llmcap show up automatically."""
        root = r"C:\llmcap"
        found = []
        try:
            if os.path.isfile(os.path.join(root, "llama-server.exe")):
                found.append((self.t("srv_preset_def"),
                              os.path.join(root, "llama-server.exe")))
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames
                               if not d.startswith((".", "$"))]
                if os.path.normpath(dirpath) == os.path.normpath(root):
                    continue
                if "llama-server.exe" in filenames:
                    found.append((os.path.basename(dirpath),
                                  os.path.join(dirpath, "llama-server.exe")))
        except OSError:
            pass
        found.sort(key=lambda e: e[0].lower())
        # Keep 'default' pinned first if present.
        found.sort(key=lambda e: e[1].lower() != os.path.join(
            root, "llama-server.exe").lower())
        self._srv_preset_paths = {label: path for label, path in found}
        labels = [label for label, _ in found]
        self.srv_preset_combo.configure(values=labels)
        # Highlight the entry matching the current exe path.
        cur = os.path.normcase(os.path.normpath(
            (self.server_var.get() or "").strip()))
        sel = next((i for i, (_, p) in enumerate(found)
                    if os.path.normcase(os.path.normpath(p)) == cur), None)
        if sel is not None:
            self.srv_preset_combo.current(sel)
        else:
            self.srv_preset_combo.set("")

    def _on_server_preset(self, _event=None):
        label = self.srv_preset_combo.get()
        path = self._srv_preset_paths.get(label)
        if path:
            self.server_var.set(path)
            self.log_write(f"Server exe preset: {path}", "info")

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
        cmd += ["--temp", str(self.temp_server_var.get())]
        cmd += ["--port", str(self.port_var.get())]
        if self.flash_var.get():
            cmd += ["-fa", "on"]
        load_mode = self.load_mode_var.get().strip()
        if load_mode and load_mode != "auto":
            cmd += ["--load-mode", load_mode]
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
        if mmproj_path and self.mmproj_enabled_var.get():
            cmd += ["--mmproj", mmproj_path]
        # --- MTP draft model file: --spec-draft-model + draft-mtp spec.
        # The file only pays off with MTP speculation active, so an ON
        # toggle forces --spec-type draft-mtp (replacing none/ngram/...).
        mtp_path = self.mtp_model_var.get().strip()
        if mtp_path and self.mtp_enabled_var.get():
            cmd += ["--spec-draft-model", mtp_path]
            if "--spec-type" in cmd:
                cmd[cmd.index("--spec-type") + 1] = "draft-mtp"
            else:
                cmd += ["--spec-type", "draft-mtp"]
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
        self._set_busy_ui(True)
        self._ensure_pulse()

        self.load_start_time = time.time()
        self.model_ready = False
        self.load_status_var.set(self.t("loading_model"))
        self.load_status_label.config(bg=S("warn"), fg=S("bg"))
        self.speed_samples.clear()
        self.token_stats = {"gen": 0, "prompt": 0, "prompt_tps": None}
        self.speed_var.set("⚡ --")
        self._run_ctx_total = int(self.ctx_var.get())
        self._last_ctx = None
        self._ctx_slots_fresh = False
        self._ctx_log_est = 0
        self._pending_prompt_n = 0
        self.ctx_pill_var.set(self.t("ctx_pill_idle"))
        self._ctx_pill_color(S("fg_muted"))
        self.speed_label.config(bg=S("surface"), fg=S("info"))
        self.open_browser_btn.config(state=tk.DISABLED)

        env = self._server_env()
        if env is not None:
            self.log_write("Env: GGML_CUDA_GRAPH_OPT=1 (concurrent Q/K/V CUDA streams)", "info")
        thread = threading.Thread(target=self.run_process, args=(cmd, env), daemon=True)
        thread.start()
        self.root.after(1000, self._tick_uptime)

    def _server_env(self):
        """Environment for the llama-server child process.

        None = inherit the app's environment untouched. Otherwise a copy
        with GGML_CUDA_GRAPH_OPT=1 (concurrent Q/K/V CUDA streams) — the
        flag must reach only llama-server, never the whole app.
        """
        if not self.graph_opt_var.get():
            return None
        env = os.environ.copy()
        env["GGML_CUDA_GRAPH_OPT"] = "1"
        return env

    def run_process(self, cmd, env=None):
        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True
            )
            if env is not None:
                popen_kwargs["env"] = env
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

            # Live context counter: exact KV occupancy from /slots (n_past
            # per slot), tracked for the lifetime of THIS process object so
            # a quick restart can't leave a stale thread polling.
            ctx_thread = threading.Thread(
                target=self._poll_context, args=(self.port_var.get(),), daemon=True
            )
            ctx_thread.start()

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
        self._set_busy_ui(False)
        self._ensure_pulse()
        self.load_status_var.set(f"{self.t('model_loaded_prefix')} {elapsed_seconds:.1f}{self.t('model_loaded_suffix')}")
        self.load_status_label.config(bg=S("ok"), fg=S("bg"))
        self.log_write(f"Model finished loading in {elapsed_seconds:.1f}s.", "success")
        self.open_browser_btn.config(state=tk.NORMAL)
        # Header labels (Run Task + Inspector) must flip to RUNNING now.
        self._update_task_server_state()

    def _poll_context(self, port):
        """Accurate context counter from GET /slots.

        Verified against the live llama-server: per-slot occupancy is
        reported as n_prompt_tokens (grows during decode as prompt+decoded
        and keeps the final value after the task; older builds use n_past
        instead — both accepted). Slots that have never run omit the
        fields entirely, so a reading only counts when at least one slot
        actually reported a number — never paint a fake 0.
        Bounded to the process that spawned this thread; 1s cadence."""
        proc = self.process
        url = f"http://127.0.0.1:{port}/slots"
        fails = 0
        na_shown = False
        while proc is not None and proc.poll() is None:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                slots = data if isinstance(data, list) else (data.get("slots") or [])
                used = 0
                got = False
                for s in slots:
                    if not isinstance(s, dict):
                        continue
                    try:
                        v = s.get("n_prompt_tokens")
                        if v is None:
                            v = s.get("n_past")
                        if v is None:
                            continue
                        v = int(v)
                        # Builds that keep n_prompt_tokens at prompt length
                        # while decoding expose growth only in n_decoded;
                        # add it just then (this build already includes
                        # decoded, where dec <= v never triggers).
                        dec = 0
                        if s.get("is_processing"):
                            nt = s.get("next_token") or []
                            if isinstance(nt, list) and nt and isinstance(nt[0], dict):
                                dec = int(nt[0].get("n_decoded") or 0)
                        if dec > v:
                            v += dec
                        used += v
                        got = True
                    except (TypeError, ValueError):
                        continue
                fails = 0
                if got:
                    total = self._run_ctx_total or 0
                    self.root.after(
                        0, lambda u=used, t=total: self._render_ctx(u, t, True))
            except Exception:
                fails += 1
                # Only surface n/a once the model is actually up — during
                # load the endpoint simply isn't answering yet.
                if fails == 3 and not na_shown and self.model_ready:
                    na_shown = True
                    self.root.after(
                        0, lambda: self.ctx_pill_var.set(self.t("ctx_pill_na")))
            time.sleep(1.0)

    def _ctx_pill_color(self, color):
        for w in (getattr(self, "ctx_pill_lbl", None),
                  getattr(self, "task_ctx_lbl", None)):
            if w is not None:
                try:
                    w.config(fg=color)
                except tk.TclError:
                    pass

    def _render_ctx(self, used, total, fresh=False):
        """Paint the context pill: used / total · free, green while >30%
        of the window remains, amber under 30%, red under 10%. fresh=True
        marks a real /slots reading (vs the stdout fallback estimate)."""
        if not total or total <= 0:
            return
        if fresh:
            self._ctx_slots_fresh = True
        self._last_ctx = (used, total)
        left = max(0, total - used)
        free_w = self.t("ctx_pill_free")
        pre = "Ctx" if self.lang == "en" else "کانتکست"
        self.ctx_pill_var.set(f"{pre} {used:,} / {total:,} · {left:,} {free_w}")
        frac = left / total
        color = (S("danger") if frac < 0.10
                 else S("warn") if frac < 0.30 else S("ok"))
        self._ctx_pill_color(color)

    def _fmt_tok(self, n):
        """1234 -> '1.2K', 356 -> '356'."""
        return f"{n/1000:.1f}K" if n >= 10000 else str(n)

    def _account_tokens(self, n, is_prompt):
        if is_prompt:
            self.token_stats["prompt"] += n
            self._pending_prompt_n = n
        else:
            self.token_stats["gen"] += n
            # Fallback context estimate for builds whose /slots never
            # reports occupancy: llama-server re-sends the full
            # conversation as the prompt each request, so last prompt +
            # last generation ≈ tokens now sitting in the KV cache.
            est = (self._pending_prompt_n or 0) + n
            if est > 0:
                self._ctx_log_est = est
                if not self._ctx_slots_fresh and self._run_ctx_total:
                    self._render_ctx(est, self._run_ctx_total)
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
            return
        seg.append(f"{self._fmt_tok(st['gen'])} {self.t('speed_gen')}")
        seg.append(f"{self._fmt_tok(st['prompt'])} {self.t('speed_prm')}")
        self.speed_var.set("⚡ " + "  ·  ".join(seg))
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
        self._set_busy_ui(False)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self.model_ready = False
        self.load_status_var.set("Model not loaded")
        self.load_status_label.config(bg=S("surface"), fg=S("fg"))
        self.open_browser_btn.config(state=tk.DISABLED)
        self.uptime_var.set("")
        self._run_ctx_total = 0
        self._last_ctx = None
        self._ctx_slots_fresh = False
        self._ctx_log_est = 0
        self._pending_prompt_n = 0
        self.ctx_pill_var.set(self.t("ctx_pill_idle"))
        self._ctx_pill_color(S("fg_muted"))
        # Server died — both server-state headers go back to not running.
        try:
            self._update_task_server_state()
        except (tk.TclError, AttributeError):
            pass

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
