# llama.cpp GUI

> [نسخه فارسی](README.md)

A simple desktop app for running `llama-server` with local GGUF models, with no web interface required.

## Features

- Start and stop `llama-server` with options for context, threads, batch, flash attention, KV cache, and more
- Get suggested settings from the actual GGUF model metadata
- Monitor GPU, RAM, CPU, disk usage, and running processes
- Run prompts and view the response, timing, and token speed
- See a token-speed chart and switch between Persian and English

## Requirements

- Windows 10 or 11
- Python 3.9+ with Tkinter
- `llama-server.exe` from [llama.cpp](https://github.com/ggml-org/llama.cpp/releases)
- A `.gguf` model file
- NVIDIA drivers for GPU information

## Run

```powershell
python -m pip install -r requirements.txt
python llama_runner.py
```

Installing `psutil` is optional, but it improves CPU and RAM detection.

## Quick guide

1. In **Server Setup**, choose `llama-server.exe` and your model file.
2. Pick a preset or let the app analyze the model.
3. Start the server.
4. Enter a prompt in **Run Task** and run it.
5. View the response and token speed in **Task Inspector**.

| Tab | Purpose |
|---|---|
| Server Setup | Select the model, configure the server, and view logs |
| GPU Processes | Monitor hardware and processes |
| Run Task | Run prompts and view responses |
| Task Inspector | Show the token-speed chart |

## Troubleshooting

- **No GPU information:** install or update NVIDIA drivers and make sure `nvidia-smi` is available.
- **The server will not start:** check the executable path, server log, and GPU memory usage. Reduce context or batch size if needed.
- **The model is not recognized:** make sure the file is a valid GGUF model.

## Files

- `llama_runner.py`: the complete application
- `requirements.txt`: optional `psutil` dependency
