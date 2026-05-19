# REFRACT

Multi-agent code intelligence platform. Paste code, get back a fully refactored, tested, and documented version of it, with every step streamed live to the browser.

## How it works

Five LLM agents run in sequence. Each one receives the original code, targeted instructions from the orchestrator, and a summary of what previous agents did.

```
Lead Programmer      analyzes code, produces a structured execution plan
Syntax Fixer         PEP8, style, and linting fixes
Performance Optimizer  algorithmic rewrites, Big-O improvements
Test Generator       generates pytest unit tests, runs them, reports pass/fail
Docstring Writer     Google-style docstrings and inline comments
```

Results stream token-by-token to the UI via Server-Sent Events. The code editor updates live as each agent completes its pass.

## Stack

- **Backend**: FastAPI, Uvicorn, SSE streaming
- **Frontend**: Vanilla JS, Monaco Editor, Marked.js
- **LLM**: Provider-agnostic (Anthropic, Lava, or any OpenAI-compatible endpoint)

## Setup

```bash
git clone https://github.com/sm1else-bot/refract
cd refract

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add your API key to .env

uvicorn main:app --reload
# Open http://localhost:8000
```

## LLM Providers

Refract supports three backends, switchable via the `LLM_PROVIDER` environment variable.

### Anthropic

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

### Lava (OpenAI-compatible, cheap)

```env
LLM_PROVIDER=lava
LAVA_API_KEY=your_key
LAVA_MODEL=deepseek-chat
```

### Local inference via Ollama

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

```env
LLM_PROVIDER=openai
OPENAI_COMPAT_BASE_URL=http://localhost:11434/v1
OPENAI_COMPAT_MODEL=qwen2.5-coder:7b
OPENAI_COMPAT_API_KEY=ollama
```

### Local inference via vLLM

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

```env
LLM_PROVIDER=openai
OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
OPENAI_COMPAT_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
```

A 4-bit quantized 7B model requires roughly 4.5GB of VRAM.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `lava`, or `openai` |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Model ID |
| `OPENAI_COMPAT_BASE_URL` | `http://localhost:8000/v1` | Base URL for OpenAI-compatible endpoint |
| `OPENAI_COMPAT_MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Model name |
| `OPENAI_COMPAT_API_KEY` | `token` | API key (ignored by most local servers) |
| `LAVA_API_KEY` | | Lava API key |
| `LAVA_BASE_URL` | `https://api.lava.so/v1` | Lava base URL |
| `LAVA_MODEL` | `deepseek-chat` | Lava model ID |
