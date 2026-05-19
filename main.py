"""
REFRACT — Multi-Agent Code Intelligence Platform
FastAPI backend with SSE streaming pipeline.

LLM Backend is provider-agnostic:
  - "anthropic"  → Anthropic API (default)
  - "openai"     → Any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio)
  - "lava"       → Lava API (https://api.lava.so/v1) — OpenAI-compatible

Environment variables:
  LLM_PROVIDER              anthropic | openai | lava   (default: anthropic)
  ANTHROPIC_API_KEY         your Anthropic key
  ANTHROPIC_MODEL           default: claude-3-5-haiku-20241022
  OPENAI_COMPAT_BASE_URL    default: http://localhost:8000/v1  (vLLM default)
  OPENAI_COMPAT_MODEL       default: Qwen/Qwen2.5-Coder-7B-Instruct
  OPENAI_COMPAT_API_KEY     default: token (Ollama ignores this)
  LAVA_API_KEY              your Lava access token
  LAVA_BASE_URL             default: https://api.lava.so/v1
  LAVA_MODEL                default: deepseek-chat
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

import anthropic
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

LLM_PROVIDER            = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY       = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL         = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
OPENAI_COMPAT_BASE_URL  = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
OPENAI_COMPAT_MODEL     = os.getenv("OPENAI_COMPAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
OPENAI_COMPAT_API_KEY   = os.getenv("OPENAI_COMPAT_API_KEY", "token")
LAVA_API_KEY            = os.getenv("LAVA_API_KEY", "")
LAVA_BASE_URL           = os.getenv("LAVA_BASE_URL", "https://api.lava.so/v1")
LAVA_MODEL              = os.getenv("LAVA_MODEL", "deepseek-chat")
MAX_TOKENS              = 2048

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="REFRACT — Multi-Agent Code Intelligence")
app.mount("/static", StaticFiles(directory="static"), name="static")


class CodeRequest(BaseModel):
    code: str
    language: str = "python"


# ── LLM Backend ──────────────────────────────────────────────────────────────

async def _stream_anthropic(system: str, user: str) -> AsyncGenerator[str, None]:
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    async with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _stream_openai_compat(system: str, user: str) -> AsyncGenerator[str, None]:
    """
    OpenAI-compatible streaming endpoint.
    Works with: vLLM (`python -m vllm.entrypoints.openai.api_server ...`)
                Ollama (`ollama serve`, base_url = http://localhost:11434/v1)
                LM Studio, llama.cpp, etc.
    """
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST",
            f"{OPENAI_COMPAT_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_COMPAT_API_KEY}"},
            json={
                "model": OPENAI_COMPAT_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": MAX_TOKENS,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


async def _stream_lava(system: str, user: str) -> AsyncGenerator[str, None]:
    """Lava API — OpenAI-compatible endpoint at api.lava.so."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST",
            f"{LAVA_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LAVA_API_KEY}"},
            json={
                "model": LAVA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": MAX_TOKENS,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


async def llm_stream(system: str, user: str) -> AsyncGenerator[str, None]:
    if LLM_PROVIDER == "anthropic":
        async for tok in _stream_anthropic(system, user):
            yield tok
    elif LLM_PROVIDER == "lava":
        async for tok in _stream_lava(system, user):
            yield tok
    else:
        async for tok in _stream_openai_compat(system, user):
            yield tok


# ── Agent Definitions ─────────────────────────────────────────────────────────

AGENTS: dict[str, dict] = {
    "orchestrator": {
        "name": "Lead Programmer",
        "color": "#a78bfa",
        "system": (
            "You are the Lead Programmer in a multi-agent code intelligence system. "
            "Your job is to analyze submitted code and produce a precise execution plan "
            "for your specialist agents.\n\n"
            "Respond ONLY in valid JSON — no markdown fences, no preamble, no trailing text:\n"
            "{\n"
            '  "analysis": "2-3 sentence overall assessment",\n'
            '  "syntax_instructions": "exact style/lint issues to fix",\n'
            '  "performance_instructions": "specific bottlenecks and optimization targets",\n'
            '  "test_instructions": "which functions to test, edge cases to cover",\n'
            '  "docstring_instructions": "which functions need docs, what to highlight"\n'
            "}"
        ),
    },
    "syntax": {
        "name": "Syntax Fixer",
        "color": "#38bdf8",
        "system": (
            "You are the Syntax and Style specialist in a multi-agent code intelligence system.\n"
            "Fix all style, formatting, and linting issues per the Lead Programmer's instructions.\n\n"
            "Output format:\n"
            "1. Bullet-point summary of every issue fixed\n"
            "2. Complete corrected code in a fenced code block\n\n"
            "Be conservative: fix only what's wrong, don't restructure logic."
        ),
    },
    "performance": {
        "name": "Performance Optimizer",
        "color": "#fb923c",
        "system": (
            "You are the Performance Optimization specialist in a multi-agent code intelligence system.\n"
            "Improve algorithmic efficiency and runtime characteristics per the Lead Programmer's instructions.\n\n"
            "Actively replace inefficient patterns with idiomatic alternatives:\n"
            "- Manual sort loops → sorted() or list.sort()\n"
            "- Append loops → list/dict/set comprehensions\n"
            "- Manual index iteration → slicing, zip, enumerate\n"
            "- Redundant variables and intermediate lists → inline expressions\n"
            "- Repeated len() / division recalculated in loops → hoist to a constant\n"
            "Do not be conservative — rewrite aggressively for clarity and speed.\n\n"
            "Output format:\n"
            "1. Analysis of bottlenecks found (include Big-O comparisons where relevant)\n"
            "2. Fully rewritten optimized code in a fenced code block\n\n"
            "Explain your trade-offs clearly."
        ),
    },
    "test": {
        "name": "Test Generator",
        "color": "#4ade80",
        "system": (
            "You are the Test Generation specialist in a multi-agent code intelligence system.\n"
            "Write comprehensive pytest unit tests per the Lead Programmer's instructions.\n\n"
            "Output format:\n"
            "1. Testing strategy (which scenarios and why)\n"
            "2. Complete pytest test file in a fenced code block\n\n"
            "Cover: happy path, edge cases, boundary conditions, error/exception cases. "
            "Use parametrize where appropriate.\n\n"
            "IMPORTANT: The source code is available as 'solution.py' in the same directory. "
            "Always import from it like: `from solution import func_name`"
        ),
    },
    "docstring": {
        "name": "Docstring Writer",
        "color": "#f472b6",
        "system": (
            "You are the Documentation specialist in a multi-agent code intelligence system.\n"
            "Add Google-style docstrings and inline comments per the Lead Programmer's instructions.\n\n"
            "Output format:\n"
            "Complete code with all docstrings and comments added, in a fenced code block.\n\n"
            "Google docstring format: Args, Returns, Raises, Example sections. "
            "Be thorough but avoid restating what the code obviously does."
        ),
    },
}

PIPELINE_ORDER = ["syntax", "performance", "test", "docstring"]


# ── Test Runner ───────────────────────────────────────────────────────────────

async def _run_pytest(source_code: str, test_code: str) -> dict:
    def _blocking():
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as f:
                f.write(source_code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w", encoding="utf-8") as f:
                f.write(test_code)
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_solution.py", "-v", "--tb=short", "--no-header", "-q"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                timeout=30,
                env=env,
            )
            return {
                "passed": result.returncode == 0,
                "output": (result.stdout + result.stderr).strip(),
                "returncode": result.returncode,
            }
    return await asyncio.get_event_loop().run_in_executor(None, _blocking)


# ── SSE Helpers ───────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run_pipeline(code: str, language: str) -> AsyncGenerator[str, None]:
    """
    Orchestration loop:
      1. Lead Programmer analyzes the code and produces a JSON plan.
      2. Each specialist agent runs in sequence, receiving the plan + prior outputs.
      3. SSE events are emitted for real-time UI updates.
    """

    # ── Step 1: Orchestrator ──
    yield _sse("agent_start", {"agent": "orchestrator"})

    plan_raw = ""
    async for tok in llm_stream(
        system=AGENTS["orchestrator"]["system"],
        user=f"Analyze this {language} code and produce the execution plan:\n\n```{language}\n{code}\n```",
    ):
        plan_raw += tok
        yield _sse("agent_token", {"agent": "orchestrator", "token": tok})

    yield _sse("agent_done", {"agent": "orchestrator"})

    # Parse plan — be lenient with model output
    plan: dict = {}
    try:
        plan = json.loads(plan_raw.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", plan_raw, re.DOTALL)
        if match:
            try:
                plan = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    fallback_instructions = {
        "syntax_instructions":      "Fix all style and linting issues.",
        "performance_instructions": "Optimize for performance where possible.",
        "test_instructions":        "Generate comprehensive unit tests.",
        "docstring_instructions":   "Add complete Google-style docstrings.",
    }

    # ── Step 2: Specialist agents ──
    # current_code evolves as each code-producing agent improves it.
    # This ensures every agent works on the latest version, not always the original.
    current_code = code
    agent_outputs: dict[str, str] = {}

    # Agents that rewrite code and whose output should advance current_code.
    CODE_AGENTS = {"syntax", "performance", "docstring"}

    for agent_id in PIPELINE_ORDER:
        yield _sse("agent_start", {"agent": agent_id})

        instruction_key = f"{agent_id}_instructions"
        instructions = plan.get(instruction_key, fallback_instructions[instruction_key])

        # Each agent receives the latest version of the code, not always the original.
        context_parts = [
            f"Language: {language}",
            f"\nCurrent code:\n```{language}\n{current_code}\n```",
            f"\nLead Programmer's instructions for you:\n{instructions}",
        ]
        if agent_outputs:
            # Strip fenced code blocks from prior outputs — the code is already
            # incorporated into current_code above, so repeating it (truncated)
            # just confuses the model.
            context_parts.append("\n\nPrior agent notes (code already applied above):")
            for prev_id, prev_out in agent_outputs.items():
                notes = re.sub(r"```[\s\S]*?```", "[code block]", prev_out).strip()
                notes = notes[:1200] + ("…" if len(notes) > 1200 else "")
                context_parts.append(f"\n— {AGENTS[prev_id]['name']} —\n{notes}")

        user_msg = "\n".join(context_parts)

        output = ""
        async for tok in llm_stream(system=AGENTS[agent_id]["system"], user=user_msg):
            output += tok
            yield _sse("agent_token", {"agent": agent_id, "token": tok})

        agent_outputs[agent_id] = output

        # Advance current_code if this agent produced a code block.
        if agent_id in CODE_AGENTS:
            code_blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", output)
            if code_blocks:
                current_code = code_blocks[-1].rstrip()

        yield _sse("agent_done", {"agent": agent_id})

        # After test agent: run the generated tests against current_code.
        if agent_id == "test":
            code_blocks = re.findall(r"```[^\n]*\n([\s\S]*?)```", output)
            yield _sse("test_running", {})
            if code_blocks:
                try:
                    results = await _run_pytest(current_code, code_blocks[-1].rstrip())
                except Exception as e:
                    results = {"passed": False, "output": str(e), "returncode": -1}
            else:
                results = {"passed": False, "output": "No test code block found in agent output.", "returncode": -1}
            yield _sse("test_results", results)

    yield _sse("pipeline_complete", {"total_agents": len(PIPELINE_ORDER) + 1})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/process")
async def process_code(req: CodeRequest):
    return StreamingResponse(
        run_pipeline(req.code, req.language),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.get("/api/test-run")
async def test_run():
    """Debug endpoint: runs a trivial pytest to verify the test runner works."""
    code = "def add(a, b):\n    return a + b\n"
    test = "from solution import add\ndef test_add():\n    assert add(1, 2) == 3\n"
    result = await _run_pytest(code, test)
    return result


@app.get("/api/health")
async def health():
    model = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else (LAVA_MODEL if LLM_PROVIDER == "lava" else OPENAI_COMPAT_MODEL)
    return {"status": "ok", "provider": LLM_PROVIDER, "model": model}
