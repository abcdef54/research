"""
LocalMind — GSM8K ablation harness.

CURRENT EXPERIMENT (Experiment 1: Self-Consistency).
  Baseline established: Qwen2.5-7B-Instruct on GSM8K-100 with temp=0, the RAW question only, no
  system prompt, no structured output = 91/100. Structured output and system prompts each
  measurably HURT accuracy, so this experiment uses NEITHER. It isolates one variable only:
  test-time sampling (generate N -> majority vote).

  Graph:  setup -> generate -> finalize     (no rank, no select, no verify, no refinement)
    - generate : N independent PLAIN-TEXT samples of the raw question (HumanMessage only).
    - finalize : self-consistency majority vote over each sample's extracted final number.

  Modes (width = N samples):  low = 1 (reproduces the 91% baseline), medium = 3, high = 5.

  TEMPERATURE (important): low must run at temperature 0 (greedy) to reproduce the 91% baseline;
  medium/high MUST run at temperature > 0 (e.g. 0.7), otherwise the N samples are identical and
  the majority vote collapses to width=1. Temperature is taken from the eval config, per mode.

EXPERIMENT 2 (Can Ranking Beat Majority Vote?) — ADDED, fully config-flag gated.
  MOTIVATION (read this first): the ranker is a DOMAIN-GENERAL selector, not merely a GSM8K
  accuracy tweak. Majority voting fundamentally depends on ANSWER EQUALITY — it can only tally
  votes when candidate outputs are directly comparable (for GSM8K: identical final numbers). That
  assumption breaks for almost every general task: open-ended QA, summarization, code generation,
  RAG systems, research agents, essay generation — there is no canonical key to bucket-and-count,
  so "majority" is undefined. An LLM ranker has no such requirement: it compares candidates on
  their merits and picks one, regardless of whether they share a normalizable answer. GSM8K is used
  here ONLY because its exact-match answers make majority vote and Pass@N well-defined, giving a
  clean head-to-head baseline; a win here is evidence for a selector we can carry to the open-ended
  tasks above, where majority vote cannot even be applied.

  Same generator as Experiment 1 (plain text, HumanMessage only, NO system prompt, NO structured
  output). ONLY the selector changes; everything upstream is byte-identical. Driven by the config
  flag `rank_mode`:
    - "majority"            (default) -> `finalize` node  = self-consistency majority vote (Exp 1, unchanged)
    - "rank_no_reasoning"   (Exp 2A)  -> `rank_select` node = ONE plain-text ranking call -> Top-1
    - "rank_with_reasoning" (Exp 2B)  -> `rank_select` node = ONE structured {reasoning, winner} call -> Top-1

  Graph:  setup -> generate -> [route_selector by rank_mode] -> (finalize | rank_select | tournament_select) -> END
  The ranker makes EXACTLY ONE LLM call (0 at width=1). There is ONE generation round, NO later
  rounds, NO tournament, NO verifier, NO critic, NO refinement, NO sparse children — the selector
  is isolated. In 2B "reasoning" means STRUCTURED OUTPUT only (a `reasoning` field, emitted FIRST),
  NOT a hidden chain-of-thought and NOT an extra call. The chosen reasoning is stored on the winning
  candidate (`Candidate.rank_reasoning`) and surfaced in state.

  Benchmark widths (paper eval only, no architectural change): low=1, medium=3, high=5, extra=7, max=9.

  NB: the PARKED tournament `rank`/`_rank_group`/`select` below are the OLD iterative-search ranker
  and remain unused. Experiment 2's one-shot ranker (`rank_no_reasoning` / `rank_with_reasoning` +
  the `rank_select` node) is SEPARATE and is the only ranking code that runs.

EXPERIMENT 3A (Can Tournament Ranking Beat One-Shot Ranking?) — ADDED, fully config-flag gated.
  MOTIVATION: Experiment 2 found one-shot N-way ranking works at Width=3 but DEGRADES at Width=5 —
  the ranker saturates when asked to compare too many candidates in a single call. Experiment 3A
  keeps the SAME generator and the SAME one-shot rankers and changes ONLY how they are CALLED:
  instead of ranking all N candidates at once, it runs a HIERARCHICAL TOURNAMENT where every ranking
  call compares at most MAX_RANK_GROUP (=3) candidates. Pure orchestration — no new prompt, no
  regeneration, no verifier/critic, no abstention, no extra generation round. Driven by `rank_mode`:
    - "tournament_no_reasoning"   -> `tournament_select` node, per-group rank_no_reasoning  (3A-no)
    - "tournament_with_reasoning" -> `tournament_select` node, per-group rank_with_reasoning (3A-rsn)
  Tournament uses EVEN benchmark widths for balanced brackets (TOURNAMENT_MODE_CONFIG): low=1,
  medium=4, high=6, extra=8, max=10. Adds tournament_rounds / tournament_rank_calls /
  tournament_max_group_size; every Experiment 1/2 metric is preserved. The Experiment 1 (majority,
  `finalize`) and Experiment 2 (one-shot, `rank_select`) paths are UNCHANGED — tournament is an
  ADDITIONAL selector, not a replacement.

PARKED FOR LATER ABLATIONS (kept in the file, but NOT wired into the graph and NOT called):
  the full search-round generator (`_generate_search_round`), `rank` / `_rank_group`, `select`,
  `_pick_parents`, `_build_child_candidate_prompt`, `build_prompt`, `structured`,
  `ResponseWithThoughts`, `RankedCandidate` / `RankingResult`, `DIVERSITY_NUDGE`, `rank_prompt`,
  and the `route_*` helpers. To restore the search loop: re-add the rank/select nodes + edges and
  point `generate` at `_generate_search_round`.

Node convention: each node returns a PARTIAL update of AgentState; LangGraph merges it.
"""

import os
import re
import json
import time
import random
import asyncio
import hashlib
import dotenv
from datetime import date
from pathlib import Path
from functools import lru_cache
from uuid import uuid4
from collections import OrderedDict, Counter
from typing import TypedDict, Annotated, Literal, Dict, Optional

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

dotenv.load_dotenv()

# Where the JSON prompt files live. Overridable so this single file can be moved freely.
# Defaults to <this file's parent>/system_prompt to match the original layout.
SYSTEM_PROMPT_PATH = os.getenv(
    "LOCALMIND_PROMPT_DIR",
    os.path.join(Path(os.path.abspath(__file__)).parent, "system_prompt"),
)

# Max candidates compared in a single ranking call (PARKED rank node only; unused in Experiment 1).
RANK_GROUP = 4

# Experiment 3A: a single tournament ranking call may compare AT MOST this many candidates. Exp 2
# found one-shot ranking degrades as candidate count grows, so the bracket caps every call at 2–3
# candidates. Distinct from the PARKED RANK_GROUP above (that one is for the old iterative ranker).
MAX_RANK_GROUP = 3


# ───────────────────────── LLM access ─────────────────────────

@lru_cache(maxsize=8)
def _build_llm(model_name: str, base_url: str, temperature: float) -> ChatOpenAI:
    """Cached so repeated nodes reuse one client. Args are hashable on purpose.

    NOTE: no fixed seed here. A fixed seed (the old seed=42) makes identical prompts return
    identical text, which collapses pass@N -> pass@1. Sampling pins a FRESH random seed per call
    via `_seeded` — only meaningful at temperature > 0 (at temp 0 decoding is greedy/deterministic
    and the seed has no effect).
    """
    return ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",
        model=model_name,
        temperature=temperature,
    )


def get_llm(
    config: RunnableConfig,
    tools_enabled: bool = False,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """Build (or reuse) the local llama.cpp-backed client.

    - model_name comes from config; base_url from env (defaults to the local llama-server).
    - tools are bound ONLY when tools_enabled is True. During testing it is False, so the
      model has no tools at all.
    """
    cfg = config["configurable"]
    model_name = cfg.get("model_name", "qwen")
    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:18000/v1")
    temp = temperature if temperature is not None else cfg.get("temperature", 0.0)

    llm = _build_llm(model_name, base_url, temp)

    if tools_enabled:
        # Lazy import so the testing path does not require the tools module at all.
        from src.backend.agents.tools import read_emails, fetch_web_page, google_search
        llm = llm.bind_tools([read_emails, fetch_web_page, google_search])

    return llm


def _seeded(llm: ChatOpenAI, seed: int) -> ChatOpenAI:
    """Return a copy of the cached client pinned to a fresh `seed`, sharing the same underlying
    HTTP client (verified: model_copy does not rebuild the connection pool). Distinct seeds make
    the N samples genuinely diverse instead of duplicates (only at temperature > 0)."""
    return llm.model_copy(update={"seed": seed})


def structured(llm, schema):
    """[PARKED — unused in Experiment 1] Wrap with_structured_output (json_schema).
    Kept for later ablations. Structured output was found to HURT GSM8K accuracy (91 -> 83), so it
    is intentionally NOT used in the current experiment."""
    return llm.with_structured_output(schema, method="json_schema")


# ───────────────────────── Prompt assembly ─────────────────────────

def get_agent_instruction(node: str) -> str:
    with open(os.path.join(SYSTEM_PROMPT_PATH, "agent_prompt.json"), "r", encoding="utf-8") as f:
        return json.load(f)[node]


def get_localmind_system_instruction_with_personality(personality: str) -> str:
    with open(os.path.join(SYSTEM_PROMPT_PATH, "system_instructions.json"), "r", encoding="utf-8") as f:
        personalities = json.load(f)
    return personalities[personality].replace("{current_date_str}", date.today().isoformat())


def build_prompt(state: "AgentState", config: RunnableConfig, node: Optional[str]) -> str:
    """[PARKED — unused in Experiment 1] Assemble personality + per-node instruction + grounding.
    System prompts were found to HURT GSM8K accuracy (91 -> 83), so the current experiment sends
    HumanMessages only and never calls this. Kept for later ablations."""
    cfg = config["configurable"]
    personality = cfg["personality"]
    tools_enabled = cfg.get("tools_enabled", False)

    parts = [get_localmind_system_instruction_with_personality(personality)]
    if node:
        parts.append(get_agent_instruction(node))

    parts.append(f"### User Question: {state['user_query']}")

    if state.get("retrieved_context"):
        parts.append(f"### Retrieved Context: {state['retrieved_context']}")
    if state.get("tool_results"):
        parts.append(f"### Tool Results: {state['tool_results']}")

    if tools_enabled:
        parts.append("Answer using your own knowledge, the retrieved context, and tools when helpful.")
    else:
        parts.append("Answer using your own knowledge.")

    return "\n\n".join(parts)


# [PARKED — unused in Experiment 1] Diversity nudge for the search-round generator.
DIVERSITY_NUDGE = (
    "Solve this with a genuine, self-contained line of reasoning. Explore your own assumptions, "
    "decomposition, representation, and order of calculation rather than defaulting to the most "
    "obvious framing. Do not merely restate the question."
)


def rank_prompt(state: "AgentState") -> str:
    """[PARKED — unused in Experiment 1] System prompt for the rank node."""
    return (
        "You are comparing candidate solutions to the user's question RELATIVE to one another, "
        "to decide which are most likely correct.\n\n"
        f"### User Question: {state['user_query']}\n\n"
        "Compare their reasoning and their final answers against each other. Where candidates "
        "converge on the same final answer, that is weak evidence in its favour; where they "
        "disagree, judge which chain of reasoning and which final number actually holds up. "
        "Explain the major disagreements, then return an ordered ranking (best first) by candidate_id."
    )


def _extract_final_number(text: str) -> Optional[str]:
    """Best-effort final numeric answer from a candidate (GSM8K-style). Priority mirrors the eval
    extractor: '#### N' -> 'answer is N' -> '\\boxed{N}' -> last number in the text. Returns None
    if no number is present. Used by the majority vote (and the parked rank/refinement code)."""
    if not text:
        return None
    for pat in (
        r"####\s*(-?\d[\d,]*\.?\d*)",
        r"answer\s+is\s*:?\s*\$?\s*(-?\d[\d,]*\.?\d*)",
        r"\\boxed\{\s*(-?\d[\d,]*\.?\d*)\s*\}",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def same_answer(a, b):
    """Numeric-equivalence for extracted GSM8K answers, so semantically identical values are NOT
    counted as disagreements (e.g. '243' == '243.00', '75' == '75.00', '12' == '12.'). Falls back to
    a trimmed string compare for non-numeric answers. Used only by the rank_agreed_with_majority
    log; the selector itself is unaffected."""
    if a is None or b is None:
        return a is b

    try:
        return abs(float(a) - float(b)) < 1e-6
    except:
        return str(a).strip() == str(b).strip()


def _sample_idx(state: "AgentState", config: RunnableConfig) -> str:
    """Per-sample tag for log lines, so logs from concurrently-running GSM8K questions don't
    interleave into something that looks impossible. Prefers config['configurable']['sample_idx']
    (set by the eval harness); otherwise falls back to a short stable hash of the question so every
    log line within one invocation shares a tag even before the eval wires sample_idx. Logging only:
    not stored in state, not a metric."""
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    idx = cfg.get("sample_idx")
    if idx is not None:
        return str(idx)
    q = state.get("user_query") or ""
    return hashlib.sha1(q.encode("utf-8")).hexdigest()[:6] if q else "?"


# ── TEMPORARY token-usage debug logging (console only; nothing stored in state/CSV/metrics) ──
# Remove this block when context-usage inspection is no longer needed.

def _estimate_tokens(text: str) -> int:
    """Approximate token count when the API doesn't report usage. Prefers tiktoken; falls back to a
    ~4-chars/token heuristic. Estimate only — used purely for the temporary debug print."""
    if not text:
        return 0
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def _token_usage_from_message(msg) -> Optional[tuple]:
    """(prompt, completion, total) from an LLM response message if the API exposed usage, else None.
    Checks usage_metadata first, then response_metadata['token_usage'] / ['usage']."""
    if msg is None:
        return None
    um = getattr(msg, "usage_metadata", None)
    if um:
        p, c, t = um.get("input_tokens"), um.get("output_tokens"), um.get("total_tokens")
        if p is not None or c is not None or t is not None:
            p, c = p or 0, c or 0
            return p, c, (t if t is not None else p + c)
    rm = getattr(msg, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage")
    if tu:
        p = tu.get("prompt_tokens", tu.get("input_tokens"))
        c = tu.get("completion_tokens", tu.get("output_tokens"))
        t = tu.get("total_tokens")
        if p is not None or c is not None or t is not None:
            p, c = p or 0, c or 0
            return p, c, (t if t is not None else p + c)
    return None


def _log_token_usage(idx: str, label: str, msg=None, prompt_text: str = "", completion_text: str = "") -> None:
    """TEMPORARY: print prompt/completion/total tokens for one LLM call. Uses real usage metadata
    when available, otherwise an approximate tokenizer estimate. Console only — stores nothing."""
    usage = _token_usage_from_message(msg)
    if usage is not None:
        p, c, t = usage
    else:
        p = _estimate_tokens(prompt_text)
        c = _estimate_tokens(completion_text)
        t = p + c
    print(f"[IDX {idx}] {label}")
    print(f"Prompt Tokens: {p}")
    print(f"Completion Tokens: {c}")
    print(f"Total Tokens: {t}\n")


# ───────────────────────── Data structures ─────────────────────────

class Candidate(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)  # stable identity for lineage
    answer: str = ""
    trace: str = ""                                       # empty in Experiment 1 (no reasoning trace)
    parents: list[str] = Field(default_factory=list)      # empty in Experiment 1 (no lineage)
    rank_reasoning: Optional[str] = None                  # Exp 2B: ranker's reasoning, set on the winner


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str

    # routing / mode
    message_intent: Literal["chat", "knowledge", "code"]   # kept for the future; always 'chat' in testing
    reasoning_mode: Literal["low", "medium", "high", "extra", "max"]

    # grounding context (unused in chat-only testing, kept for forward-compat)
    retrieved_context: str
    tool_results: str

    # search
    pool: list[Candidate]
    best: Optional[Candidate]

    # loop control
    iteration: int
    max_iterations: int
    width: int                 # number of independent samples
    budget_remaining: int
    final_answer: Optional[str]
    sampled_answers: list[str]
    sampled_numbers: list[Optional[str]]
    vote_distribution: dict
    unique_answers: int

    # selector (Experiment 2) — logged for every mode (majority included) for a uniform table
    rank_mode: str                          # "majority" | "rank_no_reasoning" | "rank_with_reasoning"
    rank_latency: float                     # seconds spent in the ranking call (0.0 for majority / width=1)
    rank_calls: int                         # ranking LLM calls (0 for majority / width=1, else exactly 1)
    winner_candidate_id: Optional[str]      # id of the selected candidate
    rank_reasoning: Optional[str]           # 2B reasoning text; None for majority / 2A (nullable)
    rank_parse_failed: bool                 # ranker reply couldn't be parsed to a label (fallback used); robustness
    rank_agreed_with_majority: bool         # ranker's final answer == the (no-LLM) majority winner's; divergence

    # selector (Experiment 3A — hierarchical tournament). Populated ONLY by `tournament_select`;
    # absent (read as 0 by the eval) for majority / one-shot runs, so those paths stay unchanged.
    tournament_rounds: int                  # bracket rounds executed (0 if no ranking call, e.g. width=1)
    tournament_rank_calls: int              # ranking LLM calls made across the tournament (== rank_calls)
    tournament_max_group_size: int          # largest candidate group submitted to a single ranking call


# ───────────── Structured output schemas (PARKED — unused in Experiment 1) ─────────────

class ResponseWithThoughts(BaseModel):
    reasoning: str = Field(..., description="Key reasoning steps worked through BEFORE the response.")
    response: str = Field(..., description="The final answer to the user's request.")


class RankedCandidate(BaseModel):
    candidate_id: str = Field(..., description="The integer label of the candidate being placed.")
    rank: int = Field(..., description="1 = best. Strictly increasing, no ties.")
    rationale: str = Field(default="", description="Why this candidate sits at this rank.")


class RankingResult(BaseModel):
    reasoning: str = Field(..., description="Compare candidates against each other BEFORE ordering them.")
    rankings: list[RankedCandidate] = Field(..., description="Every candidate, ordered best-first by `rank`.")


# ───────────── Structured output schema (ACTIVE — Experiment 2B only) ─────────────

class RankWithReasoning(BaseModel):
    """Experiment 2B one-shot ranking output. Exactly ONE structured call; no extra LLM calls.

    Field order is INTENTIONAL and must stay `reasoning` then `winner`: prior experiments indicate
    schema field ordering influences generation quality, and emitting the justification FIRST lets
    it condition the final pick. `winner` is the single candidate label (e.g. 'A', 'B', 'C')."""
    reasoning: str = Field(..., description="Compare the candidates and justify which is correct, BEFORE naming the winner.")
    winner: str = Field(..., description="The single letter label (e.g. 'A', 'B', 'C') of the most likely correct candidate.")


# ───────────────────────── Mode / difficulty config ─────────────────────────

# Experiment 1: width = number of self-consistency samples; no loop, so iterations = 0.
# budget = number of generation calls (= width).
MODE_CONFIG = {
    "low":    {"max_width": 1, "max_iterations": 0, "max_budget": 1},   # reproduces the raw baseline
    "medium": {"max_width": 3, "max_iterations": 0, "max_budget": 3},
    "high":   {"max_width": 5, "max_iterations": 0, "max_budget": 5},
    "extra":  {"max_width": 7, "max_iterations": 0, "max_budget": 7},   # Experiment 2 benchmark width only
    "max":    {"max_width": 9, "max_iterations": 0, "max_budget": 9},   # Experiment 2 benchmark width only
}

# Experiment 3A: tournament ranking uses EVEN widths so brackets split into balanced groups of 2–3
# (e.g. 4->[2,2], 6->[3,3], 8->[2,2,2,2], 10->[3,3,2,2]). Selected by `setup` ONLY when rank_mode is
# a tournament mode; the standard MODE_CONFIG above is untouched so Experiment 1/2 widths are unchanged.
TOURNAMENT_MODE_CONFIG = {
    "low":    {"max_width": 1,  "max_iterations": 0, "max_budget": 1},   # width 1 -> 0 ranking calls
    "medium": {"max_width": 4,  "max_iterations": 0, "max_budget": 4},
    "high":   {"max_width": 6,  "max_iterations": 0, "max_budget": 6},
    "extra":  {"max_width": 8,  "max_iterations": 0, "max_budget": 8},
    "max":    {"max_width": 10, "max_iterations": 0, "max_budget": 10},
}

# Internal estimate (PARKED governor; only consulted when bypass_governor=False — eval uses True).
STEP_CONFIG = {
    "single": {"width": 1, "iterations": 0, "budget": 1},
    "few":    {"width": 1, "iterations": 1, "budget": 3},
    "multi":  {"width": 2, "iterations": 2, "budget": 8},
    "deep":   {"width": 4, "iterations": 3, "budget": 30},
}


def get_config(mode: str, steps: str, bypass_governor: bool = False, tournament: bool = False) -> Dict:
    """mode = effort ceiling, reasoning_steps = effort needed; run with min(ceiling, needed).
    With bypass_governor (eval), the mode ceiling drives compute directly so modes compare uniformly.
    tournament=True selects TOURNAMENT_MODE_CONFIG (Experiment 3A even widths) instead of MODE_CONFIG;
    the two tables share the same mode keys, so every other code path is unaffected."""
    ceil = (TOURNAMENT_MODE_CONFIG if tournament else MODE_CONFIG)[mode]
    if bypass_governor:
        return {"max_iterations": ceil["max_iterations"], "width": ceil["max_width"], "budget": ceil["max_budget"]}
    need = STEP_CONFIG[steps]
    return {
        "max_iterations": min(ceil["max_iterations"], need["iterations"]),
        "width": min(ceil["max_width"], need["width"]),
        "budget": min(ceil["max_budget"], need["budget"]),
    }


# ───────────────────────── Nodes (ACTIVE) ─────────────────────────

def setup(state: AgentState, config: RunnableConfig) -> dict:
    """Pure-Python entry (NO llm call). Reads mode + governor settings, seeds the loop state.
    RETURNS: user_query, message_intent, reasoning_mode, width, max_iterations, budget_remaining,
             iteration, pool, best, rank_mode."""
    cfg = config["configurable"]
    mode = cfg["reasoning_mode"]
    bypass = cfg.get("bypass_governor", False)
    steps = cfg.get("reasoning_steps", "deep")  # only used when bypass=False
    # Experiment 3A: tournament modes use the EVEN-width table (balanced brackets); all other modes
    # (majority / one-shot) keep the standard MODE_CONFIG widths, so their behavior is unchanged.
    rank_mode = cfg.get("rank_mode", "majority")
    is_tournament = rank_mode in ("tournament_no_reasoning", "tournament_with_reasoning")
    loop_cfg = get_config(mode, steps, bypass, tournament=is_tournament)

    return {
        "user_query": state["messages"][-1].content,
        "message_intent": "chat",
        "reasoning_mode": mode,
        "width": loop_cfg["width"],
        "max_iterations": loop_cfg["max_iterations"],
        "budget_remaining": loop_cfg["budget"],
        "iteration": 0,
        "pool": [],
        "best": None,
        # Experiment 2/3A selector flag (default "majority" => Experiment-1 path is unchanged).
        "rank_mode": rank_mode,
    }


async def generate(state: AgentState, config: RunnableConfig) -> dict:
    """Experiment 1 — Self-Consistency sampling. Produce `width` INDEPENDENT samples of the RAW
    user question: HumanMessage(s) only, NO system prompt, NO structured output, plain text. Each
    sample uses a fresh random seed (only meaningful at temperature > 0; see MODE_CONFIG note).
    RETURNS: pool, iteration, budget_remaining.

    (The full search-round generator — structured output, diversity nudge, sparse refinement — is
    preserved unused as `_generate_search_round` for later ablations.)"""
    width = state["width"]
    llm = get_llm(config)  # plain client; temperature from config (0.0 for low, >0 for medium/high)

    idx = _sample_idx(state, config)
    print(f"\n[IDX {idx}] ==================== Generate Node (Self-Consistency) ====================\n")
    print(f"[IDX {idx}] Question: {state['user_query']}")
    print(f"[IDX {idx}] Width (N samples): {width}")

    # Raw question ONLY — no SystemMessage, no build_prompt, no DIVERSITY_NUDGE, no structured().
    prompts = [state["messages"] for _ in range(width)]
    responses = await asyncio.gather(*[
        _seeded(llm, random.randint(0, 2**31 - 1)).ainvoke(p) for p in prompts
    ])

    pool = []
    _gen_prompt_text = " ".join(
        m.content for m in state["messages"] if isinstance(getattr(m, "content", None), str)
    )
    for i, r in enumerate(responses):
        content = r.content if isinstance(r.content, str) else str(r.content)
        pool.append(Candidate(answer=content, trace="", parents=[]))
        print(f"[IDX {idx}] Sample {i}: id={pool[-1].id[:8]} final={_extract_final_number(content)!r}")
        # TEMPORARY token-usage debug (one block per generation call)
        _log_token_usage(idx, "Generate Call", msg=r, prompt_text=_gen_prompt_text, completion_text=content)
    print(f"\n[IDX {idx}] ==========================================================================\n")

    return {
        "pool": pool,
        "iteration": state["iteration"] + 1,
        "budget_remaining": state["budget_remaining"] - len(prompts),  # budget == generation calls
    }


def finalize(state: AgentState, config: RunnableConfig) -> dict:
    """Self-Consistency majority vote. Extract each candidate's final number, tally by value, and
    return the EARLIEST candidate that produced the most-voted value. Deterministic: ties between
    values break by earliest first-appearance; ties within the winning value by earliest candidate.
    (width=1 reduces to "return the single sample" — i.e. the raw baseline.)

    SCOPE LIMIT (why Experiment 2's ranker exists): this selector is only definable when candidate
    outputs are directly comparable — here, identical extracted final numbers. That answer-equality
    assumption does NOT generalize to open-ended QA, summarization, code generation, RAG, research
    agents, or essay generation, where there is no canonical key to bucket-and-count. For those, use
    the domain-general `rank_select` selector (rank_mode != "majority"). GSM8K is the comparison
    ground precisely because exact-match answers make this vote — and Pass@N — well-defined.
    RETURNS: final_answer, messages, sampled_answers, sampled_numbers, vote_distribution,
             unique_answers, and the selector-log fields (rank_mode='majority', rank_latency=0,
             rank_calls=0, winner_candidate_id, rank_reasoning=None, rank_parse_failed=False,
             rank_agreed_with_majority=True)."""
    pool = state.get("pool", [])
    idx = _sample_idx(state, config)
    print(f"\n[IDX {idx}] ==================== Finalize Node (Majority Vote) ====================\n")
    if not pool:
        print(f"[IDX {idx}] Empty pool.")
        return {
            "final_answer": "",
            "messages": [AIMessage(content="")],
            "sampled_answers": [],
            "sampled_numbers": [],
            "vote_distribution": {},
            "unique_answers": 0,
            "rank_mode": "majority",
            "rank_latency": 0.0,
            "rank_calls": 0,
            "winner_candidate_id": None,
            "rank_reasoning": None,
            "rank_parse_failed": False,
            "rank_agreed_with_majority": True,
        }

    # Group candidates by extracted final value, preserving first-appearance order.
    groups: "OrderedDict[Optional[str], list[Candidate]]" = OrderedDict()
    for c in pool:
        val = _extract_final_number(c.answer)
        groups.setdefault(val, []).append(c)

    print(f"[IDX {idx}] Vote tally (value -> count):")
    for val, members in groups.items():
        print(f"[IDX {idx}]   {val!r}: {len(members)}")

    # Highest count wins. On a tie keep the value seen FIRST: OrderedDict iterates in
    # first-appearance order and we only replace on a STRICTLY greater count.
    winning_members, winning_val = None, None
    for val, members in groups.items():
        if winning_members is None or len(members) > len(winning_members):
            winning_members, winning_val = members, val

    winner = winning_members[0]   # earliest candidate among those with the winning value
    answer = winner.answer
    print(f"\n[IDX {idx}] Winner value: {winning_val!r} ({len(winning_members)} vote(s)) -> candidate {winner.id[:8]}")
    print(f"\n[IDX {idx}] =======================================================================\n")
    sampled_answers = [
        c.answer
        for c in state["pool"]
    ]

    sampled_numbers = [
        _extract_final_number(c.answer)
        for c in state["pool"]
    ]

    counter = Counter(sampled_numbers)
    vote_distribution = dict(counter)
    unique_answers = len(set(sampled_numbers))

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "sampled_answers": sampled_answers,
        "sampled_numbers": sampled_numbers,
        "vote_distribution": vote_distribution,
        "unique_answers": unique_answers,
        # Uniform selector logging — majority makes no ranking call.
        "rank_mode": "majority",
        "rank_latency": 0.0,
        "rank_calls": 0,
        "winner_candidate_id": winner.id,
        "rank_reasoning": None,
        "rank_parse_failed": False,            # majority never parses a ranker reply
        "rank_agreed_with_majority": True,     # majority trivially agrees with itself
    }


# ───────────────────── Experiment 2 selector (ACTIVE — one-shot ranking) ─────────────────────
# A DOMAIN-GENERAL selector. Unlike majority vote — which can only tally when candidates share a
# normalizable answer (identical GSM8K final numbers) and is therefore undefined for open-ended QA,
# summarization, code generation, RAG, research agents, or essay generation — an LLM ranker compares
# candidates on their merits and picks one with no equality assumption. GSM8K is used only because
# its exact-match answers let us benchmark this ranker head-to-head against majority vote and Pass@N.
#
# Mechanics: Generate N -> rank ALL candidates in ONE call -> return Top-1. No tournament, no
# refinement, no verifier, no critic, no sparse children. The generator (above) is untouched: ONLY
# the selector changes. Two sub-modes via config `rank_mode`: "rank_no_reasoning" (2A) and
# "rank_with_reasoning" (2B). Both make EXACTLY ONE ranking LLM call (zero at width=1). These are
# SEPARATE from the PARKED tournament `rank`/`_rank_group` below.

def _candidate_labels(n: int) -> list[str]:
    """Stable A, B, C, … labels (benchmark widths 1..9 stay within A..I)."""
    return [chr(ord("A") + i) for i in range(n)]


def _build_ranking_block(user_query: str, candidates: list["Candidate"], labels: list[str]) -> str:
    """Question + each candidate's FULL plain-text answer under a letter label. Deliberately does
    NOT inject extracted numbers or vote counts, so the ranker judges the candidates themselves and
    stays independent of majority voting. Showing whole answers (not a normalized key) is also what
    keeps this selector TASK-AGNOSTIC — it works the same for prose, code, or summaries, where no
    comparable answer key exists."""
    parts = [f"Question:\n{user_query}\n"]
    for label, c in zip(labels, candidates):
        parts.append(f"Candidate {label}:\n{c.answer}\n")
    return "\n".join(parts)


def _no_reasoning_prompt(user_query: str, candidates: list["Candidate"], labels: list[str]) -> str:
    """Exp 2A prompt: choose one label, plain text only, no explanation / no reasoning field."""
    opts = " or ".join(labels)
    return (
        _build_ranking_block(user_query, candidates, labels)
        + "\nDetermine which answer is most likely correct.\n"
        + "Consider:\n"
        + "- arithmetic correctness\n"
        + "- logical consistency\n"
        + "- whether the answer fully addresses the question\n"
        + f"Respond ONLY with a single letter: {opts}.\n"
    )


def _with_reasoning_prompt(user_query: str, candidates: list["Candidate"], labels: list[str]) -> str:
    """Exp 2B prompt: same comparison, but the model fills the {reasoning, winner} schema."""
    opts = ", ".join(labels)
    return (
        _build_ranking_block(user_query, candidates, labels)
        + "\nChoose the single candidate whose final answer is most likely correct.\n"
        + f"`winner` must be exactly one of these labels: {opts}.\n"
        + "First work through your reasoning comparing the candidates, then give the winner."
    )


def _parse_letter_choice(text: Optional[str], labels: list[str]) -> Optional[str]:
    """Map a free-text response to one of `labels`. Tolerates 'B', 'Candidate B', 'B.', 'The answer
    is B', etc. Reading-order scan so the FIRST stated label wins. Returns None if nothing matches
    (caller decides the fallback — never falls back to majority voting)."""
    if not text:
        return None
    label_set = set(labels)
    up = text.strip().upper()
    if up in label_set:                       # exact single-letter reply
        return up
    m = re.search(r"\b([A-Z])\b", up)         # first standalone letter token
    if m and m.group(1) in label_set:
        return m.group(1)
    for ch in up:                             # first valid label char in reading order
        if ch in label_set:
            return ch
    return None


async def rank_no_reasoning(
    user_query: str, candidates: list["Candidate"], llm, idx: str = "?"
) -> "tuple[Candidate, Optional[str], int, bool]":
    """Experiment 2A. ONE plain-text ranking call over ALL candidates; parse the letter; return the
    winning Candidate. No structured output, no reasoning field, no extra calls. Domain-general: it
    compares the candidate outputs themselves and needs no answer-equality key, so it applies equally
    to non-numeric tasks (QA, summaries, code, essays) where majority vote is undefined.
    RETURNS: (winner, rank_reasoning=None, rank_calls=1, parse_failed). `parse_failed` is True when
    the reply maps to no valid label — a fallback candidate is still returned. Independent of majority."""
    labels = _candidate_labels(len(candidates))
    label_to_cand = dict(zip(labels, candidates))
    prompt = _no_reasoning_prompt(user_query, candidates, labels)

    resp = await llm.ainvoke([HumanMessage(content=prompt)])   # exactly ONE call
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    # TEMPORARY token-usage debug
    _log_token_usage(idx, "Rank Call", msg=resp, prompt_text=prompt, completion_text=text)

    label = _parse_letter_choice(text, labels)
    parse_failed = label is None
    if parse_failed:
        print(f"[IDX {idx}] [rank_no_reasoning] unparseable response {text!r}; falling back to candidate {labels[0]}")
        label = labels[0]
    return label_to_cand[label], None, 1, parse_failed


async def rank_with_reasoning(
    user_query: str, candidates: list["Candidate"], llm, idx: str = "?"
) -> "tuple[Candidate, Optional[str], int, bool]":
    """Experiment 2B. ONE structured ranking call over ALL candidates using the RankWithReasoning
    schema ({reasoning, winner}, reasoning FIRST). "Reasoning" here is the STRUCTURED FIELD only —
    not a hidden chain-of-thought, not a critic, not an extra call.
    RETURNS: (winner, rank_reasoning, rank_calls=1, parse_failed). `parse_failed` is True when the
    `winner` field maps to no valid label — a fallback candidate is still returned."""
    labels = _candidate_labels(len(candidates))
    label_to_cand = dict(zip(labels, candidates))
    prompt = _with_reasoning_prompt(user_query, candidates, labels)

    chain = structured(llm, RankWithReasoning)                 # exactly ONE structured call
    result: RankWithReasoning = await chain.ainvoke([HumanMessage(content=prompt)])
    # TEMPORARY token-usage debug. Structured output returns the parsed object (no raw message),
    # so real usage isn't exposed here -> approximate estimate from the prompt + serialized result.
    _log_token_usage(idx, "Rank Call", msg=None, prompt_text=prompt, completion_text=result.model_dump_json())

    label = _parse_letter_choice(result.winner, labels)
    parse_failed = label is None
    if parse_failed:
        print(f"[IDX {idx}] [rank_with_reasoning] unparseable winner {result.winner!r}; falling back to candidate {labels[0]}")
        label = labels[0]
    return label_to_cand[label], result.reasoning, 1, parse_failed


def _self_consistency_metrics(pool: list["Candidate"]):
    """Same per-sample metrics finalize logs, factored out so the ranking path reports them too
    (this is what lets the eval compute Pass@N / Selection Gap / Majority Failures uniformly).
    NOTE: these are GSM8K-benchmark instrumentation (they rely on numeric answer extraction). The
    ranker itself does NOT consume them — it never sees vote counts — so it carries over unchanged
    to tasks where these numeric metrics are meaningless."""
    sampled_answers = [c.answer for c in pool]
    sampled_numbers = [_extract_final_number(c.answer) for c in pool]
    vote_distribution = dict(Counter(sampled_numbers))
    unique_answers = len(set(sampled_numbers))
    return sampled_answers, sampled_numbers, vote_distribution, unique_answers


def _majority_winner(pool: list["Candidate"]) -> Optional["Candidate"]:
    """The self-consistency majority winner, computed with `finalize`'s EXACT logic and NO LLM call:
    group candidates by extracted final number in first-appearance order, take the value with the
    most votes (ties -> earliest value seen, via OrderedDict + strictly-greater replacement), and
    return the EARLIEST candidate carrying that value. Used by `rank_select` to log
    `rank_agreed_with_majority` without re-running any model. Returns None for an empty pool.
    (Kept byte-equivalent to finalize; the smoke test asserts the two pick the same candidate.)"""
    if not pool:
        return None
    groups: "OrderedDict[Optional[str], list[Candidate]]" = OrderedDict()
    for c in pool:
        groups.setdefault(_extract_final_number(c.answer), []).append(c)
    winning_members = None
    for _, members in groups.items():
        if winning_members is None or len(members) > len(winning_members):
            winning_members = members
    return winning_members[0]


async def rank_select(state: AgentState, config: RunnableConfig) -> dict:
    """Experiment 2 selector node. Generation already produced `pool`; here we rank ALL candidates
    in ONE call and return Top-1. This is the DOMAIN-GENERAL selector: it requires no answer-equality
    key, so unlike `finalize`'s majority vote it remains well-defined for open-ended QA, summarization,
    code generation, RAG, research agents, and essay generation. GSM8K is benchmarked here only
    because exact-match answers make the majority-vote / Pass@N comparison possible.
    Sub-mode from config `rank_mode`:
      - "rank_with_reasoning" -> structured {reasoning, winner}  (2B)
      - otherwise             -> plain-text single-letter choice (2A)
    Width=1 short-circuits (one candidate, no call). Preserves every Experiment-1 metric and adds
    rank_mode / rank_latency / rank_calls / winner_candidate_id / rank_reasoning, plus
    rank_parse_failed (robustness: ranker reply couldn't be parsed, fallback used) and
    rank_agreed_with_majority (divergence: ranker's final answer vs the no-LLM majority winner).
    RETURNS: final_answer, messages, the self-consistency metrics, and the selector-log fields."""
    cfg = config["configurable"]
    rank_mode = state.get("rank_mode") or cfg.get("rank_mode", "rank_no_reasoning")
    pool = state.get("pool", [])
    user_query = state.get("user_query", "")
    idx = _sample_idx(state, config)

    print(f"\n[IDX {idx}] ==================== Rank-Select Node (Experiment 2) ====================\n")
    print(f"[IDX {idx}] Rank mode: {rank_mode}")
    print(f"[IDX {idx}] Candidates: {len(pool)}")

    # Log the pool we are about to rank (id + extracted final number). Lets us verify the selected
    # candidate actually exists in THIS question's pool when async logs interleave across samples.
    for candidate in pool:
        print(f"[IDX {idx}] id={candidate.id[:8]} final={_extract_final_number(candidate.answer)}")

    sampled_answers, sampled_numbers, vote_distribution, unique_answers = _self_consistency_metrics(pool)

    if not pool:
        print(f"[IDX {idx}] Empty pool.")
        print(f"\n[IDX {idx}] =========================================================================\n")
        return {
            "final_answer": "",
            "messages": [AIMessage(content="")],
            "sampled_answers": [],
            "sampled_numbers": [],
            "vote_distribution": {},
            "unique_answers": 0,
            "rank_mode": rank_mode,
            "rank_latency": 0.0,
            "rank_calls": 0,
            "winner_candidate_id": None,
            "rank_reasoning": None,
            "rank_parse_failed": False,
            "rank_agreed_with_majority": True,
        }

    rank_latency = 0.0
    rank_calls = 0
    rank_reasoning: Optional[str] = None
    rank_parse_failed = False

    if len(pool) <= 1:
        # Single candidate: nothing to select (selection gap is 0 by construction). No LLM call.
        winner = pool[0]
        print(f"[IDX {idx}] Single candidate; no ranking call needed.")
    else:
        rank_temp = cfg.get("rank_temperature", 0.0)   # greedy ranker by default -> reproducible
        llm = get_llm(config, temperature=rank_temp)
        t0 = time.perf_counter()
        if rank_mode == "rank_with_reasoning":
            winner, rank_reasoning, rank_calls, rank_parse_failed = await rank_with_reasoning(user_query, pool, llm, idx)
        else:  # "rank_no_reasoning" — the default ranking sub-mode
            winner, rank_reasoning, rank_calls, rank_parse_failed = await rank_no_reasoning(user_query, pool, llm, idx)
        rank_latency = time.perf_counter() - t0

    # Agreement with majority vote, computed deterministically (NO extra LLM call) via finalize's
    # exact logic. "Same final answer" uses same_answer() so 243 == 243.00, 12 == 12., etc.
    majority_winner = _majority_winner(pool)
    rank_agreed_with_majority = (
        majority_winner is not None
        and same_answer(
            _extract_final_number(majority_winner.answer),
            _extract_final_number(winner.answer),
        )
    )

    # Persist the ranker's reasoning ON the selected candidate (2B). 2A leaves it None. Copy rather
    # than mutate the pooled object so the original `pool` entry is untouched. id is preserved.
    winner = winner.model_copy(update={"rank_reasoning": rank_reasoning})
    answer = winner.answer

    print(f"[IDX {idx}] Winner: id={winner.id[:8]} final={_extract_final_number(answer)}")
    if rank_reasoning:
        preview = rank_reasoning.strip().replace("\n", " ")
        print(f"[IDX {idx}] Rank reasoning: {preview[:160]}{'…' if len(preview) > 160 else ''}")
    print(f"[IDX {idx}] Rank calls: {rank_calls} | Rank latency: {rank_latency:.3f}s")
    print(f"[IDX {idx}] Parse failed: {rank_parse_failed} | Agreed with majority: {rank_agreed_with_majority}")
    print(f"\n[IDX {idx}] =========================================================================\n")

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "sampled_answers": sampled_answers,
        "sampled_numbers": sampled_numbers,
        "vote_distribution": vote_distribution,
        "unique_answers": unique_answers,
        "rank_mode": rank_mode,
        "rank_latency": rank_latency,
        "rank_calls": rank_calls,
        "winner_candidate_id": winner.id,
        "rank_reasoning": rank_reasoning,
        "rank_parse_failed": rank_parse_failed,
        "rank_agreed_with_majority": rank_agreed_with_majority,
    }


# ───────────────────── Experiment 3A selector (ACTIVE — hierarchical tournament) ─────────────────────
# Pure ORCHESTRATION over the Experiment 2 one-shot rankers (rank_no_reasoning / rank_with_reasoning):
# NO new prompt, NO regeneration, NO verifier/critic, NO abstention, NO extra generation round — the
# generator above is untouched and ONLY the selector changes. Motivated by Exp 2's finding that
# one-shot N-way ranking degrades as candidate count grows: instead of one N-way call, run a balanced
# tournament where every ranking call compares at most MAX_RANK_GROUP (=3) candidates.

def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (>= 1). Choosing a power-of-2 group count makes every later round
    halve cleanly into a balanced binary bracket (winners stay a power of 2 after round 1)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _split_balanced(items: list, num_groups: int) -> list[list]:
    """Split `items` into `num_groups` contiguous groups whose sizes differ by at most 1 (larger
    groups first). E.g. 10 into 4 -> [3,3,2,2], 8 into 4 -> [2,2,2,2]. Empty groups are dropped
    (only possible if num_groups > len(items), which the caller avoids)."""
    n = len(items)
    base, rem = divmod(n, num_groups)
    groups, start = [], 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        groups.append(items[start:start + size])
        start += size
    return [g for g in groups if g]


def _tournament_groups(candidates: list["Candidate"]) -> list[list["Candidate"]]:
    """One round's bracket split. If everything already fits in a single call (<= MAX_RANK_GROUP),
    return one group. Otherwise pick the SMALLEST power-of-2 group count G whose largest group is
    still <= MAX_RANK_GROUP, then split as evenly as possible. This reproduces the plan's brackets
    exactly: 4 -> [2,2], 6 -> [3,3], 8 -> [2,2,2,2], 10 -> [3,3,2,2]; and degrades gracefully for
    odd/other counts (a leftover singleton just takes a bye)."""
    n = len(candidates)
    if n <= MAX_RANK_GROUP:
        return [list(candidates)]
    min_groups = (n + MAX_RANK_GROUP - 1) // MAX_RANK_GROUP   # ceil(n / MAX_RANK_GROUP)
    num_groups = _next_pow2(min_groups)
    return _split_balanced(list(candidates), num_groups)


async def tournament_rank(
    user_query: str,
    candidates: list["Candidate"],
    rank_mode: str,
    llm,
    idx: str = "?",
) -> "tuple[Candidate, Optional[str], int, bool, int, int]":
    """Experiment 3A. Hierarchical tournament selector — an ORCHESTRATION layer over the existing
    one-shot rankers; it does NOT introduce a new ranking prompt. Each ranking call sees at most
    MAX_RANK_GROUP candidates, decomposing the N-way comparison that Exp 2 found degrades at larger N.
    Algorithm: shuffle once -> split into a balanced bracket (`_tournament_groups`) -> rank each group
    with the chosen one-shot ranker -> advance winners -> recurse until one candidate remains.

    rank_mode: "tournament_with_reasoning" -> per-group rank_with_reasoning (structured {reasoning,
    winner}); anything else -> per-group rank_no_reasoning (plain-text single letter).

    RETURNS: (global_winner, winner_reasoning, total_rank_calls, parse_failed, tournament_rounds,
              tournament_max_group_size).
      - winner_reasoning: the DECIDING (final-round) call's reasoning for the with-reasoning mode,
        else None. It is the justification for the global winner's last match.
      - total_rank_calls: real LLM ranking calls only (groups of >= 2); a bye (size 1) costs nothing.
      - parse_failed: True if ANY group call failed to parse a label (a fallback candidate was used).
      - tournament_max_group_size: the largest group actually submitted to a ranking call (0 if none).
    Independent of majority voting; needs no answer-equality key, so it carries over to non-numeric
    tasks exactly like the one-shot ranker it wraps."""
    group_ranker = rank_with_reasoning if rank_mode == "tournament_with_reasoning" else rank_no_reasoning

    survivors = list(candidates)

    # Global display labels (A, B, C, …) are pinned to the INPUT order — the same order the
    # tournament_select node prints under "Candidates:" — so the bracket log is reconcilable with
    # that listing and with the winner id printed downstream. Assigned BEFORE the shuffle on purpose:
    # the shuffle only randomizes bracket SEEDING (who meets whom), never a candidate's identity
    # label. LOG-ONLY, and independent of the local A/B/C labels each one-shot ranking call assigns
    # internally to its own 2–3 group members. (Pinning to the shuffled order instead made the log
    # impossible to reconcile with the pool listing — a candidate listed first could print as "B".)
    labels = {c.id: lbl for c, lbl in zip(candidates, _candidate_labels(len(candidates)))}

    def lbl(c: "Candidate") -> str:
        return labels.get(c.id, "?")

    random.shuffle(survivors)   # randomize bracket seeding only; identity labels already fixed above

    print(f"\n[IDX {idx}] ================ Tournament Ranking ================\n")

    total_calls = 0
    rounds = 0
    max_group_size = 0
    parse_failed_any = False
    # Reasoning that advanced each surviving candidate in its most recent match (id -> reasoning).
    # The global winner's entry is the deciding final-round reasoning.
    advance_reasoning: "dict[str, Optional[str]]" = {}

    while len(survivors) > 1:
        rounds += 1
        groups = _tournament_groups(survivors)
        print(f"[IDX {idx}] Round {rounds}")
        next_survivors: list[Candidate] = []
        for gi, group in enumerate(groups, start=1):
            if len(group) == 1:
                # Bye: a lone candidate advances with no ranking call (only happens off the even
                # benchmark widths; 4/6/8/10 never produce a bye).
                print(f"[IDX {idx}] Group {gi}: {lbl(group[0])} (bye)")
                print(f"[IDX {idx}] Winner: {lbl(group[0])}\n")
                next_survivors.append(group[0])
                continue

            max_group_size = max(max_group_size, len(group))
            print(f"[IDX {idx}] Group {gi}:")
            if len(group) == 2:
                print(f"[IDX {idx}] {lbl(group[0])} vs {lbl(group[1])}")
            else:                                   # 3-way (MAX_RANK_GROUP); list vertically
                for c in group:
                    print(f"[IDX {idx}] {lbl(c)}")

            winner, reasoning, calls, parse_failed = await group_ranker(user_query, group, llm, idx)
            total_calls += calls
            parse_failed_any = parse_failed_any or parse_failed
            advance_reasoning[winner.id] = reasoning
            print(f"[IDX {idx}] Winner: {lbl(winner)}\n")
            next_survivors.append(winner)
        survivors = next_survivors

    global_winner = survivors[0]
    winner_reasoning = advance_reasoning.get(global_winner.id)

    print(f"[IDX {idx}] Global Winner: {lbl(global_winner)} (id={global_winner.id[:8]})\n")
    print(f"[IDX {idx}] Tournament Rounds: {rounds}")
    print(f"[IDX {idx}] Tournament Rank Calls: {total_calls}")
    print(f"[IDX {idx}] Tournament Max Group Size: {max_group_size}")
    print(f"[IDX {idx}] ====================================================\n")

    return global_winner, winner_reasoning, total_calls, parse_failed_any, rounds, max_group_size


async def tournament_select(state: AgentState, config: RunnableConfig) -> dict:
    """Experiment 3A selector node. Generation already produced `pool`; here we select Top-1 via a
    HIERARCHICAL TOURNAMENT (`tournament_rank`) instead of one N-way ranking call — the response to
    Exp 2's finding that one-shot ranking degrades as candidate count grows. Pure orchestration over
    the one-shot rankers; the generator is untouched. Sub-mode from config `rank_mode`:
      - "tournament_with_reasoning" -> per-group rank_with_reasoning (structured {reasoning, winner})
      - "tournament_no_reasoning"   -> per-group rank_no_reasoning (plain-text single letter)
    Width=1 short-circuits (one candidate, no call, 0 rounds). Preserves EVERY Experiment 1/2 metric
    and adds tournament_rounds / tournament_rank_calls / tournament_max_group_size. `rank_calls`
    equals `tournament_rank_calls` (both count the tournament's ranking LLM calls), so the existing
    eval aggregation keeps working unchanged.
    RETURNS: final_answer, messages, the self-consistency metrics, the selector-log fields, and the
    three tournament metrics."""
    cfg = config["configurable"]
    rank_mode = state.get("rank_mode") or cfg.get("rank_mode", "tournament_no_reasoning")
    pool = state.get("pool", [])
    user_query = state.get("user_query", "")
    idx = _sample_idx(state, config)

    print(f"\n[IDX {idx}] ============ Tournament-Select Node (Experiment 3A) ============\n")
    print(f"[IDX {idx}] Rank mode: {rank_mode}")
    print(f"[IDX {idx}] Candidates: {len(pool)}")
    for label, candidate in zip(_candidate_labels(len(pool)), pool):
        print(f"[IDX {idx}] {label}: id={candidate.id[:8]} final={_extract_final_number(candidate.answer)}")

    sampled_answers, sampled_numbers, vote_distribution, unique_answers = _self_consistency_metrics(pool)

    if not pool:
        print(f"[IDX {idx}] Empty pool.")
        print(f"\n[IDX {idx}] ================================================================\n")
        return {
            "final_answer": "",
            "messages": [AIMessage(content="")],
            "sampled_answers": [],
            "sampled_numbers": [],
            "vote_distribution": {},
            "unique_answers": 0,
            "rank_mode": rank_mode,
            "rank_latency": 0.0,
            "rank_calls": 0,
            "winner_candidate_id": None,
            "rank_reasoning": None,
            "rank_parse_failed": False,
            "rank_agreed_with_majority": True,
            "tournament_rounds": 0,
            "tournament_rank_calls": 0,
            "tournament_max_group_size": 0,
        }

    rank_latency = 0.0
    rank_calls = 0
    rank_reasoning: Optional[str] = None
    rank_parse_failed = False
    tournament_rounds = 0
    tournament_max_group_size = 0

    if len(pool) <= 1:
        # Single candidate: nothing to select (no LLM call, no bracket).
        winner = pool[0]
        print(f"[IDX {idx}] Single candidate; no tournament needed.")
    else:
        rank_temp = cfg.get("rank_temperature", 0.0)   # greedy ranker by default -> reproducible
        llm = get_llm(config, temperature=rank_temp)
        t0 = time.perf_counter()
        (winner, rank_reasoning, rank_calls, rank_parse_failed,
         tournament_rounds, tournament_max_group_size) = await tournament_rank(
            user_query, pool, rank_mode, llm, idx
        )
        rank_latency = time.perf_counter() - t0

    # Agreement with majority vote, computed deterministically (NO extra LLM call), same as rank_select.
    majority_winner = _majority_winner(pool)
    rank_agreed_with_majority = (
        majority_winner is not None
        and same_answer(
            _extract_final_number(majority_winner.answer),
            _extract_final_number(winner.answer),
        )
    )

    # Persist the deciding reasoning ON the selected candidate (with-reasoning mode); copy, don't mutate.
    winner = winner.model_copy(update={"rank_reasoning": rank_reasoning})
    answer = winner.answer

    print(f"[IDX {idx}] Winner: id={winner.id[:8]} final={_extract_final_number(answer)}")
    if rank_reasoning:
        preview = rank_reasoning.strip().replace("\n", " ")
        print(f"[IDX {idx}] Rank reasoning: {preview[:160]}{'…' if len(preview) > 160 else ''}")
    print(f"[IDX {idx}] Tournament rounds: {tournament_rounds} | rank calls: {rank_calls} | max group: {tournament_max_group_size}")
    print(f"[IDX {idx}] Rank latency: {rank_latency:.3f}s")
    print(f"[IDX {idx}] Parse failed: {rank_parse_failed} | Agreed with majority: {rank_agreed_with_majority}")
    print(f"\n[IDX {idx}] ================================================================\n")

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "sampled_answers": sampled_answers,
        "sampled_numbers": sampled_numbers,
        "vote_distribution": vote_distribution,
        "unique_answers": unique_answers,
        "rank_mode": rank_mode,
        "rank_latency": rank_latency,
        "rank_calls": rank_calls,                       # == tournament_rank_calls
        "winner_candidate_id": winner.id,
        "rank_reasoning": rank_reasoning,
        "rank_parse_failed": rank_parse_failed,
        "rank_agreed_with_majority": rank_agreed_with_majority,
        "tournament_rounds": tournament_rounds,
        "tournament_rank_calls": rank_calls,
        "tournament_max_group_size": tournament_max_group_size,
    }


def route_selector(state: AgentState) -> Literal["finalize", "rank_select", "tournament_select"]:
    """Selector router. Config flag `rank_mode` (resolved into state by `setup`) chooses the selector:
      - "tournament_no_reasoning" / "tournament_with_reasoning" -> `tournament_select` (Exp 3A bracket)
      - "rank_no_reasoning"       / "rank_with_reasoning"       -> `rank_select`       (Exp 2 one-shot)
      - anything else (default "majority")                      -> `finalize`          (Exp 1 majority)
    Existing modes are unchanged: only the two new tournament modes route to the new node."""
    mode = state.get("rank_mode", "majority")
    if mode in ("tournament_no_reasoning", "tournament_with_reasoning"):
        return "tournament_select"
    if mode in ("rank_no_reasoning", "rank_with_reasoning"):
        return "rank_select"
    return "finalize"


# ───────────────────────── PARKED nodes / helpers (NOT wired into the graph) ─────────────────────────
# Everything below is retained for later ablations. None of it runs in Experiment 1.

async def _generate_search_round(state: AgentState, config: RunnableConfig) -> dict:
    """[PARKED] The full structured search-round generator: round-0 diverse sampling + sparse,
    disagreement-driven refinement, with the elitist best carried into the pool. Unused in
    Experiment 1 (structured output + diversity nudge hurt accuracy). Restore by routing `generate`
    here and re-adding the rank/select nodes + edges."""
    cfg = config["configurable"]
    tools_enabled = cfg.get("tools_enabled", False)
    base_llm = get_llm(config, tools_enabled=tools_enabled)
    system_prompt = build_prompt(state, config, "generator")
    width = state["width"]

    if not state["pool"]:
        round0_msgs = [SystemMessage(content=f"{system_prompt}\n\n{DIVERSITY_NUDGE}")] + state["messages"]
        prompts = [round0_msgs for _ in range(width)]
        parents: list[list[Candidate]] = [[] for _ in range(width)]
    else:
        n = len(state["pool"])
        parents = _pick_parents(state["pool"], cfg["parents_per_child"], n)
        prompts = [
            [SystemMessage(content=system_prompt)]
            + state["messages"]
            + [HumanMessage(content=_build_child_candidate_prompt(group))]
            for group in parents
        ]

    responses = await asyncio.gather(*[
        structured(_seeded(base_llm, random.randint(0, 2**31 - 1)), ResponseWithThoughts).ainvoke(p)
        for p in prompts
    ])

    pool = [
        Candidate(answer=r.response, trace=r.reasoning, parents=[c.id for c in parents[i]])
        for i, r in enumerate(responses)
    ]
    if state.get("best") is not None:
        pool = pool + [state["best"].model_copy()]   # elitist memory carried into next ranking

    return {
        "pool": pool,
        "iteration": state["iteration"] + 1,
        "budget_remaining": state["budget_remaining"] - len(prompts),
    }


def _pick_parents(pool: list[Candidate], parents_per_child: int, num_children: int) -> list[list[Candidate]]:
    """[PARKED] Sparse parent connectivity: each child gets a small, uniformly-sampled subset."""
    p = min(parents_per_child, len(pool))
    return [random.sample(pool, p) for _ in range(num_children)]


def _build_child_candidate_prompt(parents: list[Candidate]) -> str:
    """[PARKED] Disagreement-driven refinement prompt (verifier-free)."""
    blocks = []
    for i, parent in enumerate(parents):
        num = _extract_final_number(parent.answer) or _extract_final_number(parent.trace)
        blocks.append(
            f"### Candidate {i}\n"
            f"Answer: {parent.answer}\n"
            f"Reasoning: {parent.trace}\n"
            f"Final numeric answer: {num if num is not None else '(none found)'}"
        )
    return (
        "Below are candidate solutions to the user's request.\n\n"
        + "\n\n".join(blocks)
        + "\n\nThese candidate solutions disagree. Determine where their reasoning diverges, "
        "identify the conflicting assumptions or calculations, recompute the disputed steps, and "
        "produce a single improved answer. Synthesize the correct reasoning — do not simply copy "
        "any one candidate."
    )


async def rank(state: AgentState, config: RunnableConfig) -> dict:
    """[PARKED] Relative ranking via one-shot tournament. Unused in Experiment 1."""
    pool = state["pool"]
    if len(pool) <= 1:
        return {"pool": pool}
    llm = get_llm(config, temperature=0.0)
    if len(pool) <= RANK_GROUP:
        return {"pool": await _rank_group(pool, state, llm)}
    groups = [pool[i:i + RANK_GROUP] for i in range(0, len(pool), RANK_GROUP)]
    ranked_groups = [await _rank_group(g, state, llm) for g in groups]
    if len(ranked_groups) == 1:
        return {"pool": ranked_groups[0]}
    winners = [g[0] for g in ranked_groups]
    ranked_winners = await _rank_group(winners, state, llm)
    winner_rank = {c.id: i for i, c in enumerate(ranked_winners)}
    ordered_groups = sorted(ranked_groups, key=lambda g: winner_rank[g[0].id])
    return {"pool": [c for group in ordered_groups for c in group]}


async def _rank_group(candidates: list[Candidate], state: AgentState, llm) -> list[Candidate]:
    """[PARKED] One ranking call over <= RANK_GROUP candidates; robust id mapping. Unused now."""
    if len(candidates) <= 1:
        return list(candidates)
    labeled = {str(i): c for i, c in enumerate(candidates)}
    blocks = []
    for i, c in enumerate(candidates):
        num = _extract_final_number(c.answer) or _extract_final_number(c.trace)
        blocks.append(
            f"### Candidate {i}\nAnswer: {c.answer}\nReasoning: {c.trace}\n"
            f"Final numeric answer: {num if num is not None else '(none found)'}"
        )
    prompt = (
        rank_prompt(state) + "\n\n" + "\n\n".join(blocks)
        + f"\n\nRank all {len(candidates)} candidates by candidate_id (the integer label), best first."
    )
    chain = structured(llm, RankingResult)
    result: RankingResult = await chain.ainvoke([SystemMessage(content=prompt)])
    ordered, seen = [], set()
    for r in sorted(result.rankings, key=lambda x: x.rank):
        c = labeled.get(str(r.candidate_id).strip())
        if c is not None and c.id not in seen:
            ordered.append(c); seen.add(c.id)
    for c in candidates:
        if c.id not in seen:
            ordered.append(c); seen.add(c.id)
    return ordered


def select(state: AgentState) -> dict:
    """[PARKED] Keep top-K by rank; record rank-1 as global best (elitist). Unused in Experiment 1."""
    if not state["pool"]:
        return {"pool": []}
    k = max(2, state["width"])
    survivors = state["pool"][:k]
    global_best = survivors[0].model_copy()
    return {"best": global_best, "pool": survivors}


def route_after_generate(state: AgentState) -> str:
    """[PARKED] Loop entry router (unused: Experiment 1 wires generate -> finalize directly)."""
    return "rank" if state["max_iterations"] > 0 else "finalize"


def route_after_select(state: AgentState) -> str:
    """[PARKED] Loop stop router (unused in Experiment 1)."""
    out_of_budget = (
        state.get("budget_remaining", 0) <= 0
        or state.get("iteration", 0) >= state.get("max_iterations", 0)
    )
    return "finalize" if out_of_budget else "generate"


# ───────────── Build graph (setup -> generate -> [route_selector] -> finalize | rank_select) ─────────────

builder = StateGraph(AgentState)
builder.add_node("setup", setup)
builder.add_node("generate", generate)
builder.add_node("finalize", finalize)          # Experiment 1 selector: majority vote (default path)
builder.add_node("rank_select", rank_select)    # Experiment 2 selector: one-shot ranking (2A / 2B)
builder.add_node("tournament_select", tournament_select)  # Experiment 3A selector: hierarchical tournament

builder.add_edge(START, "setup")
builder.add_edge("setup", "generate")
# Config-flag bypass: `rank_mode` selects the selector. Default ("majority") preserves the
# Experiment-1 path generate -> finalize exactly; one-shot ranking diverts to rank_select; tournament
# ranking (Exp 3A) diverts to tournament_select. The first two selectors are unchanged.
builder.add_conditional_edges(
    "generate",
    route_selector,
    {"finalize": "finalize", "rank_select": "rank_select", "tournament_select": "tournament_select"},
)
builder.add_edge("finalize", END)
builder.add_edge("rank_select", END)
builder.add_edge("tournament_select", END)

GRAPH_RANKER = builder.compile()


# ───────────────────────── Local smoke test ─────────────────────────

def _fresh_state(question: str) -> AgentState:
    return {
        "messages": [{"role": "user", "content": question}],
        "user_query": "",
        "message_intent": "chat",
        "reasoning_mode": "low",
        "retrieved_context": "",
        "tool_results": "",
        "pool": [],
        "best": None,
        "iteration": 0,
        "max_iterations": 0,
        "width": 1,
        "budget_remaining": 1,
        "final_answer": None,
        "sampled_answers": [],
        "sampled_numbers": [],
        "vote_distribution": {},
        "unique_answers": 0,
        "rank_mode": "majority",
        "rank_latency": 0.0,
        "rank_calls": 0,
        "winner_candidate_id": None,
        "rank_reasoning": None,
        "rank_parse_failed": False,
        "rank_agreed_with_majority": True,
        "tournament_rounds": 0,
        "tournament_rank_calls": 0,
        "tournament_max_group_size": 0,
    }


if __name__ == "__main__":
    config = {
        "configurable": {
            "model_name": "qwen",
            "personality": "general",      # unused in Experiment 1 (no system prompt)
            "reasoning_mode": "low",        # low=1; medium=3; high=5; extra=7; max=9
            "bypass_governor": True,
            "tools_enabled": False,
            "temperature": 0.0,             # low: 0.0 reproduces baseline. medium/high: use >0 (e.g. 0.7)
            # Experiment 2 (selector). Default "majority" = Experiment 1. To run ranking instead:
            #   "rank_mode": "rank_no_reasoning",   # 2A: one plain-text ranking call
            #   "rank_mode": "rank_with_reasoning", # 2B: one structured {reasoning, winner} call
            # Experiment 3A (hierarchical tournament; uses EVEN widths 1/4/6/8/10):
            #   "rank_mode": "tournament_no_reasoning",   # 3A: per-group rank_no_reasoning
            #   "rank_mode": "tournament_with_reasoning", # 3A: per-group rank_with_reasoning
            #   "rank_temperature": 0.0,            # ranker decoding temp (greedy by default)
            "rank_mode": "majority",
        }
    }
    result = asyncio.run(GRAPH_RANKER.ainvoke(_fresh_state("What is 17 * 24?"), config=config))
    print(result["final_answer"])