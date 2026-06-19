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
import random
import asyncio
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
    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:18888/v1")
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


# ───────────────────────── Data structures ─────────────────────────

class Candidate(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)  # stable identity for lineage
    answer: str = ""
    trace: str = ""                                       # empty in Experiment 1 (no reasoning trace)
    parents: list[str] = Field(default_factory=list)      # empty in Experiment 1 (no lineage)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str

    # routing / mode
    message_intent: Literal["chat", "knowledge", "code"]   # kept for the future; always 'chat' in testing
    reasoning_mode: Literal["low", "medium", "high"]

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


# ───────────────────────── Mode / difficulty config ─────────────────────────

# Experiment 1: width = number of self-consistency samples; no loop, so iterations = 0.
# budget = number of generation calls (= width).
MODE_CONFIG = {
    "low":    {"max_width": 1, "max_iterations": 0, "max_budget": 1},   # reproduces the raw baseline
    "medium": {"max_width": 3, "max_iterations": 0, "max_budget": 3},
    "high":   {"max_width": 5, "max_iterations": 0, "max_budget": 5},
}

# Internal estimate (PARKED governor; only consulted when bypass_governor=False — eval uses True).
STEP_CONFIG = {
    "single": {"width": 1, "iterations": 0, "budget": 1},
    "few":    {"width": 1, "iterations": 1, "budget": 3},
    "multi":  {"width": 2, "iterations": 2, "budget": 8},
    "deep":   {"width": 4, "iterations": 3, "budget": 30},
}


def get_config(mode: str, steps: str, bypass_governor: bool = False) -> Dict:
    """mode = effort ceiling, reasoning_steps = effort needed; run with min(ceiling, needed).
    With bypass_governor (eval), the mode ceiling drives compute directly so modes compare uniformly."""
    ceil = MODE_CONFIG[mode]
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
             iteration, pool, best."""
    cfg = config["configurable"]
    mode = cfg["reasoning_mode"]
    bypass = cfg.get("bypass_governor", False)
    steps = cfg.get("reasoning_steps", "deep")  # only used when bypass=False
    loop_cfg = get_config(mode, steps, bypass)

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

    print("\n==================== Generate Node (Self-Consistency) ====================\n")
    print(f"Question: {state['user_query']}")
    print(f"Width (N samples): {width}")

    # Raw question ONLY — no SystemMessage, no build_prompt, no DIVERSITY_NUDGE, no structured().
    prompts = [state["messages"] for _ in range(width)]
    responses = await asyncio.gather(*[
        _seeded(llm, random.randint(0, 2**31 - 1)).ainvoke(p) for p in prompts
    ])

    pool = []
    for i, r in enumerate(responses):
        content = r.content if isinstance(r.content, str) else str(r.content)
        pool.append(Candidate(answer=content, trace="", parents=[]))
        print(f"Sample {i}: final_number={_extract_final_number(content)!r}")
    print("\n==========================================================================\n")

    return {
        "pool": pool,
        "iteration": state["iteration"] + 1,
        "budget_remaining": state["budget_remaining"] - len(prompts),  # budget == generation calls
    }


def finalize(state: AgentState) -> dict:
    """Self-Consistency majority vote. Extract each candidate's final number, tally by value, and
    return the EARLIEST candidate that produced the most-voted value. Deterministic: ties between
    values break by earliest first-appearance; ties within the winning value by earliest candidate.
    (width=1 reduces to "return the single sample" — i.e. the raw baseline.)
    RETURNS: final_answer, messages."""
    pool = state.get("pool", [])
    print("\n==================== Finalize Node (Majority Vote) ====================\n")
    if not pool:
        print("Empty pool.")
        return {"final_answer": "", "messages": [AIMessage(content="")]}

    # Group candidates by extracted final value, preserving first-appearance order.
    groups: "OrderedDict[Optional[str], list[Candidate]]" = OrderedDict()
    for c in pool:
        val = _extract_final_number(c.answer)
        groups.setdefault(val, []).append(c)

    print("Vote tally (value -> count):")
    for val, members in groups.items():
        print(f"  {val!r}: {len(members)}")

    # Highest count wins. On a tie keep the value seen FIRST: OrderedDict iterates in
    # first-appearance order and we only replace on a STRICTLY greater count.
    winning_members, winning_val = None, None
    for val, members in groups.items():
        if winning_members is None or len(members) > len(winning_members):
            winning_members, winning_val = members, val

    winner = winning_members[0]   # earliest candidate among those with the winning value
    answer = winner.answer
    print(f"\nWinner value: {winning_val!r} ({len(winning_members)} vote(s)) -> candidate {winner.id[:8]}")
    print("\n=======================================================================\n")
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
    }


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


# ───────────────────────── Build graph (Experiment 1: setup -> generate -> finalize) ─────────────────────────

builder = StateGraph(AgentState)
builder.add_node("setup", setup)
builder.add_node("generate", generate)
builder.add_node("finalize", finalize)

builder.add_edge(START, "setup")
builder.add_edge("setup", "generate")
builder.add_edge("generate", "finalize")
builder.add_edge("finalize", END)

GRAPH = builder.compile()


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
    }


if __name__ == "__main__":
    config = {
        "configurable": {
            "model_name": "qwen",
            "personality": "general",      # unused in Experiment 1 (no system prompt)
            "reasoning_mode": "low",        # low=1 sample; medium=3; high=5
            "bypass_governor": True,
            "tools_enabled": False,
            "temperature": 0.0,             # low: 0.0 reproduces baseline. medium/high: use >0 (e.g. 0.7)
        }
    }
    result = asyncio.run(GRAPH.ainvoke(_fresh_state("What is 17 * 24?"), config=config))
    print(result["final_answer"])