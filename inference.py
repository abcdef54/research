from datasets import load_dataset
import asyncio
import re
import os
import csv
import argparse
from tqdm import tqdm
import time 
import math

from graph import GRAPH, AgentState

os.environ["LANGCHAIN_TRACING_V2"] = "false"

gsm8k = load_dataset("openai/gsm8k", "main")
test_set = gsm8k["test"]

print(f"GSM8K Test Size: {len(test_set)}")

GSM8K_SUFFIX = (
    "\n\nEnd your response with the final answer on its own line "
    "in this exact format:\n#### <number>"
)

_NUM = r"[-+]?\d[\d,]*\.?\d*"

SEM = None


def extract_gsm8k_label(ground_truth: str) -> float | None:
    match = re.search(
        r"####\s*([-+]?\d[\d,]*\.?\d*)",
        ground_truth,
    )

    return (
        float(match.group(1).replace(",", ""))
        if match
        else None
    )


def extract_final_number(text: str) -> float | None:
    if not text:
        return None

    text = text.strip()

    match = re.search(
        rf"####\s*({_NUM})",
        text,
    )

    if not match:
        match = re.search(
            rf"(?:the\s+)?answer\s+is\s+\$?\s*({_NUM})",
            text,
            re.IGNORECASE,
        )

    if not match:
        match = re.search(
            rf"\\boxed\{{({_NUM})\}}",
            text,
        )

    if match:
        return float(
            match.group(1).replace(",", "")
        )

    numbers = re.findall(
        rf"(?<!\w)({_NUM})(?!\w)",
        text,
    )

    if numbers:
        return float(
            numbers[-1].replace(",", "")
        )

    return None



async def solve(
    question: str,
    model_name: str,
    reasoning_mode: str,
    temperature: float,
):
    config = {
        "configurable": {
            "model_name": model_name,
            "personality": "general",
            "tools_enabled": False,
            "reasoning_mode": reasoning_mode,
            "bypass_governor": True,
            "temperature": temperature,
        }
    }

    init: AgentState = {
        "messages": [
            {
                "role": "user",
                "content":
                    question + GSM8K_SUFFIX,
            }
        ],

        "message_intent": "chat",

        "retrieved_context": "",
        "tool_results": "",

        "pool": [],
        "best": None,

        "iteration": 0,
        "max_iterations": 0,
        "width": 1,
        "budget_remaining": 1,

        "final_answer": None,

        # NEW FIELDS FROM graph.py
        "sampled_answers": [],
        "sampled_numbers": [],
        "vote_distribution": {},
        "unique_answers": 0,
    }

    result = await GRAPH.ainvoke(
        init,
        config=config,
    )

    # RETURN ENTIRE STATE
    return result


async def evaluate_sample(
    idx,
    sample,
    model_name,
    reasoning_mode,
    temperature,
):
    q = sample["question"]
    gt = extract_gsm8k_label(
        sample["answer"]
    )

    try:
        

        async with SEM:
            start = time.perf_counter()
            state = await solve(
                q,
                model_name,
                reasoning_mode,
                temperature,
            )

        latency = (
            time.perf_counter()
            - start
        )

        pred_text = state.get(
            "final_answer",
            ""
        )

        pred = extract_final_number(
            pred_text
        )

        sampled_answers = state.get(
            "sampled_answers",
            [],
        )

        sampled_numbers = state.get(
            "sampled_numbers",
            [],
        )

        vote_distribution = state.get(
            "vote_distribution",
            {},
        )

        unique_answers = state.get(
            "unique_answers",
            0,
        )

        majority_correct = (
            pred is not None
            and gt is not None
            and abs(pred - gt) < 1e-6
        )

        # Oracle Accuracy
        oracle_correct = False

        if gt is not None:
            for x in sampled_numbers:
                if (
                    x is not None
                    and abs(float(x) - gt)
                    < 1e-6
                ):
                    oracle_correct = True
                    break

        total_votes = sum(
            vote_distribution.values()
        )

        max_votes = (
            max(
                vote_distribution.values()
            )
            if vote_distribution
            else 0
        )

        agreement_ratio = (
            max_votes / total_votes
            if total_votes > 0
            else 0.0
        )

        # Vote Entropy
        vote_entropy = 0.0

        if total_votes > 0:
            for v in vote_distribution.values():
                p = v / total_votes
                vote_entropy -= (
                    p * math.log2(p)
                )

        invalid_samples = sum(
            x is None
            for x in sampled_numbers
        )
        votes = sorted(
            vote_distribution.values(),
            reverse=True,
        )

        if len(votes) >= 2:
            vote_margin = (
                votes[0] - votes[1]
            )
        elif len(votes) == 1:
            vote_margin = votes[0]
        else:
            vote_margin = 0

        is_unanimous = (
            agreement_ratio == 1.0
        )

        majority_failed = (
            oracle_correct
            and not majority_correct
        )

        width = len(sampled_numbers)

        return {
            "idx": idx,
            "question": q,

            "ground_truth": gt,
            "prediction": pred,

            # Main metrics
            "correct": majority_correct,
            "oracle_correct": oracle_correct,
            "majority_failed": majority_failed,

            # Experiment configuration
            "model": model_name,
            "mode": reasoning_mode,
            "temperature": temperature,
            "width": width,

            # Diversity metrics
            "unique_answers": unique_answers,
            "agreement_ratio": agreement_ratio,
            "vote_entropy": vote_entropy,
            "vote_margin": vote_margin,
            "is_unanimous": is_unanimous,
            "invalid_samples": invalid_samples,

            # Performance
            "latency_sec": latency,

            # Analysis
            "sampled_numbers": sampled_numbers,
            "vote_distribution": vote_distribution,

            # Optional (large CSV)
            "sampled_answers": sampled_answers,

            # Final answer text
            "response": pred_text,
        }

    except Exception as e:
        return {
            "idx": idx,
            "question": q,
            "ground_truth": gt,
            "prediction": None,
            "majority_failed": 0,

            "correct": False,
            "oracle_correct": False,

            "unique_answers": 0,
            "agreement_ratio": 0.0,
            "vote_entropy": 0.0,
            "invalid_samples": 0,
            "is_unanimous": False,
            "vote_margin": 0,

            "latency_sec": 0.0,

            "sampled_numbers": [],
            "vote_distribution": {},
            "sampled_answers": [],

            "response": str(e),
        }


async def evaluate(
    dataset,
    model_name,
    reasoning_mode,
    temperature,
    csv_path,
):
    tasks = [
        evaluate_sample(
            i,
            sample,
            model_name,
            reasoning_mode,
            temperature,
        )
        for i, sample
        in enumerate(dataset)
    ]

    results = []

    correct = 0
    oracle_correct = 0
    completed = 0

    total_latency = 0.0
    total_unique = 0
    total_agreement = 0.0
    total_entropy = 0.0
    total_invalid = 0
    correct = 0
    majority_failed_count = 0
    total_vote_margin = 0.0
    total_unanimous = 0

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "question",

                "ground_truth",
                "prediction",

                "correct",
                "oracle_correct",
                "majority_failed",

                "model",
                "mode",
                "temperature",
                "width",

                "unique_answers",
                "agreement_ratio",
                "vote_entropy",
                "vote_margin",
                "is_unanimous",
                "invalid_samples",

                "latency_sec",

                "sampled_numbers",
                "vote_distribution",
                "sampled_answers",

                "response",
            ],
        )

        writer.writeheader()

        pbar = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
        )

        for coro in pbar:
            result = await coro

            results.append(result)

            completed += 1

            correct += int(
                result["correct"]
            )

            oracle_correct += int(
                result["oracle_correct"]
            )

            majority_failed_count += int(
                result["majority_failed"]
            )

            total_vote_margin += result[
                "vote_margin"
            ]

            total_unanimous += int(
                result["is_unanimous"]
            )

            total_latency += result[
                "latency_sec"
            ]

            total_unique += result[
                "unique_answers"
            ]

            total_agreement += result[
                "agreement_ratio"
            ]

            total_entropy += result[
                "vote_entropy"
            ]

            total_invalid += result[
                "invalid_samples"
            ]

            accuracy = (
                correct / completed
            )

            oracle_acc = (
                oracle_correct
                / completed
            )

            writer.writerow(result)
            f.flush()

            pbar.set_postfix(
                correct=correct,
                oracle=oracle_correct,
                acc=f"{accuracy:.4f}",
                oracle_acc=f"{oracle_acc:.4f}",
            )

    final_acc = (
        correct / completed
        if completed > 0
        else 0.0
    )

    final_oracle = (
        oracle_correct / completed
        if completed > 0
        else 0.0
    )

    avg_vote_margin = (
        total_vote_margin
        / completed
        if completed > 0
        else 0.0
    )

    unanimous_ratio = (
        total_unanimous
        / completed
        if completed > 0
        else 0.0
    )

    selection_gap = (
        final_oracle
        - final_acc
    )

    avg_latency = (
        total_latency / completed
        if completed > 0
        else 0.0
    )

    avg_unique = (
        total_unique / completed
        if completed > 0
        else 0.0
    )

    avg_agreement = (
        total_agreement / completed
        if completed > 0
        else 0.0
    )

    avg_entropy = (
        total_entropy / completed
        if completed > 0
        else 0.0
    )

    avg_invalid = (
        total_invalid / completed
        if completed > 0
        else 0.0
    )

    print("\n================ RESULTS ================\n")

    print(
        f"Final Accuracy: "
        f"{final_acc:.4f}"
    )

    print(
        f"Oracle Accuracy (Pass@N): "
        f"{final_oracle:.4f}"
    )

    print(
        f"Selection Gap: "
        f"{selection_gap:.4f}"
    )

    print(
        f"Majority Failures: "
        f"{majority_failed_count}"
    )

    print(
        f"Average Unique Answers: "
        f"{avg_unique:.4f}"
    )

    print(
        f"Average Agreement Ratio: "
        f"{avg_agreement:.4f}"
    )

    print(
        f"Average Vote Entropy: "
        f"{avg_entropy:.4f}"
    )

    print(
        f"Average Vote Margin: "
        f"{avg_vote_margin:.4f}"
    )

    print(
        f"Unanimous Ratio: "
        f"{unanimous_ratio:.4f}"
    )

    print(
        f"Average Invalid Samples: "
        f"{avg_invalid:.4f}"
    )

    print(
        f"Average Latency: "
        f"{avg_latency:.4f}s"
    )

    print("\n=========================================\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="qwen2.5-3b-q4",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "low",
            "medium",
            "high",
        ],
        default="low",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="100, 150, or -1 for full GSM8K",
    )

    parser.add_argument(
        "--sem",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--out",
        default=None,
    )

    args = parser.parse_args()

    SEM = asyncio.Semaphore(
        args.sem
    )

    if args.samples == -1:
        dataset = test_set
    else:
        dataset = test_set.select(
            range(
                min(
                    args.samples,
                    len(test_set),
                )
            )
        )

    csv_path = (
        args.out
        or
        f"gsm8k_"
        f"{args.model}_"
        f"{args.mode}_"
        f"t{args.temp}_"
        f"n{len(dataset)}.csv"
    )

    asyncio.run(
        evaluate(
            dataset=dataset,
            model_name=args.model,
            reasoning_mode=args.mode,
            temperature=args.temp,
            csv_path=csv_path,
        )
    )