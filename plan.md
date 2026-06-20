# Experiment 2

# Can Ranking Beat Majority Vote?

## Background and Findings

### Baseline Results (Qwen2.5-3B-Instruct Q4)

Low (Width=1)
Final Accuracy: 84.69%
Pass@N: 84.69%
Selection Gap: 0%

Medium (Width=3)
Final Accuracy: 85.75%
Pass@N: 92.19%
Selection Gap: 6.44%
Majority Failures: 85 questions

High (Width=5)
Final Accuracy: 87.04%
Pass@N: 93.93%
Selection Gap: 6.90%
Majority Failures: 91 questions

These results indicate:

1. The generator is strong.
2. Correct answers frequently exist among generated samples.
3. Majority voting is the bottleneck.

Approximately 6-7% of GSM8K questions already contain a correct answer somewhere in the generated samples, but majority voting still selects an incorrect answer.

Therefore the next experiment should isolate the selection mechanism.

---

# Research Question

Can LLM ranking outperform majority voting?

---

# Experimental Principle

ONLY the selector changes.

Everything else remains identical.

Generator:

* No system prompt
* No identity prompts
* Human messages only
* Plain text outputs
* No structured outputs
* Same temperature settings as Self-Consistency experiments.

No refinement.
No verifier.
No critic.
No sparse children generation.

Pipeline:

Generate N
↓
One-shot Ranking
↓
Return Top-1

There is only ONE generation round.

There are NO later rounds.

There is NO tournament ranking in this experiment.

Tournament ranking belongs to the future iterative sparse-refinement architecture.

This experiment uses:

Generate N
↓
Rank all candidates in ONE call
↓
Select winner

because the purpose is to isolate whether ranking itself can beat majority vote.

---

# Add Higher Benchmarking Modes

Current:

low
width=1

medium
width=3

high
width=5

Add:

extra
width=7

max
width=9

These modes exist only for benchmarking and paper evaluation.

No architectural changes.

No refinement.

No extra nodes.

Only wider generation.

Expected benchmark:

Width:
1
3
5
7
9

for:

Accuracy
Pass@N
Selection Gap
Latency

---

# Experiment 2A

# One-Shot Ranking WITHOUT Structured Reasoning

Pipeline:

Generate N
↓
Rank All
↓
Return Top-1

Ranking output should be plain text only.

Example prompt:

Question:
...

Candidate A:
...

Candidate B:
...

Candidate C:
...

Choose the single answer that is most likely correct.

Respond ONLY with:

A

or

B

or

C

No explanation.

No reasoning field.

No structured output.

No additional LLM calls.

Implementation:

rank_no_reasoning()

This function should:

* receive all candidates in one call
* return only winner ids
* parse plain text responses
* be completely independent from majority voting.

---

# Experiment 2B

# One-Shot Ranking WITH Structured Reasoning

Definition:

Reasoning here means STRUCTURED OUTPUT.

Reasoning does NOT mean:

* hidden chain-of-thought
* an additional critic node
* an additional LLM call.

The ranker still makes only ONE call.

The only difference is that the response schema now includes a reasoning field.

Example:

{
"reasoning":
"Candidate B correctly computes the intermediate value while Candidate A makes an arithmetic error.",
"winner":
"B"
}

IMPORTANT:

Put the reasoning field FIRST.

The schema order should be:

reasoning
winner

NOT:

winner
reasoning

because previous experiments suggest field ordering can influence generation quality.

Implementation:

rank_with_reasoning()

Store:

rank_reasoning

inside the selected candidate.

---

# Metrics

Keep all existing metrics.

Additionally log:

rank_mode:

* majority
* rank_no_reasoning
* rank_with_reasoning

rank_latency

rank_calls

winner_candidate_id

rank_reasoning
(nullable for no-reasoning mode)

---

# Main Comparison Table

Compare:

Majority Vote
vs
Ranking Without Reasoning
vs
Ranking With Reasoning

under:

Width:
1
3
5
7
9

Metrics:

Final Accuracy
Pass@N
Selection Gap
Majority Failures
Latency

---

# Hypothesis

Because:

Pass@N:
93.93%

Accuracy:
87.04%

Selection Gap:
6.90%

the ranker has substantial room for improvement.

Even partial recovery of the selection gap would demonstrate that ranking is a stronger selector than majority voting and would justify proceeding to the future sparse-refinement architecture.
