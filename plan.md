# Experiment 3B - Maximum Group Size = 3 Tournament Ranking

## Motivation

The previous Experiment 3A used a maximum tournament group size of 2:

* Medium (N=4): 2-2 → 3 ranking calls
* High (N=6): 3-3 → 3 ranking calls
* Extra (N=8): 2-2-2-2 → 7 ranking calls

Results showed severe selector degradation as ranking depth increased:

### Extra Mode (2-2-2-2)

* Final Accuracy: 84.84%
* Oracle Accuracy: 95.30%
* Selection Gap: 10.46%
* Average Rank Calls: 6.96
* Average Rank Latency: 73.68s

This strongly suggests that the bottleneck is no longer the generator but the ranker itself. Oracle performance continues to increase with larger candidate pools, meaning the correct answer is usually present somewhere in the generated candidates. However, repeated ranking decisions introduce compounding errors that prevent the selector from choosing the correct answer.

---

## New Hypothesis

The ranker may have a "sweet spot" at **3 candidates per comparison**.

Evidence:

### One-Shot Ranking (N=3)

Medium reasoning mode:

* Accuracy: 85.90%
* Selection Gap: 5.99%
* Stable performance
* Outperformed the raw baseline
* Nearly matched majority voting

This indicates that Qwen-3B can effectively compare **3 candidates simultaneously**.

In contrast:

### Tournament Medium (2-2)

Accuracy collapsed:

* Stupid: 81.27%
* Smart: 84.31%

This suggests that reducing comparisons to only 2 candidates at a time may not actually help. Instead, increasing ranking depth and accumulating selector errors appears to dominate any cognitive benefit.

---

## Experiment 3B Objective

Reduce tournament depth by increasing the maximum group size from:

```python
MAX_RANK_GROUP = 2
```

to:

```python
MAX_RANK_GROUP = 3
```

The goal is to:

1. Reduce total ranking calls.
2. Reduce compounded ranker errors.
3. Keep per-call cognitive load within a range already proven to work well (N=3).

---

## New Tournament Structures

### Medium Reasoning Mode

Previous:

```text
Generate 4
2-2
3 rank calls
```

New:

```text
Generate 3
One-shot ranking
1 rank call
```

This effectively becomes Experiment 2's medium one-shot ranker.

Expected:

* Accuracy ≈ 85.9%
* Selection gap ≈ 6%

---

### High Reasoning Mode

Previous:

```text
Generate 6
3-3
3 rank calls
```

New:

```text
Generate 5
3-2
2 rank calls
```

Structure:

Round 1:

```text
[A B C]
[D E]
```

Round 2:

```text
Winner(ABC)
Winner(DE)
```

Expected benefits:

* 33% fewer rank calls
* Lower selector error accumulation
* Similar or improved accuracy

---

### Extra Reasoning Mode

Previous:

```text
Generate 8
2-2-2-2
7 rank calls
```

New:

```text
Generate 7
3-3-1
2 rank calls
```

Structure:

Round 1:

```text
[A B C]
[D E F]
[G]
```

Round 2:

```text
Winner(ABC)
Winner(DEF)
G
```

If only one candidate remains in a group:

* No ranking call is required.
* The candidate automatically advances.

Expected benefits:

* Rank calls reduced from ~7 to ~2
* Massive reduction in compounded selector errors
* Lower latency
* Better selector utilization

---

## Architectural Principle

Maximum candidates evaluated per rank call:

```text
≤ 3
```

The selector should never compare:

* 4 candidates
* 5 candidates
* 7 candidates
* 9 candidates

because previous experiments indicate ranking quality deteriorates as candidate count increases.

---

## Metrics to Track

Continue logging:

* Final Accuracy
* Oracle Accuracy (Pass@N)
* Selection Gap
* Majority Failures
* Average Unique Answers
* Average Agreement Ratio
* Average Vote Entropy
* Average Vote Margin
* Unanimous Ratio
* Average Invalid Samples
* Average Latency
* Average Rank Calls
* Average Rank Latency
* Rank Parse Failures
* Rank Parse Failure Ratio
* Rank Disagreement Ratio
* Average Tournament Rounds
* Average Tournament Rank Calls
* Max Tournament Group Size

No new metrics are required.

---

## Research Question

Does reducing tournament depth by increasing the maximum comparison group size from 2 to 3 improve selector performance by reducing compounded ranking errors while remaining within the ranker's effective cognitive capacity?
