# Local LLM Benchmark Report
**Task:** 9 historical date questions  
**Runs:** 3 per model  
**Machine:** MacBook Air 15-inch M2, 8GB RAM

---

## Summary

| Metric | Value |
|---|---|
| Models tested | 10 |
| Questions | 9 |
| Runs per model | 3 |
| Best model | mistral (9/9) |
| Models that refused | deepseek-coder, deepseek-r1, qwen3 |

---

## Accuracy Results

| Model | Score | % |
|---|---|---|
| mistral | 9/9 | 100% |
| llama3.1 | 8/9 | 89% |
| llava | 8/9 | 89% |
| Qwen2.5-coder | 6/9 | 67% |
| phi3 | 4/9 | 44% |
| llama3.2 | 2/9 | 22% |
| gemma3:1b | 0/9 | 0% |
| deepseek-r1:8b | 0/9 | 0% (refused) |
| deepseek-coder | 0/9 | 0% (refused) |
| qwen3 | 0/9 | 0% (refused) |

---

## Response Time

| Model | Avg (s) | Std Dev (s) | Notes |
|---|---|---|---|
| deepseek-coder | 2.0 | 0.0 | Fastest — but refused all questions |
| llama3.2 | 3.0 | 0.0 | |
| gemma3:1b | 3.0 | 0.0 | Fast but fully unreliable |
| phi3 | 3.7 | 0.6 | |
| llama3.1 | 6.3 | 0.6 | |
| llava | 7.0 | 0.0 | |
| Qwen2.5-coder | 7.0 | 0.0 | |
| mistral | 7.7 | 0.6 | Best accuracy, consistent timing |
| deepseek-r1:8b | 16.7 | 15.1 | High variance — thinking mode |
| qwen3 | 21.3 | 16.0 | High variance — thinking mode |

---

## Notable Observations

### mistral — perfect score
Only model to achieve 9/9. Timing was consistent across all three runs (7–8s). Clear winner for factual date recall on this hardware.

### gemma3:1b — confident but wrong
Answered all 9 questions confidently and got every single one wrong. At 3s it's the fastest real responder, but completely unreliable for factual tasks. More dangerous than a refusal.

### phi3 — invalid date output
Returned `2011-01-55` for the Steve Jobs question — day 55 doesn't exist. This goes beyond a factual error into hallucination of impossible values.

### deepseek-coder — expected refusal
A coding-specific model. Returning `"-"` for all 9 historical questions is reasonable behaviour — the task is outside its design scope.

### qwen3 and deepseek-r1 — timing variance
Both models showed extreme run-to-run variance driven by unpredictable activation of thinking mode:
- **qwen3:** 5s → 22s → 37s (avg 21.3s, std ±16s)
- **deepseek-r1:** 6s → 34s → 10s (avg 16.7s, std ±15s)

The 0/9 scores likely reflect output format issues — the thinking mode may be producing reasoning that the test script didn't parse as a date, rather than genuine factual ignorance.

---

## Verdict

For factual recall tasks on an M2 8GB machine, **mistral is the best option** among the models tested — perfect accuracy, predictable speed, and no surprises. `llama3.1` is a strong second choice at 8/9 with slightly faster responses.

Avoid `gemma3:1b` for any task requiring factual accuracy. The thinking models (`qwen3`, `deepseek-r1`) need prompt engineering to ensure they output clean structured dates before being fairly evaluated.