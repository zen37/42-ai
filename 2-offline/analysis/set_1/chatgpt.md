# Offline model benchmark report

## Scope

This report analyzes three uploaded files:

- `9q_duration.csv` — runtime per model/run
- `9q_responses.csv` — expected answers for 9 benchmark questions
- `9q_results.csv` — model outputs for 10 offline models, each tested 3 times

## Benchmark setup

- **Questions:** 9
- **Models:** 10
- **Runs per model:** 3
- **Total model attempts:** 270
- **Evaluation rule:** exact string match against the expected answer
- **Answer format tested:** date strings in `yyyy-mm-dd`

## Executive summary

The clearest result is that **all tested models were fully deterministic** on this benchmark: each model gave the same answer to the same question in all 3 runs. The ranking therefore comes down to **accuracy, omissions, and speed**, not run-to-run variance.

### Top takeaways

1. **`mistral:latest` was the best overall model** in this benchmark: **27/27 exact matches (100%)** with no missing outputs.
2. **`llava:latest` and `llama3.1:latest` were strong second-tier performers** at **24/27 exact matches (88.9%)**.
3. **`Qwen2.5-coder:latest` was usable but clearly behind the top group** at **18/27 exact matches (66.7%)**.
4. **`phi3:latest` and `llama3.2:latest` showed limited factual reliability** on this date-only task.
5. **`gemma3:1b` answered every question but got none exactly right**.
6. **`deepseek-coder:latest`, `deepseek-r1:8b`, and `qwen3:latest` returned no usable answers at all** in these recorded results.
7. In raw speed, **`deepseek-coder:latest` was fastest**, but since it produced no valid answers, that speed is not useful. Among the accurate models, **`llama3.1:latest` offered the best speed/accuracy tradeoff**, while **`mistral:latest` offered the best absolute accuracy**.

## Accuracy ranking

| Model                 |   Exact matches |   Total attempts |   Accuracy % |   Answered |   Missing |
|:----------------------|----------------:|-----------------:|-------------:|-----------:|----------:|
| mistral:latest        |              27 |               27 |        100   |         27 |         0 |
| llava:latest          |              24 |               27 |         88.9 |         27 |         0 |
| llama3.1:latest       |              24 |               27 |         88.9 |         24 |         3 |
| Qwen2.5-coder:latest  |              18 |               27 |         66.7 |         24 |         3 |
| phi3:latest           |              12 |               27 |         44.4 |         21 |         6 |
| llama3.2:latest       |               6 |               27 |         22.2 |         21 |         6 |
| gemma3:1b             |               0 |               27 |          0   |         27 |         0 |
| deepseek-coder:latest |               0 |               27 |          0   |          0 |        27 |
| deepseek-r1:8b        |               0 |               27 |          0   |          0 |        27 |
| qwen3:latest          |               0 |               27 |          0   |          0 |        27 |

### Accuracy interpretation

- **Best overall:** `mistral:latest`
- **Best speed/accuracy balance:** `llama3.1:latest`
- **Best vision-capable performer in this set:** `llava:latest`
- **Best coding-oriented performer in this set:** `Qwen2.5-coder:latest`
- **Not usable on this benchmark as recorded:** `deepseek-coder:latest`, `deepseek-r1:8b`, `qwen3:latest`

## Speed ranking

Measured runtime is based on the recorded `measured_seconds` field for 9 questions per run.

| Model                 |   Runs |   Avg measured s / 9Q |   Avg s / question |   Avg warmup s |
|:----------------------|-------:|----------------------:|-------------------:|---------------:|
| deepseek-coder:latest |      3 |                  2    |               0.22 |       0        |
| gemma3:1b             |      3 |                  3    |               0.33 |       0        |
| llama3.2:latest       |      3 |                  3    |               0.33 |       0        |
| phi3:latest           |      3 |                  3.67 |               0.41 |       0.666667 |
| llama3.1:latest       |      3 |                  6.33 |               0.7  |       2        |
| llava:latest          |      3 |                  7    |               0.78 |       1        |
| Qwen2.5-coder:latest  |      3 |                  7    |               0.78 |       1        |
| mistral:latest        |      3 |                  7.67 |               0.85 |       1.66667  |
| deepseek-r1:8b        |      3 |                 16.67 |               1.85 |       1.66667  |
| qwen3:latest          |      3 |                 21.33 |               2.37 |       1.33333  |

### Speed interpretation

- The fastest models by measured runtime were:
  - `deepseek-coder:latest`
  - `gemma3:1b`
  - `llama3.2:latest`

- However, those speed numbers need context:
  - `deepseek-coder:latest` produced no usable answers.
  - `gemma3:1b` produced answers, but none matched exactly.
  - `llama3.2:latest` was fast, but low-accuracy.

- Among the models that were actually competitive on accuracy:
  - **`llama3.1:latest`** was the strongest practical compromise.
  - **`mistral:latest`** was a little slower, but perfect on this test.
  - **`llava:latest`** and **`Qwen2.5-coder:latest`** had similar runtime.

## Question difficulty

This table shows how many models got each question right at least once.

| Question                                                                          |   models_correct |   accuracy_pct |
|:----------------------------------------------------------------------------------|-----------------:|---------------:|
| What day did Martin Luther King Jr. deliver his "I Have a Dream" speech?          |                6 |             60 |
| When was Albert Einstein born?                                                    |                5 |             50 |
| On what day was Anne Frank born?                                                  |                5 |             50 |
| When did Nelson Mandela become President of South Africa?                         |                5 |             50 |
| When did Amelia Earhart disappear during her attempt to circumnavigate the globe? |                4 |             40 |
| What date marks the death of Steve Jobs?                                          |                4 |             40 |
| What is the birth date of Marie Curie?                                            |                3 |             30 |
| When did Queen Elizabeth II ascend to the throne?                                 |                3 |             30 |
| On what date did Leonardo da Vinci die?                                           |                2 |             20 |

### Hardest and easiest questions

- **Easiest question:** Martin Luther King Jr. speech date
- **Hardest question:** Leonardo da Vinci death date
- Other relatively hard items:
  - Queen Elizabeth II accession date
  - Marie Curie birth date

This suggests that even when models handle modern well-known dates reasonably well, they can still fail on:
- less frequently recalled historical dates
- dates with month/day confusion
- cases where the model prefers a plausible-looking but wrong date

## Determinism and consistency

A notable result: **every model was perfectly stable across all 3 runs** for every question.

That means:
- no model improved or degraded between runs
- there was no observable sampling randomness in the recorded outputs
- the benchmark reflects each model's **fixed factual pattern**, not noisy generation behavior

This is useful because it makes the findings easier to trust:
- if a model missed a question, it missed it every time
- if a model hallucinated a date, it hallucinated the same date every time
- if a model omitted an answer, it omitted it every time

## Representative misses

| Model                 | Example miss                            | Model output   |
|:----------------------|:----------------------------------------|:---------------|
| llava:latest          | What is the birth date of Marie Curie?  | 1867-05-04     |
| llama3.1:latest       | On what date did Leonardo da Vinci die? | -              |
| Qwen2.5-coder:latest  | On what date did Leonardo da Vinci die? | 1520-05-23     |
| phi3:latest           | On what date did Leonardo da Vinci die? | 1519-07-02     |
| llama3.2:latest       | On what date did Leonardo da Vinci die? | -              |
| gemma3:1b             | On what date did Leonardo da Vinci die? | 1519-03-10     |
| deepseek-coder:latest | On what date did Leonardo da Vinci die? | -              |
| deepseek-r1:8b        | On what date did Leonardo da Vinci die? | -              |
| qwen3:latest          | On what date did Leonardo da Vinci die? | -              |

## Model-by-model observations

### `mistral:latest`
- **Best performer overall**
- 100% exact match accuracy
- No missing answers
- Not the fastest, but the most reliable in this dataset

### `llava:latest`
- Strong accuracy at 88.9%
- No missing answers
- Main weakness here was a confidently wrong date for Marie Curie

### `llama3.1:latest`
- Same accuracy tier as `llava:latest`
- Faster than `llava:latest` and `mistral:latest`
- Main issue was one repeated omission for Leonardo da Vinci's death date

### `Qwen2.5-coder:latest`
- Respectable but clearly behind the top 3
- Missed or got wrong several historical dates
- Decent practical performance, but not the best factual date model in this set

### `phi3:latest`
- Middle-to-lower tier
- Produced some correct answers, but also malformed or incorrect dates
- Reliability is too uneven for date-extraction style tasks

### `llama3.2:latest`
- Fast, but low accuracy
- Seems better suited as a lightweight fallback than as a factual benchmark winner

### `gemma3:1b`
- Interesting failure mode: it answered all questions, but none exactly matched
- That points to strong formatting compliance, but poor factual recall on this benchmark

### `deepseek-coder:latest`, `deepseek-r1:8b`, `qwen3:latest`
- All three recorded only `-` outputs throughout
- Based on these files alone, they were not usable in this evaluation
- This may indicate a prompting/output-parsing issue rather than pure model incapability, but the benchmark result itself is still zero usable accuracy

## Recommendations

### If your priority is best factual reliability
Choose **`mistral:latest`**.

### If your priority is best balance of speed and quality
Choose **`llama3.1:latest`**.

### If your priority is multimodal potential with still-strong results
Choose **`llava:latest`**.

### If your priority is coding-first and you want one of the tested coder models
Choose **`Qwen2.5-coder:latest`**, but expect weaker factual date performance than `mistral` or `llama3.1`.

### If your priority is smallest/fastest lightweight fallback
Use **`llama3.2:latest`**, but not for accuracy-critical tasks.

## Caveats

- This is a **small benchmark**: only 9 questions, all date-based.
- The scoring is **strict exact match**, so near-miss dates still count as wrong.
- Some zero-score models may have suffered from **pipeline or parsing issues** rather than purely poor underlying knowledge.
- Because the benchmark is narrow, the conclusions are strongest for **short factual date retrieval**, not for broad reasoning, coding, or creative tasks.

## Bottom line

For this exact benchmark, the ranking is simple:

1. **`mistral:latest`** — best overall
2. **`llama3.1:latest`** and **`llava:latest`** — strong second tier
3. **`Qwen2.5-coder:latest`** — usable but clearly weaker
4. Everything else — either too inaccurate or not producing usable answers in the recorded runs

If your goal is a default offline factual assistant for short date questions, **`mistral:latest` is the winner in these results**.
