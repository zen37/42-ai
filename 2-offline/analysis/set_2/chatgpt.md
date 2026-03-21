# Offline model benchmark report (9 questions)

## Scope and data integrity

- Files analyzed: `9q_duration.csv`, `9q_responses.csv`, `9q_results.csv`.
- The uploaded data contains **9 questions** and **2 recorded runs per model**.
- That means each model has a maximum of **18 scored answers** in this benchmark, not 27.
- The benchmark is **text-only** and checks **exact string match** against the expected `yyyy-mm-dd` answers.
- A response of `-` was treated as a **missing answer**.

## Executive summary

- **Best overall:** `mistral:latest` with **18/18** exact matches (**100.0%**).
- **Best speed/quality balance:** `mistral:latest` among the high-accuracy models, based on the lowest average measured time while staying at or above 80% accuracy.
- **Second tier:** `llama3.1:latest` and `llava:latest` at **16/18** exact matches each.
- **Run-to-run behavior:** every model returned the **same answers in both recorded runs**.
- **Main failure pattern:** several models did not answer at all and returned only `-` for every question.

## Model ranking

| model                 |   exact_matches |   total_answers |   accuracy_pct |   missing_answers |   avg_measured_seconds |   sec_per_question | same_answers_both_runs   |
|:----------------------|----------------:|----------------:|---------------:|------------------:|-----------------------:|-------------------:|:-------------------------|
| mistral:latest        |              18 |              18 |          100   |                 0 |                    7   |              0.778 | True                     |
| llama3.1:latest       |              16 |              18 |           88.9 |                 2 |                    6   |              0.667 | True                     |
| llava:latest          |              16 |              18 |           88.9 |                 0 |                    7   |              0.778 | True                     |
| Qwen2.5-coder:latest  |              12 |              18 |           66.7 |                 2 |                    7   |              0.778 | True                     |
| gemma3:4b             |              10 |              18 |           55.6 |                 4 |                    6   |              0.667 | True                     |
| phi3:latest           |               8 |              18 |           44.4 |                 4 |                    3.5 |              0.389 | True                     |
| llava-phi3:latest     |               8 |              18 |           44.4 |                 2 |                    4   |              0.444 | True                     |
| llama3.2:latest       |               4 |              18 |           22.2 |                 4 |                    3   |              0.333 | True                     |
| moondream:latest      |               0 |              18 |            0   |                18 |                    1   |              0.111 | True                     |
| deepseek-coder:latest |               0 |              18 |            0   |                18 |                    2   |              0.222 | True                     |
| gemma3:1b             |               0 |              18 |            0   |                 0 |                    3   |              0.333 | True                     |
| qwen3:latest          |               0 |              18 |            0   |                18 |                    4.5 |              0.5   | True                     |
| qwen3:4b              |               0 |              18 |            0   |                18 |                    6   |              0.667 | True                     |
| deepseek-r1:8b        |               0 |              18 |            0   |                18 |                   16.5 |              1.833 | True                     |


## What the results say

### 1) Strongest performers
- `mistral:latest` was the clear winner: perfect accuracy, no missing answers, and stable output across both runs.
- `llama3.1:latest` matched the second-best score with no missing answers, which is stronger than the same score achieved through occasional abstention.
- `llava:latest` matched that score too, but did so with missing answers rather than a wrong date on one item.

### 2) Coding model performance on date recall
- `Qwen2.5-coder:latest` reached **12/18** exact matches. It handled many modern dates correctly but missed or skipped several history items.
- `deepseek-coder:latest` returned only `-` in both runs, so this benchmark does not show useful factual recall for that setup.

### 3) Small or multimodal models
- `llava:latest` performed strongly even though this benchmark did not use images.
- `moondream:latest` and `llava-phi3:latest` are vision-oriented families, but here they were being tested only as text answerers.
- `gemma3:1b` is notable because it answered every question in the required date format, but **every answer was wrong**.

### 4) Zero-score models are not all the same
- Some zero-score models failed by **non-answering**: `deepseek-coder:latest`, `deepseek-r1:8b`, `moondream:latest`, `qwen3:4b`, and `qwen3:latest` all returned only `-`.
- `gemma3:1b` also scored zero, but for a different reason: it returned a date for every prompt, just never the correct one.

## Question difficulty

| question                                                                          |   exact_matches |   total_answers |   accuracy_pct |   missing_answers |
|:----------------------------------------------------------------------------------|----------------:|----------------:|---------------:|------------------:|
| On what date did Leonardo da Vinci die?                                           |               4 |              28 |           14.3 |                16 |
| When did Queen Elizabeth II ascend to the throne?                                 |               6 |              28 |           21.4 |                14 |
| What date marks the death of Steve Jobs?                                          |               8 |              28 |           28.6 |                12 |
| When did Amelia Earhart disappear during her attempt to circumnavigate the globe? |               8 |              28 |           28.6 |                12 |
| What is the birth date of Marie Curie?                                            |              10 |              28 |           35.7 |                10 |
| On what day was Anne Frank born?                                                  |              12 |              28 |           42.9 |                14 |
| When did Nelson Mandela become President of South Africa?                         |              14 |              28 |           50   |                10 |
| When was Albert Einstein born?                                                    |              14 |              28 |           50   |                10 |
| What day did Martin Luther King Jr. deliver his "I Have a Dream" speech?          |              16 |              28 |           57.1 |                10 |


Interpretation:
- **Hardest question:** “On what date did Leonardo da Vinci die?” with only **4/28** exact matches.
- **Easiest question:** “What day did Martin Luther King Jr. deliver his "I Have a Dream" speech?” with **16/28** exact matches.
- Renaissance and royal-history dates were harder than the modern 20th-century biography questions.

## Notable examples

### `mistral:latest`
- No errors. It matched all 9 answers in both runs.

### `llama3.1:latest`
- Missed: “On what date did Leonardo da Vinci die?” → expected `1519-05-02`, produced `-`.

### `llava:latest`
- Missed: “What is the birth date of Marie Curie?” → expected `1867-11-07`, produced `1867-05-04`.

### `Qwen2.5-coder:latest`
- Missed: “When did Queen Elizabeth II ascend to the throne?” → expected `1952-02-06`, produced `-`.
- Missed: “On what date did Leonardo da Vinci die?” → expected `1519-05-02`, produced `1520-05-23`.
- Missed: “When did Amelia Earhart disappear during her attempt to circumnavigate the globe?” → expected `1937-07-02`, produced `1937-07-24`.

### `gemma3:1b`
- Wrong but formatted: “When was Albert Einstein born?” → expected `1879-03-14`, produced `1907-05-14`.
- Wrong but formatted: “What day did Martin Luther King Jr. deliver his "I Have a Dream" speech?” → expected `1963-08-28`, produced `1963-01-20`.
- Wrong but formatted: “When did Queen Elizabeth II ascend to the throne?” → expected `1952-02-06`, produced `1952-09-01`.
- Wrong but formatted: “On what date did Leonardo da Vinci die?” → expected `1519-05-02`, produced `1519-03-10`.

## Recommendations

- For this exact benchmark, `mistral:latest` is the best local baseline.
- If you want a stronger speed/quality compromise, `llama3.1:latest` is attractive because it was slightly faster than `mistral:latest` while staying near the top.
- `llava:latest` is also strong, but its vision capability is irrelevant here, so this result should not be overinterpreted.
- Investigate why several models returned only `-`; that looks more like a prompting or pipeline issue than a pure knowledge comparison.
- If you want a more representative benchmark, expand beyond 9 questions and include categories other than biography dates.

## Caveats

- This is a very small benchmark: **9 prompts only**.
- It measures **exact-match date recall**, not broader reasoning or coding ability.
- A model that returns `-` is heavily penalized here, even if its underlying knowledge might be better than the benchmark suggests.
