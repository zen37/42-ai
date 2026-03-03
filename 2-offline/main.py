import requests
import csv
import os
import time
from datetime import datetime, timezone
import re

# --------------------------
# Config
# --------------------------
OLLAMA_URL = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_URL}/api/generate"
TAGS_URL = f"{OLLAMA_URL}/api/tags"

INPUT_FILE = "files/9q.txt"
OUTPUT_FILE = "files/9q_results.csv"

# Date-only extraction: YYYY-MM-DD or "-"
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 900
MAX_RETRIES = 3

# Enforced cool-down between models (5 minutes)
GAP_SECONDS = 180


# --------------------------
# Ollama helpers
# --------------------------
def list_ollama_models() -> list[str]:
    """
    Lists locally available Ollama models via /api/tags.
    """
    r = requests.get(TAGS_URL, timeout=(CONNECT_TIMEOUT_S, 60))
    r.raise_for_status()
    data = r.json()
    models = [m["name"] for m in data.get("models", []) if "name" in m]
    models.sort()
    return models


def evict_model(model_name: str) -> None:
    """
    Force-unload a model from memory (best-effort).
    """
    try:
        requests.post(
            GENERATE_URL,
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=(CONNECT_TIMEOUT_S, 60),
        ).raise_for_status()
        print(f"[{model_name}] evicted (keep_alive=0)")
    except Exception as e:
        print(f"[WARN] eviction failed for {model_name}: {e}")


def cooldown_wait(seconds: int) -> None:
    """
    Wait between models to ensure they don't linger in memory.
    (Counts down in 10-second steps so you see progress.)
    """
    if seconds <= 0:
        return

    print(f"Cooling down for {seconds} seconds...")
    step = 60
    remaining = seconds
    while remaining > 0:
        sleep_for = step if remaining >= step else remaining
        time.sleep(sleep_for)
        remaining -= sleep_for
        if remaining > 0:
            print(f"  {remaining} seconds remaining...")
    print("Cooldown finished.")


# --------------------------
# Benchmark logic (DATE ONLY)
# --------------------------
def ask_model(model_name: str, question: str) -> str:
    """
    Returns only:
      - YYYY-MM-DD (extracted from model output), OR
      - "-" if not found / request fails
    """
    prompt = f"""Return EXACTLY one date in YYYY-MM-DD format.
If the answer is unknown, return -.
Do not include any other text.

Question: {question.strip()}
Answer:"""

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                GENERATE_URL,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 16,  # hard cap to prevent rambling
                        "stop": ["\n"],  # stop after first line (best-effort)
                    },
                },
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            match = DATE_RE.search(raw)
            return match.group(1) if match else "-"

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last_err = e
            time.sleep(2 * attempt)

    print(f"[WARN] {model_name}: request failed after retries: {last_err}")
    return "-"


# --------------------------
# CSV helpers
# --------------------------
def read_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_unique_column_name(headers: list[str], base_name: str) -> str:
    """
    Always add a new column per run:
      model, model_2, model_3, ...
    """
    if base_name not in headers:
        return base_name

    counter = 2
    while f"{base_name}_{counter}" in headers:
        counter += 1
    return f"{base_name}_{counter}"


def load_existing_results(path: str):
    if not os.path.exists(path):
        return [], []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    return rows[0], rows[1:]


def save_results(path: str, headers: list[str], rows: list[list[str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def append_duration_row(
    input_file: str, model: str, warmup_s: int, measured_s: int, question_count: int
):
    """
    Writes to: files/<input_base>_duration.csv
    """
    folder = os.path.dirname(input_file)
    base = os.path.splitext(os.path.basename(input_file))[0]
    duration_file = os.path.join(folder, f"{base}_duration.csv")

    file_exists = os.path.exists(duration_file)
    dt = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(duration_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "datetime_utc",
                    "model",
                    "warmup_seconds",
                    "measured_seconds",
                    "questions",
                ]
            )
        writer.writerow([dt, model, warmup_s, measured_s, question_count])


# --------------------------
# Main
# --------------------------
def main():
    questions = read_questions(INPUT_FILE)
    if not questions:
        print("No questions found.")
        return

    models = list_ollama_models()
    if not models:
        print("No Ollama models found (/api/tags returned empty).")
        return

    print(f"Found {len(models)} models:")
    for m in models:
        print(f" - {m}")

    headers, rows = load_existing_results(OUTPUT_FILE)

    # First run: initialize with just questions
    if not headers:
        headers = ["Question"]
        rows = [[q] for q in questions]

    # Ensure all questions exist
    existing_questions = {row[0] for row in rows}
    for q in questions:
        if q not in existing_questions:
            rows.append([q] + [""] * (len(headers) - 1))

    # Fast lookup: question -> row index
    row_index = {rows[i][0]: i for i in range(len(rows))}

    for idx, model_name in enumerate(models, start=1):
        print(f"\n=== ({idx}/{len(models)}) Model: {model_name} ===")

        # Ensure true cold start for THIS model
        evict_model(model_name)

        # Add a new run column for this model
        col_name = get_unique_column_name(headers, model_name)
        headers.append(col_name)
        for row in rows:
            row.append("")
        model_col_index = len(headers) - 1

        # Warmup (1 question)
        print(f"[{model_name}] Running warmup (1 question)...")
        warmup_start = time.perf_counter()
        _ = ask_model(model_name, questions[0])
        warmup_s = round(time.perf_counter() - warmup_start)
        print(f"[{model_name}] Warmup finished in {warmup_s} s")

        # Measured pass
        print(f"[{model_name}] Running measured pass ({len(questions)} questions)...")
        measured_start = time.perf_counter()

        for q in questions:
            ans = ask_model(model_name, q)
            r_i = row_index.get(q)
            if r_i is not None:
                rows[r_i][model_col_index] = ans

        measured_s = round(time.perf_counter() - measured_start)
        print(f"[{model_name}] Measured run finished in {measured_s} s")

        # Save progress after each model (so an interruption doesn't lose results)
        save_results(OUTPUT_FILE, headers, rows)
        append_duration_row(
            INPUT_FILE, model_name, warmup_s, measured_s, len(questions)
        )
        print(f"[{model_name}] Saved results + duration.")

        # Evict to avoid influencing next model
        evict_model(model_name)

        # Wait 5 minutes between models (except after the last one)
        if idx < len(models):
            cooldown_wait(GAP_SECONDS)

    print(f"\nDone.")
    print(f"Results:   {OUTPUT_FILE}")
    print(f"Durations: {os.path.splitext(INPUT_FILE)[0]}_duration.csv")


if __name__ == "__main__":
    main()
