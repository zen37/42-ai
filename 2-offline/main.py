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

# Exclude models larger than this
MAX_MODEL_SIZE_GB = 6.0
MAX_MODEL_SIZE_BYTES = int(MAX_MODEL_SIZE_GB * 1024 * 1024 * 1024)

# Enforced cool-down between models (3 minutes)
GAP_SECONDS = 180

# Date-only extraction: YYYY-MM-DD or "-"
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 900
MAX_RETRIES = 3


# --------------------------
# Ollama helpers
# --------------------------
def list_ollama_models(max_size_bytes: int) -> list[dict]:
    """
    Returns a list of dicts: [{"name": "...", "size": <bytes>}, ...]
    filtered to size <= max_size_bytes (if size is available).
    """
    r = requests.get(TAGS_URL, timeout=(CONNECT_TIMEOUT_S, 60))
    r.raise_for_status()
    data = r.json()

    out = []
    for m in data.get("models", []):
        name = m.get("name")
        size = m.get("size")  # typically bytes
        if not name:
            continue

        if isinstance(size, int):
            if size <= max_size_bytes:
                out.append({"name": name, "size": size})
        else:
            # keep unknown-size models by default
            out.append({"name": name, "size": None})

    out.sort(key=lambda x: x["name"].lower())
    return out


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


def format_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    gb = n / (1024**3)
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = n / (1024**2)
    return f"{mb:.0f} MB"


def can_generate(model_name: str) -> tuple[bool, str]:
    """
    Capability probe: try a tiny generate call.
    Returns (True, "") if model can generate, otherwise (False, reason).
    """
    try:
        r = requests.post(
            GENERATE_URL,
            json={
                "model": model_name,
                "prompt": "Return -",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2, "stop": ["\n"]},
            },
            timeout=(CONNECT_TIMEOUT_S, 30),
        )

        if r.status_code >= 400:
            # Ollama often returns {"error": "..."}
            try:
                reason = r.json().get("error", r.text)
            except Exception:
                reason = r.text
            return False, f"HTTP {r.status_code}: {reason}"

        data = r.json()
        if "error" in data:
            return False, str(data["error"])

        return True, ""

    except requests.exceptions.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


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
                        "num_predict": 16,
                        "stop": ["\n"],
                    },
                },
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            )

            # Graceful handling of non-suitable models / errors
            if response.status_code >= 400:
                try:
                    err = response.json().get("error", response.text)
                except Exception:
                    err = response.text
                print(f"[WARN] {model_name}: HTTP {response.status_code}: {err}")
                return "-"

            data = response.json()
            if "error" in data:
                print(f"[WARN] {model_name}: {data['error']}")
                return "-"

            raw = (data.get("response") or "").strip()
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

        except Exception as e:
            print(f"[WARN] {model_name}: unexpected error: {e}")
            return "-"

    print(f"[WARN] {model_name}: request failed after retries: {last_err}")
    return "-"


# --------------------------
# CSV helpers
# --------------------------
def read_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_unique_column_name(headers: list[str], base_name: str) -> str:
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

    models = list_ollama_models(MAX_MODEL_SIZE_BYTES)
    if not models:
        print("No Ollama models found under the size threshold.")
        return

    print(f"Size threshold: <= {MAX_MODEL_SIZE_GB} GB")
    print(f"Found {len(models)} models under threshold:")
    for m in models:
        print(f" - {m['name']} ({format_bytes(m['size'])})")

    headers, rows = load_existing_results(OUTPUT_FILE)

    if not headers:
        headers = ["Question"]
        rows = [[q] for q in questions]

    existing_questions = {row[0] for row in rows}
    for q in questions:
        if q not in existing_questions:
            rows.append([q] + [""] * (len(headers) - 1))

    row_index = {rows[i][0]: i for i in range(len(rows))}

    for idx, m in enumerate(models, start=1):
        model_name = m["name"]
        print(
            f"\n=== ({idx}/{len(models)}) Model: {model_name} ({format_bytes(m['size'])}) ==="
        )

        # Ensure true cold start for THIS model
        evict_model(model_name)

        # Capability probe (skip non-suitable models reliably)
        ok, reason = can_generate(model_name)
        if not ok:
            print(f"[SKIP] {model_name} cannot generate via /api/generate: {reason}")
            # Evict just in case, then continue
            evict_model(model_name)
            if idx < len(models):
                cooldown_wait(GAP_SECONDS)
            continue

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

        save_results(OUTPUT_FILE, headers, rows)
        append_duration_row(
            INPUT_FILE, model_name, warmup_s, measured_s, len(questions)
        )
        print(f"[{model_name}] Saved results + duration.")

        # Evict to avoid influencing next model
        evict_model(model_name)

        # Gap between models (except last)
        if idx < len(models):
            cooldown_wait(GAP_SECONDS)

    print(f"\nDone.")
    print(f"Results:   {OUTPUT_FILE}")
    print(f"Durations: {os.path.splitext(INPUT_FILE)[0]}_duration.csv")


if __name__ == "__main__":
    main()
