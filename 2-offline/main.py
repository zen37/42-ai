import requests
import csv
import os
import time
from datetime import datetime, timezone

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"  # Change to your model name if different
INPUT_FILE = "files/10q.txt"
OUTPUT_FILE = "files/10q_responses.csv"


def ask_model(question: str) -> str:
    prompt = (
        question.strip()
        + "\n\nRespond with ONLY the answer. No explanation. Provide the date in the format yyyy-mm-dd. If you don't know repsond with -"
    )

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=120,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def read_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_unique_column_name(headers, base_name):
    if base_name not in headers:
        return base_name

    counter = 2
    while f"{base_name}_{counter}" in headers:
        counter += 1

    return f"{base_name}_{counter}"


def load_existing_results(path):
    if not os.path.exists(path):
        return [], []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    return rows[0], rows[1:]


def save_results(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def append_duration_row(input_file: str, model: str, duration_ms: int):
    # Build duration filename: same directory, base name + "_duration.csv"
    folder = os.path.dirname(input_file)
    base = os.path.splitext(os.path.basename(input_file))[0]
    duration_file = os.path.join(folder, f"{base}_duration.csv")

    file_exists = os.path.exists(duration_file)

    # ISO datetime in UTC (stable). If you want local time, remove tz=timezone.utc.
    dt = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(duration_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["datetime", "model", "duration_ms"])
        writer.writerow([dt, model, duration_ms])


# ... everything above unchanged ...


def main():
    questions = read_questions(INPUT_FILE)
    headers, rows = load_existing_results(OUTPUT_FILE)

    if not headers:
        headers = ["Question"]
        rows = [[q] for q in questions]

    existing_questions = {row[0] for row in rows}
    for q in questions:
        if q not in existing_questions:
            rows.append([q] + [""] * (len(headers) - 1))

    new_column_name = get_unique_column_name(headers, MODEL_NAME)
    headers.append(new_column_name)

    for row in rows:
        row.append("")

    model_index = len(headers) - 1

    # ✅ PURE INFERENCE TIMER (only model calls)
    t0 = time.perf_counter()

    for question in questions:
        answer = ask_model(question)

        for row in rows:
            if row[0] == question:
                row[model_index] = answer
                break

    duration_s = round(time.perf_counter() - t0)  # no decimals

    # ✅ I/O happens AFTER timing
    save_results(OUTPUT_FILE, headers, rows)
    append_duration_row(INPUT_FILE, MODEL_NAME, duration_s)

    print(f"\nSaved results to {OUTPUT_FILE}")
    print(
        f"Saved duration to {os.path.splitext(INPUT_FILE)[0]}_duration.csv: {duration_s} s"
    )


if __name__ == "__main__":
    main()
