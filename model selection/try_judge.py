"""Try three teaching cases locally. This does not train or benchmark the judge."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
HERE = Path(__file__).resolve().parent
choice = json.loads((HERE / "model_choice.json").read_text())
model_path = HERE / choice["local_directory"]
if not (HERE / "download_manifest.json").is_file():
    raise RuntimeError("Run download_model.py first.")
cases = json.loads((HERE / "teaching_cases.json").read_text())
rubric = (HERE / "judge_instructions.txt").read_text()
started = time.perf_counter()
model, tokenizer = load(str(model_path), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
load_seconds = time.perf_counter() - started
results = []
for case in cases:
    messages = [{"role": "system", "content": rubric}, {"role": "user", "content": json.dumps(case["input"], ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_tokens = len(tokenizer.encode(prompt))
    if input_tokens > 2048:
        raise ValueError("This teaching input is too long; nothing was cut off.")
    started = time.perf_counter()
    answer = generate(model, tokenizer, prompt=prompt, max_tokens=16, sampler=make_sampler(temp=0.0), verbose=False)
    elapsed = time.perf_counter() - started
    decision = answer.strip()
    valid = decision in choice["allowed_choices"]
    results.append({"id": case["id"], "raw_output": answer, "valid_format": valid, "matches_teaching_key": valid and decision == case["expected"], "input_tokens": input_tokens, "seconds": elapsed})
    print(case["id"], "->", repr(answer), "| expected:", case["expected"], flush=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = HERE / "results" / ("teaching_run_" + stamp + ".json")
output.parent.mkdir(exist_ok=True)
report = {"model_id": choice["model_id"], "revision": choice["revision"], "training_started": False, "kind": "teaching_smoke_check_not_accuracy_benchmark", "load_seconds": load_seconds, "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3, "results": results}
output.write_text(json.dumps(report, indent=2) + "\n")
print("Saved:", output)
