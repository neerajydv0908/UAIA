"""Step 13 starts with consented records, not automatic weight updates."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .records import Run

LABELS = {"useful", "not_useful", "factual_issue", "citation_issue", "unclear"}


def save_run(run: Run, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit saves never silently overwrite a previous experiment.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(run.model_dump_json(indent=2) + "\n")


def record_feedback(run: Run, path: Path, label: str, note: str, consent: bool,
                    allow_training: bool = False) -> dict:
    if not consent:
        raise ValueError("Feedback storage requires --consent")
    if label not in LABELS or len(note) > 2000:
        raise ValueError("Choose a valid feedback label and a note under 2000 characters")
    record = {
        "run_id": run.id, "answer_hash": run.final_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label, "note": note, "storage_consent": True,
        "training_consent": allow_training, "review_status": "pending_human_review",
        "training_eligible": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record
