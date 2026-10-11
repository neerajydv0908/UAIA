"""Preserve baseline v1 once, then verify its protected files before new work."""
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = Path("model selection/baseline_v1.lock.json")
RUN = Path("model selection/results/baseline_20261006T100253287762Z")
SNAPSHOT = Path("model selection/frozen_baseline_v1")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protect(path):
    path.chmod(0o555 if path.is_dir() else 0o444)
    if sys.platform == "darwin":
        os.chflags(path, path.stat().st_flags | stat.UF_IMMUTABLE)


def verify(root=ROOT, lock=LOCK):
    manifest = json.loads((root / lock).read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Baseline file missing or changed: " + relative)
        if path.stat().st_mode & 0o222:
            raise ValueError("Baseline file is writable: " + relative)
        if manifest["mac_immutable"] and not path.stat().st_flags & stat.UF_IMMUTABLE:
            raise ValueError("Baseline file lost its Mac lock: " + relative)
    return {"status": "passed", "protected_files": len(manifest["files"]), "baseline_id": manifest["baseline_id"]}


def freeze():
    if (ROOT / LOCK).exists():
        return verify()
    plan = json.loads((ROOT / RUN / "plan.json").read_text())
    audit = json.loads((ROOT / RUN / "artifact_verification.json").read_text())
    for name, expected in audit["files_sha256"].items():
        if sha256(ROOT / RUN / name) != expected:
            raise ValueError("Baseline result changed before freezing: " + name)
    model = ROOT / "model selection"
    if sha256(model / "baseline.py") != plan["code_sha256"]:
        raise ValueError("Baseline runner changed before freezing.")
    if (model / "judge_instructions.txt").read_text() != plan["rubric"]:
        raise ValueError("Baseline prompt changed before freezing.")
    originals = list((ROOT / RUN).iterdir())
    originals += [model / name for name in ("baseline.py", "test_baseline.py", "judge_instructions.txt", "BASELINE.md")]
    copies = originals + [model / "requirements.lock.txt", model / "download_manifest.json"]
    snapshot = ROOT / SNAPSHOT
    snapshot.mkdir(exist_ok=False)
    for source in copies:
        shutil.copyfile(source, snapshot / source.name)
    (snapshot / "model_choice_at_run.json").write_text(json.dumps(plan["model"], indent=2) + "\n")
    receipt = json.loads((model / "download_manifest.json").read_text())
    if receipt["revision"] != plan["model"]["revision"]:
        raise ValueError("The model download revision no longer matches the baseline.")
    for item in receipt["files"]:
        path = model / plan["model"]["local_directory"] / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Original model file changed: " + item["file"])
        originals.append(path)
    files = originals + list(snapshot.iterdir())
    manifest = {"baseline_id": "qwen_helpsteer2_v2_validation_baseline_v1", "mac_immutable": sys.platform == "darwin",
                "files": {str(path.relative_to(ROOT)): sha256(path) for path in files},
                "policy": "Keep baseline v1 fixed. New experiments use separate output folders and adapter files. No full model copy is made."}
    (ROOT / LOCK).write_text(json.dumps(manifest, indent=2) + "\n")
    for path in files + [ROOT / LOCK]:
        protect(path)
    protect(ROOT / RUN)
    protect(snapshot)
    return verify()


if __name__ == "__main__":
    if sys.argv[1:] == ["--freeze"]:
        print(json.dumps(freeze(), indent=2))
    elif not sys.argv[1:]:
        print(json.dumps(verify(), indent=2))
    else:
        raise SystemExit("Use baseline_guard.py [--freeze].")
