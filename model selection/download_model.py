"""Download and check the selected public model. This does not train it."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import HfApi, snapshot_download
HERE = Path(__file__).resolve().parent
choice = json.loads((HERE / "model_choice.json").read_text())
destination = HERE / choice["local_directory"]
info = HfApi(token=False).model_info(choice["model_id"], revision=choice["revision"], files_metadata=True)
if info.sha != choice["revision"]:
    raise RuntimeError("The model version did not match our choice.")
allowed = ["*.json", "*.safetensors", "merges.txt", "LICENSE", "README.md"]
print("Downloading the original model files (about 8 GB).", flush=True)
snapshot_download(repo_id=choice["model_id"], revision=choice["revision"], local_dir=destination, allow_patterns=allowed, token=False, max_workers=3)
records = []
for item in info.siblings:
    path = destination / item.rfilename
    if not path.is_file():
        continue
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
    actual = sha.hexdigest()
    expected = item.lfs.sha256 if item.lfs else None
    if path.stat().st_size != item.size or (expected and actual != expected):
        raise RuntimeError("File verification failed: " + item.rfilename)
    records.append({"file": item.rfilename, "bytes": item.size, "sha256": actual, "publisher_sha256": expected})
index = json.loads((destination / "model.safetensors.index.json").read_text())
required = set(index["weight_map"].values())
if not required.issubset({item["file"] for item in records}):
    raise RuntimeError("One or more weight files are missing.")
receipt = {"model_id": choice["model_id"], "revision": info.sha, "retrieved_at": datetime.now(timezone.utc).isoformat(), "files": records}
(HERE / "download_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("Verified", len(records), "files. No training has run.")
