# Baseline v1 is fixed

Protected on October 6, 2026. The reference result remains **30/60 original-order preference matches, 31/60 swapped-order matches, and 2/20 original-order ties**. The baseline was not rerun or replaced during training preparation.

## What is protected?

The [manifest](baseline_v1.lock.json) fingerprints 35 files: the original successful run, the original baseline runner and tests, its judging prompt and report, all 12 original downloaded model files, and a small separate snapshot of the experiment files and settings. The manifest itself and the run/snapshot directories are protected too.

All these files are read-only. On this Mac, they also have the **user-immutable flag**, a filesystem lock that prevents ordinary edits, replacement, or deletion until someone deliberately removes that lock. A **SHA-256 fingerprint** is a repeatable identifier for a file's contents; the guard checks both fingerprints and file protection.

The small snapshot is in `model selection/frozen_baseline_v1`. It preserves the exact model configuration from the original run, the dependency lock, download receipt, code, and results. The original 8 GB model is protected in its existing location rather than duplicated. Snapshot copies preserve their original bytes, including historical relative links; read the original [baseline report](BASELINE.md) for its working links and commands.

These are local protections and a local snapshot, not an off-device backup or protection against someone deliberately unlocking the files.

## How later work uses it

Before preparation or a model probe, the new code calls the guard. A missing file, altered content, removed Mac lock, or writable protected file stops the check. After the GPU probe, we ran the guard again successfully.

Future trained additions must be saved in separate **adapter files**. An adapter is a small collection of learned changes used alongside the original model. It must not be merged into or saved over the protected model. Future evaluation writes a new result folder and compares against the preserved sample and scores.

Live project status, such as `model_choice.json`, can still change. The original configuration that belongs to this baseline is preserved separately. New experiments do not rewrite the historical baseline report.

## Check it yourself

From the project folder:

```bash
python3 "model selection/baseline_guard.py"
```

This checks the baseline and prints a passed/failed result. It reads the model files to verify their fingerprints, so it can take several seconds. It makes no model calls, downloads, or training updates.

The one-time `--freeze` command was already run. Running it again verifies the existing baseline instead of replacing it.
