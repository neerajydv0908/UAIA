# First real model run

**Later update — October 6:** the first [development baseline](BASELINE.md) is locked and the [dry training-readiness probe](TRAINING_READINESS.md) passed. The three exercises below remain the historical setup check; actual weight-update and checkpoint checks are next.

Recorded 2026-10-05. **Qwen3-4B-Instruct-2507 is downloaded and ran locally on a Mac. No training has run.**

## What happened, simply

We loaded the already-trained Qwen model, showed it three small exercises, and saved its actual answers. It did not use a paid API. Our program did not change the model's learned numbers.

| Teaching exercise | Expected | Actual output | Matches teaching key |
| --- | --- | --- | --- |
| library_A | A | `A` | True |
| library_B_swapped | B | `B` | True |
| equal_answers | tie_preference | `B` | False |

2 of 3 teaching keys matched. These are authored, deliberately simple examples. This is a setup check, not a meaningful accuracy score or proof of evidence-checking reliability.

The equivalent-duration exercise returned B instead of the expected tie. We keep this result visible. It tells us to examine tie behavior during the development baseline; it does not tell us the cause or overall model quality. We did not keep changing the prompt until this teaching example passed.

## Measured on this run

- Downloaded files: 12; total 8,060,915,998 bytes (about 8.06 GB).
- All three weight files matched the publisher's SHA-256 fingerprints. Smaller files have saved local fingerprints and checked sizes.
- Model loading time: 3.34 seconds.
- Peak memory reported by MLX: 7.67 GiB. This is MLX's peak allocation, not all memory used by the process or the computer.
- library_A: 185 input tokens; 3.65 seconds for its decision.
- library_B_swapped: 185 input tokens; 0.61 seconds for its decision.
- equal_answers: 183 input tokens; 0.58 seconds for its decision.

A token is a small piece of text. First-run timings can include warm-up work; these few timings do not predict full training time.

[Actual saved result](results/teaching_run_20261006T041018219907Z.json) · [Download receipt](download_manifest.json) · [Exact tool versions](requirements.lock.txt)

## What is finished and what remains

- Finished: model selection, fixed source version, official files, separate Mac environment, download verification, and local teaching run.
- Before training: measure the unchanged judge on a development sample, count dataset token lengths, set the input and memory budget, and convert only approved training records into the trainer's format.
- Then: try a short LoRA training run, compare before/after behavior, and expand only if the result helps.

**Phase 2 has started successfully; its full baseline and training-fit checks remain.** The product still uses its existing rule checker until the learned judge is evaluated and integrated.
