# Ready for the first small training trial

**Later October 6 update:** [the first actual training trial](TRAINING_TRIAL.md) is complete. The measurements below remain the historical dry checks, during which no weights were updated.

Measured October 6, 2026. **The baseline is protected, training data is formatted, and the local gradient/memory probe passed. No learned weights were updated and no trained checkpoint was saved.**

This completes the data-length and dry compatibility checks in Phase 2. Phase 3 is the first controlled fine-tuning trial. Fine-tuning gives an existing model extra practice.

## What we did, in order

1. **Protected the baseline.** Its original answers, scores, prompt, code, and model files remain unchanged. Read [how the lock works](BASELINE_LOCK.md).
2. **Read the training portion.** We checked all 6,623 training rows against their release fingerprint. No calibration or final-test records were opened. Training group IDs were also checked against the already-used frozen baseline sample.
3. **Built teaching conversations.** Each contains judging instructions, a question and candidate answers, followed by the expected preference as the assistant's target. Reviewer notes and the expected label stay outside the input message. In a future training run, that final target supplies the learning signal; it is never provided as a hint during evaluation.
4. **Measured both answer orders.** We exchanged the answer texts and the A/B labels together. Ties stay ties. Both versions remain in the training portion.
5. **Set a small starting limit.** Both orientations must fit 2,048 tokens, including the expected decision and conversation endings. A token is a small piece of text read by the model. Longer pairs are listed and set aside, not cut or deleted.
6. **Prepared a trial.** A fixed ID-based ordering selected 100 distinct training question groups, with two presentations per group: 200 teaching conversations. This file is prepared; it has not been used to update weights.
7. **Tested training calculations locally.** We loaded the original Qwen model, froze its weights, added small temporary LoRA matrices, allocated the optimizer's state, and calculated gradients on the shortest and longest fitting examples.
8. **Checked the result.** The error and gradients were finite, target counts matched, memory stayed below the chosen guideline, and the temporary adapter's before/after fingerprints were identical. The optimizer recorded zero update steps. The baseline guard passed again afterward.

## What the length scan found

| Measurement | Result |
| --- | ---: |
| Original training pairs scanned | 6,623 |
| Presentations measured, including both orders | 13,246 |
| Shortest / median / longest complete presentation | 169 / 885 / 3,522 tokens |
| 95th / 99th percentile length | 1,731 / 2,317 tokens |
| Pairs fitting the first trial's 2,048-token cap in both orders | 6,489 — about 98.0% |
| Pairs set aside for exceeding that cap | 134 |
| Truncated examples | 0 |
| Initial trial | 100 groups, 200 presentations |

The 100 original trial labels are 26 A, 45 B, and 29 ties. After adding the reversed presentations, there are 71 A targets, 71 B targets, and 58 ties. This is a reproducibly selected small trial, not a claim about the label proportions in future user questions.

[Length profile](results/training_readiness_20261006T101929653463Z/profile.json) · [Every pair's length and eligibility](results/training_readiness_20261006T101929653463Z/lengths.jsonl) · [Trial IDs and groups](results/training_readiness_20261006T101929653463Z/trial_index.json)

## What “the training probe passed” means

**LoRA** adds a small set of adjustable numbers beside selected parts of the existing model. This setup adds 327,680 numbers across the query/value attention projections in the last four model blocks. The original model weights stay frozen. The names of the trainable arrays were checked to exclude ordinary base-model weights.

A **gradient** describes a potential change that could reduce the numerical learning error, called the **loss**. Calculating gradients does not apply those changes. An **optimizer** is the component that would apply them during training. We allocated Adam's state to include that memory in the probe, but did not call its update operation.

| Actual probe | Complete / padded tokens | Calculation time | Peak MLX memory |
| --- | ---: | ---: | ---: |
| Shortest fitting example | 169 / 193 | 1.98 seconds | 7.87 GiB |
| Longest fitting example | 2,048 / 2,048 | 8.32 seconds | 9.42 GiB |

The complete supervised worker, including file verification and loading, took 19.32 seconds. Model loading took 2.68 seconds. **MLX peak memory is a library allocation measurement, not total Mac memory.** The 18 GiB MLX setting is a guideline, not a hard whole-process cap. The worker had a hard 600-second deadline.

This establishes that the chosen formatting, LoRA setup, answer-only loss, backward calculation, and optimizer-state allocation work for these probe lengths on this Mac. It does **not** establish full-run memory behavior, optimizer-update correctness, checkpoint saving/reloading, useful learning, or a quality improvement. Those are the next trial's checks.

[Measured probe](results/training_readiness_20261006T101929653463Z/probe_result.json) · [Worker outcome](results/training_readiness_20261006T101929653463Z/probe_run.json) · [Settings](results/training_readiness_20261006T101929653463Z/training_config.json)

## Two compatibility issues we caught

The first preparation attempt stopped because Transformers 5.18.0 returns a dictionary from its raw chat tokenizer by default. We now explicitly request token-ID lists and verify that the prompt is an exact prefix of the complete teaching conversation. MLX's tokenizer wrapper already supplied this option in the baseline, so the baseline was unaffected. The [failed preparation record](results/training_readiness_20261006T101745687513Z/failure.json) remains separate.

The installed MLX-LM 0.32.0 trainer's default loss included one extra padding position with our padded batches. Padding consists of filler tokens added to align batch lengths; these must not become teaching targets. A tiny test expected two target tokens, counted three with the original boundary, and counted exactly two after our correction. The new `answer_only_loss` wrapper uses the last real token as the end boundary. Third-party code was not edited. **Future training for this prepared run must use this corrected loss, not invoke the default trainer loss unchanged.**

The real model probes also checked exact target counts: four tokens for the short conversation and three for the long one, including conversation endings but excluding input and padding.

## The next controlled trial

The saved proposal is 20 optimizer updates with one presentation per batch, a 2,048-token cap, LoRA rank 8, four final blocks, learning rate 0.00001, checkpoint recomputation, and a 900-second worker deadline. The local optimizer method is Adam. These are conservative starting settings, not settings shown to produce the best judge.

The next trial must:

1. Recheck the protected baseline and prepared-file fingerprints.
2. Use only the prepared training conversations and the corrected answer-only loss.
3. Apply a small number of updates to LoRA arrays, leaving base weights frozen.
4. Save a separate adapter checkpoint and confirm that it reloads correctly.
5. Evaluate the resulting model using the fixed baseline question set and both answer orders, writing a new result folder. Keep the original scores intact.
6. Compare preference matches, ties, order stability, invalid outputs, time, and memory. Keep or reject the trained experiment based on the evidence; training is not automatically improvement.

Twenty updates are a check that training and saving work, not a promise of a strong judge. Validation remains development data; the final test stays reserved for a fixed later version. Public preference labels do not establish source-grounded factual correctness. No paid API or Colab session was used in this milestone.

## Run and read the code

Preparation makes a new timestamped folder and does not train:

```bash
"model selection/.venv/bin/python" "model selection/prepare_training.py"
```

The probe accepts a newly prepared folder and needs access to the Mac GPU:

```bash
"model selection/.venv/bin/python" "model selection/probe_training.py" "PATH_TO_NEW_PREPARED_FOLDER"
```

`PATH_TO_NEW_PREPARED_FOLDER` is a placeholder to replace with the path printed by preparation. The probe refuses to overwrite an earlier probe in the same folder.

The new offline tests need no GPU:

```bash
"model selection/.venv/bin/python" -m unittest discover -s "model selection" -p test_training_readiness.py -v
```

Five tests passed: swapped labels and hidden metadata; preserved ties and training-only inputs; tokenizer shape/prefix checks; percentile calculation; and baseline tampering/missing/write-permission detection. The actual GPU probe separately checked the padding mask and gradients.
