# Binary judge training: readiness before the learning-rate sweep

This milestone builds and probes `train_judge.py`. It checks training mathematics, memory use, and the ability to resume. It does **not** establish a better judge. Original Qwen JSON remains the quality reference: **239/311 clear validation matches, 76.8%** under [baseline v2](BASELINE_V2.md).

**Measured outcome: the full-length readiness gate failed.** The all-36-block setup exceeded the 18-GiB guideline at **1,225 tokens**, with **18.518 GiB** recorded. The planned 2,045-token stress pair was therefore **not reached**. Do not launch the learning-rate sweep yet.

A separate short-input control **passed exact checkpoint resumption**: two continuous updates and a fresh-process replay of the second update produced identical adapter, optimizer, random-state, and per-presentation loss/gradient measurements. This establishes the tested restart mechanism on short inputs, not readiness for the full training pool.

| Check | Measured result |
| --- | --- |
| Offline tests | 11 passed: three new data tests plus eight existing v2 tests |
| Hand-worked loss / final-position gradient | Loss error 3.81 × 10⁻⁹; maximum gradient error 0 |
| Accumulation / keyed-dropout recomputation | Maximum gradient or updated-parameter difference 0 |
| Final-only output head | Tiny gradient difference 0; first real update's adapter, optimizer, and random state exactly match the full-position version |
| Original full-position memory probe | One update completed; stopped during update two above 18 GiB; failing peak was not retained |
| Optimized full-length memory probe | One update completed; stopped during update two at 1,225 tokens and 18.518 GiB |
| Short control | Eight pairs, 16 presentations, 284–778 tokens; two continuous updates passed |
| Short-control memory | 15.076 GiB peak, including the fresh-process replay |
| Short-control speed | 79.99 seconds for two updates; 5.00 seconds per presentation, at 570.2 mean prompt tokens |
| Fresh-process restart | Exact state fingerprints and second-update measurements; one replayed update |
| Post-training direct-versus-cached score check | Not reached in the full probe; skipped with the long-stress section in the short control |
| Full sweep / new judge accuracy | Not run / not measured |

The short-input timing is **not** an overnight estimate for the 2,000-pair pool, whose average prompt is 896.8 tokens and includes inputs that exceed the measured memory limit. Across the two failed Qwen attempts and the short control/replay, **five optimizer applications completed**. The two final branches each end at update two; they are engineering probes, not promoted judges.

Read the [full-length failure](results/binary_readiness_20261007T032119111598Z/continuous_failure.json), [short-control plan](results/binary_readiness_20261007T032609258145Z/plan.json), [exact restart comparison](results/binary_readiness_20261007T032609258145Z/resume_comparison.json), and [machine-readable summary](results/binary_readiness_20261007T032609258145Z/measurement_summary.json). Earlier plans, source snapshots, partial results, and checkpoints are preserved.

## What happens after what?

1. **Keep the measured task.** Use exactly baseline-v2's binary instruction and JSON input. Each prompt contains the question and two answers. The correct preference and reviewer notes stay outside the prompt.
2. **Choose training pairs.** Start with A/B labels only. Under the new 2,048-token prompt limit, 5,046 of 5,147 clear training pairs qualify. Exclude over-length question groups without truncating text. Select 2,000 pairs in 1,997 question groups, keeping every clear pair in a selected group together. Each pair will appear in both answer orders: 4,000 presentations for one pass. This is a prepared pool; that full pass has not run.
3. **Keep checking data separate.** Prepare a fixed 100-pair, 100-group validation monitor. It is never a teaching input. The full validation protocol still uses all 396 pairs with a 4,096-token input cap. Calibration and final-test contents remain unopened.
4. **Add small trainable matrices.** LoRA adds learned values to the frozen original model. The proposed setup covers seven projections in each of 36 blocks: 252 modules, 504 trainable arrays, and 33,030,144 learned values. Rank 16 controls the adapter matrix width; scale 20 controls how strongly its output contributes. Five-percent dropout temporarily removes random training values.
5. **Read the decision scores.** Process the whole prompt through the transformer, then calculate vocabulary scores only for its final position. Read A and B and normalize between those two choices. A target of 0 means A; 1 means B. Swapping answers flips the target.
6. **Calculate a penalty.** Two-way cross-entropy is the negative logarithm of the probability assigned to the preferred answer. For example, giving the correct answer probability 0.25 produces loss `-ln(0.25) = 1.3863`. Lower is better. Earlier prompt tokens, padding, and literal tie labels are not extra targets.
7. **Accumulate before changing weights.** A gradient describes how a learned value affects the penalty. Process four question pairs in both orders, adding one eighth of each presentation's gradient. Then Adam, the optimizer, changes adapter weights once. Eight presentations are one update, not eight updates.
8. **Save enough to continue.** Save adapter weights, Adam's state, learning-rate settings and step, private dropout random state, the original plan fingerprint, and the next data position. Only complete folders become named checkpoints. An interrupted partial folder is not treated as a finished save.
9. **Prove a small restart works.** Run two updates continuously. In a separate Python process, restore after update one and replay only update two. Require exact adapter, optimizer, and random-state fingerprints and identical recorded losses.
10. **Use measured resource needs to plan the sweep.** The planned full-length check uses both orders of the longest eligible pair, 2,045 tokens each, with dry gradients and no extra optimizer update. It was not reached because the preceding 1,225-token presentation crossed the guideline. The later short-input control explicitly skipped that already-failed readiness path.

## Important implementation fixes found by the checks

The first synthetic check used an import name absent from the installed MLX namespace. That attempt loaded no Qwen model. The corrected import exposed a more consequential issue: with ordinary implicit-random dropout, the installed checkpoint helper's recomputed gradients differed from direct gradients. In a fixed tiny example, the maximum LoRA-B gradient difference was **1.9143257**; a dropout-zero control matched exactly.

An attempted global-state restoration then failed because MLX 0.32.3 exposes its global random state as a read-only sentinel. The final implementation instead assigns explicit random keys to each dropout module before each presentation. Forward calculation and backward recomputation reuse those keys. Only the private next-key state advances between presentations and is restored from disk. This follows MLX's explicit-key random API; it does not modify installed library files. [MLX random documentation](https://ml-explore.github.io/mlx/build/html/python/random.html), [versioned random-state implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/random.cpp).

The first Qwen probe completed one update, then exceeded the **18-GiB MLX guideline** during update two. Its guard did not retain the failing peak value, so only “greater than 18 GiB” is established. The replacement avoids projecting every prompt position into the full vocabulary; it projects only the final hidden position. Tiny-model loss and gradient comparisons show zero gradient difference. A real-model score check runs before training. Both versions also produced exactly matching adapter, optimizer, and random-state fingerprints after their first update on the same eight presentations. The optimization preserved that observed update but still failed the full-length memory gate at 1,225 tokens. It is not a new quality result.

The Mac has 32 GiB of unified memory. The 18-GiB setting is an MLX guideline checked during work, not a hard limit on total computer memory. The full-readiness plans set wall-clock deadlines of two minutes for tiny mathematical checks, twenty minutes for the continuous probe, and ten minutes for a resumed probe. The later short-input control tightened those last two limits to ten and six minutes. There are no automatic retries; corrected attempts receive separate plans and folders.

## What is implemented, and what remains next?

Implemented: preparation, exact selected prompt reuse, final-position binary loss, all-block LoRA setup, paired accumulation, a warm-up/cosine scheduler, private dropout keys, complete saved state, bounded readiness workers, and the small mathematical/data tests. The protected earlier training scripts are unchanged.

**Next: investigate and reduce backward-pass memory while preserving the binary objective, then repeat the full-length readiness gate.** Its failure is a resource result, not proof that the selected training settings cannot improve quality. Keep the 18-GiB guideline; do not silently shorten the training pool or raise the memory allowance to claim this setup passed. The future learning-rate sweep also still needs its monitored training loop and a saved wall-time budget after a representative setup fits. The chosen candidates are `1e-5`, `2e-5`, and `5e-5`, each for 150 updates from identical initialization and data order. That is 1,200 presentations per rate, not a full pass over the 2,000-pair pool. Monitor the fixed 100 validation pairs in both orders at the declared checkpoints, rank by their averaged-probability log-loss, and evaluate the selected checkpoint on full validation. Preserve the best measured checkpoint and the original Qwen reference; any early-stop rule must be fixed before the longer run.

The probe uses the middle candidate's schedule: 50 warm-up updates, then cosine decay to ten percent of the peak by update 150. Only the first two schedule positions are exercised on Qwen in the completed control; tests separately check the warm-up/decay boundaries. There is no strength weighting, full-data run, probability calibration, final-test evaluation, or product integration in this readiness milestone. Saved progress does not prove that arbitrary interrupted operating-system writes are recoverable; the tested contract is resumption from a completed update-boundary checkpoint.

The product demo still compares saved answers using quotation rules. These scripts make real local model calculations and tiny adapter updates. Neither is foundation-model pretraining.

The final [artifact verification](results/binary_readiness_20261007T032609258145Z/readiness_verification.json) keeps the passing audit separate from the failed memory gate. It verifies the preserved attempt chain, complete checkpoint files, exact restart comparison, unchanged historical experiments, and all 643 source-guide lines. No training process remains running.
