# Training memory fixed: full-length readiness passed

**Measured October 7, 2026:** the final trainer completed both updates, both orders of the longest eligible **2,045-token** pair, and an exact restart in a fresh Python process. Peak MLX memory was **10.216 GiB**, below the unchanged **18-GiB** guideline. The learning-rate sweep has not run. Original Qwen JSON remains the quality reference: **239/311 clear validation matches (76.8%)**.

The fix adds one explicit evaluation step in [train_judge.py](train_judge.py). The training task, model precision, 36 LoRA blocks, seven projections per block, rank 16, dropout 0.05, eight presentations per update, and 2,048-token cap stay the same. The full model is still frozen; only the adapter learns.

## What was using the memory?

MLX can postpone calculations until their values are needed. Our old check built the calculation for the gradient's size before asking MLX to finish the gradients themselves. In this measured setup, evaluating that combined calculation used much more temporary memory. A **gradient** tells the optimizer how each trainable number affects the loss; its **norm** is a single number summarizing the gradient's size.

We now finish the loss and gradients first with `mx.eval(loss, gradients, model._judge_random)`, then construct and evaluate the norm. This is an execution-order change. The same first update produces exactly the same adapter weights, optimizer state, and private random state as the original implementation. We have measured the benefit in this setup; we have not established a universal claim about every MLX calculation.

## What happens after what?

1. Read one training prompt: question, answer A, answer B, using the selected JSON format.
2. Assign this presentation's dropout keys. Recalculation reuses those keys.
3. Calculate A/B decision scores and the loss, the penalty for preferring the wrong answer.
4. Calculate the gradients. **Finish this calculation now**, so the later diagnostic does not extend the same large computation.
5. Calculate their norm and check that loss, gradients, and memory are acceptable.
6. Add one eighth of the gradients to the running total. Repeat for four pairs in both orders.
7. Update the adapter once, then save weights, optimizer state, random state, schedule position, and data position.
8. In a separate process, reload the first saved update and reproduce the second. Require exactly matching numerical state and per-presentation measurements.

## What actually ran?

| Check | Measured result |
| --- | --- |
| Original first update, same eight presentations | 15.223 GiB peak |
| Final fixed first update | 9.170 GiB; exact original adapter/optimizer/random state |
| Final two-update probe | 16 presentations, 358–1,543 tokens; 118.10 seconds |
| Long-input stress | Both orders at 2,045 tokens; no extra optimizer update |
| Highest final-probe memory | 10.216 GiB, including the stress checks |
| Fresh-process restart | Exact adapter, optimizer, private random state, second-update loss, and all eight loss/norm measurements |
| Offline tests | 11 passed |
| Synthetic GPU mathematics | Passed loss, gradient, accumulation, dropout recomputation, schedule, and save/restore checks |
| Direct versus baseline-v2 cached scoring | Difference 0 on the checked prompt; this one near-certain decision is a limited compatibility check |
| New full-validation score / learning-rate sweep | Not measured / not run |

The first candidate compiled the gradient calculation using MLX. Its small mathematical tests passed, but the full probe still failed at **1,225 tokens and 18.518 GiB**, the same recorded failure as before. That approach was not retained in the active code. Its plan, source snapshots, checks, failure, and first checkpoint are preserved in [the failed candidate](results/training_memory_20261007T082656184836Z/plan.json).

The explicit evaluation boundary then passed at full length in [a separately planned trial](results/training_memory_20261007T090030703700Z/plan.json). We removed the unused compilation/comparison code and repeated the full probe with the final minimal implementation. The two successful versions have exactly matching numerical state after update two. The final [plan](results/training_memory_20261007T090838442782Z/plan.json), [measurements](results/training_memory_20261007T090838442782Z/measurement_summary.json), and [restart comparison](results/training_memory_20261007T090838442782Z/resume_comparison.json) record the results. Six Qwen optimizer applications completed across this milestone: one in the failed candidate, two in the first successful trial, two in the final repeat, and one replay. These are engineering checks, not six independently trained judges.

## Timing and limits

The final 16-presentation update timing averages **7.38 seconds per presentation** at **891.8 mean prompt tokens**. The prepared 2,000-pair pool averages 896.8 tokens. Multiplying the measured time by 4,000 presentations gives roughly **8.2 hours for one pass**, before validation, checkpoint overhead, or machine variability. This is a planning estimate, not a completed overnight run or a guaranteed duration.

The 18-GiB value is an MLX memory guideline checked by the runner, not a hard ceiling on total Mac memory. Passing the representative batch and longest eligible pair does not test every possible example. Keep the runtime guards during longer training. Each attempt had a saved plan, a two-minute mathematics-check limit, a twenty-minute continuous-probe limit, and a ten-minute resumed-probe limit where applicable. There were no automatic retries, downloads, paid calls, or calibration/final-test reads. The prepared 100-pair validation monitor remains unscored.

## Next step

Implement and preregister the monitored learning-rate sweep: `1e-5`, `2e-5`, and `5e-5`, 150 updates each from identical initialization and data order. Score the fixed 100-pair validation monitor in both orders at update 100 and 150; keep complete checkpoints and select by averaged-probability log-loss. Save the wall-time limits and stopping rules before launching, then run full validation on the selected checkpoint. A longer 2,000-pair run follows that comparison.

The product still uses saved demo answers and quotation rules. This milestone concerns judge fine-tuning, not foundation-model pretraining. Read [Technical_gist.md](../docs/technical_gist.md) or [the original readiness report](BINARY_TRAINING_READINESS.md). Earlier failed results remain historical facts.

The [verification receipt](results/training_memory_20261007T090838442782Z/memory_verification.json) records unchanged historical artifacts, complete checkpoints, exact resumption and the passing full-length result.
