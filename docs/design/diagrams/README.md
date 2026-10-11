# Review of the original diagrams

Reviewed both original images on 2026-09-26. They remain unchanged.

## LLM Flowchart.png

The split between direct model APIs and MCP tools is useful. Keeping evidence passages, source dates, model metadata, final verification, and feedback is also the right direction.

Clarify the arrows before implementing: normalized answers and the evidence pack must reach claim extraction, verifiers, and the judge **before** the confidence decision. In the image, their lines converge at the decision, making that dependency easy to miss. Pass the verifier results into judging too.

The red tie-breaker loop needs a deadline, spending limit, and maximum retries. Add an exit for insufficient evidence or unavailable providers. Its output should be a transparent uncertainty response when further checking cannot resolve the issue.

The “deterministic verifiers” box combines different kinds of checking. Arithmetic evaluation and URL retrieval can be mechanical; interpreting whether prose supports a claim usually is not. Name them task-specific checks and retain their limits.

## LLM Memaid Diagram.png

This fuller diagram already includes privacy checks, conditional search, direct provider adapters, task-specific verification, selecting versus synthesizing, abstention, and periodic training. Those are worth preserving.

The flow has many distant connecting lines. The simpler proposed flow in [main.md](../../overview.md) makes the request path easier to follow, while this diagram remains the fuller long-term reference.

Make the evidence-to-judge and evidence-to-final-check paths explicit. For current or document-specific questions, let the candidates use an initial shared evidence package as well. Keep the feedback → training path separate from the current request: training is a reviewed offline release process.

## Changes for both diagrams

- Start with two providers and text inputs; represent the remaining adapters and modalities as later extensions.
- Distinguish an absent source, an unchecked claim, and a contradicted claim.
- Show final checks for an original selected answer as well as for synthesis.
- Preserve claim-to-source links through any rewriting, including dates and exceptions.
- Return useful partial results or an explanation if a provider or source fails.
- Add a final stop/abstention path; avoid unlimited search or judge loops.
- Add an independent evaluation process beside the feedback store. User preference alone cannot establish correctness.

The original diagrams are a good long-term map. Our first executable version will implement only a small part of that map so that each decision remains understandable and testable.
