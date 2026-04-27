# Current SABER Results Snapshot

This file records the working state from the turquoise experiments. It is not a final paper table.

## Gemma 4 E4B

Current upload target:

- `GestaltLabs/Gemma-4-E4B-SABER`

Best current aggressive candidate from the local autotune table:

- run: `gemma4_e4b_auto_svd_nh60_a825_g14`
- quick keyword refusal: `0.00%`
- mean KLD: about `0.4327`
- residual refusal score: about `3.9394`

Balanced candidate:

- run: `gemma4_e4b_auto_svd_a825_g14`
- quick keyword refusal: about `2.04%`
- mean KLD: about `0.3164`
- residual refusal score: about `3.6821`

## Ornstein-Hermes-3.6-27B

Base model:

- `GestaltLabs/Ornstein-Hermes-3.6-27b`

Quick 30-prompt boundary search found:

- `a250_g6`: `4/30` refusals, about `13.33%`
- `a350_g8`: `3/30` refusals, about `10.00%`
- `a450_g10`: `1/30` refusals, about `3.33%`
- `a550_g12` and stronger settings: `0/30` refusals in quick checks

Caution: 30 prompts is too small for final claims. Expanded refusal eval is now queued/running at 349 prompts per selected candidate.

## Metric Notes

- Use `KLD`, not perplexity, for the drift number in this workflow.
- Treat KLD comparatively within the same base/eval setup.
- Selection target is a Pareto point: first get refusal rate to zero on a sufficiently large eval, then choose the lowest-drift candidate among the zero-refusal candidates.
