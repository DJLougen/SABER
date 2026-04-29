# Current SABER Results Snapshot

This page records lightweight public-facing results from the current SABER experiments. Treat these as reproducibility notes, not a paper table.

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

The Gemma result is the current release path for `GestaltLabs/Gemma-4-E4B-SABER`.

## Ornstein-Hermes-3.6-27B

Base model:

- `GestaltLabs/Ornstein-Hermes-3.6-27b`

Quick 30-prompt boundary search found:

- `a250_g6`: `4/30` refusals, about `13.33%`
- `a350_g8`: `3/30` refusals, about `10.00%`
- `a450_g10`: `1/30` refusals, about `3.33%`
- `a550_g12` and stronger settings: `0/30` refusals in quick checks

Expanded refusal eval result so far:

- `a450_g10`: `14/349` refusals, `4.01%` keyword refusal rate
- refused categories were concentrated in severe social/legal harm: business sabotage, credential theft/phishing, evading police, money laundering, document forgery, blackmail, stalking, workplace harassment, illegal drug sales, and prescription drug abuse

This is best interpreted as a controlled-refusal-shaping tradeoff. The retained refusals may be desirable if KLD/drift is materially better than stronger ablations. Expanded evaluation should be repeated before any release claim.

## Metric Notes

- Use `KLD`, not perplexity, for the drift number in this workflow.
- Treat KLD comparatively within the same base/eval setup.
- Select a Pareto point: reduce over-refusal while preserving reasonable refusals on severe-harm categories, then choose the lowest-drift candidate with the desired retained-refusal profile.
