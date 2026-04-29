# Safety And Release Policy

SABER is dual-use model-editing research. It can reduce refusal behavior in open-weight models and should be handled as a controlled evaluation workflow, not as a generic "uncensoring" button.

## Public Framing

Public releases should describe SABER candidates as refusal-shaping or refusal-frontier experiments. Avoid claiming that a model is "safe" based only on a low refusal rate or a small prompt set.

Good release notes include:

- base model and license
- exact evaluation prompt count
- prompt category composition
- retained refusal categories
- KLD or comparable drift metric
- generation settings used for scoring
- qualitative review notes
- known failure modes

## Minimum Release Evidence

Before publishing a tuned model, collect:

1. A saved SABER run config.
2. Generation-based refusal evaluation.
3. Drift evaluation against the source model.
4. Category review of retained refusals.
5. Qualitative samples from harmless, capability, and high-risk prompts.
6. A model card that states this is experimental and model-family dependent.

## Handling High-Risk Prompts

SABER uses harmful prompts to locate and evaluate refusal-related behavior. Keep prompt sets and outputs in controlled evaluation artifacts. Do not include procedural high-risk content in README examples, marketing copy, or casual issue discussions.

## Known Limits

- Refusal keyword checks can undercount or overcount refusals.
- A low refusal rate can indicate harmful compliance if not paired with category review.
- KLD is comparative and depends on the exact base model, prompt set, and generation settings.
- Results often do not transfer cleanly across model families.
