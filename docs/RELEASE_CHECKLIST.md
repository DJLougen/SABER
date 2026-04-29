# Release Checklist

Use this checklist before pushing a SABER-edited model or writing a public model card.

## Required

- [ ] Base model repo and license are documented.
- [ ] SABER commit hash or release snapshot is recorded.
- [ ] Run config is saved with extraction method, selected layers, direction count, alpha values, and iterations.
- [ ] Evaluation prompt count and category composition are documented.
- [ ] Generation settings are recorded.
- [ ] Refusal evaluation result is saved.
- [ ] KLD or comparable drift metric is saved.
- [ ] Retained refusal categories are reviewed and summarized.
- [ ] Qualitative samples are reviewed.
- [ ] Model card states limitations and dual-use context.

## Recommended

- [ ] Compare at least three candidates on a refusal/KLD frontier.
- [ ] Prefer the lowest-drift candidate that meets the desired behavior profile.
- [ ] Re-run a small harmless/capability sample after quantization.
- [ ] Verify uploaded files and card metadata on Hugging Face.
- [ ] Keep large artifacts out of git.

## Do Not Claim

- Do not claim general safety from a refusal percentage alone.
- Do not claim zero-risk behavior from keyword checks.
- Do not report a candidate without the prompt count and evaluation setup.
- Do not imply the method preserves every capability without evidence.
