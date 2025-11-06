# AGENT.md

## Research Mode (scope & constraints)

- This repository is for **research/prototyping**, not production.
- Optimize for **clarity, iteration speed, and reproducibility**, not architecture.
- Keep changes **simple and minimal**: avoid heavy abstractions, DI frameworks, plug-in systems, or multi-layer class hierarchies.
- Only extract a function if it's reused (≥2 places) or improves clarity significantly.
- Prefer explicit code to "clever" patterns; readability over concision.

## Language & Output

- Perform internal reasoning in **English**.
- **All user-visible output must be in Japanese** (explanations, code comments, PR text).
- Do not include step-by-step reasoning in the final output; if needed, add up to 3 concise Japanese bullets.

## Reproducibility

- Provide a one-command run path and pin dependencies.
- Set random seeds and print them; log key config + commit hash once at start.
- Add a **smoke test** (e.g., runs one training/eval step) rather than full unit test coverage.

## Deliverables Style

- When proposing edits, output a diff/patch and a short Japanese rationale.
- Do not introduce new top-level dependencies unless necessary; justify in one sentence.
- Do not translate code identifiers, API names, CLI flags, errors, or stack traces.