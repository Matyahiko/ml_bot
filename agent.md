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

## Dependency Management
- When adding or changing dependencies, explicitly explain the reason.
- For new library introductions, justify the necessity in one sentence.
- Do not translate code identifiers, API names, CLI flags, errors, or stack traces.

## Testing Approach
- Start with basic function-level tests first.
- Use insights discovered during testing as improvement points for next steps.
- Set random seeds and print them; log key config + commit hash once at start.

## Knowledge Gap Identification
- When unclear specifications or missing concepts are detected, explicitly point out and suggest:
  - Potential oversights in research design
  - Standard practices in the relevant field that may have been missed
  - Conceptual gaps that could affect reproducibility or validity
- Provide brief explanations with references to established methods or literature when a