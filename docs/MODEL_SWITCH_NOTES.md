# Model Switch Notes

Notes for the switch from `gpt-4.1` to `gpt-5-mini` in `src/multi_lingual_qac/qac_generation/openai_qa.py`.

## Previous `gpt-4.1` Setup

- `DEFAULT_GENERATION_MODEL = "gpt-4.1"`
- `DEFAULT_QUALITY_MODEL = "gpt-4.1"`
- `DEFAULT_SUPPORT_MODEL = "gpt-4.1"`
- `DEFAULT_TRANSLATION_MODEL = "gpt-4.1"`
- Explicit temperatures were used:
  - English generation: `temperature=0.3`
  - English checks and quality checks: `temperature=0`
  - Translation: `temperature=0.2`

## Current `gpt-5-mini` Setup

- `DEFAULT_GENERATION_MODEL = "gpt-5-mini"`
- `DEFAULT_QUALITY_MODEL = "gpt-5-mini"`
- `DEFAULT_SUPPORT_MODEL = "gpt-5-mini"`
- `DEFAULT_TRANSLATION_MODEL = "gpt-5-mini"`
- Explicit `temperature` arguments were removed because `gpt-5-mini` does not support the same temperature usage as the `gpt-4.1` setup.
- Reasoning effort was added:
  - English generation: `DEFAULT_GENERATION_REASONING_EFFORT = "medium"`
  - Checks and translation: `DEFAULT_REASONING_EFFORT = "low"`

## Wikidata qrels relevance judge (`gpt-5-nano`)

Separate from the Q&A stack above:

- **`src/multi_lingual_qac/qac_generation/label_wikidata_qrels.py`** uses **`DEFAULT_QRELS_JUDGE_MODEL = "gpt-5-nano"`** for batched “which passages answer this question?” calls when labeling `queries.csv` / `qrels.csv`.
- Changing Q&A defaults in `openai_qa.py` does **not** change this judge unless you edit `label_wikidata_qrels.py` (or add a CLI override later).

## Prompt Changes Added For `gpt-5-mini`

- Stronger instruction to prefer semantic questions over easy extractive lookups.
- Explicit rejection of broad `purpose` / `advantages` wording when a narrower technical question exists.
- Explicit rejection of spec-sheet questions when a better effect / rationale / mechanism question is available.
- Added bad examples based on observed `gpt-5-mini` failures.

## Observed Quality Trend

- First `gpt-5-mini` run: cleaner but too literal and too lookup-oriented.
- Later prompt-only revisions: improved question shape and reduced broad fallbacks.
- Current state: best `gpt-5-mini` run so far, but still slightly below the best `gpt-4.1` run for pure question quality.

## Quick Revert Checklist

If we want to go back to the old `gpt-4.1` configuration:

1. In `src/multi_lingual_qac/qac_generation/openai_qa.py`, set all four default model constants back to `gpt-4.1`.
2. Remove or stop using `DEFAULT_GENERATION_REASONING_EFFORT`.
3. Restore explicit temperatures:
   - generation: `temperature=0.3`
   - English/support/quality checks: `temperature=0`
   - translation: `temperature=0.2`
4. Keep or adjust the newer prompt wording depending on whether we want `gpt-4.1` with the old prompts or `gpt-4.1` with the newer anti-literal prompt updates.

## Important Note

Going back to `gpt-4.1` is not only a model swap if we want a true apples-to-apples rollback. We also need to decide whether to keep the newer `gpt-5-mini` prompt improvements or revert those too.
