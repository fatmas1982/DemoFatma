# Reproducibility Guide

## Public evaluation-layer reproduction

1. Use the public OOF files under `data/oof/`.
2. Run `code/recompute_public_results.py`.
3. Compare recomputed outputs with `data/summary/`.

This path does not require access to prompt/source/output text and does not retrain M3 or rerun M4.

## Full authorised reproduction

Full model reruns require the two frozen workbooks whose hashes are listed in `data/RESTRICTED_INPUT_HASHES.txt`.

Recommended execution order:

1. `code/01_verify_completed_experiments.ipynb`
2. `code/02_m3_modernbert_semantic_audit.ipynb`
3. `code/03_m4_qwen3_forced_choice.ipynb`
4. `code/04_followup_source_uncertainty.ipynb`

## Frozen constants

- Grouped CV seed: 20260908
- Bootstrap seed: 20260908
- Bootstrap replicates: 2000
- M3 stability seeds: 20260909, 20260910
- Prompt clusters remain grouped across all six SLM outputs.
- Legacy free-generation M4 outputs are diagnostic only and must not be used as final performance estimates.
