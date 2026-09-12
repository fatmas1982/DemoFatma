# Semantic Coherence Failure Recovery in Small Language Models

This repository accompanies the manuscript **"From Reliable Human Annotation to Computational Recovery of Semantic Coherence Failures in Small Language Models"**.

It contains the executable analysis notebooks, model-output prediction files that can be shared publicly, aggregate result tables, reproducibility documentation, and frozen-input hashes used in the study.

## Scope

The study evaluates four hierarchical computational targets:

1. **T1 — Failure detection:** `failure` vs `no_failure`
2. **T2 — Grounding involvement:** `semantic_only` vs `grounding_involved`
3. **T3 — Mechanism family:** `Core_SCF` vs `OVR`
4. **T4 — Non-zero severity:** severity `1/2/3`

The computational comparison includes majority, lexical, latent-semantic, supervised contextual (ModernBERT), and target-isolated zero-shot open-weight LLM judging (Qwen3-8B).

## Repository structure

- `code/01_verify_completed_experiments.ipynb` — frozen baseline/result verification.
- `code/02_m3_modernbert_semantic_audit.ipynb` — ModernBERT, semantic-leakage sensitivity audit, stability, LOTO, and diagnostics.
- `code/03_m4_qwen3_forced_choice.ipynb` — final target-isolated forced-choice Qwen3-8B comparator.
- `code/04_followup_source_uncertainty.ipynb` — task-conditioned source utility and human–machine uncertainty analyses.
- `code/followup_source_uncertainty.py` — standalone version of the follow-up analyses.
- `code/recompute_public_results.py` — integrity checks and recomputation of public M3/M4 metrics, paired contrasts, contract compliance, and McNemar tests.
- `data/summary/` — aggregate tables reported or used in the manuscript/supplement.
- `data/oof/` — public case-ID-level OOF predictions without prompt/source/model-output text.
- `data/RESTRICTED_INPUT_HASHES.txt` — SHA-256 hashes of the two frozen restricted input workbooks.
- `DATA_AVAILABILITY.md` — exact public/restricted boundary.
- `REPRODUCIBILITY.md` — execution order and frozen seeds/revisions.

## Data-availability boundary

The public repository **does not contain** case-level prompt text, source/context text, generated response text, original annotator workbooks, or adjudication records. These materials are restricted by institutional and model-output redistribution constraints and are not claimed to be publicly redistributable.

The public release instead provides case IDs, prompt-cluster IDs, task labels, folds, gold target IDs, held-out predictions, model probabilities where permitted, aggregate result tables, exact analysis/model code, frozen hashes, model revisions, seeds, and protocol documentation.

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

## Frozen computational settings

- Primary grouped CV seed: `20260908`
- M3 stability seeds: `20260909`, `20260910`
- Prompt-cluster bootstrap: `2000` replicates, seed `20260908`
- Semantic-audit encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- M3: `answerdotai/ModernBERT-base`, revision `c0e44438c79d5a72972fe8b14dbbc822418356c9`
- M4: `Qwen/Qwen3-8B`, revision `e8bbd8252970581ea5b08b6a5b3e668adaf3161a`

## Reproducibility note

The public OOF files are sufficient to reproduce the reported **evaluation-layer** metrics and paired comparisons without re-running the GPU models. Full end-to-end retraining/inference additionally requires the restricted frozen input workbooks whose hashes are supplied in `data/RESTRICTED_INPUT_HASHES.txt`.

## Citation

If you use this repository, please cite the associated manuscript. A machine-readable citation template is provided in `CITATION.cff` and should be updated with the final journal bibliographic details after publication.

## Repository URL

https://github.com/fatmas1982/DemoFatma
