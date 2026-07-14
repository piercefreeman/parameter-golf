# autoresearch

This file defines the operating rules for an autonomous LoRA-focused research loop in this repository.
It is modeled after Andrej Karpathy's `autoresearch/program.md`, but adapted to Parameter Golf, the current Runpod setup, and this repo's training scripts.

Source inspiration: https://github.com/karpathy/autoresearch/blob/master/program.md

## Goal

Drive down validation `val_bpb` as far as possible on the Parameter Golf setup.
The main objective is the best reproducible final `final_int8_zlib_roundtrip_exact val_bpb`.

This is a LoRA-centric search, not a generic random sweep.
The agent should reason over previous runs, propose the next experiment, execute it, and keep only changes that genuinely help.

## Scope

For this research loop:

- Primary CUDA execution target: `train_gpt.py`
- Primary idea family: LoRA-based variants and their interactions with model shape and optimization
- Read for context before starting: `README.md`, `AGENTS.md`, `train_gpt.py`, and `train_gpt_mlx.py`
- Do not change dataset export or validation definitions while running the loop
- Do not add dependencies
- Keep the code understandable; tiny wins are not worth ugly hacks

Repo-specific note:

- Final submission prep should still respect the repo instruction to update `train_gpt_mlx.py`
- During CUDA-side research on Runpod, it is acceptable to iterate in `train_gpt.py` first, then port winning ideas later

## Hard Constraints

- Per-run training budget is 10 minutes of wall-clock training time
- The artifact target is 16,000,000 bytes total for code plus compressed model
- The code itself should stay below 1,500 lines
- The metric to optimize is validation `val_bpb`
- Do not leak validation information into training
- Avoid brute-force seed fishing; tune ideas, not lottery tickets
- If a run crashes, log it clearly and move on unless the fix is trivial

## Setup

Before starting a session:

1. Create a fresh branch: `autoresearch/<date>-lora` or similar
2. Verify the dataset and tokenizer exist locally
3. Create a run ledger at `.research/results.tsv`
4. Create a rolling session report at `.research/session_report.md`
5. Create per-run directories under `.research/runs/<run_id>/`

The TSV should be tab-separated with this header:

```tsv
commit	branch	val_bpb	val_loss	artifact_mb	status	description
```

Status must be one of:

- `keep`
- `discard`
- `crash`

Do not commit the TSV or the rolling report. They are session state, not source code.

## First Runs

The first two runs should establish calibration:

1. Run the current repo LoRA baseline as-is
2. Run one true no-LoRA baseline with `LINEAR_IMPL=linear`

After that, stay focused on the LoRA implementation unless the evidence strongly suggests LoRA is the wrong direction.

## LoRA Search Heuristics

Prioritize experiments in roughly this order:

1. LoRA structure
   - `LINEAR_IMPL`
   - `LORA_RANK`
   - `LORA_ALPHA`
   - where LoRA is applied, if new targeting knobs are added later
2. LoRA + model-shape interaction
   - `NUM_LAYERS`
   - `MODEL_DIM`
   - `MLP_MULT`
   - `NUM_KV_HEADS`
3. LoRA + optimization interaction
   - `MATRIX_LR`
   - `SCALAR_LR`
   - `TIED_EMBED_LR`
   - `MUON_MOMENTUM`
   - `MUON_BACKEND_STEPS`
   - `WARMDOWN_ITERS`
4. Sequence/batch tradeoffs
   - `TRAIN_SEQ_LEN`
   - `TRAIN_BATCH_TOKENS`
5. Cleanup or simplification
   - remove complexity that does not pay for itself

General policy:

- Change one concept at a time unless two knobs are tightly coupled
- Prefer stable, interpretable moves over chaotic multi-knob jumps
- A win under ~0.001 `val_bpb` is not worth much added complexity
- A simplification with equal metrics is a real win
- A crash from a typo or shape bug can be fixed once and re-run
- A fundamentally bad idea should be logged and abandoned quickly

## Experiment Loop

Loop forever until the session budget expires:

1. Read `.research/session_report.md` and the last few TSV rows
2. Inspect the current best kept commit and the most recent failed ideas
3. Form exactly one next hypothesis
4. Edit `train_gpt.py` for that hypothesis, or apply a tightly scoped env/config change
5. Commit the experiment before running it
6. Run the trainer with stdout and stderr redirected into a per-run log
7. Parse the final summary, especially:
   - `final_int8_zlib_roundtrip_exact val_loss`
   - `final_int8_zlib_roundtrip_exact val_bpb`
   - compressed model size
8. Append the result to `.research/results.tsv`
9. Update `.research/session_report.md`
10. If the run improved the best valid `val_bpb`, keep the commit and advance
11. If it did not improve, revert to the previous best commit and try again

## Logging Rules

Every run directory should contain at minimum:

- `run.log`
- `env.json`
- `summary.json`
- `notes.md`
- a copy of the trainer diff or commit hash

The rolling report should contain:

- current best run and commit
- best-history timeline
- recent experiments and outcomes
- active hypotheses
- recurring failure modes
- notes on what the agent believes next

Descriptions in the TSV should be short and concrete, for example:

- `baseline fixed_lora`
- `linear baseline no lora`
- `lora rank 8 alpha 8`
- `lora rank 32 with lower matrix lr`
- `deeper narrower lora model`
- `longer seq len with smaller batch`

Make sure to also log out to wandb under rdiehlmartinez and a project called parameter-golf-autoresearch. Pick a good run name for each experiment.

## Keep / Discard Policy

Keep a run if:

- it improves the best valid `val_bpb`, or
- it matches the best result with materially simpler code

Discard a run if:

- `val_bpb` is worse or flat without a simplicity win, or
- the artifact limit is broken, or
- the run is unstable or obviously brittle

Crash if:

- the script errors out
- the run times out badly
- the model OOMs
- metrics cannot be parsed reliably

## Timeouts

A single run should finish within the expected training budget plus modest startup/eval overhead.
If a run drifts far past that window, kill it, record `crash`, and move on.

## Agentic Planner Policy

The planner should not merely sweep static presets.
After each run it should:

1. Read the latest log and summary
2. Compare the run to the current best
3. Explain why the result likely happened
4. Propose the next experiment as a response to the evidence
5. Launch the next run only after writing that reasoning into the session report

The planner is allowed to be bold, but it must stay legible.
The ideal loop is autonomous, empirical, and cumulative.
