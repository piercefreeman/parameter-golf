#!/usr/bin/env bash
set -euo pipefail

# 1xH100 mini run: keep the baseline model/training defaults, but scale the
# global token budget to 1/8 of the 8xH100 default.
export RUN_ID="${RUN_ID:-mini_1xh100}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

exec uv run torchrun --standalone --nproc_per_node=1 train_gpt.py
