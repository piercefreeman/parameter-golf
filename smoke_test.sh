#! /bin/bash
RUN_ID=mlx_smoke \
    ITERATIONS=200 \
    TRAIN_BATCH_TOKENS=8192 \
    VAL_LOSS_EVERY=0 \
    VAL_BATCH_SIZE=8192 \
    uv run train_gpt_mlx.py
