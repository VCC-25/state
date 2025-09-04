#!/bin/bash

DIR_NAME="first_run"

uv run state tx predict \
    --checkpoint "final.ckpt" \
    --output_dir "competition/${DIR_NAME}/" \
    --profile full

