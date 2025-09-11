#!/bin/bash

# Run the evaluation command
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep \
    -i "competition/dyno005/prediction.h5ad" \
    -g "competition_support_set/gene_names.csv"
