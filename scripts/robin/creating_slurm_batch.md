Just go through this:
#!/bin/bash
#SBATCH --job-name=state_infer
#SBATCH --partition=main  # or whatever partition is available
#SBATCH --qos=standard        # or whatever QoS is available
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=infer_%j.out
#SBATCH --error=infer_%j.err
#filepath: /home/sayar99/scp-ed/arc-state/state/scripts/robin/infer_n.sh

# What to do
if unsure, do screen, run 
    ```
    #for dryrun
    salloc \
        --job-name=dry_run \
        --partition=scavenger,gpu \
        --qos=standard \
        --mem=1G \
        --time=00:03:00 \
        --cpus-per-task=1
    ```
    followed by your bash (without any SBATCH)

    after that repeat with correct salloc details:

    ```
    salloc \
        --gres=gpu:a5000:4 \
        --job-name=pipe_dan_train \
        --partition=scavenger,gpu \
        --qos=standard \
        --mem=90G \
        --time=10:00:00 \
        --cpus-per-task=8
    ```