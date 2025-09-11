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
# filepath: /home/sayar99/scp-ed/arc-state/state/scripts/robin/infer_n.sh

