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




#what to infer? 

#jurkat (does it really matter?)
#replogle gwps (covers a good bit ) 

#most importantly: 
#arc h1 300 -> for heldout data + other data
#how do we infer only for a part of the dataset? 

#then directly afterwards run cell-eval (but how to get agg results)? 

#is state 

#!/bin/bash

# Run the inference command
# # NOTE: model_dir looks for a ./checkpoints/final.ckpt to infer on
#split for the held out data
model_dir=results/esm2-3-en-and-decoder-layers
pred_folder=$model_dir/predictions

LAST_CKPT=4000
START=400
STEP_SIZE=400

mkdir -p $pred_folder

for ckpt in $(seq $START $STEP_SIZE $LAST_CKPT); do
  state tx infer \
    --model_dir "$model_dir" \
    --adata datasets/base_dataset/control_template/competition_train.h5ad \
    --output $pred_folder/prediction_val_$ckpt.h5ad \
    --pert_col "target_gene" \
    --checkpoint $ckpt &
done

wait



#split the inferred data based on this and ..
[fewshot."replogle_h1.rpe1"]
val = [ "ACAT2", "ARID1A", "ATP6V0C", "CLDN6", "CLDN7", "ETV4", "HDAC8", "HTATSF1", "IDE", "INSIG1", "JAZF1", "JMJD8", "KDM2B", "KLF10", "LZTR1", "MAP3K7", "MAT2A", "MED1", "METTL14", "NDUFB4", "PHF14", "RNF20", "SLC25A3", "SMAGP", "SMARCB1", "SSBP1", "TCF3", "USF2", "WAC", "ZNF714",]
test = [ "ACLY", "BRD9", "C1QBP", "CAST", "CENPB", "CENPO", "COX4I1", "DHCR24", "GNG12", "IKBKG", "KAT2A", "KDM1A", "KLHDC2", "MAPKAPK2", "MED12", "MED13", "METTL3", "MGST1", "MTFR1", "NIT1", "OXA1L", "PMS1", "PRDM14", "RAF1", "SALL4", "SMARCA4", "TARBP2", "TET1", "UQCRB", "ZNF581",]

[fewshot."replogle_h1.competition_train"]
val = [ "ACAT2", "ARID1A", "ATP6V0C", "CLDN6", "CLDN7", "ETV4", "HDAC8", "HTATSF1", "IDE", "INSIG1", "JAZF1", "JMJD8", "KDM2B", "KLF10", "LZTR1", "MAP3K7", "MAT2A", "MED1", "METTL14", "NDUFB4", "PHF14", "RNF20", "SLC25A3", "SMAGP", "SMARCB1", "SSBP1", "TCF3", "USF2", "WAC", "ZNF714",]
test = [ "ACLY", "BRD9", "C1QBP", "CAST", "CENPB", "CENPO", "COX4I1", "DHCR24", "GNG12", "IKBKG", "KAT2A", "KDM1A", "KLHDC2", "MAPKAPK2", "MED12", "MED13", "METTL3", "MGST1", "MTFR1", "NIT1", "OXA1L", "PMS1", "PRDM14", "RAF1", "SALL4", "SMARCA4", "TARBP2", "TET1", "UQCRB", "ZNF581",]



uv run -m cell_eval run \
    -ap competition/dyno005/prediction_val.h5ad \
    -ar competition/dyno005/holdout_ground_truth_val.h5ad \
    -o competition/dyno005/results \
    --pert-col target_gene \
    --control-pert non-targeting \
    --profile vcc \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads 8 \
    --batch-size 16

