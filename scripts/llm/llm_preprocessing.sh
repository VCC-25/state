
# Create output directory if it doesn't exist
mkdir -p ../datasets/base_dataset/hvgs

# Preprocess each dataset individually
state tx preprocess_train \
  --adata ../datasets/base_datasetn_train.h5 \
  --output ../datasets/base_datasetcompetition_train.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/base_dataset
  --output ../datasets/base_dataset
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/base_dataset
  --output ../datasets/base_dataset/hvgs/preprocessed_training_data_rpe1.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/base_dataset/jurkat.h5 \
  --output ../datasets/base_dataset/hvgs/preprocessed_training_data_jurkat.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/base_dataset/k562.h5 \
  --output ../datasets/base_dataset/hvgs/preprocessed_training_data_k562.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/base_dataset/hepg2.h5 \
  --output ../datasets/base_dataset/hvgs/preprocessed_training_data_hepg2.h5ad \
  --num_hvgs 2000
