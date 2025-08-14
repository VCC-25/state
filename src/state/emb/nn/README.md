# StateEmbeddingModel - Enhanced with Graph Embeddings

## Overview

The `StateEmbeddingModel` is a PyTorch Lightning module designed for the Virtual Cell Challenge that predicts cellular responses to genetic and chemical perturbations. This model combines **ESM (Evolutionary Scale Modeling) protein embeddings** with **graph-based embeddings** from multiple biological networks to create rich, context-aware representations of genes and cellular states.

## Key Features

### 🧬 Multi-Modal Gene Representations
- **ESM Protein Embeddings**: Leverages evolutionary information from protein sequences
- **Graph Embeddings**: Incorporates biological network knowledge from:
  - STRING Database (protein-protein interactions)
  - Gene Ontology (GO) networks
  - Reactome pathways
  - SCGPT-derived experimental graphs

### 🏗️ Architecture Components
- **Transformer Encoder**: Processes gene sequences with attention mechanisms
- **Flash Attention Support**: Optional high-performance attention implementation
- **Skip Blocks**: Residual connections for better gradient flow
- **Positional Encoding**: Captures sequential information in gene representations
- **Dynamic Binary Decoder**: Adapts to varying input dimensions

### 🎯 Task-Specific Features
- **Context Generalization**: Predicts effects in held-out cell types (H1 embryonic stem cells)
- **Dataset Correction**: Handles batch effects across multiple datasets
- **Perturbation Ranking**: Evaluates model predictions against experimental data

## Configuration

### Graph Embeddings Configuration

```python
# Enable graph embeddings in your config
cfg.model.use_graph_embeddings = True
cfg.model.graph_dim = 64  # Dimension of graph embeddings

# Configure multiple graph sources
cfg.model.graph_config = {
    "experimental_graph": {
        "type": "scgpt_derived",
        "args": {"mode": "top_5"}
    },
    "string_db": {
        "type": "string",
        "args": {"confidence": 0.7}
    },
    "gene_ontology": {
        "type": "go",
        "args": {"namespace": "biological_process"}
    },
    "reactome": {
        "type": "reactome",
        "args": {"species": "homo_sapiens"}
    }
}
```

### Model Parameters

```python
model = StateEmbeddingModel(
    token_dim=5120,          # ESM embedding dimension
    d_model=512,             # Transformer hidden dimension
    nhead=8,                 # Number of attention heads
    d_hid=2048,             # Feed-forward dimension
    nlayers=6,              # Number of transformer layers
    output_dim=128,         # Output prediction dimension
    dropout=0.1,            # Dropout rate
    use_graph_embeddings=True,  # Enable graph integration
    graph_dim=64,           # Graph embedding dimension
    compiled=True           # Use torch.compile for optimization
)
```

## Core Functionality

### 1. Gene Embedding Integration

The model creates rich gene representations by combining:

```python
def get_gene_embedding(self, genes):
    # Get ESM protein embeddings
    protein_embeds = self.load_protein_embeddings(genes)
    
    if self.use_graph_embeddings:
        # Get graph embeddings from multiple sources
        graph_embeds = self.get_graph_embeddings(genes)
        # Combine ESM + Graph embeddings
        combined_embeds = torch.cat([protein_embeds, graph_embeds], dim=-1)
        return self.gene_embedding_layer(combined_embeds)
    else:
        return self.gene_embedding_layer(protein_embeds)
```

### 2. Multi-Source Graph Embeddings

The model integrates knowledge from multiple biological databases:

- **STRING Database**: Protein-protein interaction networks
- **Gene Ontology**: Functional annotation networks  
- **Reactome**: Metabolic and signaling pathways
- **Experimental Graphs**: Data-driven gene relationships

### 3. Forward Pass with Enhanced Embeddings

```python
def forward(self, src, mask, gene_names=None):
    if self.use_graph_embeddings and gene_names:
        # Create ESM + Graph combined embeddings
        combined_embeds = self.get_gene_embedding(gene_names)
        src = combined_embeds.expand(batch_size, -1, -1)
    
    # Apply transformer encoder
    output = self.transformer_encoder(src)
    
    # Extract representations and make predictions
    embedding = output[:, 0, :]  # CLS token
    predictions = self.decoder(embedding)
    
    return predictions, embedding, dataset_embedding
```

## Training Process

### Loss Functions
- **Cross-Entropy**: For binary classification tasks
- **MSE**: For regression tasks
- **Wasserstein**: For distribution matching
- **MMD**: Maximum Mean Discrepancy
- **Graph Consistency Loss**: Ensures embeddings respect biological relationships

### Validation Metrics
- **Differential Expression**: Overlap with experimentally validated gene rankings
- **Perturbation Correlation**: Pearson correlation of predicted vs. actual responses
- **Cross-Cell-Type Generalization**: Performance on held-out cell types

## Usage Example

```python
import lightning as L
from model import StateEmbeddingModel

# Initialize model with graph embeddings
model = StateEmbeddingModel(
    token_dim=5120,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    cfg=config,
    use_graph_embeddings=True
)

# Set up trainer
trainer = L.Trainer(
    max_epochs=100,
    devices=1,
    accelerator="gpu"
)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)

# Make predictions
predictions = model.predict_perturbation_effects(test_data)
```

## File Structure Requirements

```
project/
├── state/
│   ├── graphs/
│   │   ├── string/
│   │   │   ├── string_gene_names.npy
│   │   │   └── v11.5.parquet
│   │   ├── go/
│   │   │   ├── go_gene_names.npy
│   │   │   └── go_graph_adjacency_matrix.npy
│   │   └── reactome/
│   │       ├── reactome_gene_names.npy
│   │       └── reactome_graph_adjacency_matrix.npy
│   └── embeddings/
│       └── protein_embeddings.pt
└── model.py
```

## Key Improvements

### 🚀 Performance Enhancements
- **Flash Attention**: Optional high-performance attention implementation
- **Torch Compile**: JIT compilation for faster inference
- **Dynamic Binary Decoder**: Adapts to varying input dimensions
- **Memory Optimization**: Efficient handling of large embedding matrices

### 🔬 Biological Integration
- **Multi-Network Knowledge**: Combines protein, pathway, and experimental data
- **Graph Consistency Loss**: Ensures embeddings respect biological relationships
- **Cross-Dataset Generalization**: Handles batch effects across studies

### 📊 Evaluation Framework
- **Comprehensive Validation**: Multiple metrics for model assessment
- **Cross-Cell-Type Testing**: Evaluates generalization capabilities
- **Real-Time Monitoring**: Continuous validation during training

## Dependencies

```bash
# Core ML libraries
torch >= 2.0.0
lightning >= 2.0.0
transformers
flash-attn  # Optional for Flash Attention

# Scientific computing
numpy
pandas
scanpy
scipy

# Biological data
string-db
goatools
reactome2py
```

## Virtual Cell Challenge Context

This model is specifically designed for the 2025 Virtual Cell Challenge, focusing on:

- **Context Generalization**: Predicting effects in H1 human embryonic stem cells
- **Perturbation Modeling**: Understanding genetic and chemical interventions
- **Cross-Dataset Learning**: Leveraging multiple experimental datasets
- **Biological Interpretability**: Incorporating known biological networks

The enhanced graph embedding functionality provides the model with rich biological context, improving its ability to generalize across cell types and predict novel perturbation effects.

## Contributing

When extending this model:

1. **Graph Sources**: Add new biological networks in `_get_single_graph_embeddings()`
2. **Embedding Types**: Extend `get_gene_embedding()` for new embedding modalities
3. **Loss Functions**: Add domain-specific losses in `shared_step()`
4. **Validation**: Implement new metrics in validation methods


# Building AI Models to Predict Cellular Responses: A Deep Dive into Graph-Enhanced Transformers

*How we're combining protein language models with biological networks to predict how cells respond to genetic and chemical perturbations*

## The Challenge: Understanding Cellular Responses

Imagine you're a biologist trying to understand what happens when you knock out a specific gene in a cell, or when you treat cells with a new drug. Will the cell survive? Will it change its behavior? Which other genes will be affected? These questions are fundamental to biology and medicine, but answering them experimentally for every possible perturbation is practically impossible.

This is where the **Virtual Cell Challenge** comes in - a competition to build AI models that can predict how cells respond to genetic and chemical perturbations. Our approach combines cutting-edge machine learning with deep biological knowledge to create models that don't just memorize experimental data, but actually understand the underlying biology.

## The Big Idea: Cells as Language, Biology as Grammar

At the heart of our approach is a powerful insight: **we can treat cellular states like sentences in a language**. Each gene is like a word, and the expression levels of all genes together form a "sentence" that describes the cell's current state. When we perturb a cell (by knocking out a gene or adding a drug), we're essentially asking: "How does this sentence change?"

But here's the key innovation: just like human language has grammar rules, biology has its own "grammar" - the networks of protein interactions, metabolic pathways, and regulatory relationships that govern how genes affect each other. Our model learns both the "language" of cellular states and the "grammar" of biological networks.

## The Architecture: A Tale of Two Embeddings

### ESM Embeddings: The Protein Language Model

Our foundation is **ESM (Evolutionary Scale Modeling)**, a transformer model trained on millions of protein sequences. Think of ESM as "BERT for proteins" - it has learned to understand the "language" of protein sequences by studying evolutionary patterns across all of life.

```python
# ESM gives us rich protein representations
protein_embeds = self.load_protein_embeddings(gene_names)
# Each gene gets a 5120-dimensional vector encoding evolutionary knowledge
```

ESM embeddings capture incredible biological insight. For example, proteins that fold into similar structures get similar embeddings, even if their sequences look different. This evolutionary knowledge is crucial because protein structure determines function.

### Graph Embeddings: The Network Knowledge

But protein sequences alone aren't enough. Genes don't act in isolation - they form complex networks of interactions. This is where our **graph embeddings** come in. We integrate knowledge from multiple biological databases:

- **STRING Database**: Who talks to whom? (protein-protein interactions)
- **Gene Ontology**: What does each gene do? (functional annotations)
- **Reactome**: How do pathways work? (metabolic and signaling networks)
- **Experimental Graphs**: What do the data tell us? (co-expression patterns)

```python
def get_graph_embeddings(self, genes):
    """Combine knowledge from multiple biological networks"""
    all_graph_embeddings = []
    
    # Get embeddings from each network type
    string_embeds = self._get_string_embeddings(genes)      # Interactions
    go_embeds = self._get_go_embeddings(genes)             # Functions  
    reactome_embeds = self._get_reactome_embeddings(genes) # Pathways
    
    # Combine into unified graph representation
    combined = torch.stack([string_embeds, go_embeds, reactome_embeds]).mean(dim=0)
    return combined
```

### The Fusion: ESM + Graphs = Biological Understanding

The magic happens when we combine these two types of knowledge:

```python
def get_gene_embedding(self, genes):
    # Get evolutionary knowledge from protein sequences
    protein_embeds = self.load_protein_embeddings(genes)
    
    if self.use_graph_embeddings:
        # Get network knowledge from biological databases
        graph_embeds = self.get_graph_embeddings(genes)
        
        # Combine both types of knowledge
        combined_embeds = torch.cat([protein_embeds, graph_embeds], dim=-1)
        return self.gene_embedding_layer(combined_embeds)
```

This fusion gives our model a much richer understanding of each gene. It knows not just what the protein looks like (from ESM), but also how it fits into the broader biological network (from graphs).

## The Transformer: Learning Cellular Grammar

With our enhanced gene representations, we feed everything into a **transformer encoder** - the same architecture that powers ChatGPT, but adapted for biology:

```python
def forward(self, gene_sequences, gene_names=None):
    # Create rich gene embeddings (ESM + Graph)
    if self.use_graph_embeddings and gene_names:
        combined_embeds = self.get_gene_embedding(gene_names)
        src = combined_embeds.expand(batch_size, -1, -1)
    
    # Let the transformer learn cellular grammar
    output = self.transformer_encoder(src)
    
    # Extract the cell state representation
    cell_embedding = output[:, 0, :]  # CLS token
    
    return cell_embedding
```

The transformer learns to understand how genes interact within a cell. The attention mechanism allows it to focus on the most relevant gene-gene relationships for each prediction task.

## The Training: Learning from Perturbation Data

We train our model on large datasets of cellular perturbation experiments. Each training example shows the model:

- **Before**: The baseline gene expression in control cells
- **Perturbation**: What we did (knocked out gene X, added drug Y)  
- **After**: How gene expression changed in response

The model learns to predict the "after" state given the "before" state and perturbation information.

### Multiple Loss Functions for Different Aspects

We use several loss functions to capture different aspects of cellular responses:

```python
# Different ways to measure prediction quality
if self.cfg.loss.name == "cross_entropy":
    criterion = BCEWithLogitsLoss()  # Binary gene up/down
elif self.cfg.loss.name == "wasserstein":
    criterion = WassersteinLoss()    # Distribution matching
elif self.cfg.loss.name == "mmd":
    criterion = MMDLoss()            # Statistical similarity

# Plus our graph consistency loss
graph_loss = self.compute_graph_consistency_loss(embeddings, batch)
total_loss = prediction_loss + graph_loss
```

The **graph consistency loss** is particularly important - it ensures that genes that are close in biological networks also have similar learned representations.

## The Challenge: Generalization Across Cell Types

Here's where things get really interesting. The Virtual Cell Challenge asks us to predict perturbation effects in **H1 human embryonic stem cells** - a cell type the model has never seen during training. This tests whether our model has learned general principles of cellular biology, not just memorized specific cell types.

Our approach tackles this through:

### 1. Rich Biological Priors
By incorporating graph embeddings, we give the model strong biological priors about how genes should behave based on known networks. This helps it generalize to new contexts.

### 2. Dataset Correction
We include special "dataset tokens" that help the model learn to separate dataset-specific batch effects from true biological signals:

```python
if self.dataset_token is not None:
    dataset_token = self.dataset_token.expand(batch_size, -1).unsqueeze(1)
    gene_sequence = torch.cat((gene_sequence, dataset_token), dim=1)
```

### 3. Cross-Cell-Type Validation
During training, we continuously test the model's ability to predict perturbation effects in held-out cell types:

```python
def _compute_val_perturbation(self, current_step):
    # For each cell type, train on others and test on this one
    for holdout_cell_type in adata.obs["cell_type"].unique():
        train_data = adata[adata.obs["cell_type"] != holdout_cell_type]
        test_data = adata[adata.obs["cell_type"] == holdout_cell_type]
        
        # Measure how well we generalize
        correlation = compute_pearson_delta(predicted, actual)
        self.log("validation/perturbation_correlation", correlation)
```

## Performance Optimizations: Making It Fast

Training these models requires serious computational power. We've included several optimizations:

### Flash Attention
```python
if use_flash and FlashTransformerEncoderLayer is not None:
    print("!!! Using Flash Attention !!!")
    layers = [FlashTransformerEncoderLayer(...) for _ in range(nlayers)]
    self.transformer_encoder = FlashTransformerEncoder(layers)
```

Flash Attention reduces memory usage and speeds up training significantly for long sequences.

### Torch Compile
```python
if compiled:
    self.transformer_encoder = torch.compile(self.transformer_encoder)
    self.decoder = torch.compile(self.decoder)
```

PyTorch's JIT compilation can provide substantial speedups.

### Dynamic Architecture
```python
# Create binary decoder dynamically based on actual input size
if self.binary_decoder is None or self.binary_decoder_input_size != actual_dim:
    self.binary_decoder = nn.Sequential(
        SkipBlock(actual_dim),
        SkipBlock(actual_dim), 
        nn.Linear(actual_dim, 1, bias=True),
    ).to(self.device)
```

The model adapts its architecture based on the actual dimensions it encounters, making it flexible and robust.

## Evaluation: How Do We Know It Works?

We evaluate our model on multiple biological metrics:

### 1. Differential Expression Overlap
Does the model predict the same genes as experimentally important that biologists find through traditional differential expression analysis?

```python
# Compare top-k genes from model vs. experimental ranking
de_metrics = compute_gene_overlap_cross_pert(
    predicted_rankings, true_rankings, k=top_k
)
```

### 2. Perturbation Correlation
How well does the model predict the magnitude and direction of gene expression changes?

```python
correlation = compute_pearson_delta(predicted_effects, real_effects)
```

### 3. Cross-Cell-Type Generalization
This is the big test - can the model predict effects in cell types it's never seen?

## Why This Matters: From Research to Medicine

The ability to predict cellular responses has enormous implications:

- **Drug Discovery**: Test thousands of drug candidates computationally before expensive lab experiments
- **Personalized Medicine**: Predict how a patient's specific genetic background will respond to treatments
- **Understanding Disease**: Model how disease-causing mutations affect cellular function
- **Bioengineering**: Design cells with desired properties for manufacturing or therapy

## The Future: What's Next?

This work represents just the beginning. Future directions include:

- **Larger Models**: Scaling up to handle entire genomes and complex multi-cellular systems
- **Temporal Dynamics**: Predicting how cellular responses unfold over time
- **Multi-Modal Integration**: Combining gene expression with protein levels, metabolite concentrations, and imaging data
- **Causal Discovery**: Moving beyond correlation to understand causal mechanisms

## Conclusion: Biology Meets AI

Our StateEmbeddingModel represents a new paradigm in computational biology - one that combines the pattern recognition power of large language models with the structured knowledge of biological networks. By treating cells as language and biology as grammar, we're building AI systems that don't just predict, but understand.

The Virtual Cell Challenge is pushing the boundaries of what's possible when we bring together cutting-edge AI and deep biological knowledge. As these models improve, they'll become increasingly powerful tools for understanding life itself.

*The future of biology is computational, and the future of AI is biological.*

---

## Technical Deep Dive: Key Code Insights

For the technically inclined, here are some key implementation details:

### Dynamic Dimension Handling
```python
# The model adapts to varying input dimensions
expected_dim = self.output_dim + self.d_model + self.z_dim
actual_dim = combine.shape[-1]

if self.binary_decoder is None or self.binary_decoder_input_size != actual_dim:
    self.binary_decoder = nn.Sequential(
        SkipBlock(actual_dim),
        SkipBlock(actual_dim),
        nn.Linear(actual_dim, 1, bias=True),
    ).to(self.device)
```

### Multi-Source Graph Integration
```python
def _get_single_graph_embeddings(self, genes, graph_type, graph_args):
    """Get embeddings for a single graph type."""
    if graph_type == "string":
        return self._compute_string_graph_embedding(gene, graph_args)
    elif graph_type == "go":
        return self._compute_go_graph_embedding(gene, graph_args)
    elif graph_type == "reactome":
        return self._compute_reactome_graph_embedding(gene, graph_args)
    # ... more graph types
```

### Sophisticated Learning Rate Scheduling
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
    
    # Warmup + Cosine annealing
    lr_schedulers = [
        LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, eta_min=max_lr * 0.3, T_max=total_steps)
    ]
    
    return ChainedScheduler(lr_schedulers)
```

This combination of biological insight and technical sophistication is what makes modern computational biology so exciting!