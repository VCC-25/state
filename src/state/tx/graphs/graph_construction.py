"""
Graph construction module for STATE perturbation encoding.
Adds graph-based perturbation representations.
"""

import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class StateGraphBuilder:
    """Graph builder for STATE perturbation encoding."""
    
    def __init__(self, pert2id: Dict[str, int], cache_dir: str = "./graphs"):
        """
        Initialize the graph builder.
        
        Args:
            pert2id: Mapping from perturbation names to indices
            cache_dir: Directory containing graph data files
        """
        self.pert2id = pert2id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create reverse mapping for convenience
        self.id2pert = {idx: pert for pert, idx in pert2id.items()}
        
        logger.info(f"Initialized StateGraphBuilder with {len(pert2id)} perturbations")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def create_graph(self, graph_type: str, graph_args: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Create graph based on type.
        
        Args:
            graph_type: Type of graph to create ("go", "string", "reactome", "scgpt_derived", "dense")
            graph_args: Arguments for graph creation
            
        Returns:
            Tuple of (edge_index, edge_weight, num_nodes)
        """
        graph_args = graph_args or {}
        
        if graph_type == "go":
            return self._create_go_graph(graph_args)
        elif graph_type == "string":
            return self._create_string_graph(graph_args)
        elif graph_type == "reactome":
            return self._create_reactome_graph(graph_args)
        elif graph_type == "scgpt_derived":
            return self._create_scgpt_graph(graph_args)
        elif graph_type == "dense":
            return self._create_dense_graph()
        else:
            raise ValueError(f"Unknown graph type: {graph_type}. Available types: {list(self.pert2id.keys())}")
    
    def _create_go_graph(self, graph_args: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Create Gene Ontology-based graph."""
        graph_args = graph_args or {}
        
        # Load GO relationships
        go_file = self.cache_dir / "go" / "go_top_50.csv"
        if not go_file.exists():
            logger.warning(f"GO file not found: {go_file}")
            logger.info("Creating fallback dense graph for GO")
            return self._create_dense_graph()
        
        logger.info(f"Loading GO graph from {go_file}")
        network = pd.read_csv(go_file)
        return self._process_network(network, graph_args)
    
    def _create_string_graph(self, graph_args: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Create STRING database-based graph."""
        graph_args = graph_args or {}
        
        # Try different STRING file formats
        string_files = [
            self.cache_dir / "string" / "v11.5.parquet",
            self.cache_dir / "string" / "string_v11.5.parquet",
            self.cache_dir / "string" / "protein_links.csv"
        ]
        
        string_file = None
        for file_path in string_files:
            if file_path.exists():
                string_file = file_path
                break
        
        if string_file is None:
            logger.warning(f"STRING file not found in {self.cache_dir / 'string'}")
            logger.info("Creating fallback dense graph for STRING")
            return self._create_dense_graph()
        
        logger.info(f"Loading STRING graph from {string_file}")
        if string_file.suffix == ".parquet":
            network = pd.read_parquet(string_file)
        else:
            network = pd.read_csv(string_file)
        
        return self._process_network(network, graph_args)
    
    def _create_reactome_graph(self, graph_args: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Create Reactome pathway-based graph."""
        graph_args = graph_args or {}
        
        # Load Reactome relationships
        reactome_file = self.cache_dir / "reactome" / "uniprot_pathways.txt"
        if not reactome_file.exists():
            logger.warning(f"Reactome file not found: {reactome_file}")
            logger.info("Creating fallback dense graph for Reactome")
            return self._create_dense_graph()
        
        logger.info(f"Loading Reactome graph from {reactome_file}")
        
        # Load the protein-pathway mappings (UniProt IDs to Reactome pathways)
        protein_pathway_df = pd.read_csv(reactome_file, sep='\t', header=None,
                                       names=['protein_id', 'pathway_id', 'url', 'pathway_name', 'evidence', 'species'])
        
        # Filter for human proteins only
        human_mappings = protein_pathway_df[protein_pathway_df['species'] == 'Homo sapiens'].copy()
        logger.info(f"📊 Loaded {len(human_mappings):,} human protein-pathway mappings")
        logger.info(f"📊 Unique proteins: {human_mappings['protein_id'].nunique():,}")
        logger.info(f"📊 Unique pathways: {human_mappings['pathway_id'].nunique():,}")
        
        # Create gene-gene connections based on shared pathway memberships
        gene_connections = []
        
        # Group proteins by pathway
        pathway_proteins = human_mappings.groupby('pathway_id')['protein_id'].apply(set).to_dict()
        
        # Create gene-gene edges based on shared pathways
        from collections import defaultdict
        shared_pathways = defaultdict(set)
        
        for pathway_id, proteins in pathway_proteins.items():
            if len(proteins) > 1:  # Only pathways with multiple proteins
                for protein1 in proteins:
                    for protein2 in proteins:
                        if protein1 < protein2:  # Avoid duplicates
                            shared_pathways[(protein1, protein2)].add(pathway_id)
        
        # Create gene-gene connections with weights based on shared pathways
        for (protein1, protein2), shared_pathway_set in shared_pathways.items():
            # Weight based on number of shared pathways
            weight = len(shared_pathway_set)
            
            # Map UniProt IDs to gene names (simplified mapping)
            gene1 = protein1  # Use UniProt ID as gene name for now
            gene2 = protein2
            
            gene_connections.append({
                'regulator': gene1,
                'target': gene2,
                'weight': weight
            })
        
        if gene_connections:
            # Create DataFrame from gene connections
            network = pd.DataFrame(gene_connections)
            logger.info(f"📊 Created {len(gene_connections):,} gene-gene connections from Reactome pathways")
            return self._process_network(network, graph_args)
        else:
            # Fallback to dense graph if no gene connections created
            logger.warning("No gene connections created from Reactome pathways, using dense graph")
            return self._create_dense_graph()
    
    def _create_scgpt_graph(self, graph_args: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Create scGPT-derived similarity graph."""
        graph_args = graph_args or {}
        
        # Check for pre-computed scGPT embeddings
        experimental_data_dir = self.cache_dir / "experimental_data"
        gene_embeddings_file = experimental_data_dir / "gene_embeddings.npy"
        gene_names_file = experimental_data_dir / "gene_names.npy"
        
        if gene_embeddings_file.exists() and gene_names_file.exists():
            logger.info("Loading pre-computed scGPT gene embeddings...")
            gene_embeddings = np.load(gene_embeddings_file)
            gene_names = np.load(gene_names_file, allow_pickle=True)
            
            logger.info(f"Loaded embeddings shape: {gene_embeddings.shape}")
            logger.info(f"Number of genes: {len(gene_names)}")
            
            # Create gene name to index mapping
            gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
            
            # Filter to only perturbation genes that exist in the embeddings
            available_pert_genes = [gene for gene in self.pert2id.keys() if gene in gene_name_to_idx]
            missing_genes = [gene for gene in self.pert2id.keys() if gene not in gene_name_to_idx]
            
            if missing_genes:
                logger.warning(f"Warning: {len(missing_genes)} perturbation genes not found in embeddings: {missing_genes[:5]}...")
            
            if not available_pert_genes:
                logger.warning("No perturbation genes found in the embeddings! Creating dense graph.")
                return self._create_dense_graph()
            
            logger.info(f"Using {len(available_pert_genes)} perturbation genes with embeddings")
            
            # Get embeddings for available perturbation genes
            pert_indices = [gene_name_to_idx[gene] for gene in available_pert_genes]
            pert_embeddings = gene_embeddings[pert_indices]
            
            # Compute pairwise cosine similarities
            similarities = cosine_similarity(pert_embeddings)
            
            # Take absolute values as in the paper
            abs_similarities = np.abs(similarities)
            
            # Convert to edge list format
            edges = []
            for i, gene1 in enumerate(available_pert_genes):
                for j, gene2 in enumerate(available_pert_genes):
                    if i != j:  # No self-loops
                        similarity = abs_similarities[i, j]
                        edges.append({
                            'regulator': gene1,
                            'target': gene2, 
                            'weight': similarity
                        })
            
            network = pd.DataFrame(edges)
            logger.info(f"Created scGPT graph with {len(network)} edges from {len(available_pert_genes)} genes")
            return self._process_network(network, graph_args)
        else:
            logger.warning("Pre-computed scGPT embeddings not found. Creating dense graph.")
            return self._create_dense_graph()
    
    def _create_dense_graph(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Create fully connected graph."""
        num_nodes = len(self.pert2id)
        logger.info(f"Creating dense graph with {num_nodes} nodes")
        
        # Create all possible edges (excluding self-loops)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add bidirectional edges
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
        
        logger.info(f"Dense graph created with {edge_index.size(1)} edges")
        return edge_index, edge_weight, num_nodes
    
    def _process_network(self, network: pd.DataFrame, graph_args: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Process network DataFrame into PyG format.
        
        Args:
            network: DataFrame with network data
            graph_args: Processing arguments
            
        Returns:
            Tuple of (edge_index, edge_weight, num_nodes)
        """
        graph_args = graph_args or {}
        reduce2perts = graph_args.get("reduce2perts", True)
        reduce2positive = graph_args.get("reduce2positive", False)
        norm_weights = graph_args.get("norm_weights", False)
        mode = graph_args.get("mode", None)
        
        logger.info(f"Processing network with {len(network)} edges")
        
        # Check which column naming convention is used
        if "regulator" in network.columns and "target" in network.columns:
            source_col = "regulator"
            target_col = "target"
        elif "source" in network.columns and "target" in network.columns:
            source_col = "source"
            target_col = "target"
        else:
            raise ValueError("Network must have either regulator/target or source/target columns")

        # Check and rename importance column if it exists
        if "importance" in network.columns:
            network = network.rename(columns={"importance": "weight"})

        # Rename columns to standard format
        if source_col != "regulator":
            network = network.rename(columns={source_col: "regulator", target_col: "target"})

        # Reduce to edges between perturbation genes
        if reduce2perts:
            network["regulator"] = network["regulator"].map(self.pert2id)
            network["target"] = network["target"].map(self.pert2id)
            network = network.dropna()
            num_nodes = len(self.pert2id)
            logger.info(f"Reduced to {len(network)} edges between perturbation genes")
        else:
            # For non-perturbation reduction, we'd need gene2id mapping
            # For now, we'll use pert2id as fallback
            network["regulator"] = network["regulator"].map(self.pert2id)
            network["target"] = network["target"].map(self.pert2id)
            network = network.dropna()
            num_nodes = len(self.pert2id)

        # Reduce to positive weights only
        if reduce2positive:
            network = network[network["weight"] > 0]
            network = network.reset_index(drop=True)
            logger.info(f"Reduced to {len(network)} positive edges")

        # Normalize the weights to [0,1]
        if norm_weights:
            network["weight"] = abs(network["weight"].transform(lambda x: x / x.max()))
            logger.info("Normalized weights to [0,1]")

        # Determine the mode of edge selection
        if mode is not None:
            if "_" in mode:
                mode_name, arg = mode.split("_")
                arg = int(arg)
            else:
                mode_name = mode
                arg = None

            if mode_name == "top" and arg is not None:
                network = (
                    network.groupby("target")
                    .apply(lambda x: x.nlargest(arg, ["weight"]))
                    .reset_index(drop=True)
                )
                logger.info(f"Selected top {arg} edges per target")

            elif mode_name == "threshold" and arg is not None:
                network = (
                    network.groupby("target")
                    .apply(lambda x: x[x["weight"] >= arg])
                    .reset_index(drop=True)
                )
                logger.info(f"Selected edges with weight >= {arg}")

            elif "percentile" in mode_name and arg is not None:
                if "abs" in mode_name:
                    threshold = np.percentile(network["weight"].abs(), arg)
                    network = network[network["weight"].abs() >= threshold]
                else:
                    threshold = np.percentile(network["weight"], arg)
                    network = network[network["weight"] >= threshold]
                network = network.reset_index(drop=True)
                logger.info(f"Selected edges in {arg}th percentile")

        # Package the graph into PyG format
        edge_index = torch.tensor(
            [network["regulator"].to_numpy(), network["target"].to_numpy()],
            dtype=torch.long,
        )
        edge_weight = torch.tensor(network["weight"].to_numpy(), dtype=torch.float)
        
        logger.info(f"Final graph: {edge_index.size(1)} edges, {num_nodes} nodes")
        return edge_index, edge_weight, num_nodes
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about available graphs and perturbation mapping."""
        return {
            "num_perturbations": len(self.pert2id),
            "perturbation_mapping": self.pert2id,
            "cache_directory": str(self.cache_dir),
            "available_graph_types": ["go", "string", "scgpt_derived", "dense"]
        } 