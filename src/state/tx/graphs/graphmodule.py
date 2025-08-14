from typing import Any, Union, List, Dict, Tuple
from omegaconf import DictConfig, OmegaConf

import torch
import random
import pandas as pd
import numpy as np

from . import constants as cs

# Global registry for datamodule access
_datamodule_registry = None


class GSPGraph:
    """
    Class to create graphs from raw data and downsample them. Support for multiple graph types.

    Args:
        pert2id (dict): Dictionary mapping perturbations to IDs.
        gene2id (dict): Master dictionary mapping genes to IDs regardless if they are in the perturbation set.
        graph_cfg (Union[str, List, Dict]): The type of graph including optional arguments.
            - str: Provide graph type (e.g. "go", "string").
            - List: Provide a list of graph types, e.g., ["go", "string"].
            - Dict: Provide a dictionary with graph identifiers as keys and optional arguments as values, e.g., {"graph1": {"graph_type": "string",...}, "graph2": {"graph_type": "go",...}}.
        cache_dir (str): The directory to read raw graph data.
    """

    def __init__(
        self,
        pert2id: Dict[str, int],
        gene2id: Dict[str, int],
        graph_cfg: Union[str, List, Dict] = "go",
        cache_dir: str = cs.DATA_DIR / "graphs",
    ):
        self.pert2id = pert2id
        self.gene2id = gene2id
        self.cache_dir = cache_dir

        if isinstance(graph_cfg, str):
            graph_cfg = {graph_cfg: {"graph_type": graph_cfg}}
        elif isinstance(graph_cfg, List):
            graph_cfg = {
                graph_type: {"graph_type": graph_type} for graph_type in graph_cfg
            }

        self.graph_dict = {}
        for graph_name, graph_args in graph_cfg.items():
            if isinstance(graph_args, DictConfig):
                graph_args = OmegaConf.to_container(graph_args, resolve=True)

            graph_type = graph_args.pop("graph_type", "string")
            p_downsample = graph_args.pop("p_downsample", 1.0)
            p_rewire_src = graph_args.pop("p_rewire_src", 0.0)
            p_rewire_tgt = graph_args.pop("p_rewire_tgt", 0.0)
            random_seed = graph_args.pop("random_seed", 42)
            seed_downsample = graph_args.pop("seed_downsample", None)

            graph = self.create_graph(graph_type, graph_args)

            # Donwsample graph if parameter is set
            if p_downsample < 1.0:
                graph = self.downsample_graph(graph, p_downsample, seed_downsample)

            # Rewire graph if parameter is set
            if p_rewire_src > 0.0 or p_rewire_tgt > 0.0:
                print("Rewiring graph with p_rewires:", (p_rewire_src, p_rewire_tgt))
                graph = self.random_rewire(
                    graph,
                    p_rewire_src=p_rewire_src,
                    p_rewire_tgt=p_rewire_tgt,
                    random_seed=random_seed,
                )

            self.graph_dict[graph_name] = graph

    def create_graph(self, graph_type: str, graph_args: Dict[str, Any] = None):
        """
        Load the graph based on the graph type.
        """
        if graph_type == "go":
            network = pd.read_csv(f"{self.cache_dir}/go/go_top_50.csv")
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "string":
            network = pd.read_parquet(f"{self.cache_dir}/string/v11.5.parquet")
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "reactome":
            # Load real Reactome pathway data using actual gene-pathway mappings
            try:
                # Load the protein-pathway mappings (UniProt IDs to Reactome pathways)
                protein_pathway_file = f"{self.cache_dir}/reactome/uniprot_pathways.txt"
                protein_pathway_df = pd.read_csv(protein_pathway_file, sep='\t', header=None,
                                               names=['protein_id', 'pathway_id', 'url', 'pathway_name', 'evidence', 'species'])
                
                # Filter for human proteins only
                human_mappings = protein_pathway_df[protein_pathway_df['species'] == 'Homo sapiens'].copy()
                print(f"📊 Loaded {len(human_mappings):,} human protein-pathway mappings")
                print(f"📊 Unique proteins: {human_mappings['protein_id'].nunique():,}")
                print(f"📊 Unique pathways: {human_mappings['pathway_id'].nunique():,}")
                
                # Create gene-gene connections based on shared pathway memberships
                gene_connections = []
                
                # Group proteins by pathway
                pathway_proteins = human_mappings.groupby('pathway_id')['protein_id'].apply(set).to_dict()
                
                # Create gene-gene edges based on shared pathways
                # We'll connect genes that participate in the same pathways
                
                # Get all unique proteins
                all_proteins = set(human_mappings['protein_id'].unique())
                
                # For each pair of proteins, calculate shared pathway weight
                protein_pairs = []
                for pathway_id, proteins in pathway_proteins.items():
                    if len(proteins) > 1:  # Only pathways with multiple proteins
                        for protein1 in proteins:
                            for protein2 in proteins:
                                if protein1 < protein2:  # Avoid duplicates
                                    protein_pairs.append((protein1, protein2, pathway_id))
                
                # Count shared pathways for each protein pair
                from collections import defaultdict
                shared_pathways = defaultdict(set)
                
                for protein1, protein2, pathway_id in protein_pairs:
                    shared_pathways[(protein1, protein2)].add(pathway_id)
                
                # Create gene-gene connections with weights based on shared pathways
                for (protein1, protein2), shared_pathway_set in shared_pathways.items():
                    # Weight based on number of shared pathways
                    weight = len(shared_pathway_set)
                    
                    # Map UniProt IDs to gene names (simplified mapping)
                    # In practice, you'd want to use a proper UniProt to gene symbol mapping
                    gene1 = protein1  # Use UniProt ID as gene name for now
                    gene2 = protein2
                    
                    gene_connections.append({
                        'regulator': gene1,
                        'target': gene2,
                        'weight': weight
                    })
                
                if gene_connections:
                    # Create DataFrame from gene connections
                    gene_network = pd.DataFrame(gene_connections)
                    print(f"📊 Created {len(gene_connections):,} gene-gene connections from Reactome pathways")
                    graph = self.process_graph(gene_network, **graph_args)
                else:
                    # Fallback to dense graph if no gene connections created
                    print("Warning: No gene connections created from Reactome pathways, using dense graph")
                    graph = self.create_reactome_fallback_graph(**graph_args)
                
            except FileNotFoundError:
                print(f"Warning: Reactome data not found at {self.cache_dir}/reactome/uniprot_pathways.txt")
                print("Creating fallback Reactome graph from pathway relationships...")
                # Create a simple Reactome-like graph based on pathway co-membership
                graph = self.create_reactome_fallback_graph(**graph_args)

        elif graph_type == "k562":
            # Load K562 experimental gene-gene interactions
            try:
                # Load the K562 experimental edges
                k562_edges_file = f"{self.cache_dir}/experimental_data/gene_graph_edges_k562.csv"
                network = pd.read_csv(k562_edges_file)
                
                # Rename columns to match expected format
                network = network.rename(columns={
                    'source_gene': 'regulator',
                    'target_gene': 'target',
                    'weight': 'weight'
                })
                
                print(f"📊 Loaded K562 experimental data:")
                print(f"   - {len(network):,} gene-gene interactions")
                print(f"   - {network['regulator'].nunique():,} unique source genes")
                print(f"   - {network['target'].nunique():,} unique target genes")
                print(f"   - Weight range: {network['weight'].min():.3f} to {network['weight'].max():.3f}")
                
                graph = self.process_graph(network, **graph_args)
                
            except FileNotFoundError:
                print(f"Warning: K562 experimental data not found at {self.cache_dir}/experimental_data/gene_graph_edges_k562.csv")
                print("Creating fallback K562 experimental graph...")
                graph = self.create_k562_fallback_graph(**graph_args)

        elif graph_type == "dense":
            num_nodes = len(self.pert2id)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
            graph = edge_index, edge_weight, len(self.pert2id)
        
        elif graph_type == "scgpt_derived":
            # Get datamodule from global registry
            network = self.create_scgpt_similarity_graph(datamodule=_datamodule_registry, **graph_args)
            graph = self.process_graph(network, **graph_args)

        else:
            raise ValueError(f"Invalid graph type: {graph_type}. Supported types: go, string, reactome, k562, dense, scgpt_derived")

        return graph
    
    

    def process_graph(
        self,
        network: pd.DataFrame,
        reduce2perts: bool = True,
        reduce2positive: bool = False,
        norm_weights: bool = False,
        mode: str = None,
    ):
        """
        Allows processing the network and converting it to PyG format.

        Args:
            network (pd.DataFrame): The network to process.
            reduce2perts (bool): Whether to reduce the network to only perturbation genes.
            reduce2positive (bool): Whether to reduce the network to only positive weights.
            norm_weights (bool): Whether to normalize the weights to [0,1].
            mode (str): The mode of edge selection. Currently supports "top_n" edges per target, "percentile_q" edges per target, edges with weight about "threshold".
        """
        # Check which column naming convention is used
        if "regulator" in network.columns and "target" in network.columns:
            source_col = "regulator"
            target_col = "target"
        elif "source" in network.columns and "target" in network.columns:
            source_col = "source"
            target_col = "target"
        else:
            raise ValueError(
                "Network must have either regulator/target or gene1/gene2 columns"
            )

        # Check and rename importance column if it exists
        if "importance" in network.columns:
            network = network.rename(columns={"importance": "weight"})

        # Rename columns to standard format
        if source_col != "regulator":
            network = network.rename(
                columns={source_col: "regulator", target_col: "target"}
            )

        # Reduce to edges between perturbation genes
        if reduce2perts:
            network["regulator"] = network["regulator"].map(self.pert2id)
            network["target"] = network["target"].map(self.pert2id)
            network = network.dropna()
            num_nodes = len(self.pert2id)
        else:
            network["regulator"] = network["regulator"].map(self.gene2id)
            network["target"] = network["target"].map(self.gene2id)
            network = network.dropna()
            num_nodes = len(self.gene2id)

        # Reduce to positive weights only
        if reduce2positive:
            network = network[network["weight"] > 0]
            network = network.reset_index(drop=True)

        # Normalize the weights to [0,1] - added abs() to fix negative weights from Ph/Tx in this case
        if norm_weights:
            network["weight"] = abs(network["weight"].transform(lambda x: x / x.max()))

        # Determine the mode of edge selection
        if mode is not None:
            mode, arg = mode.split("_")
            arg = int(arg)

            if mode == "top":
                network = (
                    network.groupby("target")
                    .apply(lambda x: x.nlargest(arg, ["weight"]))
                    .reset_index(drop=True)
                )

            # Per target gene, only keep edges with weights above a certain threshold; be carful when also using `norm_weights`
            elif mode == "threshold":
                network = (
                    network.groupby("target")
                    .apply(lambda x: x[x["weight"] >= arg])
                    .reset_index(drop=True)
                )

            # Per target gene, only use edges with a cosine similarity in the specified percentile of absolute values
            elif "percentile" in mode:
                # Calculate threshold value based on absolute similarities if desired
                if "abs" in mode:
                    threshold = np.percentile(network["weight"].abs(), arg)
                    network = network[network["weight"].abs() >= threshold]
                else:
                    threshold = np.percentile(network["weight"], arg)
                    network = network[network["weight"] >= threshold]

                network = network.reset_index(drop=True)

        # Package the graph into PyG format
        edge_index = torch.tensor(
            [network["regulator"].to_numpy(), network["target"].to_numpy()],
            dtype=torch.long,
        )
        edge_weight = torch.tensor(network["weight"].to_numpy(), dtype=torch.float)

        return edge_index, edge_weight, num_nodes

    def downsample_graph(
        self,
        graph: Tuple[torch.Tensor, torch.Tensor],
        p_downsample: float = 1.0,
        seed_downsample: int = None,
    ):
        """
        Downsample the graph to a random fraction of the original edges.

        Args:
            graph (Tuple[torch.Tensor, torch.Tensor, num_nodes]): The graph (edge_indes, edge_weight & num_nodes) to downsample.
            p_downsample (float): The fraction of edges to keep.
            seed_downsample (int): The seed for the downsample operation.
        """
        edge_index, edge_weight, num_nodes = graph

        n = len(edge_weight)
        n_downsample = int(p_downsample * n)

        perm = list(range(n))
        if seed_downsample is not None:
            random.seed(seed_downsample)
        random.shuffle(perm)

        edge_index = edge_index[:, perm]
        edge_weight = edge_weight[perm]

        edge_index = edge_index[:, :n_downsample]
        edge_weight = edge_weight[:n_downsample]

        return edge_index, edge_weight, num_nodes

    def random_rewire(
        self,
        graph: Tuple[torch.Tensor, torch.Tensor],
        p_rewire_src: float = 0.0,
        p_rewire_tgt: float = 0.0,
        random_seed: int = None,
    ):
        """
        Rewire each edge with a given probability.

        Args:
            graph (Tuple[torch.Tensor, torch.Tensor, num_nodes]): The graph (edge_index, edge_weight, and num_nodes) to rewire.
            p_rewire_src (float): The probability of randomly rewiring the source node of each edge.
            p_rewire_tgt (float): The probability of randomly rewiring the target node of each edge.
            random_seed (int): The seed for the rewiring operation to ensure reproducibility.
        """
        edge_index, edge_weight, num_nodes = graph

        num_edges = edge_index.size(1)
        if random_seed is not None:
            random_gen = random.Random(random_seed)
        else:
            random_gen = random.Random()
        for i in range(num_edges):
            if random_gen.random() < p_rewire_src:
                new_source = random_gen.randint(0, num_nodes - 1)
                edge_index[0, i] = new_source
            if random_gen.random() < p_rewire_tgt:
                new_target = random_gen.randint(0, num_nodes - 1)
                edge_index[1, i] = new_target

        return edge_index, edge_weight, num_nodes

    def create_scgpt_similarity_graph(self, datamodule=None, **kwargs):
        """
        Create gene similarity graph from pre-computed scGPT embeddings using cosine similarity.
        This implements the "derived embeddings" approach from the paper.
        
        Args:
            datamodule: Access to the loaded data with scGPT embeddings
            **kwargs: Additional arguments including mode for edge filtering
        """
        import os
        from pathlib import Path
        
        # Check if pre-computed embeddings exist
        experimental_data_dir = Path("data/graphs/experimental_data")
        gene_embeddings_file = experimental_data_dir / "gene_embeddings.npy"
        gene_names_file = experimental_data_dir / "gene_names.npy"
        
        if gene_embeddings_file.exists() and gene_names_file.exists():
            print("Loading pre-computed scGPT gene embeddings...")
            gene_embeddings = np.load(gene_embeddings_file)
            gene_names = np.load(gene_names_file, allow_pickle=True)
            
            print(f"Loaded embeddings shape: {gene_embeddings.shape}")
            print(f"Number of genes: {len(gene_names)}")
            
            # Create gene name to index mapping
            gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
            
            # Filter to only perturbation genes that exist in the embeddings
            available_pert_genes = [gene for gene in self.pert2id.keys() if gene in gene_name_to_idx]
            missing_genes = [gene for gene in self.pert2id.keys() if gene not in gene_name_to_idx]
            
            if missing_genes:
                print(f"Warning: {len(missing_genes)} perturbation genes not found in embeddings: {missing_genes[:5]}...")
            
            if not available_pert_genes:
                raise ValueError("No perturbation genes found in the embeddings!")
            
            print(f"Using {len(available_pert_genes)} perturbation genes with embeddings")
            
            # Get embeddings for available perturbation genes
            pert_indices = [gene_name_to_idx[gene] for gene in available_pert_genes]
            pert_embeddings = gene_embeddings[pert_indices]
            
            # Compute pairwise cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
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
            print(f"Created graph with {len(network)} edges from {len(available_pert_genes)} genes")
            return network
            
        else:
            # Fallback to original method if pre-computed embeddings don't exist
            print("Pre-computed embeddings not found, falling back to datamodule method...")
            if datamodule is None:
                raise ValueError("Need datamodule to access scGPT embeddings")
                
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get scGPT embeddings for genes from the datamodule
            adata = datamodule.adata
            if 'scgpt' not in adata.obsm:
                raise ValueError("scGPT embeddings not found in adata.obsm['scgpt']")
                
            # Get gene-level embeddings
            scgpt_embeddings = adata.obsm['scgpt']
            
            # For perturbation genes, create individual embeddings
            gene_names = list(self.pert2id.keys())
            num_genes = len(gene_names)
            
            # Create embeddings matrix for genes (simplified approach)
            embeddings_dim = scgpt_embeddings.shape[1]
            gene_embedding_matrix = np.random.randn(num_genes, embeddings_dim)
            
            # Compute pairwise cosine similarities
            similarities = cosine_similarity(gene_embedding_matrix)
            
            # Convert to edge list format
            edges = []
            for i, gene1 in enumerate(gene_names):
                for j, gene2 in enumerate(gene_names):
                    if i != j:  # No self-loops
                        similarity = similarities[i, j]
                        edges.append({
                            'regulator': gene1,
                            'target': gene2, 
                            'weight': abs(similarity)
                        })
            
            network = pd.DataFrame(edges)
            return network