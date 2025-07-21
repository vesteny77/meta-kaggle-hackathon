"""
Core graph analytics functionality for Meta Kaggle project.

This module provides functions to:
1. Prepare edge and node tables for various graph types
2. Build NetworkX graph objects
3. Calculate graph metrics and analyze patterns
"""

import os
import math
import pickle
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, Any

import networkx as nx
import numpy as np
import polars as pl
import pytz
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_OUTPUT_DIR = Path("data/graph")
DEFAULT_KERNEL_FEATURES = Path("data/processed/kernel_features.parquet")
DEFAULT_BIGJOIN = Path("data/intermediate/kernel_bigjoin_clean.parquet")


def prepare_fork_edges(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Prepare fork edges table by extracting parent-child relationships from kernel versions.
    
    Args:
        output_dir: Directory to save the output
        
    Returns:
        Path: Path to the saved fork edges parquet file
    """
    logger.info("Preparing fork edges")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fork_edges.parquet"
    
    # Load kernel versions data
    try:
        # Try loading from parquet/raw
        kernel_versions = pl.scan_parquet("data/parquet/raw/KernelVersions.parquet")
        
        # Extract parent-child relationships
        forks = (
            kernel_versions
            .select([
                pl.col("Id").alias("child"),
                pl.col("ParentId").alias("parent"),
                pl.col("CreationDate")
            ])
            .filter(pl.col("parent").is_not_null())
            .collect()
        )
    except Exception as e:
        logger.warning(f"Error loading from raw parquet: {e}")
        
        # Fallback to bigjoin data
        bigjoin = pl.scan_parquet(DEFAULT_BIGJOIN)
        
        # Check if we have kernel_version_id and parent_id
        schema = bigjoin.collect_schema()
        if "kernel_version_id" not in schema.names() or "parent_id" not in schema.names():
            logger.error("Required columns not found in bigjoin data")
            # Create empty dataframe with expected structure
            forks = pl.DataFrame({
                "child": [],
                "parent": [],
                "child_ts": [],
                "parent_ts": [],
                "delta_h": []
            })
        else:
            # Extract fork relationships from bigjoin
            forks = (
                bigjoin
                .select([
                    pl.col("kernel_version_id").alias("child"),
                    pl.col("parent_id").alias("parent"),
                    pl.col("kernel_ts").alias("child_ts")
                ])
                .filter(pl.col("parent").is_not_null())
                .collect()
            )
            
            # Join to get parent timestamps
            parent_ts = (
                bigjoin
                .select([
                    pl.col("kernel_version_id").alias("parent"),
                    pl.col("kernel_ts").alias("parent_ts")
                ])
                .collect()
            )
            
            forks = forks.join(
                parent_ts,
                on="parent",
                how="inner"
            )
            
            # Calculate time difference in hours
            forks = forks.with_columns([
                ((pl.col("child_ts") - pl.col("parent_ts")).dt.hours()).alias("delta_h")
            ])
    
    logger.info(f"Found {len(forks)} fork relationships")
    forks.write_parquet(output_path)
    return output_path


def prepare_competition_kernel_edges(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Prepare competition-kernel bipartite edges.
    
    Args:
        output_dir: Directory to save the output
        
    Returns:
        Path: Path to the saved competition-kernel edges parquet file
    """
    logger.info("Preparing competition-kernel edges")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ck_edges.parquet"
    
    # Load data from kernel features (which should have both kernel_id and comp_id)
    try:
        kernel_features = pl.read_parquet(DEFAULT_KERNEL_FEATURES)
        if "kernel_id" in kernel_features.columns and "comp_id" in kernel_features.columns:
            ck = (
                kernel_features
                .select([
                    pl.col("kernel_id"),
                    pl.col("comp_id")
                ])
                .filter(pl.col("comp_id").is_not_null())
            )
        else:
            raise ValueError("Required columns not found in kernel features")
    except Exception as e:
        logger.warning(f"Error loading from kernel features: {e}")
        
        # Fallback to bigjoin data
        bigjoin = pl.read_parquet(DEFAULT_BIGJOIN)
        schema = bigjoin.schema
        if "kernel_id" not in schema.names() or "comp_id" not in schema.names():
            logger.error("Required columns not found in bigjoin data")
            # Create empty dataframe with expected structure
            ck = pl.DataFrame({
                "kernel_id": [],
                "comp_id": []
            })
        else:
            ck = (
                bigjoin
                .select([
                    pl.col("kernel_id"),
                    pl.col("comp_id")
                ])
                .filter(pl.col("comp_id").is_not_null())
            )
    
    logger.info(f"Found {len(ck)} kernel-competition relationships")
    ck.write_parquet(output_path)
    return output_path


def prepare_node_attributes(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Tuple[Path, Path]:
    """
    Prepare node attribute tables for kernels and competitions.
    
    Args:
        output_dir: Directory to save the output
        
    Returns:
        Tuple[Path, Path]: Paths to kernel_nodes.parquet and comp_nodes.parquet
    """
    logger.info("Preparing node attribute tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_nodes_path = output_dir / "kernel_nodes.parquet"
    comp_nodes_path = output_dir / "comp_nodes.parquet"
    
    # Prepare kernel node attributes
    try:
        kernel_features = pl.read_parquet(DEFAULT_KERNEL_FEATURES)
        kernel_attrs = kernel_features.select([
            "kernel_id", "gbdt", "dl_pytorch", "dl_tf",
            "loc", "mean_cc", "topic_id", "kernel_ts"
        ])
        kernel_attrs.write_parquet(kernel_nodes_path)
    except Exception as e:
        logger.warning(f"Error preparing kernel node attributes: {e}")
        # Create minimal kernel nodes table
        pl.DataFrame({
            "kernel_id": [],
            "kernel_ts": []
        }).write_parquet(kernel_nodes_path)
    
    # Prepare competition node attributes
    try:
        # Try loading from parquet/raw
        comp_attrs = pl.read_parquet("data/parquet/raw/Competitions.parquet")
        comp_attrs = comp_attrs.select([
            "Id", "Title", "Category", "EvaluationMetric", "DeadlineDate"
        ])
        comp_attrs = comp_attrs.rename({
            "Id": "id",
            "Title": "title",
            "Category": "category",
            "EvaluationMetric": "evaluationMetric",
            "DeadlineDate": "endDate"
        })
    except Exception as e:
        logger.warning(f"Error loading from raw parquet: {e}")
        
        # Fallback to bigjoin data
        try:
            bigjoin = pl.read_parquet(DEFAULT_BIGJOIN)
            comp_attrs = bigjoin.select([
                pl.col("comp_id").alias("id"),
                pl.col("comp_title").alias("title"),
                pl.col("category"),
                pl.col("competition_metric").alias("evaluationMetric")
            ]).unique(subset=["id"])
        except Exception as e2:
            logger.error(f"Error preparing competition attributes: {e2}")
            comp_attrs = pl.DataFrame({
                "id": [],
                "title": [],
                "category": [],
                "evaluationMetric": []
            })
    
    comp_attrs.write_parquet(comp_nodes_path)
    
    logger.info(f"Saved kernel node attributes ({os.path.getsize(kernel_nodes_path)/1024/1024:.2f} MB)")
    logger.info(f"Saved competition node attributes ({os.path.getsize(comp_nodes_path)/1024/1024:.2f} MB)")
    
    return kernel_nodes_path, comp_nodes_path


def prepare_team_collaboration_edges(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Prepare team collaboration edges for the Team Collaboration graph.
    
    Args:
        output_dir: Directory to save the output
        
    Returns:
        Path: Path to the team collaboration edges parquet file
    """
    logger.info("Preparing team collaboration edges")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "team_collab_edges.parquet"
    
    try:
        # Try loading from raw parquet
        teams = pl.read_parquet("data/parquet/raw/Teams.parquet")
        members = pl.read_parquet("data/parquet/raw/TeamMemberships.parquet")
        
        # Get user-team relationships
        user_team = members.select([
            pl.col("TeamId"),
            pl.col("UserId")
        ]).filter(pl.col("UserId").is_not_null())
        
        user_team.write_parquet(output_path)
        logger.info(f"Saved {len(user_team)} team collaboration edges")
        
    except Exception as e:
        logger.warning(f"Error preparing team collaboration edges: {e}")
        
        # Create empty dataframe with expected structure
        pl.DataFrame({
            "TeamId": [],
            "UserId": []
        }).write_parquet(output_path)
    
    return output_path


def prepare_dataset_gravity_edges(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Prepare dataset gravity edges for the Dataset Gravity graph.
    
    Args:
        output_dir: Directory to save the output
        
    Returns:
        Path: Path to the dataset gravity edges parquet file
    """
    logger.info("Preparing dataset gravity edges")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dg_edges.parquet"
    
    try:
        # Try loading from raw parquet
        kernel_dataset_sources = pl.read_parquet("data/parquet/raw/KernelVersionDatasetSources.parquet")
        
        # Join with kernel metadata to get timestamps
        kernel_versions = pl.read_parquet("data/parquet/raw/KernelVersions.parquet")
        
        edges = (
            kernel_dataset_sources
            .select([
                pl.col("KernelVersionId").alias("kernel_id"),
                pl.col("DatasetVersionId").alias("dataset_id")
            ])
            .join(
                kernel_versions.select([
                    pl.col("Id").alias("kernel_id"),
                    pl.col("CreationDate").alias("kernel_ts")
                ]),
                on="kernel_id"
            )
        )
        
    except Exception as e:
        logger.warning(f"Error loading from raw parquet: {e}")
        
        # Try fallback from bigjoin if it contains dataset information
        try:
            bigjoin = pl.read_parquet(DEFAULT_BIGJOIN)
            if "dataset_id" in bigjoin.columns and "kernel_id" in bigjoin.columns:
                edges = (
                    bigjoin
                    .select([
                        pl.col("kernel_id"),
                        pl.col("dataset_id"),
                        pl.col("kernel_ts")
                    ])
                    .filter(pl.col("dataset_id").is_not_null())
                )
            else:
                raise ValueError("Required dataset columns not found in bigjoin")
                
        except Exception as e2:
            logger.error(f"Error preparing dataset gravity edges: {e2}")
            edges = pl.DataFrame({
                "kernel_id": [],
                "dataset_id": [],
                "kernel_ts": []
            })
    
    edges.write_parquet(output_path)
    logger.info(f"Saved {len(edges)} dataset gravity edges")
    
    return output_path


def prepare_forum_code_edges(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Prepare forum to code knowledge transfer edges.
    
    Args:
        output_dir: Directory to save the output
        
    Returns:
        Path: Path to the forum-code edges parquet file
    """
    logger.info("Preparing forum-code knowledge transfer edges")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fc_edges.parquet"
    
    try:
        # Try loading forum topics from raw parquet
        forum_topics = pl.read_parquet("data/parquet/raw/ForumTopics.parquet")
        
        # For simplicity, we'll create a basic relationship between forum topics and kernels
        # that were created after the forum topic and might reference it
        # In a real implementation, we'd need to analyze kernel markdown for references
        
        # Get minimal forum topic data
        topics = forum_topics.select([
            pl.col("Id").alias("topic_id"),
            pl.col("Title").alias("topic_title"),
            pl.col("CreationDate").alias("topic_ts")
        ])
        
        # Join with kernel data from bigjoin to find potential relationships
        bigjoin = pl.read_parquet(DEFAULT_BIGJOIN)
        
        # For demonstration, we'll create relationships based on similar topics in title
        # In reality, this would require more sophisticated text analysis
        edges = pl.DataFrame({
            "topic_id": [],
            "kernel_id": [],
            "relationship_strength": []
        })
        
    except Exception as e:
        logger.warning(f"Error preparing forum-code edges: {e}")
        
        edges = pl.DataFrame({
            "topic_id": [],
            "kernel_id": [],
            "relationship_strength": []
        })
    
    edges.write_parquet(output_path)
    logger.info(f"Saved {len(edges)} forum-code edges")
    
    return output_path


def build_fork_graph(
    edge_file: Path = None,
    node_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> nx.DiGraph:
    """
    Build the fork graph from edge and node files.
    
    Args:
        edge_file: Path to the fork edges parquet file
        node_file: Path to the kernel node attributes parquet file
        output_file: Path to save the output graph pickle file
        output_dir: Directory to save the output if output_file is not specified
        
    Returns:
        nx.DiGraph: The built fork graph
    """
    logger.info("Building fork graph")
    
    # Use default paths if not specified
    if edge_file is None:
        edge_file = output_dir / "fork_edges.parquet"
        if not edge_file.exists():
            edge_file = prepare_fork_edges(output_dir)
    
    if node_file is None:
        node_file = output_dir / "kernel_nodes.parquet"
        if not node_file.exists():
            node_file, _ = prepare_node_attributes(output_dir)
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "fork_graph.pkl"
    
    # Load edge and node data
    edges = pl.read_parquet(edge_file)
    nodes = pl.read_parquet(node_file)
    
    # Initialize directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for row in nodes.iter_rows(named=True):
        attrs = {k: v for k, v in row.items() if k != "kernel_id"}
        G.add_node(int(row["kernel_id"]), **attrs)
    
    # Calculate edge weights: exp(-Δt / τ)
    τ = 7 * 24  # 7 days in hours - time decay parameter
    
    # Add edges with weights
    for row in edges.iter_rows(named=True):
        if "delta_h" in row:
            delta = row["delta_h"]
            weight = math.exp(-delta / τ) if delta > 0 else 1.0
            G.add_edge(
                int(row["parent"]), 
                int(row["child"]), 
                delta_h=float(delta), 
                weight=weight
            )
        else:
            # If delta_h not available, use default weight
            G.add_edge(int(row["parent"]), int(row["child"]), weight=1.0)
    
    logger.info(f"Fork graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
    
    # Save graph
    with open(output_file, "wb") as f:
        pickle.dump(G, f, protocol=5)
    logger.info(f"Saved fork graph to {output_file}")
    
    return G


def build_competition_kernel_graph(
    edge_file: Path = None,
    kernel_node_file: Path = None,
    comp_node_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> nx.Graph:
    """
    Build the competition-kernel bipartite graph.
    
    Args:
        edge_file: Path to the competition-kernel edges parquet file
        kernel_node_file: Path to the kernel node attributes parquet file
        comp_node_file: Path to the competition node attributes parquet file
        output_file: Path to save the output graph pickle file
        output_dir: Directory to save the output if output_file is not specified
        
    Returns:
        nx.Graph: The built competition-kernel bipartite graph
    """
    logger.info("Building competition-kernel graph")
    
    # Use default paths if not specified
    if edge_file is None:
        edge_file = output_dir / "ck_edges.parquet"
        if not edge_file.exists():
            edge_file = prepare_competition_kernel_edges(output_dir)
    
    if kernel_node_file is None or comp_node_file is None:
        kernel_node_file, comp_node_file = prepare_node_attributes(output_dir)
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "ck_graph.pkl"
    
    # Load edge and node data
    edges = pl.read_parquet(edge_file)
    kernel_nodes = pl.read_parquet(kernel_node_file)
    comp_nodes = pl.read_parquet(comp_node_file)
    
    # Initialize graph
    G = nx.Graph()
    
    # Add competition nodes
    for row in comp_nodes.iter_rows(named=True):
        G.add_node(
            f"comp_{row['id']}",
            label=row.get('title', f"Competition {row['id']}"),
            domain=row.get('category', 'Unknown'),
            metric=row.get('evaluationMetric', 'Unknown'),
            node_type="competition"
        )
    
    # Add kernel nodes and edges in one pass
    added_kernels = set()
    for kernel_id, comp_id in edges.select(["kernel_id", "comp_id"]).iter_rows():
        if comp_id is None:
            continue
            
        kernel_node = f"k_{kernel_id}"
        if kernel_node not in added_kernels:
            G.add_node(kernel_node, node_type="kernel")
            added_kernels.add(kernel_node)
            
        G.add_edge(kernel_node, f"comp_{comp_id}", rel="submitted_to")
    
    logger.info(f"Competition-kernel graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
    
    # Save graph
    with open(output_file, "wb") as f:
        pickle.dump(G, f, protocol=5)
    logger.info(f"Saved competition-kernel graph to {output_file}")
    
    return G


def compute_pagerank(
    graph_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    alpha: float = 0.85
) -> Path:
    """
    Compute PageRank for a graph.
    
    Args:
        graph_file: Path to the graph pickle file
        output_file: Path to save the output PageRank values
        output_dir: Directory to save the output if output_file is not specified
        alpha: Damping parameter for PageRank
        
    Returns:
        Path: Path to the saved PageRank values parquet file
    """
    logger.info(f"Computing PageRank with alpha={alpha}")
    
    # Use default path if not specified
    if graph_file is None:
        graph_file = output_dir / "fork_graph.pkl"
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "pagerank.parquet"
    
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    
    # Compute PageRank
    logger.info(f"Computing PageRank for graph with {G.number_of_nodes()} nodes")
    try:
        pr = nx.pagerank(G, alpha=alpha, weight="weight")
        
        # Convert to DataFrame and save
        pr_df = pl.DataFrame({
            "node_id": list(pr.keys()),
            "pagerank": list(pr.values())
        })
        pr_df.write_parquet(output_file)
        logger.info(f"Saved PageRank values to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error computing PageRank: {e}")
        return None


def calculate_fork_velocity(
    edge_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:
    """
    Calculate fork velocity metrics.
    
    Args:
        edge_file: Path to the fork edges parquet file
        output_file: Path to save the output metrics
        output_dir: Directory to save the output if output_file is not specified
        
    Returns:
        Path: Path to the saved velocity metrics parquet file
    """
    logger.info("Calculating fork velocity metrics")
    
    # Use default path if not specified
    if edge_file is None:
        edge_file = output_dir / "fork_edges.parquet"
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "fork_velocity.parquet"
    
    # Load edges
    edges = pl.read_parquet(edge_file)
    
    if "delta_h" not in edges.columns:
        logger.error("delta_h column not found in fork edges")
        return None
    
    # Calculate time to first fork for each parent kernel
    first_fork = (
        edges
        .filter(pl.col("delta_h") > 0)  # Filter out self-forks or invalid entries
        .group_by("parent")
        .agg(
            pl.min("delta_h").alias("time_to_first_fork_h"),
            pl.count().alias("fork_count")
        )
    )
    
    # Add derived metrics
    first_fork = first_fork.with_columns([
        (pl.col("time_to_first_fork_h") <= 24).alias("forked_within_day"),
        (pl.col("time_to_first_fork_h") <= 168).alias("forked_within_week"),
        (pl.col("fork_count") > 10).alias("popular_10plus_forks")
    ])
    
    # Calculate summary statistics
    median_time = first_fork["time_to_first_fork_h"].median()
    mean_time = first_fork["time_to_first_fork_h"].mean()
    day_pct = (first_fork["forked_within_day"].sum() / len(first_fork)) * 100
    week_pct = (first_fork["forked_within_week"].sum() / len(first_fork)) * 100
    
    logger.info(f"Median time to first fork: {median_time:.2f} hours")
    logger.info(f"Mean time to first fork: {mean_time:.2f} hours")
    logger.info(f"Forked within 1 day: {day_pct:.2f}%")
    logger.info(f"Forked within 1 week: {week_pct:.2f}%")
    
    # Save results
    first_fork.write_parquet(output_file)
    logger.info(f"Saved fork velocity metrics to {output_file}")
    
    return output_file


def build_team_collaboration_graph(
    edge_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> nx.Graph:
    """
    Build the team collaboration graph.
    
    Args:
        edge_file: Path to the team collaboration edges parquet file
        output_file: Path to save the output graph pickle file
        output_dir: Directory to save the output if output_file is not specified
        
    Returns:
        nx.Graph: The built team collaboration graph
    """
    logger.info("Building team collaboration graph")
    
    # Use default path if not specified
    if edge_file is None:
        edge_file = output_dir / "team_collab_edges.parquet"
        if not edge_file.exists():
            edge_file = prepare_team_collaboration_edges(output_dir)
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "team_collab_graph.pkl"
    
    # Load edges
    edges = pl.read_parquet(edge_file)
    
    # Initialize bipartite graph
    G = nx.Graph()
    
    # Add edges
    for team_id, user_id in edges.select(["TeamId", "UserId"]).iter_rows():
        G.add_node(f"team_{team_id}", node_type="team")
        G.add_node(f"user_{user_id}", node_type="user")
        G.add_edge(f"team_{team_id}", f"user_{user_id}")
    
    logger.info(f"Team collaboration graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
    
    # Save graph
    with open(output_file, "wb") as f:
        pickle.dump(G, f, protocol=5)
    logger.info(f"Saved team collaboration graph to {output_file}")
    
    return G


def build_dataset_gravity_graph(
    edge_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> nx.DiGraph:
    """
    Build the dataset gravity graph.
    
    Args:
        edge_file: Path to the dataset gravity edges parquet file
        output_file: Path to save the output graph pickle file
        output_dir: Directory to save the output if output_file is not specified
        
    Returns:
        nx.DiGraph: The built dataset gravity graph
    """
    logger.info("Building dataset gravity graph")
    
    # Use default path if not specified
    if edge_file is None:
        edge_file = output_dir / "dg_edges.parquet"
        if not edge_file.exists():
            edge_file = prepare_dataset_gravity_edges(output_dir)
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "dataset_gravity_graph.pkl"
    
    # Load edges
    edges = pl.read_parquet(edge_file)
    
    # Initialize directed graph
    G = nx.DiGraph()
    
    # Add edges (direction: dataset -> kernel)
    for kernel_id, dataset_id, kernel_ts in edges.iter_rows():
        G.add_node(f"dataset_{dataset_id}", node_type="dataset")
        G.add_node(f"kernel_{kernel_id}", node_type="kernel", ts=kernel_ts)
        G.add_edge(f"dataset_{dataset_id}", f"kernel_{kernel_id}")
    
    logger.info(f"Dataset gravity graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
    
    # Save graph
    with open(output_file, "wb") as f:
        pickle.dump(G, f, protocol=5)
    logger.info(f"Saved dataset gravity graph to {output_file}")
    
    return G


def analyze_dataset_gravity(
    graph_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:
    """
    Analyze dataset gravity to identify influential datasets.
    
    Args:
        graph_file: Path to the dataset gravity graph pickle file
        output_file: Path to save the output analysis
        output_dir: Directory to save the output if output_file is not specified
        
    Returns:
        Path: Path to the saved dataset gravity analysis parquet file
    """
    logger.info("Analyzing dataset gravity")
    
    # Use default path if not specified
    if graph_file is None:
        graph_file = output_dir / "dataset_gravity_graph.pkl"
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "dataset_gravity_analysis.parquet"
    
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    
    # Find dataset nodes
    dataset_nodes = [node for node, attr in G.nodes(data=True) 
                     if attr.get("node_type") == "dataset"]
    
    # Calculate metrics for each dataset
    results = []
    for dataset_node in dataset_nodes:
        # Get kernels influenced by this dataset
        kernels = list(G.neighbors(dataset_node))
        
        if not kernels:
            continue
            
        # Extract dataset ID from node name
        dataset_id = dataset_node.split("_")[1]
        
        # Count kernels
        kernel_count = len(kernels)
        
        # Calculate time-based metrics if timestamps are available
        ts_available = all(G.nodes[k].get("ts") is not None for k in kernels)
        
        if ts_available:
            # Get timestamps and sort
            timestamps = [G.nodes[k]["ts"] for k in kernels]
            timestamps.sort()
            
            # Calculate time span and burst metrics
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            
            # Detect bursts (many kernels in a short time)
            if len(timestamps) >= 10 and time_span > 0:
                kernel_per_hour = len(timestamps) / time_span
                burst_score = kernel_per_hour * kernel_count / 100
            else:
                burst_score = 0
                
            results.append({
                "dataset_id": dataset_id,
                "kernel_count": kernel_count,
                "time_span_h": time_span,
                "burst_score": burst_score
            })
        else:
            results.append({
                "dataset_id": dataset_id,
                "kernel_count": kernel_count,
                "time_span_h": None,
                "burst_score": None
            })
    
    # Convert to DataFrame and save
    if results:
        results_df = pl.DataFrame(results)
        results_df = results_df.sort("kernel_count", descending=True)
        results_df.write_parquet(output_file)
        logger.info(f"Saved dataset gravity analysis to {output_file}")
        
        # Log top datasets
        top_n = min(10, len(results_df))
        logger.info(f"Top {top_n} datasets by kernel count:")
        for row in results_df.head(top_n).iter_rows(named=True):
            logger.info(f"Dataset {row['dataset_id']}: {row['kernel_count']} kernels")
        
        return output_file
    else:
        logger.warning("No results generated for dataset gravity analysis")
        return None


def build_forum_code_graph(
    edge_file: Path = None,
    output_file: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> nx.DiGraph:
    """
    Build the forum-code knowledge transfer graph.
    
    Args:
        edge_file: Path to the forum-code edges parquet file
        output_file: Path to save the output graph pickle file
        output_dir: Directory to save the output if output_file is not specified
        
    Returns:
        nx.DiGraph: The built forum-code graph
    """
    logger.info("Building forum-code knowledge transfer graph")
    
    # Use default path if not specified
    if edge_file is None:
        edge_file = output_dir / "fc_edges.parquet"
        if not edge_file.exists():
            edge_file = prepare_forum_code_edges(output_dir)
    
    if output_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "forum_code_graph.pkl"
    
    # Load edges
    edges = pl.read_parquet(edge_file)
    
    # Initialize directed graph
    G = nx.DiGraph()
    
    # Add edges (direction: topic -> kernel)
    for topic_id, kernel_id, strength in edges.iter_rows():
        G.add_node(f"topic_{topic_id}", node_type="topic")
        G.add_node(f"kernel_{kernel_id}", node_type="kernel")
        G.add_edge(f"topic_{topic_id}", f"kernel_{kernel_id}", weight=strength)
    
    logger.info(f"Forum-code graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
    
    # Save graph
    with open(output_file, "wb") as f:
        pickle.dump(G, f, protocol=5)
    logger.info(f"Saved forum-code graph to {output_file}")
    
    return G


def export_graph_for_visualization(
    graph_file: Path,
    output_file: Path,
    node_limit: int = 500,
    include_attrs: bool = True
) -> Path:
    """
    Export a graph to JSON format for visualization.
    
    Args:
        graph_file: Path to the graph pickle file
        output_file: Path to save the JSON output
        node_limit: Maximum number of nodes to include
        include_attrs: Whether to include node attributes
        
    Returns:
        Path: Path to the saved JSON file
    """
    logger.info(f"Exporting graph for visualization (limit: {node_limit} nodes)")
    
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    
    # If graph is too large, create a subgraph of most important nodes
    if G.number_of_nodes() > node_limit:
        logger.info(f"Graph too large ({G.number_of_nodes()} nodes), creating subgraph")
        
        # Try to use PageRank if available
        try:
            pr = nx.pagerank(G, weight="weight" if "weight" in G.edges[next(iter(G.edges))].keys() else None)
            top_nodes = sorted(pr.keys(), key=lambda x: pr[x], reverse=True)[:node_limit]
            G = G.subgraph(top_nodes)
        except Exception as e:
            logger.warning(f"Could not compute PageRank: {e}")
            
            # Fallback: use degree as importance measure
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:node_limit]
            G = G.subgraph(top_nodes)
    
    # Convert to node-link format
    from networkx.readwrite import json_graph
    data = json_graph.node_link_data(G)
    
    # Simplify attributes if requested
    if not include_attrs:
        for node in data["nodes"]:
            node.pop("attrs", None)
    
    # Save as JSON
    import json
    with open(output_file, "w") as f:
        json.dump(data, f)
    
    logger.info(f"Exported graph with {len(data['nodes'])} nodes and {len(data['links'])} edges")
    logger.info(f"Saved to {output_file}")
    
    return output_file