"""
Graph visualization utilities for Meta Kaggle project.

This module provides functions to create visualizations of graph analytics results.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import networkx as nx
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_GRAPH_DIR = Path("data/graph")
DEFAULT_VISUALS_DIR = Path("visuals")


def plot_fork_graph(
    graph_file: Path = None,
    output_file: Path = None,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
    node_limit: int = 100,
    node_size_attr: str = "pagerank",
    node_color_attr: str = None,
    layout: str = "spring",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> Path:
    """
    Create a visualization of the fork graph.
    
    Args:
        graph_file: Path to the graph pickle file
        output_file: Path to save the visualization
        graph_dir: Directory containing graph files
        visuals_dir: Directory to save visualizations
        node_limit: Maximum number of nodes to include
        node_size_attr: Node attribute to use for sizing nodes
        node_color_attr: Node attribute to use for coloring nodes
        layout: Graph layout algorithm ('spring', 'kamada_kawai', 'circular')
        figsize: Figure size
        dpi: Output DPI
        
    Returns:
        Path: Path to the saved visualization
    """
    logger.info("Creating fork graph visualization")
    
    # Use default paths if not specified
    if graph_file is None:
        graph_file = graph_dir / "fork_graph.pkl"
    
    if output_file is None:
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_file = visuals_dir / "fork_graph.png"
    
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    
    # If the graph is too large, select top nodes by PageRank or degree
    if G.number_of_nodes() > node_limit:
        logger.info(f"Selecting top {node_limit} nodes from graph with {G.number_of_nodes()} nodes")
        
        if node_size_attr == "pagerank":
            # Compute PageRank if not already computed
            pagerank_file = graph_dir / "pagerank.parquet"
            if pagerank_file.exists():
                pr_df = pl.read_parquet(pagerank_file)
                pr = {row["node_id"]: row["pagerank"] for row in pr_df.iter_rows(named=True)}
            else:
                pr = nx.pagerank(G, weight="weight")
            
            # Select top nodes
            top_nodes = sorted(pr.keys(), key=lambda x: pr[x], reverse=True)[:node_limit]
        else:
            # Use degree as fallback
            deg = dict(G.degree(weight="weight"))
            top_nodes = sorted(deg.keys(), key=lambda x: deg[x], reverse=True)[:node_limit]
        
        # Create subgraph
        G = G.subgraph(top_nodes)
    
    # Set up node sizes
    if node_size_attr == "pagerank":
        try:
            pr = nx.pagerank(G, weight="weight")
            node_sizes = [pr[node] * 10000 for node in G.nodes()]
        except:
            node_sizes = 100
    elif node_size_attr == "degree":
        node_sizes = [G.degree(node) * 5 for node in G.nodes()]
    else:
        node_sizes = 100
    
    # Set up node colors
    if node_color_attr == "gbdt":
        node_colors = [G.nodes[node].get("gbdt", False) for node in G.nodes()]
    elif node_color_attr == "dl_pytorch":
        node_colors = [G.nodes[node].get("dl_pytorch", False) for node in G.nodes()]
    else:
        node_colors = "skyblue"
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.3, iterations=50, weight="weight")
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight="weight")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, arrows=True, arrowsize=5)
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    
    logger.info(f"Saved fork graph visualization to {output_file}")
    
    return output_file


def plot_competition_kernel_graph(
    graph_file: Path = None,
    output_file: Path = None,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
    highlight_domain: str = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
) -> Path:
    """
    Create a visualization of the competition-kernel bipartite graph.
    
    Args:
        graph_file: Path to the graph pickle file
        output_file: Path to save the visualization
        graph_dir: Directory containing graph files
        visuals_dir: Directory to save visualizations
        highlight_domain: Domain to highlight (e.g., 'CV', 'NLP', 'Tabular')
        figsize: Figure size
        dpi: Output DPI
        
    Returns:
        Path: Path to the saved visualization
    """
    logger.info("Creating competition-kernel graph visualization")
    
    # Use default paths if not specified
    if graph_file is None:
        graph_file = graph_dir / "ck_graph.pkl"
    
    if output_file is None:
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_file = visuals_dir / "competition_kernel_graph.png"
    
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    
    # If the graph is very large, create a more manageable visualization
    if G.number_of_nodes() > 500:
        logger.info("Graph too large, creating a domain-specific subgraph")
        
        # Filter to specific domain if requested
        if highlight_domain:
            comp_nodes = [n for n, d in G.nodes(data=True) 
                          if d.get("node_type") == "competition" and d.get("domain") == highlight_domain]
            
            # Get connected kernel nodes
            kernel_nodes = []
            for comp in comp_nodes:
                kernel_nodes.extend(list(G.neighbors(comp)))
            
            # Create subgraph
            nodes = comp_nodes + kernel_nodes
            G = G.subgraph(nodes)
        
        # If still too large or no domain specified, take top competitions by degree
        if G.number_of_nodes() > 500 or not highlight_domain:
            comp_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "competition"]
            comp_degrees = {n: G.degree(n) for n in comp_nodes}
            top_comps = sorted(comp_degrees.keys(), key=lambda x: comp_degrees[x], reverse=True)[:50]
            
            # Get connected kernel nodes
            kernel_nodes = []
            for comp in top_comps:
                kernel_nodes.extend(list(G.neighbors(comp)))
            
            # Limit number of kernel nodes if necessary
            if len(kernel_nodes) > 450:
                kernel_nodes = kernel_nodes[:450]
            
            # Create subgraph
            nodes = top_comps + kernel_nodes
            G = G.subgraph(nodes)
    
    # Set up visualization
    plt.figure(figsize=figsize)
    
    # Create bipartite layout
    comp_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "competition"]
    kernel_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "kernel"]
    
    # Position nodes in bipartite layout
    pos = {}
    for i, node in enumerate(comp_nodes):
        pos[node] = np.array([0, i])
    for i, node in enumerate(kernel_nodes):
        pos[node] = np.array([1, i * len(comp_nodes) / len(kernel_nodes)])
    
    # Draw competition nodes
    nx.draw_networkx_nodes(G, pos, nodelist=comp_nodes, node_color="orange", 
                           node_size=200, alpha=0.8, node_shape="s")
    
    # Draw kernel nodes
    nx.draw_networkx_nodes(G, pos, nodelist=kernel_nodes, node_color="skyblue", 
                           node_size=50, alpha=0.6)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
    
    # Add domain labels for competitions if there aren't too many
    if len(comp_nodes) <= 20:
        labels = {n: G.nodes[n].get("label", n) for n in comp_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    
    logger.info(f"Saved competition-kernel graph visualization to {output_file}")
    
    return output_file


def plot_fork_velocity_histogram(
    velocity_file: Path = None,
    output_file: Path = None,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
) -> Path:
    """
    Create a histogram of fork velocity (time to first fork).
    
    Args:
        velocity_file: Path to the fork velocity metrics parquet file
        output_file: Path to save the visualization
        graph_dir: Directory containing graph files
        visuals_dir: Directory to save visualizations
        figsize: Figure size
        dpi: Output DPI
        
    Returns:
        Path: Path to the saved visualization
    """
    logger.info("Creating fork velocity histogram")
    
    # Use default paths if not specified
    if velocity_file is None:
        velocity_file = graph_dir / "fork_velocity.parquet"
    
    if output_file is None:
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_file = visuals_dir / "fork_velocity_histogram.png"
    
    # Load velocity data
    velocity_df = pl.read_parquet(velocity_file)
    
    # Create histogram
    plt.figure(figsize=figsize)
    
    # Filter to reasonable range (up to 30 days)
    times = velocity_df.filter(pl.col("time_to_first_fork_h") <= 30*24)["time_to_first_fork_h"].to_numpy()
    
    # Plot histogram
    plt.hist(times, bins=50, alpha=0.75, color="steelblue")
    
    # Add vertical lines for key timepoints
    median = velocity_df["time_to_first_fork_h"].median()
    plt.axvline(x=median, color="red", linestyle="--", 
                label=f"Median: {median:.1f} hours")
    plt.axvline(x=24, color="green", linestyle="-.", 
                label="24 hours (1 day)")
    plt.axvline(x=168, color="purple", linestyle="-.", 
                label="168 hours (1 week)")
    
    # Add labels and title
    plt.xlabel("Time to First Fork (hours)")
    plt.ylabel("Count")
    plt.title("Distribution of Time to First Fork")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    
    logger.info(f"Saved fork velocity histogram to {output_file}")
    
    return output_file


def plot_dataset_gravity_bubbles(
    gravity_file: Path = None,
    output_file: Path = None,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> Path:
    """
    Create a bubble chart of dataset gravity metrics.
    
    Args:
        gravity_file: Path to the dataset gravity analysis parquet file
        output_file: Path to save the visualization
        graph_dir: Directory containing graph files
        visuals_dir: Directory to save visualizations
        figsize: Figure size
        dpi: Output DPI
        
    Returns:
        Path: Path to the saved visualization
    """
    logger.info("Creating dataset gravity bubble chart")
    
    # Use default paths if not specified
    if gravity_file is None:
        gravity_file = graph_dir / "dataset_gravity_analysis.parquet"
    
    if output_file is None:
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_file = visuals_dir / "dataset_gravity_bubbles.png"
    
    # Load gravity data
    gravity_df = pl.read_parquet(gravity_file)
    
    # Sort by kernel count and select top N
    top_n = min(50, len(gravity_df))
    top_datasets = gravity_df.sort("kernel_count", descending=True).head(top_n)
    
    # Create bubble chart
    plt.figure(figsize=figsize)
    
    # Extract data
    datasets = top_datasets["dataset_id"].to_list()
    kernel_counts = top_datasets["kernel_count"].to_list()
    
    if "burst_score" in top_datasets.columns and not top_datasets["burst_score"].is_null().all():
        burst_scores = top_datasets["burst_score"].to_list()
        # Normalize for color mapping
        norm = Normalize(vmin=min(burst_scores), vmax=max(burst_scores))
        colors = cm.plasma(norm(burst_scores))
        scatter = plt.scatter(range(len(datasets)), kernel_counts, 
                              s=kernel_counts, c=burst_scores, cmap="plasma", 
                              alpha=0.7)
        plt.colorbar(scatter, label="Burst Score")
    else:
        plt.scatter(range(len(datasets)), kernel_counts, 
                    s=kernel_counts, c="steelblue", alpha=0.7)
    
    # Add dataset labels for top datasets
    for i, (dataset_id, count) in enumerate(zip(datasets[:15], kernel_counts[:15])):
        plt.annotate(f"DS-{dataset_id}", (i, count), 
                     xytext=(5, 5), textcoords="offset points", 
                     fontsize=8)
    
    # Add labels and title
    plt.xlabel("Dataset Rank")
    plt.ylabel("Kernel Count")
    plt.title("Dataset Gravity: Influence on Kernel Creation")
    plt.grid(alpha=0.3)
    plt.yscale("log")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    
    logger.info(f"Saved dataset gravity bubble chart to {output_file}")
    
    return output_file


def plot_library_diffusion(
    graph_file: Path = None,
    kernel_features_file: Path = None,
    output_file: Path = None,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
    library: str = "dl_pytorch",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
) -> Path:
    """
    Create a visualization of library diffusion through the fork graph.
    
    Args:
        graph_file: Path to the graph pickle file
        kernel_features_file: Path to the kernel features parquet file
        output_file: Path to save the visualization
        graph_dir: Directory containing graph files
        visuals_dir: Directory to save visualizations
        library: Library flag to analyze ('dl_pytorch', 'gbdt', etc.)
        figsize: Figure size
        dpi: Output DPI
        
    Returns:
        Path: Path to the saved visualization
    """
    logger.info(f"Creating {library} diffusion visualization")
    
    # Use default paths if not specified
    if graph_file is None:
        graph_file = graph_dir / "fork_graph.pkl"
    
    if kernel_features_file is None:
        kernel_features_file = Path("data/processed/kernel_features.parquet")
    
    if output_file is None:
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_file = visuals_dir / f"{library}_diffusion.png"
    
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    
    # Load kernel features
    kernel_features = pl.read_parquet(kernel_features_file)
    
    # Filter to kernels using the specified library
    if library in kernel_features.columns:
        adopters = kernel_features.filter(pl.col(library) == True)
        
        if len(adopters) == 0:
            logger.warning(f"No kernels found using {library}")
            return None
        
        # Sort by timestamp and get the earliest adopters
        if "kernel_ts" in adopters.columns:
            adopters = adopters.sort("kernel_ts")
            seeds = adopters.head(10)["kernel_id"].to_list()
        else:
            seeds = adopters.head(10)["kernel_id"].to_list()
        
        # For each adopter kernel, measure shortest path distance from seeds
        dists = []
        kernel_ids = []
        for kernel_id in tqdm(adopters["kernel_id"].to_list(), desc="Computing diffusion distances"):
            try:
                # Find shortest path from any seed
                shortest = float('inf')
                for seed in seeds:
                    try:
                        if nx.has_path(G, source=seed, target=kernel_id):
                            dist = nx.shortest_path_length(G, source=seed, target=kernel_id)
                            shortest = min(shortest, dist)
                    except:
                        continue
                
                if shortest < float('inf'):
                    dists.append(shortest)
                    kernel_ids.append(kernel_id)
            except:
                continue
        
        # Create histogram of fork hop distances
        plt.figure(figsize=figsize)
        plt.hist(dists, bins=range(1, max(dists) + 2), alpha=0.75, color="steelblue")
        
        # Add labels and title
        plt.xlabel("Fork Hops from Initial Adopters")
        plt.ylabel("Count")
        plt.title(f"Diffusion of {library} Through Fork Network")
        plt.grid(alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        
        logger.info(f"Saved {library} diffusion visualization to {output_file}")
        
        return output_file
    else:
        logger.warning(f"Library {library} not found in kernel features")
        return None


def create_animated_fork_network(
    graph_file: Path = None,
    kernel_features_file: Path = None,
    output_file: Path = None,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
    library: str = None,
    node_limit: int = 200,
    fps: int = 2
) -> Path:
    """
    Create an animated GIF showing the growth of the fork network over time.
    
    Args:
        graph_file: Path to the graph pickle file
        kernel_features_file: Path to the kernel features parquet file
        output_file: Path to save the animation
        graph_dir: Directory containing graph files
        visuals_dir: Directory to save visualizations
        library: Optional library flag to highlight ('dl_pytorch', 'gbdt', etc.)
        node_limit: Maximum number of nodes to include
        fps: Frames per second in the animation
        
    Returns:
        Path: Path to the saved animation
    """
    try:
        import imageio
    except ImportError:
        logger.error("imageio package not installed, cannot create animation")
        return None
    
    logger.info("Creating animated fork network visualization")
    
    # Use default paths if not specified
    if graph_file is None:
        graph_file = graph_dir / "fork_graph.pkl"
    
    if kernel_features_file is None:
        kernel_features_file = Path("data/processed/kernel_features.parquet")
    
    if output_file is None:
        visuals_dir.mkdir(parents=True, exist_ok=True)
        output_file = visuals_dir / "fork_network_growth.gif"
    
    # Create temp directory for frames
    frame_dir = visuals_dir / "temp_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    
    # Load kernel features to get timestamps
    kernel_features = pl.read_parquet(kernel_features_file)
    
    # If graph is too large, select a subgraph
    if G.number_of_nodes() > node_limit:
        # Try using PageRank if available
        try:
            pr = nx.pagerank(G, weight="weight")
            top_nodes = sorted(pr.keys(), key=lambda x: pr[x], reverse=True)[:node_limit]
            G = G.subgraph(top_nodes)
        except:
            # Fallback to degree
            deg = dict(G.degree(weight="weight"))
            top_nodes = sorted(deg.keys(), key=lambda x: deg[x], reverse=True)[:node_limit]
            G = G.subgraph(top_nodes)
    
    # Extract timestamps
    ts_map = {}
    for row in kernel_features.select(["kernel_id", "kernel_ts"]).iter_rows(named=True):
        if row["kernel_ts"] is not None:
            ts_map[row["kernel_id"]] = row["kernel_ts"]
    
    # Get years range
    years = set()
    for kernel_id in G.nodes():
        if kernel_id in ts_map:
            years.add(ts_map[kernel_id].year)
    
    if not years:
        logger.error("No year information available for nodes")
        return None
    
    min_year = min(years)
    max_year = max(years)
    
    # Calculate node positions once for consistent layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, weight="weight", seed=42)
    
    # Generate frames for each year
    frames = []
    for year in range(min_year, max_year + 1):
        logger.info(f"Creating frame for year {year}")
        
        # Create subgraph of nodes up to this year
        nodes = []
        for node in G.nodes():
            if node in ts_map and ts_map[node].year <= year:
                nodes.append(node)
        
        if not nodes:
            continue
            
        H = G.subgraph(nodes)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Draw nodes
        if library and library in kernel_features.columns:
            # Color nodes based on library usage
            lib_kernels = set(kernel_features.filter(pl.col(library) == True)["kernel_id"].to_list())
            node_colors = ["red" if node in lib_kernels else "skyblue" for node in H.nodes()]
            nx.draw_networkx_nodes(H, pos, node_size=50, node_color=node_colors, alpha=0.7)
        else:
            nx.draw_networkx_nodes(H, pos, node_size=50, node_color="skyblue", alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(H, pos, width=0.5, alpha=0.4, arrows=True, arrowsize=5)
        
        plt.title(f"Fork Network Growth ≤ {year}")
        plt.axis("off")
        plt.tight_layout()
        
        # Save frame
        frame_file = frame_dir / f"fork_{year}.png"
        plt.savefig(frame_file, dpi=100)
        plt.close()
        
        frames.append(imageio.imread(frame_file))
    
    # Create animated GIF
    imageio.mimsave(output_file, frames, fps=fps)
    
    logger.info(f"Saved animated fork network to {output_file}")
    
    # Clean up temporary frames
    for f in frame_dir.glob("*.png"):
        f.unlink()
    
    return output_file