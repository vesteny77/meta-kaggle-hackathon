"""
Pipeline node function definitions for graph analytics.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from kedro.pipeline import Pipeline, node

from src.graphs.core import (
    prepare_fork_edges,
    prepare_competition_kernel_edges,
    prepare_node_attributes,
    prepare_team_collaboration_edges,
    prepare_dataset_gravity_edges,
    prepare_forum_code_edges,
    build_fork_graph,
    build_competition_kernel_graph,
    build_team_collaboration_graph,
    build_dataset_gravity_graph,
    build_forum_code_graph,
    compute_pagerank,
    calculate_fork_velocity,
    analyze_dataset_gravity,
    export_graph_for_visualization
)

from src.graphs.visualization import (
    plot_fork_graph,
    plot_competition_kernel_graph,
    plot_fork_velocity_histogram,
    plot_dataset_gravity_bubbles,
    plot_library_diffusion,
    create_animated_fork_network
)


logger = logging.getLogger(__name__)


def prepare_graph_edges_node(params: Dict[str, Any]) -> Dict[str, Path]:
    """
    Node function for preparing all graph edge tables.
    
    Args:
        params: Pipeline parameters
        
    Returns:
        Dict[str, Path]: Dictionary of paths to edge files
    """
    output_dir = Path(params.get("output_dir", "data/graph"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare edge tables for different graph types
    fork_edges = prepare_fork_edges(output_dir)
    ck_edges = prepare_competition_kernel_edges(output_dir)
    kernel_nodes, comp_nodes = prepare_node_attributes(output_dir)
    tc_edges = prepare_team_collaboration_edges(output_dir)
    dg_edges = prepare_dataset_gravity_edges(output_dir)
    fc_edges = prepare_forum_code_edges(output_dir)
    
    return {
        "fork_edges": fork_edges,
        "ck_edges": ck_edges,
        "kernel_nodes": kernel_nodes,
        "comp_nodes": comp_nodes,
        "tc_edges": tc_edges,
        "dg_edges": dg_edges,
        "fc_edges": fc_edges
    }


def build_graphs_node(edge_files: Dict[str, Path], params: Dict[str, Any]) -> Dict[str, Path]:
    """
    Node function for building all graph objects.
    
    Args:
        edge_files: Dictionary of paths to edge files
        params: Pipeline parameters
        
    Returns:
        Dict[str, Path]: Dictionary of paths to graph files
    """
    output_dir = Path(params.get("output_dir", "data/graph"))
    
    # Build graphs
    fork_graph_file = output_dir / "fork_graph.pkl"
    fork_graph = build_fork_graph(
        edge_file=edge_files["fork_edges"],
        node_file=edge_files["kernel_nodes"],
        output_file=fork_graph_file
    )
    
    ck_graph_file = output_dir / "ck_graph.pkl"
    ck_graph = build_competition_kernel_graph(
        edge_file=edge_files["ck_edges"],
        kernel_node_file=edge_files["kernel_nodes"],
        comp_node_file=edge_files["comp_nodes"],
        output_file=ck_graph_file
    )
    
    tc_graph_file = output_dir / "team_collab_graph.pkl"
    tc_graph = build_team_collaboration_graph(
        edge_file=edge_files["tc_edges"],
        output_file=tc_graph_file
    )
    
    dg_graph_file = output_dir / "dataset_gravity_graph.pkl"
    dg_graph = build_dataset_gravity_graph(
        edge_file=edge_files["dg_edges"],
        output_file=dg_graph_file
    )
    
    fc_graph_file = output_dir / "forum_code_graph.pkl"
    fc_graph = build_forum_code_graph(
        edge_file=edge_files["fc_edges"],
        output_file=fc_graph_file
    )
    
    return {
        "fork_graph": fork_graph_file,
        "ck_graph": ck_graph_file,
        "tc_graph": tc_graph_file,
        "dg_graph": dg_graph_file,
        "fc_graph": fc_graph_file
    }


def compute_graph_metrics_node(graph_files: Dict[str, Path], params: Dict[str, Any]) -> Dict[str, Path]:
    """
    Node function for computing various graph metrics.
    
    Args:
        graph_files: Dictionary of paths to graph files
        params: Pipeline parameters
        
    Returns:
        Dict[str, Path]: Dictionary of paths to metric files
    """
    output_dir = Path(params.get("output_dir", "data/graph"))
    
    # Compute PageRank
    pagerank_file = compute_pagerank(
        graph_file=graph_files["fork_graph"],
        output_dir=output_dir
    )
    
    # Calculate fork velocity
    velocity_file = calculate_fork_velocity(
        edge_file=output_dir / "fork_edges.parquet",
        output_dir=output_dir
    )
    
    # Analyze dataset gravity
    gravity_file = analyze_dataset_gravity(
        graph_file=graph_files["dg_graph"],
        output_dir=output_dir
    )
    
    # Export graphs for visualization
    viz_fork_graph = export_graph_for_visualization(
        graph_file=graph_files["fork_graph"],
        output_file=output_dir / "fork_graph_viz.json",
        node_limit=500
    )
    
    viz_ck_graph = export_graph_for_visualization(
        graph_file=graph_files["ck_graph"],
        output_file=output_dir / "ck_graph_viz.json",
        node_limit=500
    )
    
    return {
        "pagerank": pagerank_file,
        "fork_velocity": velocity_file,
        "dataset_gravity": gravity_file,
        "viz_fork_graph": viz_fork_graph,
        "viz_ck_graph": viz_ck_graph
    }


def create_visualizations_node(
    graph_files: Dict[str, Path],
    metric_files: Dict[str, Path],
    params: Dict[str, Any]
) -> Dict[str, Path]:
    """
    Node function for creating graph visualizations.
    
    Args:
        graph_files: Dictionary of paths to graph files
        metric_files: Dictionary of paths to metric files
        params: Pipeline parameters
        
    Returns:
        Dict[str, Path]: Dictionary of paths to visualization files
    """
    output_dir = Path(params.get("visuals_dir", "visuals"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    fork_graph_viz = plot_fork_graph(
        graph_file=graph_files["fork_graph"],
        output_file=output_dir / "fork_graph.png"
    )
    
    ck_graph_viz = plot_competition_kernel_graph(
        graph_file=graph_files["ck_graph"],
        output_file=output_dir / "competition_kernel_graph.png"
    )
    
    velocity_hist = plot_fork_velocity_histogram(
        velocity_file=metric_files["fork_velocity"],
        output_file=output_dir / "fork_velocity_histogram.png"
    )
    
    gravity_viz = plot_dataset_gravity_bubbles(
        gravity_file=metric_files["dataset_gravity"],
        output_file=output_dir / "dataset_gravity_bubbles.png"
    )
    
    # Visualize library diffusion for PyTorch and GBDT
    pytorch_diffusion = plot_library_diffusion(
        graph_file=graph_files["fork_graph"],
        output_file=output_dir / "pytorch_diffusion.png",
        library="dl_pytorch"
    )
    
    gbdt_diffusion = plot_library_diffusion(
        graph_file=graph_files["fork_graph"],
        output_file=output_dir / "gbdt_diffusion.png",
        library="gbdt"
    )
    
    # Create animation of fork network growth
    fork_animation = create_animated_fork_network(
        graph_file=graph_files["fork_graph"],
        output_file=output_dir / "fork_network_growth.gif"
    )
    
    return {
        "fork_graph_viz": fork_graph_viz,
        "ck_graph_viz": ck_graph_viz,
        "velocity_hist": velocity_hist,
        "gravity_viz": gravity_viz,
        "pytorch_diffusion": pytorch_diffusion,
        "gbdt_diffusion": gbdt_diffusion,
        "fork_animation": fork_animation
    }


def create_pipeline(**kwargs) -> Pipeline:
    """Create the graph analytics pipeline."""
    return Pipeline(
        [
            node(
                prepare_graph_edges_node,
                inputs=["params:graphs"],
                outputs="graph_edge_files",
                name="prepare_graph_edges",
            ),
            node(
                build_graphs_node,
                inputs=["graph_edge_files", "params:graphs"],
                outputs="graph_files",
                name="build_graphs",
            ),
            node(
                compute_graph_metrics_node,
                inputs=["graph_files", "params:graphs"],
                outputs="graph_metrics",
                name="compute_graph_metrics",
            ),
            node(
                create_visualizations_node,
                inputs=["graph_files", "graph_metrics", "params:graphs"],
                outputs="graph_visualizations",
                name="create_graph_visualizations",
            ),
        ]
    )