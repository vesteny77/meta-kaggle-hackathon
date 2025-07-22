# Meta Kaggle Hackathon Visualizations

This folder contains self-contained Python scripts that generate interactive visualizations for the Meta Kaggle Hackathon project. Each script can be run independently without additional dependencies (except for the Python libraries imported within each script).

## Visualizations

1. **`1_streamgraph_library_adoption.py`**: Generates a streamgraph visualization showing the evolution of machine learning library adoption on Kaggle from 2010 to 2025.

2. **`2_heatmap_compute_roi.py`**: Creates a heatmap visualization showing how compute ROI (score improvement per execution time) has evolved over the years.

3. **`3_sankey_methods_competitions.py`**: Produces a Sankey diagram showing which machine learning methods tend to win which types of competitions.

4. **`4_choropleth_team_diversity.py`**: Visualizes the relationship between team diversity (entropy) and medal rates across different countries.

5. **`5_animated_fork_network.py`**: Generates an animated visualization showing the spread of knowledge through fork networks over time.

6. **`6_library_adoption_by_domain.py`**: Compares the adoption trends of different libraries across Computer Vision, NLP, and Tabular domains.

## Requirements

These scripts require the following Python libraries:
- plotly
- pandas
- numpy
- networkx (for the fork network visualization)

## Usage

Each script can be run independently:

```
python 1_streamgraph_library_adoption.py
```

Each script will:
1. Generate synthetic data that mimics realistic ML trends over time
2. Create an interactive visualization
3. Display the visualization in your default browser
4. Save the visualization as an HTML file in the same directory

The HTML files can be opened in any modern web browser for interactive exploration of the data.

## Notes

- All visualizations use synthetic data that simulates real-world trends in machine learning adoption and evolution on Kaggle.
- The interactive HTML files allow for exploration, hovering over elements for more information, and in some cases (like the animated fork network) playing animations.