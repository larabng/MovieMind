"""
Rename extracted plots with meaningful names based on content
"""
import os
import shutil

# Define mappings: (old_name, new_name)
plot_mappings = {
    '01_exploratory_analysis': [
        ('cell_08_plot_01.png', '01_rating_distribution.png'),
        ('cell_09_plot_00.png', '02_runtime_distribution.png'),
        ('cell_10_plot_00.png', '03_review_length_distribution.png'),
        ('cell_12_plot_00.png', '04_correlation_heatmap.png'),
        ('cell_13_plot_01.png', '05_runtime_vs_rating.png'),
        ('cell_14_plot_00.png', '06_budget_vs_revenue.png'),
        ('cell_16_plot_00.png', '07_genre_distribution.png'),
        ('cell_19_plot_00.png', '08_movies_per_year.png'),
        ('cell_20_plot_00.png', '09_rating_over_time.png'),
    ],
    '02_model_training': [
        ('cell_08_plot_01.png', '10_sentiment_distribution.png'),
        ('cell_12_plot_01.png', '11_confusion_matrix.png'),
        ('cell_18_plot_00.png', '12_residual_plots.png'),
        ('cell_22_plot_04.png', '13_elbow_plot.png'),
        ('cell_24_plot_00.png', '14_cluster_visualization_2d.png'),
        ('cell_25_plot_00.png', '15_cluster_distribution.png'),
    ],
    '03_clustering_analysis': [
        ('cell_11_plot_00.png', '16_elbow_analysis.png'),
        ('cell_16_plot_00.png', '17_cluster_pca.png'),
        ('cell_20_plot_00.png', '18_cluster_counts.png'),
        ('cell_23_plot_00.png', '19_cluster_boxplot.png'),
    ],
    '04_geo_visualization': [
        ('cell_13_plot_00.png', '20_geographic_distribution.png'),
    ]
}

def rename_plots():
    """Rename all plots with meaningful names"""

    base_dir = 'presentation/screenshots/plots'
    renamed_dir = 'presentation/screenshots/renamed'

    # Create renamed directory
    os.makedirs(renamed_dir, exist_ok=True)

    print("=" * 60)
    print("RENAMING PLOTS")
    print("=" * 60)

    total_renamed = 0

    for notebook_folder, mappings in plot_mappings.items():
        source_folder = os.path.join(base_dir, notebook_folder)

        if not os.path.exists(source_folder):
            print(f"[SKIP] Folder not found: {source_folder}")
            continue

        print(f"\nProcessing: {notebook_folder}")
        print("-" * 60)

        for old_name, new_name in mappings:
            old_path = os.path.join(source_folder, old_name)
            new_path = os.path.join(renamed_dir, new_name)

            if os.path.exists(old_path):
                shutil.copy2(old_path, new_path)
                print(f"  [OK] {old_name} -> {new_name}")
                total_renamed += 1
            else:
                print(f"  [SKIP] Not found: {old_name}")

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_renamed} plots renamed and saved to {renamed_dir}")
    print("=" * 60)

    # Create index file
    create_index(renamed_dir)

def create_index(renamed_dir):
    """Create an index file listing all plots"""

    index_content = """# MovieMind - Extracted Plots Index

## Statistical Analysis & EDA (9 plots)

### Distribution Plots
- `01_rating_distribution.png` - Histogram and boxplot of movie ratings
- `02_runtime_distribution.png` - Distribution of movie runtime
- `03_review_length_distribution.png` - Distribution of review text length

### Correlation Analysis
- `04_correlation_heatmap.png` ⭐ - Correlation matrix of movie features (vote_average, budget, revenue, etc.)
- `05_runtime_vs_rating.png` - Scatter plot showing relationship between runtime and rating

### Business Analysis
- `06_budget_vs_revenue.png` - Log-scale plot showing ROI analysis

### Genre & Temporal Analysis
- `07_genre_distribution.png` - Bar chart of top 15 genres
- `08_movies_per_year.png` - Trend of movies released over time
- `09_rating_over_time.png` - Average rating trends over years

## Machine Learning Results (11 plots)

### Sentiment Classification
- `10_sentiment_distribution.png` - Distribution of sentiment labels (positive/neutral/negative)
- `11_confusion_matrix.png` ⭐ - Sentiment classification confusion matrix

### Regression (Score Prediction)
- `12_residual_plots.png` ⭐ - Residual plots and predicted vs actual

### K-Means Clustering (Notebook 02)
- `13_elbow_plot.png` ⭐ - Elbow method from model training notebook
- `14_cluster_visualization_2d.png` ⭐ - 2D PCA visualization from model training
- `15_cluster_distribution.png` - Bar chart of cluster sizes

### K-Means Clustering (Notebook 03 - Detailed Analysis)
- `16_elbow_analysis.png` - Elbow method with silhouette scores
- `17_cluster_pca.png` - PCA visualization of clusters
- `18_cluster_counts.png` - Cluster distribution bar chart
- `19_cluster_boxplot.png` - Boxplot of scores by cluster

### Geographic Analysis
- `20_geographic_distribution.png` - Geographic distribution of movies

---

## Usage for Presentation

### For Statistical Tests Section:
- ⭐ Use: `04_correlation_heatmap.png`

### For K-Means Clustering:
- ⭐ Use: `13_elbow_plot.png` or `16_elbow_analysis.png` (shows optimal k)
- ⭐ Use: `14_cluster_visualization_2d.png` or `17_cluster_pca.png` (shows actual clusters)

### For Confusion Matrix:
- ⭐ Use: `11_confusion_matrix.png`

### For Regression Diagnostics:
- ⭐ Use: `12_residual_plots.png`

### For EDA:
- Use: `01_rating_distribution.png`, `07_genre_distribution.png`, `08_movies_per_year.png`

---

## Total: 20 plots extracted and renamed
"""

    index_path = os.path.join(renamed_dir, 'INDEX.md')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(f"\n[OK] Index file created: {index_path}")

if __name__ == "__main__":
    rename_plots()
