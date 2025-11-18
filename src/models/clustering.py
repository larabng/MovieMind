"""
Clustering Analysis for MovieMind
Performs k-means clustering on movies/reviews to find similar patterns
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieClusterer:
    """Performs clustering analysis on movies and reviews"""

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize clusterer

        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=500, max_df=0.8, min_df=2)
        self.pca = None
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame,
                        text_column: str = None,
                        numeric_columns: List[str] = None) -> np.ndarray:
        """
        Prepare features for clustering

        Args:
            df: Input dataframe
            text_column: Column containing text (optional)
            numeric_columns: List of numeric columns to use

        Returns:
            Feature matrix
        """
        features = []

        # Text features
        if text_column and text_column in df.columns:
            logger.info(f"Extracting TF-IDF features from {text_column}")
            text_data = df[text_column].fillna('')
            text_features = self.vectorizer.fit_transform(text_data)
            features.append(text_features.toarray())

        # Numeric features
        if numeric_columns:
            logger.info(f"Scaling numeric features: {numeric_columns}")
            numeric_data = df[numeric_columns].fillna(0)
            numeric_features = self.scaler.fit_transform(numeric_data)
            features.append(numeric_features)

        # Combine features
        if len(features) == 0:
            raise ValueError("No features provided for clustering")
        elif len(features) == 1:
            return features[0]
        else:
            return np.hstack(features)

    def fit(self, X: np.ndarray) -> Dict[str, float]:
        """
        Fit k-means clustering

        Args:
            X: Feature matrix

        Returns:
            Dictionary with clustering metrics
        """
        logger.info(f"Fitting k-means with {self.n_clusters} clusters")

        # Fit k-means
        self.kmeans.fit(X)
        self.is_fitted = True

        # Calculate metrics
        labels = self.kmeans.labels_
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        inertia = self.kmeans.inertia_

        metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'inertia': inertia,
            'n_clusters': self.n_clusters
        }

        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        logger.info(f"Inertia: {inertia:.2f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels

        Args:
            X: Feature matrix

        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.kmeans.predict(X)

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fit and predict in one step

        Args:
            X: Feature matrix

        Returns:
            Tuple of (cluster labels, metrics)
        """
        metrics = self.fit(X)
        labels = self.kmeans.labels_
        return labels, metrics

    def elbow_analysis(self, X: np.ndarray, max_k: int = 10) -> pd.DataFrame:
        """
        Perform elbow analysis to find optimal k

        Args:
            X: Feature matrix
            max_k: Maximum number of clusters to test

        Returns:
            DataFrame with k and corresponding metrics
        """
        logger.info(f"Performing elbow analysis for k=1 to {max_k}")

        results = []

        for k in range(1, max_k + 1):
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans_temp.fit(X)

            metrics = {
                'k': k,
                'inertia': kmeans_temp.inertia_
            }

            # Silhouette score (not defined for k=1)
            if k > 1:
                silhouette = silhouette_score(X, kmeans_temp.labels_)
                metrics['silhouette'] = silhouette
            else:
                metrics['silhouette'] = None

            results.append(metrics)

        return pd.DataFrame(results)

    def plot_elbow(self, elbow_results: pd.DataFrame):
        """
        Plot elbow curve

        Args:
            elbow_results: DataFrame from elbow_analysis()
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Inertia plot
        axes[0].plot(elbow_results['k'], elbow_results['inertia'], marker='o')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method - Inertia')
        axes[0].grid(True, alpha=0.3)

        # Silhouette plot
        silhouette_data = elbow_results[elbow_results['silhouette'].notna()]
        axes[1].plot(silhouette_data['k'], silhouette_data['silhouette'], marker='o', color='green')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Elbow Method - Silhouette Score')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def visualize_clusters_2d(self, X: np.ndarray, labels: np.ndarray,
                              title: str = "K-Means Clustering Visualization"):
        """
        Visualize clusters in 2D using PCA

        Args:
            X: Feature matrix
            labels: Cluster labels
            title: Plot title
        """
        # Apply PCA for 2D visualization
        if self.pca is None:
            self.pca = PCA(n_components=2, random_state=self.random_state)

        X_pca = self.pca.fit_transform(X)

        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis',
                            alpha=0.6, edgecolors='w', linewidth=0.5)

        # Plot cluster centers
        if hasattr(self.kmeans, 'cluster_centers_'):
            centers_pca = self.pca.transform(self.kmeans.cluster_centers_)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                       marker='X', s=300, c='red', edgecolors='black',
                       linewidth=2, label='Centroids')

        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(title)
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_cluster_summary(self, df: pd.DataFrame, labels: np.ndarray,
                           numeric_columns: List[str] = None) -> pd.DataFrame:
        """
        Get summary statistics for each cluster

        Args:
            df: Original dataframe
            labels: Cluster labels
            numeric_columns: Columns to summarize

        Returns:
            DataFrame with cluster summaries
        """
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels

        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        summary = df_with_clusters.groupby('cluster')[numeric_columns].agg(['mean', 'std', 'count'])

        return summary

    def get_top_terms_per_cluster(self, labels: np.ndarray, top_n: int = 10) -> Dict[int, List[str]]:
        """
        Get top TF-IDF terms for each cluster

        Args:
            labels: Cluster labels
            top_n: Number of top terms to return

        Returns:
            Dictionary mapping cluster to top terms
        """
        if not hasattr(self.vectorizer, 'get_feature_names_out'):
            logger.warning("Vectorizer not fitted. Cannot extract top terms.")
            return {}

        feature_names = self.vectorizer.get_feature_names_out()

        # Get cluster centers from k-means
        centers = self.kmeans.cluster_centers_

        # Assuming text features are first in the feature matrix
        n_text_features = len(feature_names)

        cluster_terms = {}

        for cluster_id in range(self.n_clusters):
            # Get text features for this cluster center
            center_text = centers[cluster_id][:n_text_features]

            # Get top terms
            top_indices = center_text.argsort()[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]

            cluster_terms[cluster_id] = top_terms

        return cluster_terms


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_blobs

    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=5,
                          random_state=42, cluster_std=1.5)

    # Create dataframe
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['text'] = [f"sample text {i} " * (i % 5 + 1) for i in range(300)]

    # Initialize clusterer
    clusterer = MovieClusterer(n_clusters=4)

    # Prepare features
    features = clusterer.prepare_features(
        df,
        text_column='text',
        numeric_columns=['feature_0', 'feature_1', 'feature_2']
    )

    # Elbow analysis
    elbow_results = clusterer.elbow_analysis(features, max_k=10)
    print("\nElbow Analysis Results:")
    print(elbow_results)

    # Plot elbow
    clusterer.plot_elbow(elbow_results)

    # Fit and predict
    labels, metrics = clusterer.fit_predict(features)

    print("\nClustering Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Visualize
    clusterer.visualize_clusters_2d(features, labels)

    # Cluster summary
    summary = clusterer.get_cluster_summary(df, labels, ['feature_0', 'feature_1'])
    print("\nCluster Summary:")
    print(summary)
