"""
Hierarchical clustering using engineered features with cosine distance
and average linkage.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from features import compute_features


def build_linkage(
    feature_df: pd.DataFrame, metric: str = "cosine", method: str = "average"
) -> Tuple[np.ndarray, list]:
    """
    Compute linkage matrix from a feature dataframe (tickers x features).
    """
    standardized = (feature_df - feature_df.mean()) / feature_df.std(ddof=0)
    standardized = standardized.fillna(0.0)
    distances = pdist(standardized.values, metric=metric)
    linkage_matrix = hierarchy.linkage(distances, method=method)
    labels = list(feature_df.index)
    return linkage_matrix, labels


def compute_clusters(
    feature_df: pd.DataFrame, distance_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Assign cluster labels based on a distance threshold.
    """
    linkage_matrix, labels = build_linkage(feature_df)
    cluster_ids = hierarchy.fcluster(
        linkage_matrix, t=distance_threshold, criterion="distance"
    )
    return pd.DataFrame({"ticker": labels, "cluster": cluster_ids})


def load_and_cluster(db_path: Optional[str] = None, distance_threshold: float = 0.5):
    feats = compute_features(db_path)
    linkage_matrix, labels = build_linkage(feats)
    clusters = compute_clusters(feats, distance_threshold=distance_threshold)
    return feats, linkage_matrix, labels, clusters


if __name__ == "__main__":
    feats, _, _, clusters = load_and_cluster()
    print(clusters.sort_values("cluster"))


