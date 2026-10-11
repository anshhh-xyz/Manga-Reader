"""
Speaker Attribution Package for Manga Task Reader
=================================================
Primary Model: Enhanced Spatial Layout Clusterer (Option 3 - Winner)
Pairwise XGBoost affinity + Constrained Agglomerative Clustering (balanced_joint_f1 = 0.3509)
"""

from .enhanced_spatial_clusterer import cluster_assign, assign_speakers

__all__ = ["cluster_assign", "assign_speakers"]
