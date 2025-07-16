"""
Semantic Feature Selection for Quantum Encoding (QRF-02).

This module implements intelligent feature selection methods that preserve semantic
information when reducing high-dimensional embeddings for quantum encoding.

Key Goals:
1. Reduce 768D embeddings to quantum-compatible dimensions (16-64 features)
2. Preserve semantic relationships and discrimination
3. Integrate with existing mRMR implementation
4. Support multiple selection strategies for quantum encoding

Based on:
- Task QRF-02: Fix Amplitude Encoding Discrimination
- Paper: "Quantum-inspired Embeddings Projection and Similarity Metrics"
- Paper: "Quantum Embedding Search for Quantum Machine Learning"
- Existing quantum_feature_selection.py mRMR implementation
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import time

# Import existing mRMR implementation
from .quantum_feature_selection import QuantumFeatureSelector, QuantumFeatureSelectionConfig

logger = logging.getLogger(__name__)


@dataclass
class SemanticSelectionConfig:
    """Configuration for semantic feature selection."""
    method: str = "hybrid"  # hybrid, pca, variance, mrmr, correlation_clustering, ranking_aware
    target_features: int = 16  # Target number of features for quantum encoding
    preserve_local_structure: bool = True  # Preserve local embedding structure
    correlation_threshold: float = 0.85  # For correlation-based clustering
    variance_threshold: float = 0.01  # Minimum variance for feature retention
    pca_explained_variance: float = 0.95  # PCA variance retention target
    semantic_preservation_weight: float = 0.7  # Weight for semantic vs compression
    use_sliding_window: bool = True  # Use sliding window for local structure
    window_overlap: float = 0.5  # Overlap ratio for sliding windows
    ranking_preservation_weight: float = 0.8  # Weight for ranking preservation in selection


@dataclass
class SelectionResult:
    """Result from semantic feature selection."""
    success: bool
    selected_indices: Optional[np.ndarray] = None
    selected_features: Optional[np.ndarray] = None
    feature_scores: Optional[np.ndarray] = None
    information_preservation: float = 0.0
    semantic_preservation: float = 0.0
    method_used: str = ""
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


class SemanticFeatureSelector:
    """
    Intelligent feature selection for quantum encoding that preserves semantic information.
    
    Integrates with existing mRMR and provides new methods optimized for semantic preservation
    in quantum state encoding.
    """
    
    def __init__(self, config: Optional[SemanticSelectionConfig] = None):
        """Initialize semantic feature selector."""
        self.config = config or SemanticSelectionConfig()
        self.selected_indices_ = None
        self.feature_scores_ = None
        self.scaler_ = None
        self.pca_ = None
        
        # Initialize mRMR selector for integration
        mrmr_config = QuantumFeatureSelectionConfig(
            method="mrmr",
            num_features=self.config.target_features,
            normalize_features=True
        )
        self.mrmr_selector = QuantumFeatureSelector(mrmr_config)
        
        logger.info(f"Initialized SemanticFeatureSelector: {self.config.method}, target={self.config.target_features}")
    
    def fit_transform(self, embeddings: np.ndarray, 
                     labels: Optional[np.ndarray] = None) -> SelectionResult:
        """
        Fit selector and transform embeddings.
        
        Args:
            embeddings: Input embeddings [n_samples, n_features]
            labels: Optional labels for supervised selection
            
        Returns:
            SelectionResult with selected features and metadata
        """
        return self.fit(embeddings, labels).transform(embeddings)
    
    def fit(self, embeddings: np.ndarray, 
            labels: Optional[np.ndarray] = None) -> 'SemanticFeatureSelector':
        """
        Fit the semantic feature selector.
        
        Args:
            embeddings: Input embeddings [n_samples, n_features]
            labels: Optional labels for supervised selection
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        logger.info(f"Fitting semantic feature selector: {embeddings.shape} -> {self.config.target_features} features")
        
        # Validate inputs
        if embeddings.shape[1] <= self.config.target_features:
            logger.warning(f"Input dimensions ({embeddings.shape[1]}) <= target ({self.config.target_features})")
            self.selected_indices_ = np.arange(embeddings.shape[1])
            return self
        
        # Standardize features for better selection
        self.scaler_ = StandardScaler()
        embeddings_scaled = self.scaler_.fit_transform(embeddings)
        
        # Select method and perform feature selection
        if self.config.method == "pca":
            result = self._pca_selection(embeddings_scaled)
        elif self.config.method == "variance":
            result = self._variance_based_selection(embeddings_scaled)
        elif self.config.method == "mrmr":
            result = self._mrmr_selection(embeddings_scaled, labels)
        elif self.config.method == "correlation_clustering":
            result = self._correlation_clustering_selection(embeddings_scaled)
        elif self.config.method == "ranking_aware":
            result = self._ranking_aware_selection(embeddings_scaled)
        elif self.config.method == "hybrid":
            result = self._hybrid_selection(embeddings_scaled, labels)
        else:
            raise ValueError(f"Unknown selection method: {self.config.method}")
        
        if result.success:
            self.selected_indices_ = result.selected_indices
            self.feature_scores_ = result.feature_scores
            
            fit_time = time.time() - start_time
            logger.info(f"Feature selection completed in {fit_time:.2f}s, "
                       f"preservation: {result.information_preservation:.3f}")
        else:
            logger.error(f"Feature selection failed: {result.error}")
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> SelectionResult:
        """
        Transform embeddings using fitted selector.
        
        Args:
            embeddings: Input embeddings to transform
            
        Returns:
            SelectionResult with selected features
        """
        if self.selected_indices_ is None:
            return SelectionResult(
                success=False,
                error="Selector not fitted. Call fit() first."
            )
        
        try:
            # Scale using fitted scaler
            embeddings_scaled = self.scaler_.transform(embeddings)
            
            # Select features
            selected_features = embeddings_scaled[:, self.selected_indices_]
            
            # Calculate information preservation
            original_var = np.var(embeddings_scaled, axis=0).sum()
            selected_var = np.var(selected_features, axis=0).sum()
            info_preservation = selected_var / original_var if original_var > 0 else 0.0
            
            return SelectionResult(
                success=True,
                selected_indices=self.selected_indices_,
                selected_features=selected_features,
                feature_scores=self.feature_scores_,
                information_preservation=info_preservation,
                method_used=self.config.method,
                metadata={
                    'original_shape': embeddings.shape,
                    'selected_shape': selected_features.shape,
                    'selection_ratio': len(self.selected_indices_) / embeddings.shape[1]
                }
            )
            
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            return SelectionResult(
                success=False,
                error=str(e)
            )
    
    def _pca_selection(self, embeddings: np.ndarray) -> SelectionResult:
        """PCA-based feature selection preserving maximum variance."""
        try:
            # Fit PCA with target components
            self.pca_ = PCA(n_components=self.config.target_features)
            pca_features = self.pca_.fit_transform(embeddings)
            
            # Get component loadings to identify most important original features
            components = np.abs(self.pca_.components_)
            
            # Select features with highest loadings across all components
            feature_importance = np.sum(components, axis=0)
            selected_indices = np.argsort(feature_importance)[-self.config.target_features:]
            
            # Calculate information preservation
            explained_variance_ratio = np.sum(self.pca_.explained_variance_ratio_)
            
            return SelectionResult(
                success=True,
                selected_indices=selected_indices,
                feature_scores=feature_importance,
                information_preservation=explained_variance_ratio,
                method_used="pca",
                metadata={
                    'explained_variance_ratio': explained_variance_ratio,
                    'n_components': self.config.target_features,
                    'pca_shape': pca_features.shape
                }
            )
            
        except Exception as e:
            logger.error(f"PCA selection failed: {e}")
            return SelectionResult(success=False, error=str(e))
    
    def _variance_based_selection(self, embeddings: np.ndarray) -> SelectionResult:
        """Variance-based selection with local structure preservation."""
        try:
            if self.config.use_sliding_window:
                # Sliding window approach to preserve local structure
                selected_indices = self._sliding_window_selection(embeddings)
            else:
                # Global variance-based selection
                feature_variances = np.var(embeddings, axis=0)
                selected_indices = np.argsort(feature_variances)[-self.config.target_features:]
            
            # Calculate scores and preservation
            feature_variances = np.var(embeddings, axis=0)
            selected_var = np.sum(feature_variances[selected_indices])
            total_var = np.sum(feature_variances)
            info_preservation = selected_var / total_var if total_var > 0 else 0.0
            
            return SelectionResult(
                success=True,
                selected_indices=selected_indices,
                feature_scores=feature_variances,
                information_preservation=info_preservation,
                method_used="variance",
                metadata={
                    'use_sliding_window': self.config.use_sliding_window,
                    'variance_preservation': info_preservation,
                    'mean_variance': np.mean(feature_variances[selected_indices])
                }
            )
            
        except Exception as e:
            logger.error(f"Variance selection failed: {e}")
            return SelectionResult(success=False, error=str(e))
    
    def _sliding_window_selection(self, embeddings: np.ndarray) -> np.ndarray:
        """Select features using sliding window to preserve local structure."""
        n_features = embeddings.shape[1]
        window_size = n_features // self.config.target_features
        overlap_size = int(window_size * self.config.window_overlap)
        step_size = max(1, window_size - overlap_size)
        
        selected_indices = []
        
        for i in range(0, n_features - window_size + 1, step_size):
            window_end = min(i + window_size, n_features)
            window_features = embeddings[:, i:window_end]
            
            # Select best feature from this window
            window_variances = np.var(window_features, axis=0)
            best_local_idx = np.argmax(window_variances)
            global_idx = i + best_local_idx
            
            if global_idx not in selected_indices:
                selected_indices.append(global_idx)
            
            if len(selected_indices) >= self.config.target_features:
                break
        
        # Fill remaining slots with highest variance features if needed
        if len(selected_indices) < self.config.target_features:
            all_variances = np.var(embeddings, axis=0)
            remaining_indices = [i for i in range(n_features) if i not in selected_indices]
            remaining_variances = [(i, all_variances[i]) for i in remaining_indices]
            remaining_variances.sort(key=lambda x: x[1], reverse=True)
            
            for i, _ in remaining_variances:
                selected_indices.append(i)
                if len(selected_indices) >= self.config.target_features:
                    break
        
        return np.array(selected_indices[:self.config.target_features])
    
    def _mrmr_selection(self, embeddings: np.ndarray, 
                       labels: Optional[np.ndarray]) -> SelectionResult:
        """mRMR selection using existing implementation."""
        try:
            if labels is None:
                # Use unsupervised approach - create pseudo-labels based on clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(10, embeddings.shape[0] // 10), random_state=42)
                labels = kmeans.fit_predict(embeddings)
            
            # Use existing mRMR selector
            selected_embeddings = self.mrmr_selector.fit_transform(embeddings, labels)
            selected_indices = self.mrmr_selector.selected_features_
            
            # Calculate information preservation
            original_var = np.var(embeddings, axis=0).sum()
            selected_var = np.var(selected_embeddings, axis=0).sum()
            info_preservation = selected_var / original_var if original_var > 0 else 0.0
            
            return SelectionResult(
                success=True,
                selected_indices=selected_indices,
                feature_scores=self.mrmr_selector.feature_scores_,
                information_preservation=info_preservation,
                method_used="mrmr",
                metadata={
                    'mrmr_scores': self.mrmr_selector.feature_scores_,
                    'labels_used': labels is not None,
                    'selection_history': self.mrmr_selector.selection_history_
                }
            )
            
        except Exception as e:
            logger.error(f"mRMR selection failed: {e}")
            return SelectionResult(success=False, error=str(e))
    
    def _correlation_clustering_selection(self, embeddings: np.ndarray) -> SelectionResult:
        """Select features by clustering highly correlated features."""
        try:
            # Compute feature correlation matrix
            correlation_matrix = np.corrcoef(embeddings.T)
            
            # Identify clusters of highly correlated features
            clusters = []
            used_features = set()
            
            for i in range(correlation_matrix.shape[0]):
                if i in used_features:
                    continue
                
                # Find features correlated with feature i
                correlated = np.where(
                    np.abs(correlation_matrix[i]) > self.config.correlation_threshold
                )[0]
                
                if len(correlated) > 1:
                    clusters.append(correlated.tolist())
                    used_features.update(correlated)
                else:
                    clusters.append([i])
                    used_features.add(i)
            
            # Select representative feature from each cluster
            selected_indices = []
            feature_scores = np.zeros(embeddings.shape[1])
            
            for cluster in clusters:
                if len(selected_indices) >= self.config.target_features:
                    break
                
                # Select feature with highest variance in cluster
                cluster_variances = np.var(embeddings[:, cluster], axis=0)
                best_idx = cluster[np.argmax(cluster_variances)]
                selected_indices.append(best_idx)
                feature_scores[best_idx] = np.max(cluster_variances)
            
            # Fill remaining slots if needed
            if len(selected_indices) < self.config.target_features:
                all_variances = np.var(embeddings, axis=0)
                remaining = [i for i in range(embeddings.shape[1]) if i not in selected_indices]
                remaining_sorted = sorted(remaining, key=lambda x: all_variances[x], reverse=True)
                
                needed = self.config.target_features - len(selected_indices)
                selected_indices.extend(remaining_sorted[:needed])
                for idx in remaining_sorted[:needed]:
                    feature_scores[idx] = all_variances[idx]
            
            selected_indices = np.array(selected_indices[:self.config.target_features])
            
            # Calculate information preservation
            original_var = np.var(embeddings, axis=0).sum()
            selected_var = np.sum(np.var(embeddings[:, selected_indices], axis=0))
            info_preservation = selected_var / original_var if original_var > 0 else 0.0
            
            return SelectionResult(
                success=True,
                selected_indices=selected_indices,
                feature_scores=feature_scores,
                information_preservation=info_preservation,
                method_used="correlation_clustering",
                metadata={
                    'n_clusters': len(clusters),
                    'avg_cluster_size': np.mean([len(c) for c in clusters]),
                    'correlation_threshold': self.config.correlation_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Correlation clustering selection failed: {e}")
            return SelectionResult(success=False, error=str(e))
    
    def _ranking_aware_selection(self, embeddings: np.ndarray) -> SelectionResult:
        """
        Ranking-aware feature selection that optimizes for semantic ordering preservation.
        
        Selects features that best preserve pairwise distance relationships between samples.
        """
        try:
            from scipy.spatial.distance import pdist, squareform
            from scipy.stats import spearmanr
            
            n_features = embeddings.shape[1]
            n_samples = embeddings.shape[0]
            
            # Compute pairwise distances for original embeddings (sample subset for efficiency)
            sample_size = min(500, n_samples)  # Limit for computational efficiency
            sample_indices = np.random.choice(n_samples, sample_size, replace=False) if n_samples > sample_size else np.arange(n_samples)
            sample_embeddings = embeddings[sample_indices]
            
            # Original pairwise cosine distances
            original_distances = pdist(sample_embeddings, metric='cosine')
            
            # Evaluate each feature's contribution to distance preservation
            feature_ranking_scores = np.zeros(n_features)
            
            # Use batching to handle large feature sets efficiently
            batch_size = min(100, n_features)
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_features = range(batch_start, batch_end)
                
                for i, feature_idx in enumerate(batch_features):
                    try:
                        # Compute distances using only this feature
                        single_feature_distances = pdist(sample_embeddings[:, [feature_idx]], metric='cosine')
                        
                        # Measure correlation with original distances
                        correlation, _ = spearmanr(original_distances, single_feature_distances)
                        feature_ranking_scores[feature_idx] = abs(correlation) if not np.isnan(correlation) else 0.0
                        
                    except Exception as feature_error:
                        logger.debug(f"Feature {feature_idx} evaluation failed: {feature_error}")
                        feature_ranking_scores[feature_idx] = 0.0
            
            # Combine ranking preservation with traditional metrics
            variance_scores = np.var(embeddings, axis=0)
            variance_scores_normalized = (variance_scores - np.min(variance_scores)) / (np.max(variance_scores) - np.min(variance_scores) + 1e-8)
            
            # Combined score: ranking preservation + variance
            ranking_weight = self.config.ranking_preservation_weight
            variance_weight = 1.0 - ranking_weight
            
            combined_scores = (ranking_weight * feature_ranking_scores + 
                             variance_weight * variance_scores_normalized)
            
            # Select top features
            selected_indices = np.argsort(combined_scores)[-self.config.target_features:]
            selected_indices = np.sort(selected_indices)  # Sort for consistency
            
            # Calculate information preservation
            original_var = np.var(embeddings, axis=0).sum()
            selected_var = np.sum(np.var(embeddings[:, selected_indices], axis=0))
            info_preservation = selected_var / original_var if original_var > 0 else 0.0
            
            # Calculate semantic preservation (distance correlation with selected features)
            try:
                selected_distances = pdist(sample_embeddings[:, selected_indices], metric='cosine')
                semantic_preservation, _ = spearmanr(original_distances, selected_distances)
                semantic_preservation = abs(semantic_preservation) if not np.isnan(semantic_preservation) else 0.0
            except:
                semantic_preservation = 0.0
            
            return SelectionResult(
                success=True,
                selected_indices=selected_indices,
                feature_scores=combined_scores,
                information_preservation=info_preservation,
                semantic_preservation=semantic_preservation,
                method_used="ranking_aware",
                metadata={
                    'ranking_scores': feature_ranking_scores,
                    'variance_scores': variance_scores_normalized,
                    'ranking_weight': ranking_weight,
                    'variance_weight': variance_weight,
                    'sample_size_used': len(sample_indices),
                    'semantic_preservation': semantic_preservation
                }
            )
            
        except Exception as e:
            logger.error(f"Ranking-aware selection failed: {e}")
            return SelectionResult(success=False, error=str(e))
    
    def _hybrid_selection(self, embeddings: np.ndarray, 
                         labels: Optional[np.ndarray]) -> SelectionResult:
        """Hybrid selection combining multiple methods."""
        try:
            # Phase 1: Variance-based pre-filtering
            variance_threshold = VarianceThreshold(threshold=self.config.variance_threshold)
            embeddings_filtered = variance_threshold.fit_transform(embeddings)
            valid_indices = variance_threshold.get_support(indices=True)
            
            logger.debug(f"Variance filtering: {embeddings.shape[1]} -> {embeddings_filtered.shape[1]} features")
            
            # Phase 2: PCA for initial reduction
            intermediate_features = min(self.config.target_features * 3, embeddings_filtered.shape[1])
            if embeddings_filtered.shape[1] > intermediate_features:
                pca = PCA(n_components=intermediate_features)
                pca_features = pca.fit_transform(embeddings_filtered)
                
                # Get most important original features from PCA
                components = np.abs(pca.components_)
                feature_importance = np.sum(components, axis=0)
                pca_indices = np.argsort(feature_importance)[-intermediate_features:]
                
                embeddings_intermediate = embeddings_filtered[:, pca_indices]
                intermediate_indices = valid_indices[pca_indices]
            else:
                embeddings_intermediate = embeddings_filtered
                intermediate_indices = valid_indices
            
            # Phase 3: mRMR for final selection
            if labels is not None and embeddings_intermediate.shape[1] > self.config.target_features:
                mrmr_config = QuantumFeatureSelectionConfig(
                    method="mrmr",
                    num_features=self.config.target_features,
                    normalize_features=False  # Already normalized
                )
                mrmr_selector = QuantumFeatureSelector(mrmr_config)
                
                final_embeddings = mrmr_selector.fit_transform(embeddings_intermediate, labels)
                final_indices = intermediate_indices[mrmr_selector.selected_features_]
                final_scores = mrmr_selector.feature_scores_
            else:
                # Fallback to variance selection
                variances = np.var(embeddings_intermediate, axis=0)
                top_variance_indices = np.argsort(variances)[-self.config.target_features:]
                final_indices = intermediate_indices[top_variance_indices]
                final_scores = variances
            
            # Calculate information preservation
            original_var = np.var(embeddings, axis=0).sum()
            selected_var = np.sum(np.var(embeddings[:, final_indices], axis=0))
            info_preservation = selected_var / original_var if original_var > 0 else 0.0
            
            return SelectionResult(
                success=True,
                selected_indices=final_indices,
                feature_scores=final_scores,
                information_preservation=info_preservation,
                method_used="hybrid",
                metadata={
                    'phase1_features': len(valid_indices),
                    'phase2_features': len(intermediate_indices),
                    'final_features': len(final_indices),
                    'used_mrmr': labels is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid selection failed: {e}")
            return SelectionResult(success=False, error=str(e))
    
    def analyze_feature_preservation(self, original_embeddings: np.ndarray,
                                   selected_embeddings: np.ndarray) -> Dict[str, float]:
        """
        Analyze how well semantic information is preserved in selected features.
        
        Returns:
            Dictionary with preservation metrics
        """
        try:
            # Information-theoretic measures
            original_var = np.var(original_embeddings, axis=0).sum()
            selected_var = np.var(selected_embeddings, axis=0).sum()
            variance_preservation = selected_var / original_var if original_var > 0 else 0.0
            
            # Pairwise distance preservation
            from scipy.spatial.distance import pdist, squareform
            from scipy.stats import spearmanr
            
            # Sample subset for efficiency on large datasets
            n_samples = min(1000, original_embeddings.shape[0])
            sample_indices = np.random.choice(original_embeddings.shape[0], n_samples, replace=False)
            
            orig_sample = original_embeddings[sample_indices]
            sel_sample = selected_embeddings[sample_indices]
            
            # Compute pairwise distances
            orig_distances = pdist(orig_sample, metric='cosine')
            sel_distances = pdist(sel_sample, metric='cosine')
            
            # Correlation between distance matrices
            distance_correlation, _ = spearmanr(orig_distances, sel_distances)
            
            return {
                'variance_preservation': variance_preservation,
                'distance_correlation': distance_correlation,
                'compression_ratio': selected_embeddings.shape[1] / original_embeddings.shape[1],
                'n_features_selected': selected_embeddings.shape[1],
                'n_samples_analyzed': n_samples
            }
            
        except Exception as e:
            logger.error(f"Feature preservation analysis failed: {e}")
            return {
                'variance_preservation': 0.0,
                'distance_correlation': 0.0,
                'compression_ratio': 0.0,
                'error': str(e)
            }


def test_semantic_feature_selection():
    """Test semantic feature selection on sample embeddings."""
    from quantum_rerank.core.embeddings import EmbeddingProcessor
    
    # Initialize components
    embedding_processor = EmbeddingProcessor()
    selector = SemanticFeatureSelector()
    
    # Test texts
    test_texts = [
        "The patient presented with acute myocardial infarction and elevated troponin levels.",
        "Diabetes mellitus type 2 requires insulin therapy and dietary management.",
        "Quantum computing uses quantum mechanical phenomena for computation.",
        "Machine learning algorithms analyze data to identify patterns.",
        "The weather today is sunny with a temperature of 75 degrees."
    ]
    
    print("Testing Semantic Feature Selection")
    print("="*50)
    
    # Generate embeddings
    embeddings = embedding_processor.encode_texts(test_texts)
    print(f"Original embeddings shape: {embeddings.shape}")
    
    # Test different selection methods
    methods = ['variance', 'pca', 'correlation_clustering', 'hybrid']
    
    for method in methods:
        print(f"\nTesting method: {method}")
        
        config = SemanticSelectionConfig(
            method=method,
            target_features=16,
            preserve_local_structure=True
        )
        selector = SemanticFeatureSelector(config)
        
        result = selector.fit_transform(embeddings)
        
        if result.success:
            print(f"  Selected features: {result.selected_features.shape}")
            print(f"  Information preservation: {result.information_preservation:.3f}")
            
            # Analyze preservation
            preservation = selector.analyze_feature_preservation(
                embeddings, result.selected_features
            )
            print(f"  Distance correlation: {preservation.get('distance_correlation', 0):.3f}")
        else:
            print(f"  FAILED: {result.error}")
    
    return embeddings, result


if __name__ == "__main__":
    # Run test if executed directly
    test_embeddings, test_result = test_semantic_feature_selection()