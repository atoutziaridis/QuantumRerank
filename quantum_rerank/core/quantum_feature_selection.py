"""
Quantum-specific feature selection methods for data-driven kernel optimization.

Implements minimum Redundancy Maximum Relevance (mRMR) and other advanced
feature selection techniques specifically designed for quantum machine learning.

Based on:
- "Data-Driven Quantum Kernel Development" documentation
- mRMR algorithm for quantum feature selection
- Light-cone feature selection for quantum circuits
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import logging
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import time

logger = logging.getLogger(__name__)

@dataclass
class QuantumFeatureSelectionConfig:
    """Configuration for quantum feature selection methods."""
    method: str = "mrmr"  # mrmr, pca, mutual_info, light_cone
    num_features: int = 32  # Number of features to select
    relevance_metric: str = "mutual_info"  # mutual_info, f_statistic, correlation
    redundancy_metric: str = "mutual_info"  # mutual_info, correlation
    scoring_function: str = "difference"  # difference, ratio
    max_qubits: int = 8  # Maximum qubits for quantum encoding
    normalize_features: bool = True  # Normalize selected features
    parallel_computation: bool = False  # Parallel feature evaluation


class QuantumFeatureSelector:
    """
    Advanced feature selection specifically designed for quantum machine learning.
    
    Implements mRMR and other techniques to select optimal features for
    quantum encoding and kernel computation.
    """
    
    def __init__(self, config: QuantumFeatureSelectionConfig = None):
        self.config = config or QuantumFeatureSelectionConfig()
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selection_history_ = []
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit feature selector and transform data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Transformed data with selected features
        """
        return self.fit(X, y).transform(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumFeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting quantum feature selector with method: {self.config.method}")
        logger.info(f"Input shape: {X.shape}, selecting {self.config.num_features} features")
        
        # Ensure we don't select more features than available
        self.config.num_features = min(self.config.num_features, X.shape[1])
        
        # Normalize features if specified
        if self.config.normalize_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
        
        # Encode labels if necessary
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        # Select features based on method
        if self.config.method == "mrmr":
            self.selected_features_ = self._mrmr_selection(X_scaled, y_encoded)
        elif self.config.method == "pca":
            self.selected_features_ = self._pca_selection(X_scaled, y_encoded)
        elif self.config.method == "mutual_info":
            self.selected_features_ = self._mutual_info_selection(X_scaled, y_encoded)
        elif self.config.method == "light_cone":
            self.selected_features_ = self._light_cone_selection(X_scaled, y_encoded)
        else:
            raise ValueError(f"Unknown feature selection method: {self.config.method}")
        
        logger.info(f"Selected features: {self.selected_features_}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using selected features.
        
        Args:
            X: Input features to transform
            
        Returns:
            Transformed data with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("Feature selector has not been fitted yet")
        
        # Apply scaling if used during fitting
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        return X_scaled[:, self.selected_features_]
    
    def _mrmr_selection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """
        Implement minimum Redundancy Maximum Relevance (mRMR) feature selection.
        
        Based on the algorithm described in the documentation.
        """
        logger.info("Running mRMR feature selection")
        
        selected_features = []
        remaining_features = list(range(X.shape[1]))
        self.selection_history_ = []
        
        # Step 1: Select most relevant feature
        relevance_scores = self._compute_relevance_scores(X, y)
        best_feature_idx = np.argmax([relevance_scores[i] for i in remaining_features])
        best_feature = remaining_features[best_feature_idx]
        
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        
        self.selection_history_.append({
            'feature': best_feature,
            'relevance': relevance_scores[best_feature],
            'redundancy': 0.0,
            'score': relevance_scores[best_feature]
        })
        
        # Step 2: Iteratively select features using mRMR criterion
        for iteration in range(self.config.num_features - 1):
            if not remaining_features:
                break
            
            best_score = float('-inf')
            best_feature = None
            best_relevance = 0.0
            best_redundancy = 0.0
            
            for feature in remaining_features:
                # Compute relevance
                relevance = relevance_scores[feature]
                
                # Compute redundancy with already selected features
                redundancy = self._compute_redundancy(
                    X[:, feature], X[:, selected_features]
                )
                
                # Compute mRMR score
                if self.config.scoring_function == "difference":
                    score = relevance - redundancy
                elif self.config.scoring_function == "ratio":
                    score = relevance / (redundancy + 1e-8)  # Avoid division by zero
                else:
                    score = relevance - redundancy
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_relevance = relevance
                    best_redundancy = redundancy
            
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                
                self.selection_history_.append({
                    'feature': best_feature,
                    'relevance': best_relevance,
                    'redundancy': best_redundancy,
                    'score': best_score
                })
            else:
                break
        
        return selected_features
    
    def _compute_relevance_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute relevance scores for all features."""
        if self.config.relevance_metric == "mutual_info":
            try:
                scores = mutual_info_classif(X, y, random_state=42)
            except Exception as e:
                logger.warning(f"Mutual info failed, using correlation: {e}")
                scores = self._compute_correlation_scores(X, y)
        elif self.config.relevance_metric == "f_statistic":
            f_scores, _ = f_classif(X, y)
            scores = f_scores
        elif self.config.relevance_metric == "correlation":
            scores = self._compute_correlation_scores(X, y)
        else:
            scores = np.ones(X.shape[1])
        
        # Handle NaN values
        scores = np.nan_to_num(scores, nan=0.0)
        return scores
    
    def _compute_correlation_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute correlation-based relevance scores."""
        scores = []
        for i in range(X.shape[1]):
            try:
                corr, _ = pearsonr(X[:, i], y)
                scores.append(abs(corr))
            except:
                scores.append(0.0)
        return np.array(scores)
    
    def _compute_redundancy(self, feature: np.ndarray, selected_features: np.ndarray) -> float:
        """Compute redundancy between a feature and selected features."""
        if selected_features.shape[1] == 0:
            return 0.0
        
        redundancies = []
        
        for i in range(selected_features.shape[1]):
            if self.config.redundancy_metric == "mutual_info":
                try:
                    # For continuous features, discretize for mutual info
                    feature_discrete = self._discretize_feature(feature)
                    selected_discrete = self._discretize_feature(selected_features[:, i])
                    
                    # Use mutual info between discretized features
                    redundancy = mutual_info_classif(
                        feature_discrete.reshape(-1, 1), 
                        selected_discrete, 
                        random_state=42
                    )[0]
                except Exception as e:
                    # Fallback to correlation
                    redundancy = abs(np.corrcoef(feature, selected_features[:, i])[0, 1])
                    if np.isnan(redundancy):
                        redundancy = 0.0
            else:  # correlation
                redundancy = abs(np.corrcoef(feature, selected_features[:, i])[0, 1])
                if np.isnan(redundancy):
                    redundancy = 0.0
            
            redundancies.append(redundancy)
        
        # Return mean redundancy
        return np.mean(redundancies)
    
    def _discretize_feature(self, feature: np.ndarray, bins: int = 5) -> np.ndarray:
        """Discretize continuous feature for mutual information computation."""
        try:
            return np.digitize(feature, bins=np.linspace(feature.min(), feature.max(), bins))
        except:
            return np.zeros_like(feature, dtype=int)
    
    def _pca_selection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Feature selection using PCA for dimensionality reduction."""
        logger.info("Running PCA-based feature selection")
        
        pca = PCA(n_components=self.config.num_features)
        pca.fit(X)
        
        # Select features with highest contribution to principal components
        components = np.abs(pca.components_)
        feature_importance = np.sum(components, axis=0)
        
        # Select top features
        selected_indices = np.argsort(feature_importance)[-self.config.num_features:]
        return sorted(selected_indices.tolist())
    
    def _mutual_info_selection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Feature selection using mutual information only."""
        logger.info("Running mutual information-based feature selection")
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        selected_indices = np.argsort(mi_scores)[-self.config.num_features:]
        return sorted(selected_indices.tolist())
    
    def _light_cone_selection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """
        Light-cone feature selection for quantum circuits.
        
        Selects features based on quantum circuit locality considerations.
        """
        logger.info("Running light-cone feature selection")
        
        # For now, implement a simplified version that groups features
        # based on potential quantum entanglement patterns
        
        num_qubits = min(self.config.max_qubits, self.config.num_features)
        features_per_qubit = self.config.num_features // num_qubits
        
        # Compute feature relevance
        relevance_scores = self._compute_relevance_scores(X, y)
        
        # Group features into qubit regions
        selected_features = []
        sorted_features = np.argsort(relevance_scores)[::-1]  # Most relevant first
        
        for qubit in range(num_qubits):
            start_idx = qubit * features_per_qubit
            end_idx = start_idx + features_per_qubit
            if qubit == num_qubits - 1:  # Last qubit gets remaining features
                end_idx = self.config.num_features
            
            qubit_features = sorted_features[start_idx:end_idx]
            selected_features.extend(qubit_features)
        
        return sorted(selected_features[:self.config.num_features])
    
    def get_feature_ranking(self) -> Dict[str, Any]:
        """Get detailed feature ranking information."""
        if self.selected_features_ is None:
            return {"message": "Feature selector not fitted yet"}
        
        ranking_info = {
            "selected_features": self.selected_features_,
            "num_selected": len(self.selected_features_),
            "selection_method": self.config.method,
            "selection_history": self.selection_history_
        }
        
        if hasattr(self, 'feature_scores_'):
            ranking_info["feature_scores"] = self.feature_scores_
        
        return ranking_info
    
    def quantum_encoding_compatibility(self, num_qubits: int) -> Dict[str, Any]:
        """
        Check compatibility of selected features with quantum encoding constraints.
        
        Args:
            num_qubits: Number of qubits available for encoding
            
        Returns:
            Compatibility analysis
        """
        if self.selected_features_ is None:
            return {"message": "Feature selector not fitted yet"}
        
        num_selected = len(self.selected_features_)
        
        # Different encoding schemes have different qubit requirements
        encoding_analysis = {
            "amplitude_encoding": {
                "required_qubits": int(np.ceil(np.log2(num_selected))),
                "feasible": int(np.ceil(np.log2(num_selected))) <= num_qubits,
                "efficiency": num_selected / (2 ** int(np.ceil(np.log2(num_selected))))
            },
            "angle_encoding": {
                "required_qubits": num_selected,
                "feasible": num_selected <= num_qubits,
                "efficiency": 1.0 if num_selected <= num_qubits else num_qubits / num_selected
            },
            "dense_angle_encoding": {
                "required_qubits": int(np.ceil(num_selected / 3)),  # 3 angles per qubit
                "feasible": int(np.ceil(num_selected / 3)) <= num_qubits,
                "efficiency": num_selected / (3 * int(np.ceil(num_selected / 3)))
            }
        }
        
        # Recommend best encoding
        feasible_encodings = [
            name for name, info in encoding_analysis.items() 
            if info["feasible"]
        ]
        
        if feasible_encodings:
            best_encoding = max(
                feasible_encodings, 
                key=lambda x: encoding_analysis[x]["efficiency"]
            )
        else:
            best_encoding = "amplitude_encoding"  # Most qubit-efficient
        
        return {
            "num_selected_features": num_selected,
            "available_qubits": num_qubits,
            "encoding_analysis": encoding_analysis,
            "recommended_encoding": best_encoding,
            "overall_feasible": len(feasible_encodings) > 0
        }


class AdaptiveQuantumFeatureSelector:
    """
    Adaptive feature selector that tries multiple methods and selects the best.
    """
    
    def __init__(self, config: QuantumFeatureSelectionConfig = None):
        self.base_config = config or QuantumFeatureSelectionConfig()
        self.methods_to_try = ["mrmr", "mutual_info", "pca"]
        self.best_selector_ = None
        self.method_comparison_ = {}
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, 
                     evaluation_metric: Callable = None) -> Tuple[np.ndarray, str]:
        """
        Try multiple feature selection methods and return the best result.
        
        Args:
            X: Input features
            y: Target labels  
            evaluation_metric: Function to evaluate feature quality
            
        Returns:
            Tuple of (transformed_data, best_method_name)
        """
        logger.info("Running adaptive feature selection")
        
        best_score = float('-inf')
        best_method = None
        best_transformed = None
        
        for method in self.methods_to_try:
            try:
                # Create selector with current method
                config = QuantumFeatureSelectionConfig(
                    method=method,
                    num_features=self.base_config.num_features,
                    max_qubits=self.base_config.max_qubits
                )
                
                selector = QuantumFeatureSelector(config)
                transformed = selector.fit_transform(X, y)
                
                # Evaluate selection quality
                if evaluation_metric is not None:
                    score = evaluation_metric(transformed, y)
                else:
                    # Default evaluation: variance of selected features
                    score = np.mean(np.var(transformed, axis=0))
                
                self.method_comparison_[method] = {
                    'score': score,
                    'selected_features': selector.selected_features_,
                    'selector': selector
                }
                
                if score > best_score:
                    best_score = score
                    best_method = method
                    best_transformed = transformed
                    self.best_selector_ = selector
                
                logger.info(f"Method {method}: score = {score:.6f}")
                
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue
        
        logger.info(f"Best feature selection method: {best_method} (score: {best_score:.6f})")
        
        return best_transformed, best_method
    
    def get_comparison_results(self) -> Dict[str, Any]:
        """Get detailed comparison of different feature selection methods."""
        return {
            "method_comparison": self.method_comparison_,
            "best_method": getattr(self.best_selector_, 'config.method', None) if self.best_selector_ else None,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on method comparison."""
        recommendations = []
        
        if not self.method_comparison_:
            return ["No methods successfully completed"]
        
        scores = {method: info['score'] for method, info in self.method_comparison_.items()}
        best_method = max(scores.keys(), key=lambda k: scores[k])
        
        recommendations.append(f"Best overall method: {best_method}")
        
        # Check if mRMR was successful
        if 'mrmr' in scores:
            recommendations.append("mRMR method available for quantum-optimized selection")
        
        # Check quantum encoding compatibility
        if self.best_selector_:
            compatibility = self.best_selector_.quantum_encoding_compatibility(8)  # Assume 8 qubits
            if compatibility.get('overall_feasible', False):
                recommendations.append(f"Selected features compatible with quantum encoding")
                recommendations.append(f"Recommended encoding: {compatibility.get('recommended_encoding')}")
        
        return recommendations