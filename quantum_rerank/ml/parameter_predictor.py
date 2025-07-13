"""
Quantum Parameter Prediction Module.

This module implements a classical MLP that predicts quantum circuit parameters 
from embeddings, enabling hybrid quantum-classical training as specified in the PRD.

Based on:
- PRD Section 3.1: Core Algorithms - Parameterized Quantum Circuits (PQC)
- PRD Section 2.2: Implementation Stack - Classical ML (PyTorch)
- Documentation: Quantum-Inspired Semantic Reranking with PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParameterPredictorConfig:
    """Configuration for quantum parameter prediction."""
    embedding_dim: int = 768  # From SentenceTransformers
    hidden_dims: List[int] = None  # Will default to [512, 256]
    n_qubits: int = 4
    n_layers: int = 2  # Quantum circuit layers
    dropout_rate: float = 0.1
    activation: str = 'relu'  # 'relu', 'tanh', 'gelu'
    parameter_range: str = 'pi'  # 'pi' for [0, π], '2pi' for [0, 2π]
    device: str = 'auto'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


class QuantumParameterPredictor(nn.Module):
    """
    Classical MLP that predicts quantum circuit parameters from embeddings.
    
    Based on PRD Section 3.1 and hybrid quantum-classical approach.
    Predicts parameters for parameterized quantum circuits (PQCs) that can
    be used to create quantum states for similarity computation.
    """
    
    def __init__(self, config: ParameterPredictorConfig = None):
        super().__init__()
        
        self.config = config or ParameterPredictorConfig()
        
        # Calculate number of parameters needed per layer
        self.params_per_layer = self._calculate_params_per_layer()
        self.total_params = self.params_per_layer * self.config.n_layers
        
        # Build MLP layers
        self.layers = self._build_mlp_layers()
        
        # Parameter output heads for different gate types
        self.parameter_heads = self._build_parameter_heads()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Parameter predictor initialized: {self.total_params} quantum parameters")
    
    def _calculate_params_per_layer(self) -> int:
        """
        Calculate number of parameters needed per quantum circuit layer.
        
        Based on standard parameterized quantum circuit structure:
        - 3 rotation parameters per qubit (RY, RZ, RY sequence)
        - Entangling parameters between adjacent qubits
        
        Returns:
            Number of parameters per layer
        """
        # Rotation parameters: 3 per qubit (RY-RZ-RY sequence)
        rotation_params = 3 * self.config.n_qubits
        
        # Entangling parameters: between adjacent qubits (RZZ gates)
        entangling_params = self.config.n_qubits - 1
        
        return rotation_params + entangling_params
    
    def _build_mlp_layers(self) -> nn.ModuleList:
        """Build the main MLP layers."""
        layers = nn.ModuleList()
        
        # Input layer
        prev_dim = self.config.embedding_dim
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return layers
    
    def _build_parameter_heads(self) -> nn.ModuleDict:
        """Build separate heads for different parameter types."""
        heads = nn.ModuleDict()
        
        final_hidden_dim = self.config.hidden_dims[-1]
        
        # Rotation parameter heads (one for each rotation type)
        heads['ry_params'] = nn.Linear(final_hidden_dim, 
                                      self.config.n_qubits * self.config.n_layers)
        heads['rz_params'] = nn.Linear(final_hidden_dim, 
                                      self.config.n_qubits * self.config.n_layers)
        heads['ry2_params'] = nn.Linear(final_hidden_dim, 
                                       self.config.n_qubits * self.config.n_layers)
        
        # Entangling parameter head (RZZ gates)
        heads['entangling_params'] = nn.Linear(final_hidden_dim, 
                                              (self.config.n_qubits - 1) * self.config.n_layers)
        
        return heads
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'tanh':
            return nn.Tanh()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()  # Default
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: embeddings -> quantum parameters.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Dictionary of parameter tensors for different gate types
        """
        # Forward through main MLP
        x = embeddings
        for layer in self.layers:
            x = layer(x)
        
        # Generate parameters through separate heads
        parameters = {}
        
        # Rotation parameters with appropriate scaling
        # Use sigmoid to bound outputs to [0,1] then scale to desired range
        parameters['ry_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['ry_params'](x))
        )
        parameters['rz_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['rz_params'](x))
        )
        parameters['ry2_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['ry2_params'](x))
        )
        
        # Entangling parameters (RZZ gates)
        parameters['entangling_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['entangling_params'](x))
        )
        
        return parameters
    
    def _scale_parameters(self, sigmoid_output: torch.Tensor) -> torch.Tensor:
        """Scale sigmoid output to appropriate parameter range."""
        if self.config.parameter_range == 'pi':
            return sigmoid_output * torch.pi
        elif self.config.parameter_range == '2pi':
            return sigmoid_output * 2 * torch.pi
        else:
            return sigmoid_output * torch.pi  # Default
    
    def get_flat_parameters(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get all parameters as a flat tensor for compatibility.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Flat parameter tensor [batch_size, total_params]
        """
        param_dict = self.forward(embeddings)
        
        # Concatenate all parameter types in consistent order
        flat_params = torch.cat([
            param_dict['ry_params'],
            param_dict['rz_params'], 
            param_dict['ry2_params'],
            param_dict['entangling_params']
        ], dim=1)
        
        return flat_params
    
    def get_structured_parameters(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get parameters organized by layer and qubit structure.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Dictionary with layer-wise parameter organization
        """
        param_dict = self.forward(embeddings)
        batch_size = embeddings.shape[0]
        
        structured_params = {
            'layers': []
        }
        
        # Organize parameters by layer
        for layer in range(self.config.n_layers):
            layer_params = {
                'rotation_params': [],
                'entangling_params': []
            }
            
            # Extract rotation parameters for this layer
            for qubit in range(self.config.n_qubits):
                param_idx = layer * self.config.n_qubits + qubit
                
                qubit_rotations = {
                    'ry': param_dict['ry_params'][:, param_idx],
                    'rz': param_dict['rz_params'][:, param_idx],
                    'ry2': param_dict['ry2_params'][:, param_idx]
                }
                layer_params['rotation_params'].append(qubit_rotations)
            
            # Extract entangling parameters for this layer
            for connection in range(self.config.n_qubits - 1):
                param_idx = layer * (self.config.n_qubits - 1) + connection
                layer_params['entangling_params'].append(
                    param_dict['entangling_params'][:, param_idx]
                )
            
            structured_params['layers'].append(layer_params)
        
        # Add metadata
        structured_params['metadata'] = {
            'batch_size': batch_size,
            'n_qubits': self.config.n_qubits,
            'n_layers': self.config.n_layers,
            'total_params': self.total_params
        }
        
        return structured_params
    
    def validate_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """
        Validate that predicted parameters are within expected ranges and shapes.
        
        Args:
            parameters: Dictionary of parameter tensors
            
        Returns:
            Validation results for each parameter type
        """
        validation_results = {}
        batch_size = next(iter(parameters.values())).shape[0]
        
        # Expected shapes
        expected_shapes = {
            'ry_params': (batch_size, self.config.n_qubits * self.config.n_layers),
            'rz_params': (batch_size, self.config.n_qubits * self.config.n_layers),
            'ry2_params': (batch_size, self.config.n_qubits * self.config.n_layers),
            'entangling_params': (batch_size, (self.config.n_qubits - 1) * self.config.n_layers)
        }
        
        for param_type, param_tensor in parameters.items():
            if param_type in expected_shapes:
                # Check shape
                shape_valid = param_tensor.shape == expected_shapes[param_type]
                
                # Check value ranges
                param_numpy = param_tensor.detach().cpu().numpy()
                values_finite = np.all(np.isfinite(param_numpy))
                
                if self.config.parameter_range == 'pi':
                    values_in_range = np.all((param_numpy >= 0) & (param_numpy <= np.pi))
                elif self.config.parameter_range == '2pi':
                    values_in_range = np.all((param_numpy >= 0) & (param_numpy <= 2 * np.pi))
                else:
                    values_in_range = np.all((param_numpy >= 0) & (param_numpy <= np.pi))
                
                validation_results[param_type] = {
                    'shape_valid': shape_valid,
                    'values_finite': values_finite,
                    'values_in_range': values_in_range,
                    'overall_valid': shape_valid and values_finite and values_in_range
                }
        
        # Overall validation
        all_valid = all(
            result['overall_valid'] for result in validation_results.values()
        )
        validation_results['overall_valid'] = all_valid
        
        return validation_results
    
    def get_parameter_statistics(self, embeddings: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about predicted parameters.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Statistics for each parameter type
        """
        with torch.no_grad():
            parameters = self.forward(embeddings)
        
        statistics = {}
        
        for param_type, param_tensor in parameters.items():
            param_numpy = param_tensor.detach().cpu().numpy()
            
            statistics[param_type] = {
                'mean': float(np.mean(param_numpy)),
                'std': float(np.std(param_numpy)),
                'min': float(np.min(param_numpy)),
                'max': float(np.max(param_numpy)),
                'range': float(np.max(param_numpy) - np.min(param_numpy))
            }
        
        return statistics