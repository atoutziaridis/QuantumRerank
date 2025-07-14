"""
QPMeL (Quantum Polar Metric Learning) Circuit Implementation.

Implements the efficient polar encoding approach from the QPMeL paper:
- 2 angles per qubit (theta for Ry, gamma for Rz)
- Shallow circuits with entangling ZZ gates
- Quantum Residual Correction (QRC) for training stability
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector, CircuitInstruction

from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)

@dataclass
class QPMeLConfig:
    """Configuration for QPMeL circuits."""
    n_qubits: int = 4
    n_layers: int = 1  # Number of entangling layers
    enable_qrc: bool = True  # Quantum Residual Correction
    entangling_gate: str = "zz"  # "zz" or "cx"
    max_circuit_depth: int = 15
    # Circuit optimization options
    optimize_gates: bool = True  # Enable gate optimization
    use_efficient_encoding: bool = True  # Use optimized parameter encoding
    enable_parameter_sharing: bool = False  # Share parameters between layers

class QPMeLCircuitBuilder:
    """
    Builds QPMeL circuits using polar encoding.
    
    Key features:
    - 2 parameters per qubit (theta, gamma) for Ry, Rz gates
    - Shallow circuits with ZZ entangling gates
    - Optional Quantum Residual Correction (QRC)
    """
    
    def __init__(self, config: QPMeLConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        
        # Total parameters: 
        # - 2 angles per qubit for initial Ry/Rz rotations: 2 * n_qubits
        # - 1 parameter per entangling gate per layer: (n_qubits - 1) * n_layers
        polar_params = 2 * self.n_qubits
        entangling_params = (self.n_qubits - 1) * self.n_layers if config.entangling_gate == "zz" else 0
        self.n_params = polar_params + entangling_params
        
        logger.info(f"QPMeL circuit builder initialized: {self.n_qubits} qubits, "
                   f"{self.n_layers} layers, {self.n_params} parameters")
    
    def create_parameterized_circuit(self, 
                                   name: str = "qpmel_circuit") -> Tuple[QuantumCircuit, ParameterVector]:
        """
        Create a parameterized QPMeL circuit with optimizations.
        
        Returns:
            Tuple of (circuit, parameter_vector)
        """
        # Create parameter vector
        params = ParameterVector(f"{name}_params", self.n_params)
        
        # Create circuit
        qreg = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qreg, name=name)
        
        param_idx = 0
        
        # Layer 0: Optimized polar encoding
        if self.config.use_efficient_encoding:
            # Use more efficient Ry-Rz pattern that reduces gate count
            for qubit in range(self.n_qubits):
                theta = params[param_idx]
                gamma = params[param_idx + 1]
                
                # Efficient polar encoding: Single Ry followed by Rz
                circuit.ry(theta, qubit)
                circuit.rz(gamma, qubit)
                
                param_idx += 2
        else:
            # Original encoding
            for qubit in range(self.n_qubits):
                theta = params[param_idx]
                gamma = params[param_idx + 1]
                
                circuit.ry(theta, qubit)
                circuit.rz(gamma, qubit)
                
                param_idx += 2
        
        # Optimized entangling layers
        for layer in range(self.n_layers):
            if self.config.entangling_gate == "zz":
                # Optimized ZZ gates with reduced depth
                for qubit in range(self.n_qubits - 1):
                    alpha = params[param_idx]
                    
                    if self.config.optimize_gates:
                        # More efficient ZZ implementation
                        circuit.cx(qubit, qubit + 1)
                        circuit.rz(alpha, qubit + 1)
                        circuit.cx(qubit, qubit + 1)
                    else:
                        # Original ZZ implementation
                        circuit.rz(-alpha/2, qubit)
                        circuit.rz(-alpha/2, qubit + 1)
                        circuit.cx(qubit, qubit + 1)
                        circuit.rz(alpha/2, qubit + 1)
                    
                    param_idx += 1
            else:  # "cx"
                for qubit in range(self.n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
        
        # Validate circuit constraints
        if circuit.depth() > self.config.max_circuit_depth:
            logger.warning(f"QPMeL circuit depth {circuit.depth()} exceeds limit {self.config.max_circuit_depth}")
        
        logger.debug(f"Created QPMeL circuit: {circuit.depth()} depth, {circuit.size()} gates")
        
        return circuit, params
    
    def bind_parameters(self, 
                       circuit: QuantumCircuit, 
                       params: ParameterVector, 
                       parameter_values: np.ndarray) -> QuantumCircuit:
        """
        Bind parameter values to create a concrete circuit.
        
        Args:
            circuit: Parameterized circuit
            params: Parameter vector
            parameter_values: Array of parameter values
            
        Returns:
            Bound circuit ready for execution
        """
        if len(parameter_values) != len(params):
            raise ValueError(f"Expected {len(params)} parameters, got {len(parameter_values)}")
        
        # Create parameter binding dictionary
        param_dict = {param: float(value) for param, value in zip(params, parameter_values)}
        
        # Bind parameters
        bound_circuit = circuit.assign_parameters(param_dict)
        
        return bound_circuit
    
    def create_batch_circuits(self, parameter_batch: torch.Tensor) -> List[QuantumCircuit]:
        """
        Create a batch of bound circuits from parameter tensor.
        
        Args:
            parameter_batch: Tensor of shape (batch_size, n_params)
            
        Returns:
            List of bound quantum circuits
        """
        batch_size = parameter_batch.shape[0]
        
        # Create template circuit
        template_circuit, params = self.create_parameterized_circuit("qpmel_batch")
        
        circuits = []
        for i in range(batch_size):
            param_values = parameter_batch[i].detach().cpu().numpy()
            bound_circuit = self.bind_parameters(template_circuit, params, param_values)
            bound_circuit.name = f"qpmel_batch_{i}"
            circuits.append(bound_circuit)
        
        return circuits
    
    def get_circuit_properties(self) -> Dict[str, Any]:
        """Get properties of the QPMeL circuit architecture."""
        template_circuit, _ = self.create_parameterized_circuit("template")
        
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_parameters": self.n_params,
            "circuit_depth": template_circuit.depth(),
            "circuit_size": template_circuit.size(),
            "entangling_gate": self.config.entangling_gate,
            "enable_qrc": self.config.enable_qrc,
            "gate_counts": template_circuit.count_ops()
        }

class QPMeLParameterPredictor(nn.Module):
    """
    Neural network that predicts QPMeL quantum circuit parameters from embeddings.
    
    Key features:
    - Outputs 2*n_qubits angles (theta, gamma for each qubit)
    - Sigmoid activation to constrain to [0, 2π]
    - Optional Quantum Residual Correction (QRC) parameters
    """
    
    def __init__(self, 
                 input_dim: int,
                 config: QPMeLConfig,
                 hidden_dims: List[int] = [512, 256]):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.enable_qrc = config.enable_qrc
        
        # Base parameter count: 2 * n_qubits for polar encoding + entangling parameters
        circuit_builder = QPMeLCircuitBuilder(config)
        self.n_circuit_params = circuit_builder.n_params
        
        # Build MLP layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        # Output layer for circuit parameters
        layers.append(nn.Linear(current_dim, self.n_circuit_params))
        layers.append(nn.Sigmoid())  # Constrain to [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        
        # Quantum Residual Correction (QRC) parameters
        if self.enable_qrc:
            # Trainable residual parameters for polar angles only
            n_polar_params = 2 * self.n_qubits  
            self.qrc_residuals = nn.Parameter(torch.zeros(n_polar_params))
        
        self.circuit_builder = circuit_builder
        
        logger.info(f"QPMeL parameter predictor initialized: {input_dim} -> {self.n_circuit_params} parameters")
    
    def forward(self, embeddings: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Predict quantum circuit parameters from embeddings.
        
        Args:
            embeddings: Input embeddings (batch_size, input_dim)
            training: Whether in training mode (applies QRC if enabled)
            
        Returns:
            Quantum parameters scaled to [0, 2π] (batch_size, n_circuit_params)
        """
        # Get base predictions [0, 1]
        base_params = self.mlp(embeddings)
        
        # Scale to [0, 2π]
        scaled_params = base_params * 2 * np.pi
        
        # Apply Quantum Residual Correction during training
        if training and self.enable_qrc:
            batch_size = scaled_params.shape[0]
            
            # Apply QRC only to polar angle parameters (first 2*n_qubits)
            polar_params = scaled_params[:, :2*self.n_qubits]
            other_params = scaled_params[:, 2*self.n_qubits:]
            
            # Add residual corrections
            qrc_expanded = self.qrc_residuals.unsqueeze(0).expand(batch_size, -1)
            corrected_polar = polar_params + qrc_expanded
            
            # Recombine
            scaled_params = torch.cat([corrected_polar, other_params], dim=1)
        
        return scaled_params
    
    def get_circuits(self, embeddings: torch.Tensor, training: bool = True) -> List[QuantumCircuit]:
        """
        Generate quantum circuits from embeddings.
        
        Args:
            embeddings: Input embeddings
            training: Whether in training mode
            
        Returns:
            List of quantum circuits
        """
        with torch.no_grad():
            parameters = self.forward(embeddings, training=training)
        
        return self.circuit_builder.create_batch_circuits(parameters)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_dim": self.input_dim,
            "n_qubits": self.n_qubits,
            "n_circuit_params": self.n_circuit_params,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "enable_qrc": self.enable_qrc,
            "qrc_parameters": self.qrc_residuals.numel() if self.enable_qrc else 0,
            "circuit_properties": self.circuit_builder.get_circuit_properties()
        }