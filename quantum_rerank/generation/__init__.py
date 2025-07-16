"""
Generation module for quantum-inspired lightweight RAG.

Provides Small Language Model (SLM) integration for edge deployment
with ultra-low memory footprint and high performance.
"""

from .slm_generator import SLMGenerator, SLMConfig, create_slm_config, validate_slm_generator

__all__ = [
    'SLMGenerator',
    'SLMConfig', 
    'create_slm_config',
    'validate_slm_generator'
]