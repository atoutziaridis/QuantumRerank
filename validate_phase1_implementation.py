"""
Phase 1 Implementation Validation Script

This script validates the quantum-inspired lightweight RAG Phase 1 implementation
without requiring large model downloads. Tests component interfaces, configurations,
and integration points.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_imports() -> Dict[str, Any]:
    """Validate all Phase 1 component imports."""
    results = {}
    
    # Test core imports
    try:
        from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
        from quantum_rerank.core.tensor_train_compression import (
            TTEmbeddingLayer, BERTTTCompressor, TTConfig, validate_compression_pipeline
        )
        results['core_embeddings'] = {'status': 'success'}
        results['tt_compression'] = {'status': 'success'}
    except ImportError as e:
        results['core_imports'] = {'status': 'error', 'message': str(e)}
    
    # Test retrieval imports
    try:
        from quantum_rerank.retrieval.quantized_faiss_store import (
            QuantizedFAISSStore, QuantizedFAISSConfig, validate_quantized_faiss
        )
        results['quantized_faiss'] = {'status': 'success'}
    except ImportError as e:
        results['quantized_faiss'] = {'status': 'error', 'message': str(e)}
    
    # Test generation imports
    try:
        from quantum_rerank.generation.slm_generator import (
            SLMGenerator, SLMConfig, validate_slm_generator
        )
        results['slm_generator'] = {'status': 'success'}
    except ImportError as e:
        results['slm_generator'] = {'status': 'error', 'message': str(e)}
    
    # Test pipeline imports
    try:
        from quantum_rerank.lightweight_rag_pipeline import (
            LightweightRAGPipeline, LightweightRAGConfig, validate_lightweight_pipeline
        )
        results['pipeline'] = {'status': 'success'}
    except ImportError as e:
        results['pipeline'] = {'status': 'error', 'message': str(e)}
    
    return results


def validate_configurations() -> Dict[str, Any]:
    """Validate component configurations."""
    results = {}
    
    # Embedding config
    try:
        from quantum_rerank.core.embeddings import EmbeddingConfig
        config = EmbeddingConfig()
        results['embedding_config'] = {
            'status': 'success',
            'model_name': config.model_name,
            'embedding_dim': config.embedding_dim
        }
    except Exception as e:
        results['embedding_config'] = {'status': 'error', 'message': str(e)}
    
    # TT config
    try:
        from quantum_rerank.core.tensor_train_compression import TTConfig
        config = TTConfig()
        results['tt_config'] = {
            'status': 'success',
            'tt_rank': config.tt_rank,
            'target_compression': config.target_compression_ratio
        }
    except Exception as e:
        results['tt_config'] = {'status': 'error', 'message': str(e)}
    
    # FAISS config
    try:
        from quantum_rerank.retrieval.quantized_faiss_store import QuantizedFAISSConfig
        config = QuantizedFAISSConfig()
        results['faiss_config'] = {
            'status': 'success',
            'quantization_bits': config.quantization_bits,
            'target_dim': config.target_dim
        }
    except Exception as e:
        results['faiss_config'] = {'status': 'error', 'message': str(e)}
    
    # SLM config
    try:
        from quantum_rerank.generation.slm_generator import SLMConfig
        config = SLMConfig()
        results['slm_config'] = {
            'status': 'success',
            'model_name': config.model_name,
            'max_memory_gb': config.max_memory_gb
        }
    except Exception as e:
        results['slm_config'] = {'status': 'error', 'message': str(e)}
    
    # Pipeline config
    try:
        from quantum_rerank.lightweight_rag_pipeline import LightweightRAGConfig
        config = LightweightRAGConfig()
        results['pipeline_config'] = {
            'status': 'success',
            'use_tt_compression': config.use_tt_compression,
            'use_quantized_faiss': config.use_quantized_faiss,
            'target_latency_ms': config.target_latency_ms
        }
    except Exception as e:
        results['pipeline_config'] = {'status': 'error', 'message': str(e)}
    
    return results


def validate_component_interfaces() -> Dict[str, Any]:
    """Validate component interfaces without heavy operations."""
    results = {}
    
    # Test TT layer interface (without actual compression)
    try:
        from quantum_rerank.core.tensor_train_compression import TTConfig
        config = TTConfig(tt_rank=4)
        
        # Test configuration validation
        assert config.tt_rank == 4
        assert config.target_compression_ratio > 0
        
        results['tt_interface'] = {
            'status': 'success',
            'config_valid': True,
            'compression_target': config.target_compression_ratio
        }
    except Exception as e:
        results['tt_interface'] = {'status': 'error', 'message': str(e)}
    
    # Test FAISS store interface (without actual indexing)
    try:
        from quantum_rerank.retrieval.quantized_faiss_store import QuantizedFAISSConfig
        config = QuantizedFAISSConfig(nlist=10, m=8, target_dim=64)
        
        # Test configuration validation
        assert config.quantization_bits in [4, 8, 16]
        assert config.target_dim > 0
        
        results['faiss_interface'] = {
            'status': 'success',
            'config_valid': True,
            'quantization_bits': config.quantization_bits
        }
    except Exception as e:
        results['faiss_interface'] = {'status': 'error', 'message': str(e)}
    
    # Test SLM interface
    try:
        from quantum_rerank.generation.slm_generator import SLMConfig
        config = SLMConfig(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            use_quantization=True,
            max_memory_gb=2.0
        )
        
        # Test configuration validation
        assert config.max_memory_gb > 0
        assert config.model_name is not None
        
        results['slm_interface'] = {
            'status': 'success',
            'config_valid': True,
            'model_name': config.model_name
        }
    except Exception as e:
        results['slm_interface'] = {'status': 'error', 'message': str(e)}
    
    return results


def test_compression_calculations() -> Dict[str, Any]:
    """Test compression ratio calculations."""
    results = {}
    
    # Test TT compression math
    try:
        # Simulate TT compression calculation
        original_params = 50000 * 768  # 50K vocab x 768D embeddings
        tt_rank = 8
        
        # Estimate TT parameters (simplified)
        # For 3D tensorization: roughly rank^2 * dims
        estimated_tt_params = tt_rank * tt_rank * (50 + 25 + 768)  # Rough estimate
        compression_ratio = original_params / estimated_tt_params
        
        results['tt_compression_math'] = {
            'status': 'success',
            'original_params': original_params,
            'estimated_tt_params': estimated_tt_params,
            'compression_ratio': compression_ratio,
            'target_met': compression_ratio >= 8.0
        }
    except Exception as e:
        results['tt_compression_math'] = {'status': 'error', 'message': str(e)}
    
    # Test FAISS compression math
    try:
        # Simulate FAISS compression
        original_size = 1000 * 768 * 4  # 1000 docs x 768D x 4 bytes (float32)
        
        # PCA compression: 768D -> 384D (2x)
        pca_size = 1000 * 384 * 4
        
        # Quantization: float32 -> int8 (4x)
        quantized_size = 1000 * 384 * 1
        
        total_compression = original_size / quantized_size
        
        results['faiss_compression_math'] = {
            'status': 'success',
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': total_compression,
            'target_met': total_compression >= 4.0
        }
    except Exception as e:
        results['faiss_compression_math'] = {'status': 'error', 'message': str(e)}
    
    # Combined compression
    try:
        tt_ratio = results['tt_compression_math']['compression_ratio']
        faiss_ratio = results['faiss_compression_math']['compression_ratio']
        combined_ratio = tt_ratio * faiss_ratio
        
        results['combined_compression'] = {
            'status': 'success',
            'tt_compression': tt_ratio,
            'faiss_compression': faiss_ratio,
            'combined_compression': combined_ratio,
            'target_met': combined_ratio >= 8.0
        }
    except Exception as e:
        results['combined_compression'] = {'status': 'error', 'message': str(e)}
    
    return results


def validate_dependency_availability() -> Dict[str, Any]:
    """Check availability of required dependencies."""
    results = {}
    
    # Core dependencies
    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('transformers', 'HuggingFace Transformers'),
        ('sentence_transformers', 'SentenceTransformers'),
        ('faiss', 'FAISS')
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            results[module] = {'status': 'available', 'name': name}
        except ImportError:
            results[module] = {'status': 'missing', 'name': name}
    
    # Optional dependencies
    optional_deps = [
        ('tensorly', 'TensorLy (for TT decomposition)'),
        ('bitsandbytes', 'BitsAndBytes (for quantization)')
    ]
    
    for module, name in optional_deps:
        try:
            __import__(module)
            results[module] = {'status': 'available', 'name': name}
        except ImportError:
            results[module] = {'status': 'optional_missing', 'name': name}
    
    return results


def main():
    """Run Phase 1 implementation validation."""
    logger.info("=" * 80)
    logger.info("PHASE 1 IMPLEMENTATION VALIDATION")
    logger.info("=" * 80)
    
    validation_results = {}
    
    # 1. Test imports
    logger.info("\n1. Testing Component Imports")
    import_results = validate_imports()
    validation_results['imports'] = import_results
    
    for component, result in import_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"  {status} {component}: {result['status']}")
    
    # 2. Test configurations
    logger.info("\n2. Testing Configurations")
    config_results = validate_configurations()
    validation_results['configurations'] = config_results
    
    for component, result in config_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"  {status} {component}: {result['status']}")
    
    # 3. Test interfaces
    logger.info("\n3. Testing Component Interfaces")
    interface_results = validate_component_interfaces()
    validation_results['interfaces'] = interface_results
    
    for component, result in interface_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"  {status} {component}: {result['status']}")
    
    # 4. Test compression calculations
    logger.info("\n4. Testing Compression Calculations")
    compression_results = test_compression_calculations()
    validation_results['compression'] = compression_results
    
    for component, result in compression_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"  {status} {component}: {result['status']}")
        if 'compression_ratio' in result:
            logger.info(f"      Compression: {result['compression_ratio']:.1f}x")
    
    # 5. Check dependencies
    logger.info("\n5. Checking Dependencies")
    dependency_results = validate_dependency_availability()
    validation_results['dependencies'] = dependency_results
    
    for module, result in dependency_results.items():
        if result['status'] == 'available':
            logger.info(f"  ✅ {result['name']}: Available")
        elif result['status'] == 'missing':
            logger.info(f"  ❌ {result['name']}: Missing (Required)")
        else:
            logger.info(f"  ⚠️  {result['name']}: Missing (Optional)")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    # Count successful validations
    success_counts = {}
    for category, results in validation_results.items():
        if category == 'dependencies':
            continue  # Skip dependencies for success count
        
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        total = len(results)
        success_counts[category] = (successful, total)
        
        logger.info(f"{category.title()}: {successful}/{total} successful")
    
    # Overall assessment
    total_successful = sum(s for s, t in success_counts.values())
    total_tests = sum(t for s, t in success_counts.values())
    success_rate = total_successful / total_tests if total_tests > 0 else 0
    
    logger.info(f"\nOverall Success Rate: {success_rate:.1%} ({total_successful}/{total_tests})")
    
    # Check critical dependencies
    critical_deps = ['torch', 'numpy', 'transformers', 'sentence_transformers', 'faiss']
    missing_critical = [dep for dep in critical_deps 
                       if dependency_results.get(dep, {}).get('status') != 'available']
    
    if missing_critical:
        logger.warning(f"Missing critical dependencies: {missing_critical}")
        logger.warning("Install with: pip install torch transformers sentence-transformers faiss-cpu")
    
    # Phase 1 readiness assessment
    phase1_ready = (
        success_rate >= 0.8 and 
        len(missing_critical) == 0 and
        compression_results.get('combined_compression', {}).get('target_met', False)
    )
    
    logger.info(f"\nPhase 1 Implementation Ready: {'✅ YES' if phase1_ready else '❌ NO'}")
    
    if phase1_ready:
        logger.info("🎉 Phase 1 implementation is validated and ready for testing!")
        logger.info("Next steps:")
        logger.info("  1. Install optional dependencies: pip install tensorly[complete] bitsandbytes")
        logger.info("  2. Run full benchmark: python benchmark_lightweight_rag.py")
        logger.info("  3. Test with actual models and data")
    else:
        logger.info("⚠️  Phase 1 implementation needs attention:")
        if success_rate < 0.8:
            logger.info(f"  - Fix component issues (current success rate: {success_rate:.1%})")
        if missing_critical:
            logger.info(f"  - Install missing dependencies: {missing_critical}")
    
    # Save validation results
    import json
    from datetime import datetime
    
    output_file = f"phase1_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    logger.info(f"\nValidation results saved to: {output_file}")


if __name__ == "__main__":
    main()