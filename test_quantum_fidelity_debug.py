"""
Test Script for QRF-01: Debug Quantum Fidelity Saturation Issue

This script runs comprehensive debugging analysis to identify and fix
quantum fidelity saturation issues in the quantum reranker.

Based on:
- Task QRF-01: Debug Quantum Fidelity Saturation Issue
- Documentation-first approach from instructions.md
"""

import sys
import os
import logging

# Add project root to path
sys.path.append('/Users/alkist/Projects/QuantumRerank')

from debug_tools.quantum_state_analyzer import run_qrf01_debug_analysis, QuantumStateAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_medical_texts_fidelity():
    """Test quantum fidelity with medical texts to reproduce the saturation issue."""
    
    # Medical texts that should show different similarities
    medical_texts = [
        # Medical content
        "The patient presented with acute myocardial infarction, elevated troponin levels, and ST-segment elevation on ECG.",
        "Diabetes mellitus type 2 is characterized by insulin resistance, hyperglycemia, and increased risk of cardiovascular complications.",
        "Magnetic resonance imaging revealed multiple sclerosis lesions in the white matter of the brain and spinal cord.",
        
        # Non-medical content  
        "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement for computational advantages.",
        "Machine learning algorithms can process large datasets to identify patterns and make predictions about future events.",
        
        # Very different content
        "The weather today is sunny with a temperature of 75 degrees and low humidity levels.",
        "I went to the grocery store to buy apples, oranges, bread, and milk for dinner tonight."
    ]
    
    print("Testing Quantum Fidelity with Medical Texts")
    print("="*60)
    
    # Run comprehensive debug analysis
    results = run_qrf01_debug_analysis(medical_texts)
    
    # Additional detailed analysis for critical pairs
    analyzer = QuantumStateAnalyzer(n_qubits=4)
    
    print("\nDetailed Pairwise Analysis:")
    print("-"*30)
    
    # Test specific pairs that should show different fidelities
    test_pairs = [
        (0, 1),  # Medical to medical (should be higher)
        (0, 3),  # Medical to quantum (should be lower)  
        (3, 4),  # Quantum to ML (should be medium)
        (0, 5),  # Medical to weather (should be very low)
        (0, 0),  # Identical (should be 1.0)
    ]
    
    for i, j in test_pairs:
        if i < len(medical_texts) and j < len(medical_texts):
            fidelity_result = analyzer.debug_fidelity_computation(
                medical_texts[i], medical_texts[j]
            )
            
            print(f"\nPair ({i}, {j}):")
            print(f"  Text 1: {medical_texts[i][:60]}...")
            print(f"  Text 2: {medical_texts[j][:60]}...")
            print(f"  Classical similarity: {fidelity_result.classical_similarity:.6f}")
            print(f"  Theoretical fidelity: {fidelity_result.theoretical_fidelity:.6f}")
            print(f"  SWAP test fidelity: {fidelity_result.swap_test_fidelity:.6f}")
            print(f"  Discrimination score: {fidelity_result.discrimination_score:.6f}")
            print(f"  Issues identified: {fidelity_result.issue_identified}")
    
    return results


def validate_swap_test_implementation():
    """Validate SWAP test implementation with known quantum states."""
    print("\n" + "="*60)
    print("SWAP Test Implementation Validation")
    print("="*60)
    
    analyzer = QuantumStateAnalyzer(n_qubits=4)
    validation_results = analyzer.validate_swap_test_with_known_states()
    
    print("\nValidation Results:")
    for test_name, result in validation_results.items():
        if test_name != 'overall_validation':
            print(f"\n{test_name.replace('_', ' ').title()}:")
            print(f"  Fidelity: {result['fidelity']:.6f}")
            print(f"  Expected: {result['expected']:.6f}")
            print(f"  Error: {result['error']:.6f}")
            print(f"  Status: {'PASS' if result['pass'] else 'FAIL'}")
    
    overall = validation_results['overall_validation']
    print(f"\nOverall Validation: {'PASS' if overall['pass'] else 'FAIL'}")
    print(f"Tests passed: {overall['passed_tests']}/{overall['total_tests']}")
    
    return validation_results


def test_amplitude_encoding_discrimination():
    """Test amplitude encoding discrimination with controlled inputs."""
    print("\n" + "="*60)
    print("Amplitude Encoding Discrimination Test")
    print("="*60)
    
    analyzer = QuantumStateAnalyzer(n_qubits=4)
    
    # Test with very different texts
    very_different_texts = [
        "Medical diagnosis requires careful examination of symptoms and laboratory results.",
        "Quantum physics equations describe wave-particle duality and uncertainty principles.",
        "The cat sat on the mat and purred contentedly in the warm sunshine."
    ]
    
    print("\nTesting amplitude encoding with very different semantic content:")
    
    for i, text in enumerate(very_different_texts):
        state_analysis = analyzer.analyze_quantum_state_preparation(text)
        
        print(f"\nText {i}: {text[:60]}...")
        print(f"  Information loss: {state_analysis.information_loss:.6f}")
        print(f"  State norm: {state_analysis.state_norm:.6f}")
        print(f"  Encoding issues: {state_analysis.metadata.get('encoding_issues', [])}")
        
        # Print amplitude statistics
        amp_stats = state_analysis.metadata.get('amplitude_statistics', {})
        print(f"  Amplitude variance: {amp_stats.get('amplitude_variance', 0):.8f}")
        print(f"  Effective rank: {amp_stats.get('effective_rank', 0):.6f}")
        print(f"  Amplitude entropy: {amp_stats.get('amplitude_entropy', 0):.6f}")
    
    # Test pairwise discrimination
    print(f"\nPairwise fidelity analysis:")
    for i in range(len(very_different_texts)):
        for j in range(i+1, len(very_different_texts)):
            fidelity_result = analyzer.debug_fidelity_computation(
                very_different_texts[i], very_different_texts[j]
            )
            
            print(f"  Pair ({i},{j}): Classical={fidelity_result.classical_similarity:.6f}, "
                  f"Quantum={fidelity_result.theoretical_fidelity:.6f}, "
                  f"Discrimination={fidelity_result.discrimination_score:.6f}")


def main():
    """Main test function for QRF-01 debugging."""
    print("QRF-01: QUANTUM FIDELITY SATURATION DEBUG TEST")
    print("="*60)
    print("Following documentation-first approach from instructions.md")
    print("Based on quantum cosine similarity and geometric similarity papers")
    print()
    
    try:
        # Step 1: Validate SWAP test implementation
        swap_validation = validate_swap_test_implementation()
        
        # Step 2: Test with medical texts (reproducing the issue)
        medical_results = test_medical_texts_fidelity()
        
        # Step 3: Test amplitude encoding discrimination
        test_amplitude_encoding_discrimination()
        
        # Step 4: Summary and recommendations
        print("\n" + "="*60)
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*60)
        
        if medical_results:
            print("\nKey Findings:")
            disc = medical_results['discrimination_analysis']
            print(f"- Quantum fidelity range: {disc['quantum_fidelity_range']:.6f}")
            print(f"- Classical similarity range: {disc['classical_similarity_range']:.6f}")
            print(f"- Discrimination ratio: {disc['discrimination_ratio']:.6f}")
            print(f"- Quantum saturation detected: {disc['quantum_saturation_detected']}")
            
            print(f"\nRecommendations:")
            for i, rec in enumerate(medical_results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # SWAP test summary
        if swap_validation:
            overall = swap_validation['overall_validation']
            print(f"\nSWAP Test Validation: {'PASS' if overall['pass'] else 'FAIL'}")
            
            if not overall['pass']:
                print("CRITICAL: SWAP test implementation has issues!")
                for test_name, result in swap_validation.items():
                    if test_name != 'overall_validation' and not result.get('pass', True):
                        print(f"  - {test_name}: Expected {result['expected']:.3f}, "
                              f"Got {result['fidelity']:.3f} (error: {result['error']:.3f})")
        
        print(f"\nNext Steps for QRF-01:")
        print(f"1. Address critical amplitude encoding issues")
        print(f"2. Implement alternative encoding methods if needed")
        print(f"3. Fix SWAP test implementation if validation failed")
        print(f"4. Test improved implementation with real medical data")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)