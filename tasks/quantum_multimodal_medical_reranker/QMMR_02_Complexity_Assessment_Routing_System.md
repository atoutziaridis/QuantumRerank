# Task QMMR-02: Complexity Assessment & Routing System

## Objective

Implement an intelligent complexity assessment and routing system that analyzes multimodal medical queries and candidates to determine whether they should be processed by the classical reranker or the quantum multimodal reranker. This system maximizes efficiency by using quantum resources only where they provide genuine advantages.

## Prerequisites

### Completed Tasks
- **QMMR-01**: Multimodal Embedding Integration Foundation
- **Tasks 01-30**: Complete QuantumRerank foundation
- **QRF-01 through QRF-05**: Quantum reranker fixes
- **Industry-standard evaluation framework**: Operational

### Required Components
- `quantum_rerank.core.multimodal_embedding_processor.MultimodalEmbeddingProcessor`
- `quantum_rerank.core.quantum_similarity_engine.QuantumSimilarityEngine`
- `quantum_rerank.evaluation.industry_standard_evaluation.IndustryStandardEvaluator`
- `quantum_rerank.evaluation.medical_relevance.MedicalDomainClassifier`

## Technical Reference

### Primary Documentation
- **PRD Section 2.2**: Hybrid Classical-Quantum Architecture
- **PRD Section 4.3**: Performance Targets (<500ms batch processing)
- **QMMR Strategic Plan**: Complexity Assessment & Routing (Section 4.3)

### Research Papers (Priority Order)
1. **Quantum Approach for Contextual Search**: Context-aware routing strategies
2. **Measuring Graph Similarity through Quantum Walks**: Complexity metrics
3. **Quantum-inspired Embeddings Projection**: Multimodal complexity assessment
4. **Quantum Geometric Model of Similarity**: Uncertainty quantification

### Existing Code References
- `quantum_rerank/evaluation/industry_standard_evaluation.py` - Evaluation framework
- `quantum_rerank/core/quantum_similarity_engine.py` - Similarity computation
- `quantum_rerank/evaluation/medical_relevance.py` - Medical domain processing

## Implementation Steps

### Step 1: Analyze Complexity Characteristics
Research and identify key complexity indicators for medical queries:

1. **Multimodal Complexity**: Multiple data types (text, clinical, images)
2. **Noise Indicators**: OCR errors, abbreviations, missing data
3. **Uncertainty Markers**: Conflicting information, ambiguous terms
4. **Medical Domain Complexity**: Specialized terminology, clinical correlations

### Step 2: Design Complexity Scoring Algorithm
Create comprehensive complexity assessment framework:

```python
@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for medical queries"""
    
    # Multimodal complexity
    modality_count: int = 0
    modality_diversity: float = 0.0
    cross_modal_dependencies: float = 0.0
    
    # Noise indicators
    ocr_error_probability: float = 0.0
    abbreviation_density: float = 0.0
    missing_data_ratio: float = 0.0
    
    # Uncertainty markers
    term_ambiguity_score: float = 0.0
    conflicting_information: float = 0.0
    diagnostic_uncertainty: float = 0.0
    
    # Medical domain complexity
    medical_terminology_density: float = 0.0
    clinical_correlation_complexity: float = 0.0
    domain_specificity: float = 0.0
    
    # Overall complexity score
    overall_complexity: float = 0.0
```

### Step 3: Implement Complexity Assessment Engine
Build modular complexity assessment system:

```python
class ComplexityAssessmentEngine:
    """
    Comprehensive complexity assessment for medical queries and candidates.
    """
    
    def __init__(self, config: ComplexityAssessmentConfig = None):
        self.config = config or ComplexityAssessmentConfig()
        
        # Initialize assessment modules
        self.multimodal_assessor = MultimodalComplexityAssessor()
        self.noise_assessor = NoiseIndicatorAssessor()
        self.uncertainty_assessor = UncertaintyAssessor()
        self.medical_assessor = MedicalDomainComplexityAssessor()
        
        # Initialize machine learning components
        self.complexity_predictor = ComplexityPredictor()
        
        # Performance monitoring
        self.assessment_stats = {
            'total_assessments': 0,
            'avg_assessment_time_ms': 0.0,
            'routing_accuracy': 0.0
        }
    
    def assess_complexity(self, query: Dict[str, Any], candidates: List[Dict[str, Any]]) -> ComplexityAssessmentResult:
        """
        Assess complexity for query and candidates with <50ms constraint.
        """
        start_time = time.time()
        
        # Assess query complexity
        query_complexity = self._assess_query_complexity(query)
        
        # Assess candidates complexity
        candidate_complexities = [
            self._assess_candidate_complexity(candidate) 
            for candidate in candidates
        ]
        
        # Compute overall complexity metrics
        overall_complexity = self._compute_overall_complexity(
            query_complexity, candidate_complexities
        )
        
        # Generate routing decision
        routing_decision = self._generate_routing_decision(overall_complexity)
        
        # Performance tracking
        elapsed = (time.time() - start_time) * 1000
        self._update_assessment_stats(elapsed)
        
        return ComplexityAssessmentResult(
            query_complexity=query_complexity,
            candidate_complexities=candidate_complexities,
            overall_complexity=overall_complexity,
            routing_decision=routing_decision,
            assessment_time_ms=elapsed
        )
    
    def _assess_query_complexity(self, query: Dict[str, Any]) -> ComplexityMetrics:
        """Assess complexity of individual query"""
        metrics = ComplexityMetrics()
        
        # Multimodal complexity
        metrics.modality_count = len([k for k in query.keys() if k in ['text', 'clinical_data', 'image']])
        metrics.modality_diversity = self.multimodal_assessor.assess_diversity(query)
        metrics.cross_modal_dependencies = self.multimodal_assessor.assess_dependencies(query)
        
        # Noise indicators
        if 'text' in query:
            metrics.ocr_error_probability = self.noise_assessor.assess_ocr_errors(query['text'])
            metrics.abbreviation_density = self.noise_assessor.assess_abbreviations(query['text'])
        
        if 'clinical_data' in query:
            metrics.missing_data_ratio = self.noise_assessor.assess_missing_data(query['clinical_data'])
        
        # Uncertainty markers
        metrics.term_ambiguity_score = self.uncertainty_assessor.assess_ambiguity(query)
        metrics.conflicting_information = self.uncertainty_assessor.assess_conflicts(query)
        metrics.diagnostic_uncertainty = self.uncertainty_assessor.assess_diagnostic_uncertainty(query)
        
        # Medical domain complexity
        metrics.medical_terminology_density = self.medical_assessor.assess_terminology_density(query)
        metrics.clinical_correlation_complexity = self.medical_assessor.assess_correlation_complexity(query)
        metrics.domain_specificity = self.medical_assessor.assess_domain_specificity(query)
        
        # Compute overall complexity
        metrics.overall_complexity = self._compute_weighted_complexity(metrics)
        
        return metrics
```

### Step 4: Implement Routing Decision Engine
Create intelligent routing system with A/B testing support:

```python
class RoutingDecisionEngine:
    """
    Intelligent routing between classical and quantum rerankers.
    """
    
    def __init__(self, config: RoutingConfig = None):
        self.config = config or RoutingConfig()
        
        # Routing thresholds
        self.quantum_threshold = 0.6  # Route to quantum if complexity > 0.6
        self.classical_threshold = 0.4  # Route to classical if complexity < 0.4
        
        # A/B testing infrastructure
        self.ab_tester = ABTestingFramework()
        
        # Performance tracking
        self.routing_stats = {
            'total_routings': 0,
            'quantum_routings': 0,
            'classical_routings': 0,
            'hybrid_routings': 0,
            'routing_accuracy': 0.0
        }
    
    def route_query(self, complexity_result: ComplexityAssessmentResult) -> RoutingDecision:
        """
        Make routing decision based on complexity assessment.
        """
        overall_complexity = complexity_result.overall_complexity.overall_complexity
        
        # A/B testing override
        if self.ab_tester.is_active():
            ab_decision = self.ab_tester.get_routing_decision(complexity_result)
            if ab_decision:
                return ab_decision
        
        # Standard routing logic
        if overall_complexity >= self.quantum_threshold:
            routing_method = RoutingMethod.QUANTUM
            confidence = min(overall_complexity, 1.0)
        elif overall_complexity <= self.classical_threshold:
            routing_method = RoutingMethod.CLASSICAL
            confidence = 1.0 - overall_complexity
        else:
            # Hybrid approach for medium complexity
            routing_method = RoutingMethod.HYBRID
            confidence = 0.5
        
        # Special cases
        routing_method = self._apply_special_routing_rules(complexity_result, routing_method)
        
        # Update statistics
        self._update_routing_stats(routing_method)
        
        return RoutingDecision(
            method=routing_method,
            confidence=confidence,
            complexity_score=overall_complexity,
            reasoning=self._generate_routing_reasoning(complexity_result, routing_method)
        )
    
    def _apply_special_routing_rules(self, complexity_result: ComplexityAssessmentResult, 
                                   default_method: RoutingMethod) -> RoutingMethod:
        """Apply domain-specific routing rules"""
        
        # Emergency medicine: Always use quantum for rapid pattern recognition
        if complexity_result.query_complexity.domain_specificity > 0.8:
            domain = self._classify_medical_domain(complexity_result)
            if domain == 'emergency_medicine':
                return RoutingMethod.QUANTUM
        
        # High uncertainty: Use quantum for confidence intervals
        if complexity_result.query_complexity.diagnostic_uncertainty > 0.7:
            return RoutingMethod.QUANTUM
        
        # Simple text-only queries: Use classical
        if complexity_result.query_complexity.modality_count == 1:
            return RoutingMethod.CLASSICAL
        
        return default_method
```

### Step 5: Implement Hybrid Reranking Pipeline
Create unified pipeline integrating classical and quantum rerankers:

```python
class HybridQuantumClassicalPipeline:
    """
    Unified pipeline combining classical and quantum reranking with intelligent routing.
    """
    
    def __init__(self, config: HybridPipelineConfig = None):
        self.config = config or HybridPipelineConfig()
        
        # Initialize components
        self.classical_reranker = ClassicalReranker()
        self.quantum_reranker = MultimodalQuantumSimilarityEngine()
        self.complexity_engine = ComplexityAssessmentEngine()
        self.routing_engine = RoutingDecisionEngine()
        
        # Performance monitoring
        self.pipeline_stats = {
            'total_queries': 0,
            'quantum_queries': 0,
            'classical_queries': 0,
            'hybrid_queries': 0,
            'avg_processing_time_ms': 0.0
        }
    
    def rerank(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
              top_k: int = 10) -> HybridRerankingResult:
        """
        Perform hybrid reranking with intelligent routing.
        """
        start_time = time.time()
        
        # Assess complexity
        complexity_result = self.complexity_engine.assess_complexity(query, candidates)
        
        # Make routing decision
        routing_decision = self.routing_engine.route_query(complexity_result)
        
        # Execute reranking based on routing decision
        if routing_decision.method == RoutingMethod.CLASSICAL:
            reranking_result = self._execute_classical_reranking(query, candidates, top_k)
        elif routing_decision.method == RoutingMethod.QUANTUM:
            reranking_result = self._execute_quantum_reranking(query, candidates, top_k)
        else:  # HYBRID
            reranking_result = self._execute_hybrid_reranking(query, candidates, top_k)
        
        # Performance tracking
        elapsed = (time.time() - start_time) * 1000
        self._update_pipeline_stats(routing_decision.method, elapsed)
        
        return HybridRerankingResult(
            reranked_candidates=reranking_result.reranked_candidates,
            routing_decision=routing_decision,
            complexity_assessment=complexity_result,
            processing_time_ms=elapsed,
            quantum_advantage_score=reranking_result.quantum_advantage_score
        )
    
    def _execute_quantum_reranking(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], 
                                  top_k: int) -> RerankingResult:
        """Execute quantum multimodal reranking"""
        similarities = []
        
        for candidate in candidates:
            similarity, metadata = self.quantum_reranker.compute_multimodal_similarity(query, candidate)
            similarities.append((candidate, similarity, metadata))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return RerankingResult(
            reranked_candidates=similarities[:top_k],
            method='quantum',
            quantum_advantage_score=self._compute_quantum_advantage_score(similarities)
        )
```

### Step 6: Implement A/B Testing Framework
Create comprehensive A/B testing for routing decisions:

```python
class ABTestingFramework:
    """
    A/B testing framework for routing decisions and performance comparison.
    """
    
    def __init__(self, config: ABTestingConfig = None):
        self.config = config or ABTestingConfig()
        
        # Test configurations
        self.active_tests = {}
        self.test_results = {}
        
        # Performance tracking
        self.test_stats = {
            'total_tests': 0,
            'active_tests': 0,
            'completed_tests': 0
        }
    
    def create_routing_test(self, test_name: str, test_config: ABTestConfig) -> ABTest:
        """
        Create A/B test for routing decisions.
        """
        test = ABTest(
            name=test_name,
            config=test_config,
            start_time=time.time(),
            metrics_collector=ABTestMetricsCollector()
        )
        
        self.active_tests[test_name] = test
        return test
    
    def get_routing_decision(self, complexity_result: ComplexityAssessmentResult) -> Optional[RoutingDecision]:
        """
        Get routing decision from active A/B tests.
        """
        for test_name, test in self.active_tests.items():
            if test.should_apply(complexity_result):
                decision = test.get_routing_decision(complexity_result)
                test.record_decision(decision)
                return decision
        
        return None
    
    def analyze_test_results(self, test_name: str) -> ABTestAnalysis:
        """
        Analyze results from completed A/B test.
        """
        if test_name not in self.test_results:
            raise ValueError(f"Test {test_name} not found in results")
        
        test_data = self.test_results[test_name]
        
        return ABTestAnalysis(
            test_name=test_name,
            classical_performance=test_data['classical_metrics'],
            quantum_performance=test_data['quantum_metrics'],
            statistical_significance=test_data['statistical_tests'],
            recommendations=test_data['recommendations']
        )
```

## Success Criteria

### Functional Requirements
- [ ] **Complexity Assessment**: Accurately assess multimodal query complexity
- [ ] **Routing Decision**: Intelligent routing between classical and quantum rerankers
- [ ] **A/B Testing**: Comprehensive testing framework for routing decisions
- [ ] **Performance Monitoring**: Real-time performance tracking and optimization
- [ ] **Hybrid Pipeline**: Unified pipeline integrating both reranking methods

### Performance Benchmarks
- [ ] **Assessment Speed**: <50ms for complexity assessment
- [ ] **Routing Accuracy**: >80% correct routing decisions
- [ ] **Overall Latency**: <500ms for batch processing (50 documents)
- [ ] **Resource Efficiency**: Optimal use of quantum resources (10-20% of queries)

### Integration Requirements
- [ ] **Seamless Integration**: Works with existing evaluation framework
- [ ] **Backward Compatibility**: Supports single-modal queries
- [ ] **API Compatibility**: Maintains existing API interfaces
- [ ] **Configuration**: Flexible configuration for different deployment scenarios

## Files to Create/Modify

### New Files
```
quantum_rerank/routing/complexity_assessment_engine.py
quantum_rerank/routing/routing_decision_engine.py
quantum_rerank/routing/hybrid_pipeline.py
quantum_rerank/routing/ab_testing_framework.py
quantum_rerank/routing/complexity_metrics.py
quantum_rerank/config/routing_config.py
```

### Modified Files
```
quantum_rerank/core/quantum_similarity_engine.py (integration)
quantum_rerank/evaluation/industry_standard_evaluation.py (A/B testing)
quantum_rerank/api/endpoints/rerank.py (hybrid pipeline)
```

### Test Files
```
tests/unit/test_complexity_assessment_engine.py
tests/unit/test_routing_decision_engine.py
tests/integration/test_hybrid_pipeline.py
tests/integration/test_ab_testing_framework.py
```

## Testing & Validation

### Unit Tests
```python
def test_complexity_assessment_engine():
    """Test complexity assessment accuracy"""
    engine = ComplexityAssessmentEngine()
    
    # Simple query (should route to classical)
    simple_query = {'text': 'headache treatment'}
    simple_candidates = [{'text': 'aspirin for headache'}]
    
    result = engine.assess_complexity(simple_query, simple_candidates)
    assert result.overall_complexity.overall_complexity < 0.4
    
    # Complex multimodal query (should route to quantum)
    complex_query = {
        'text': 'pt c/o CP w/ SOB',  # Noisy with abbreviations
        'clinical_data': {'age': 45, 'bp': '???', 'ecg': 'abnormal ST'},  # Missing data
        'image': 'chest_xray.jpg'  # Multimodal
    }
    
    result = engine.assess_complexity(complex_query, [])
    assert result.overall_complexity.overall_complexity > 0.6

def test_routing_decision_engine():
    """Test routing decision accuracy"""
    engine = RoutingDecisionEngine()
    
    # High complexity -> quantum
    high_complexity = ComplexityAssessmentResult(
        overall_complexity=ComplexityMetrics(overall_complexity=0.8)
    )
    
    decision = engine.route_query(high_complexity)
    assert decision.method == RoutingMethod.QUANTUM
    assert decision.confidence > 0.7
    
    # Low complexity -> classical
    low_complexity = ComplexityAssessmentResult(
        overall_complexity=ComplexityMetrics(overall_complexity=0.2)
    )
    
    decision = engine.route_query(low_complexity)
    assert decision.method == RoutingMethod.CLASSICAL
    assert decision.confidence > 0.7
```

### Integration Tests
```python
def test_hybrid_pipeline_end_to_end():
    """Test complete hybrid pipeline"""
    pipeline = HybridQuantumClassicalPipeline()
    
    # Test with various complexity levels
    test_cases = [
        # Simple case -> classical
        {
            'query': {'text': 'diabetes treatment'},
            'candidates': [{'text': 'metformin for diabetes'}],
            'expected_method': RoutingMethod.CLASSICAL
        },
        # Complex case -> quantum
        {
            'query': {
                'text': 'pt c/o CP w/ SOB',
                'clinical_data': {'age': 45, 'bp': '???'},
                'image': 'chest_xray.jpg'
            },
            'candidates': [{'text': 'cardiac catheterization'}],
            'expected_method': RoutingMethod.QUANTUM
        }
    ]
    
    for test_case in test_cases:
        result = pipeline.rerank(test_case['query'], test_case['candidates'])
        assert result.routing_decision.method == test_case['expected_method']
        assert result.processing_time_ms < 500  # PRD constraint
```

### Performance Validation
```python
def test_performance_constraints():
    """Validate all performance constraints"""
    engine = ComplexityAssessmentEngine()
    
    # Test assessment speed
    query = {'text': 'test query', 'clinical_data': {'age': 30}}
    candidates = [{'text': f'candidate {i}'} for i in range(50)]
    
    start_time = time.time()
    result = engine.assess_complexity(query, candidates)
    elapsed = (time.time() - start_time) * 1000
    
    assert elapsed < 50  # Assessment should be <50ms
    assert result.assessment_time_ms < 50

def test_routing_accuracy():
    """Test routing accuracy with ground truth"""
    engine = RoutingDecisionEngine()
    
    # Create test cases with known optimal routing
    test_cases = create_routing_test_cases()
    
    correct_decisions = 0
    for test_case in test_cases:
        decision = engine.route_query(test_case.complexity_result)
        if decision.method == test_case.optimal_method:
            correct_decisions += 1
    
    accuracy = correct_decisions / len(test_cases)
    assert accuracy > 0.8  # >80% accuracy requirement
```

### A/B Testing Validation
```python
def test_ab_testing_framework():
    """Test A/B testing framework functionality"""
    framework = ABTestingFramework()
    
    # Create test
    test_config = ABTestConfig(
        test_name='routing_comparison',
        control_method=RoutingMethod.CLASSICAL,
        treatment_method=RoutingMethod.QUANTUM,
        sample_size=1000
    )
    
    test = framework.create_routing_test('routing_comparison', test_config)
    assert test.name == 'routing_comparison'
    assert test.config == test_config
    
    # Simulate test execution
    for i in range(100):
        complexity_result = create_mock_complexity_result()
        decision = framework.get_routing_decision(complexity_result)
        if decision:
            assert decision.method in [RoutingMethod.CLASSICAL, RoutingMethod.QUANTUM]
```

## Expected Outputs

### Functional Outputs
- Working complexity assessment engine with multimodal support
- Intelligent routing system with >80% accuracy
- Comprehensive A/B testing framework
- Unified hybrid pipeline integrating classical and quantum rerankers

### Performance Metrics
- Complexity assessment: <50ms per query
- Routing accuracy: >80% correct decisions
- Overall processing: <500ms for 50 documents
- Resource efficiency: 10-20% of queries routed to quantum

### Integration Verification
- Seamless integration with existing evaluation framework
- Backward compatibility with single-modal queries
- API compatibility maintained
- Comprehensive monitoring and logging

## Risk Mitigation

### Technical Risks
- **Assessment accuracy**: Extensive training data and validation
- **Routing overhead**: Optimized assessment algorithms
- **Performance degradation**: Continuous monitoring and optimization

### Operational Risks
- **A/B testing complexity**: Gradual rollout with careful monitoring
- **Routing decision errors**: Conservative thresholds and fallback options
- **Resource utilization**: Dynamic threshold adjustment

## Dependencies

### Internal Dependencies
- `quantum_rerank.core.multimodal_embedding_processor.MultimodalEmbeddingProcessor`
- `quantum_rerank.core.quantum_similarity_engine.QuantumSimilarityEngine`
- `quantum_rerank.evaluation.industry_standard_evaluation.IndustryStandardEvaluator`

### External Dependencies
- `scikit-learn` (for complexity prediction)
- `numpy` (for statistical computations)
- `pandas` (for A/B testing analytics)

## Completion Criteria

This task is complete when:
1. ✅ Complexity assessment engine accurately evaluates multimodal queries
2. ✅ Routing decision engine achieves >80% accuracy in routing decisions
3. ✅ A/B testing framework enables comprehensive performance comparison
4. ✅ Hybrid pipeline integrates classical and quantum rerankers seamlessly
5. ✅ All performance constraints met (<50ms assessment, <500ms processing)
6. ✅ Integration with existing system maintains backward compatibility
7. ✅ Comprehensive test suite passes with >90% coverage

**Next Task**: QMMR-03 - Quantum Multimodal Similarity Engine