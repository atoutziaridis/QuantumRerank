"""
Intelligent Diagnostic and Troubleshooting System for QuantumRerank.

This module provides AI-powered problem diagnosis, automated troubleshooting,
performance analysis, and solution recommendation for comprehensive support
of the QuantumRerank quantum-enhanced information retrieval system.
"""

import re
import json
import time
import traceback
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import threading

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ProblemCategory(Enum):
    """Categories of problems that can occur."""
    INSTALLATION = "installation"
    CONFIGURATION = "configuration"
    QUANTUM_EXECUTION = "quantum_execution"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    API_USAGE = "api_usage"
    MEMORY = "memory"
    NETWORK = "network"
    SECURITY = "security"
    DATA_PROCESSING = "data_processing"


class ProblemSeverity(Enum):
    """Severity levels for problems."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SolutionType(Enum):
    """Types of solutions."""
    QUICK_FIX = "quick_fix"
    CONFIGURATION_CHANGE = "configuration_change"
    CODE_MODIFICATION = "code_modification"
    ENVIRONMENT_SETUP = "environment_setup"
    UPGRADE_DOWNGRADE = "upgrade_downgrade"
    DOCUMENTATION_REFERENCE = "documentation_reference"


@dataclass
class ProblemSymptom:
    """Individual problem symptom."""
    symptom_id: str
    description: str
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    severity: ProblemSeverity = ProblemSeverity.MEDIUM
    frequency: int = 1
    first_occurred: float = field(default_factory=time.time)
    last_occurred: float = field(default_factory=time.time)


@dataclass
class ProblemClassification:
    """Result of problem classification."""
    problem_type: ProblemCategory
    confidence: float
    root_causes: List[str] = field(default_factory=list)
    related_symptoms: List[str] = field(default_factory=list)
    severity: ProblemSeverity = ProblemSeverity.MEDIUM
    urgency: str = "normal"  # low, normal, high, urgent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    """Recommended solution for a problem."""
    solution_id: str
    title: str
    description: str
    solution_type: SolutionType
    steps: List[str] = field(default_factory=list)
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    estimated_time_minutes: int = 10
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    success_rate: float = 0.8
    prerequisites: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    related_docs: List[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Complete diagnosis result with recommendations."""
    problem_classification: ProblemClassification
    recommended_solutions: List[Solution]
    alternative_solutions: List[Solution] = field(default_factory=list)
    performance_analysis: Optional[Dict[str, Any]] = None
    preventive_measures: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)
    escalation_path: Optional[str] = None
    diagnosis_confidence: float = 0.0
    diagnosis_time_ms: float = 0.0


class ProblemClassifier:
    """Intelligent problem classification system."""
    
    def __init__(self):
        """Initialize problem classifier."""
        self.classification_rules = self._initialize_classification_rules()
        self.error_patterns = self._initialize_error_patterns()
        self.symptom_history = defaultdict(list)
        
        self.logger = logger
        logger.info("Initialized ProblemClassifier")
    
    def _initialize_classification_rules(self) -> Dict[ProblemCategory, Dict[str, Any]]:
        """Initialize problem classification rules."""
        return {
            ProblemCategory.INSTALLATION: {
                "keywords": ["install", "pip", "conda", "dependency", "module not found", "import error"],
                "error_patterns": [
                    r"ModuleNotFoundError",
                    r"ImportError",
                    r"pip.*failed",
                    r"conda.*error",
                    r"No module named"
                ],
                "severity_indicators": {
                    ProblemSeverity.CRITICAL: ["cannot import quantum_rerank", "installation failed completely"],
                    ProblemSeverity.HIGH: ["missing core dependencies", "version conflict"],
                    ProblemSeverity.MEDIUM: ["optional dependency missing"],
                    ProblemSeverity.LOW: ["warning during installation"]
                }
            },
            ProblemCategory.CONFIGURATION: {
                "keywords": ["config", "settings", "parameter", "invalid value", "configuration error"],
                "error_patterns": [
                    r"ConfigurationError",
                    r"Invalid.*config",
                    r"Missing.*configuration",
                    r"ValueError.*parameter"
                ],
                "severity_indicators": {
                    ProblemSeverity.CRITICAL: ["cannot start application", "invalid core configuration"],
                    ProblemSeverity.HIGH: ["quantum backend configuration error"],
                    ProblemSeverity.MEDIUM: ["suboptimal configuration"],
                    ProblemSeverity.LOW: ["configuration warning"]
                }
            },
            ProblemCategory.QUANTUM_EXECUTION: {
                "keywords": ["quantum", "circuit", "qiskit", "pennylane", "execution error", "quantum backend"],
                "error_patterns": [
                    r"QuantumError",
                    r"CircuitError",
                    r"Backend.*error",
                    r"Quantum.*failed",
                    r"QiskitError"
                ],
                "severity_indicators": {
                    ProblemSeverity.CRITICAL: ["quantum backend unavailable", "circuit execution failed"],
                    ProblemSeverity.HIGH: ["quantum simulation error", "state preparation failed"],
                    ProblemSeverity.MEDIUM: ["quantum warning", "approximate results"],
                    ProblemSeverity.LOW: ["quantum optimization suggestion"]
                }
            },
            ProblemCategory.PERFORMANCE: {
                "keywords": ["slow", "timeout", "memory", "performance", "optimization"],
                "error_patterns": [
                    r"TimeoutError",
                    r"MemoryError",
                    r"PerformanceWarning",
                    r"slow.*execution"
                ],
                "severity_indicators": {
                    ProblemSeverity.CRITICAL: ["system unresponsive", "out of memory"],
                    ProblemSeverity.HIGH: ["significant slowdown", "memory pressure"],
                    ProblemSeverity.MEDIUM: ["suboptimal performance"],
                    ProblemSeverity.LOW: ["optimization opportunity"]
                }
            },
            ProblemCategory.API_USAGE: {
                "keywords": ["api", "method", "function", "parameter", "usage error", "invalid call"],
                "error_patterns": [
                    r"TypeError",
                    r"AttributeError",
                    r"Invalid.*parameter",
                    r"Unexpected.*argument"
                ],
                "severity_indicators": {
                    ProblemSeverity.HIGH: ["core API misuse", "breaking change"],
                    ProblemSeverity.MEDIUM: ["deprecated API usage", "parameter error"],
                    ProblemSeverity.LOW: ["usage warning", "style issue"]
                }
            }
        }
    
    def _initialize_error_patterns(self) -> Dict[str, ProblemCategory]:
        """Initialize common error patterns and their categories."""
        return {
            # Installation patterns
            r"No module named ['\"]quantum_rerank['\"]": ProblemCategory.INSTALLATION,
            r"pip install.*failed": ProblemCategory.INSTALLATION,
            r"ImportError.*quantum_rerank": ProblemCategory.INSTALLATION,
            
            # Configuration patterns
            r"ConfigurationError": ProblemCategory.CONFIGURATION,
            r"Invalid configuration.*": ProblemCategory.CONFIGURATION,
            
            # Quantum execution patterns
            r"QiskitError": ProblemCategory.QUANTUM_EXECUTION,
            r"QuantumError": ProblemCategory.QUANTUM_EXECUTION,
            r"Circuit.*failed": ProblemCategory.QUANTUM_EXECUTION,
            
            # Performance patterns
            r"TimeoutError": ProblemCategory.PERFORMANCE,
            r"MemoryError": ProblemCategory.PERFORMANCE,
            r"execution.*too slow": ProblemCategory.PERFORMANCE,
            
            # API usage patterns
            r"TypeError.*parameter": ProblemCategory.API_USAGE,
            r"AttributeError.*method": ProblemCategory.API_USAGE
        }
    
    def classify_problem(self, symptoms: Dict[str, Any]) -> ProblemClassification:
        """
        Classify problem based on symptoms.
        
        Args:
            symptoms: Dictionary containing problem symptoms
            
        Returns:
            ProblemClassification with analysis results
        """
        error_message = symptoms.get("error_message", "")
        stack_trace = symptoms.get("stack_trace", "")
        description = symptoms.get("description", "")
        context = symptoms.get("context", {})
        
        # Combine all text for analysis
        full_text = f"{error_message} {stack_trace} {description}".lower()
        
        # Score each category
        category_scores = {}
        
        for category, rules in self.classification_rules.items():
            score = 0
            
            # Check keywords
            for keyword in rules["keywords"]:
                if keyword in full_text:
                    score += 1
            
            # Check error patterns
            for pattern in rules["error_patterns"]:
                if re.search(pattern, full_text, re.IGNORECASE):
                    score += 3  # Patterns have higher weight
            
            category_scores[category] = score
        
        # Check specific error patterns
        for pattern, category in self.error_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                category_scores[category] = category_scores.get(category, 0) + 5
        
        # Find best match
        if not category_scores or max(category_scores.values()) == 0:
            # Default classification
            best_category = ProblemCategory.API_USAGE
            confidence = 0.3
        else:
            best_category = max(category_scores, key=category_scores.get)
            max_score = category_scores[best_category]
            total_score = sum(category_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
        
        # Determine severity
        severity = self._determine_severity(best_category, full_text)
        
        # Extract root causes
        root_causes = self._extract_root_causes(best_category, symptoms)
        
        # Determine urgency
        urgency = self._determine_urgency(severity, context)
        
        return ProblemClassification(
            problem_type=best_category,
            confidence=confidence,
            root_causes=root_causes,
            severity=severity,
            urgency=urgency,
            metadata={
                "category_scores": category_scores,
                "analysis_keywords": self._extract_keywords(full_text)
            }
        )
    
    def _determine_severity(self, category: ProblemCategory, text: str) -> ProblemSeverity:
        """Determine problem severity based on category and indicators."""
        if category not in self.classification_rules:
            return ProblemSeverity.MEDIUM
        
        severity_indicators = self.classification_rules[category]["severity_indicators"]
        
        # Check indicators from most severe to least
        for severity in [ProblemSeverity.CRITICAL, ProblemSeverity.HIGH, ProblemSeverity.MEDIUM, ProblemSeverity.LOW]:
            indicators = severity_indicators.get(severity, [])
            for indicator in indicators:
                if indicator.lower() in text:
                    return severity
        
        return ProblemSeverity.MEDIUM
    
    def _extract_root_causes(self, category: ProblemCategory, symptoms: Dict[str, Any]) -> List[str]:
        """Extract potential root causes based on category and symptoms."""
        root_causes = []
        
        error_message = symptoms.get("error_message", "")
        context = symptoms.get("context", {})
        
        if category == ProblemCategory.INSTALLATION:
            if "permission" in error_message.lower():
                root_causes.append("Insufficient permissions for installation")
            if "network" in error_message.lower() or "timeout" in error_message.lower():
                root_causes.append("Network connectivity issues")
            if "version" in error_message.lower():
                root_causes.append("Package version conflicts")
                
        elif category == ProblemCategory.QUANTUM_EXECUTION:
            if "backend" in error_message.lower():
                root_causes.append("Quantum backend configuration or availability")
            if "circuit" in error_message.lower():
                root_causes.append("Invalid quantum circuit construction")
            if "parameter" in error_message.lower():
                root_causes.append("Invalid quantum parameters")
                
        elif category == ProblemCategory.PERFORMANCE:
            if "memory" in error_message.lower():
                root_causes.append("Insufficient memory resources")
            if "timeout" in error_message.lower():
                root_causes.append("Operation taking too long")
            if context.get("large_dataset"):
                root_causes.append("Dataset size exceeding system capacity")
        
        return root_causes
    
    def _determine_urgency(self, severity: ProblemSeverity, context: Dict[str, Any]) -> str:
        """Determine problem urgency based on severity and context."""
        if severity == ProblemSeverity.CRITICAL:
            return "urgent"
        elif severity == ProblemSeverity.HIGH:
            if context.get("production_environment"):
                return "urgent"
            return "high"
        elif severity == ProblemSeverity.MEDIUM:
            return "normal"
        else:
            return "low"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from problem text."""
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter for relevant technical terms
        relevant_words = [
            word for word in words
            if len(word) > 3 and word in [
                "quantum", "circuit", "qiskit", "pennylane", "embedding",
                "similarity", "error", "timeout", "memory", "performance",
                "config", "install", "import", "module", "backend"
            ]
        ]
        
        return list(set(relevant_words))


class SolutionRecommender:
    """Recommends solutions based on problem classification."""
    
    def __init__(self):
        """Initialize solution recommender."""
        self.solution_database = self._initialize_solution_database()
        self.solution_effectiveness = defaultdict(lambda: 0.8)  # Default success rate
        
        self.logger = logger
        logger.info("Initialized SolutionRecommender")
    
    def _initialize_solution_database(self) -> Dict[ProblemCategory, List[Solution]]:
        """Initialize database of solutions for different problem categories."""
        return {
            ProblemCategory.INSTALLATION: [
                Solution(
                    solution_id="install_basic",
                    title="Basic Installation via pip",
                    description="Install QuantumRerank using pip package manager",
                    solution_type=SolutionType.ENVIRONMENT_SETUP,
                    steps=[
                        "Ensure Python 3.8+ is installed",
                        "Upgrade pip: pip install --upgrade pip",
                        "Install QuantumRerank: pip install quantum-rerank",
                        "Verify installation: python -c \"import quantum_rerank\""
                    ],
                    code_examples=[
                        {
                            "language": "bash",
                            "code": "pip install quantum-rerank"
                        },
                        {
                            "language": "python",
                            "code": "import quantum_rerank\nprint(quantum_rerank.__version__)"
                        }
                    ],
                    estimated_time_minutes=5,
                    difficulty="beginner",
                    success_rate=0.9
                ),
                Solution(
                    solution_id="install_conda",
                    title="Installation via Conda",
                    description="Install QuantumRerank using conda package manager",
                    solution_type=SolutionType.ENVIRONMENT_SETUP,
                    steps=[
                        "Create new conda environment: conda create -n quantum-rerank python=3.9",
                        "Activate environment: conda activate quantum-rerank",
                        "Install dependencies: conda install numpy scipy",
                        "Install QuantumRerank: pip install quantum-rerank"
                    ],
                    estimated_time_minutes=10,
                    difficulty="beginner"
                ),
                Solution(
                    solution_id="install_dev",
                    title="Development Installation",
                    description="Install QuantumRerank for development",
                    solution_type=SolutionType.ENVIRONMENT_SETUP,
                    steps=[
                        "Clone repository: git clone https://github.com/quantum-rerank/quantum-rerank.git",
                        "Navigate to directory: cd quantum-rerank",
                        "Install in development mode: pip install -e .",
                        "Install development dependencies: pip install -r requirements-dev.txt"
                    ],
                    estimated_time_minutes=15,
                    difficulty="intermediate"
                )
            ],
            ProblemCategory.CONFIGURATION: [
                Solution(
                    solution_id="config_basic",
                    title="Basic Configuration Setup",
                    description="Set up basic QuantumRerank configuration",
                    solution_type=SolutionType.CONFIGURATION_CHANGE,
                    steps=[
                        "Create configuration file: quantum_rerank_config.json",
                        "Set quantum backend: 'qiskit_aer_simulator'",
                        "Configure embedding model: 'all-MiniLM-L6-v2'",
                        "Test configuration"
                    ],
                    code_examples=[
                        {
                            "language": "json",
                            "code": """{
    "quantum_backend": "qiskit_aer_simulator",
    "embedding_model": "all-MiniLM-L6-v2",
    "quantum_shots": 1024,
    "hybrid_weight": 0.5
}"""
                        },
                        {
                            "language": "python",
                            "code": """from quantum_rerank import QuantumRerankConfig
config = QuantumRerankConfig.from_file('quantum_rerank_config.json')"""
                        }
                    ],
                    estimated_time_minutes=10,
                    difficulty="beginner"
                )
            ],
            ProblemCategory.QUANTUM_EXECUTION: [
                Solution(
                    solution_id="quantum_backend_fix",
                    title="Fix Quantum Backend Issues",
                    description="Resolve quantum backend configuration and execution problems",
                    solution_type=SolutionType.CONFIGURATION_CHANGE,
                    steps=[
                        "Check available backends",
                        "Switch to simulator if hardware unavailable",
                        "Verify backend configuration",
                        "Test with simple circuit"
                    ],
                    code_examples=[
                        {
                            "language": "python",
                            "code": """from qiskit import Aer
from quantum_rerank.quantum import QuantumBackend

# List available backends
print(Aer.backends())

# Switch to simulator
backend = QuantumBackend('qiskit_aer_simulator')
print(f"Backend status: {backend.status()}")"""
                        }
                    ],
                    estimated_time_minutes=15,
                    difficulty="intermediate"
                ),
                Solution(
                    solution_id="circuit_debugging",
                    title="Debug Quantum Circuit Issues",
                    description="Identify and fix quantum circuit construction problems",
                    solution_type=SolutionType.CODE_MODIFICATION,
                    steps=[
                        "Validate circuit parameters",
                        "Check qubit count limits",
                        "Verify gate sequences",
                        "Test with simplified circuit"
                    ],
                    estimated_time_minutes=20,
                    difficulty="advanced"
                )
            ],
            ProblemCategory.PERFORMANCE: [
                Solution(
                    solution_id="memory_optimization",
                    title="Memory Usage Optimization",
                    description="Optimize memory usage for large-scale operations",
                    solution_type=SolutionType.CODE_MODIFICATION,
                    steps=[
                        "Implement batch processing",
                        "Use memory-efficient data structures",
                        "Clear intermediate variables",
                        "Monitor memory usage"
                    ],
                    code_examples=[
                        {
                            "language": "python",
                            "code": """# Batch processing example
from quantum_rerank import QuantumSimilarityEngine

engine = QuantumSimilarityEngine()

# Process in batches instead of all at once
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    results = engine.compute_similarities(query, batch)
    # Process results
    del results  # Free memory"""
                        }
                    ],
                    estimated_time_minutes=25,
                    difficulty="intermediate"
                ),
                Solution(
                    solution_id="performance_tuning",
                    title="Performance Parameter Tuning",
                    description="Tune parameters for optimal performance",
                    solution_type=SolutionType.CONFIGURATION_CHANGE,
                    steps=[
                        "Profile current performance",
                        "Adjust quantum shot counts",
                        "Optimize embedding dimensions",
                        "Balance quantum vs classical weights"
                    ],
                    estimated_time_minutes=30,
                    difficulty="advanced"
                )
            ],
            ProblemCategory.API_USAGE: [
                Solution(
                    solution_id="api_basic_usage",
                    title="Correct API Usage Patterns",
                    description="Learn proper API usage to avoid common mistakes",
                    solution_type=SolutionType.DOCUMENTATION_REFERENCE,
                    steps=[
                        "Review API documentation",
                        "Check parameter types and formats",
                        "Use proper initialization patterns",
                        "Handle exceptions appropriately"
                    ],
                    code_examples=[
                        {
                            "language": "python",
                            "code": """# Correct API usage
from quantum_rerank import QuantumSimilarityEngine

# Proper initialization
engine = QuantumSimilarityEngine(
    quantum_backend='qiskit_aer_simulator',
    embedding_model='all-MiniLM-L6-v2'
)

# Proper parameter types
query = "search query"  # string
documents = ["doc1", "doc2"]  # list of strings
top_k = 5  # integer

# Proper error handling
try:
    results = engine.compute_similarities(query, documents)
except Exception as e:
    print(f"Error: {e}")"""
                        }
                    ],
                    estimated_time_minutes=15,
                    difficulty="beginner",
                    related_docs=["api_reference", "getting_started"]
                )
            ]
        }
    
    def recommend_solutions(self, classification: ProblemClassification,
                          performance_analysis: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None) -> List[Solution]:
        """
        Recommend solutions based on problem classification.
        
        Args:
            classification: Problem classification result
            performance_analysis: Optional performance analysis
            context: Optional additional context
            
        Returns:
            List of recommended solutions
        """
        category = classification.problem_type
        
        # Get base solutions for category
        base_solutions = self.solution_database.get(category, [])
        
        # Filter and rank solutions
        recommended = []
        
        for solution in base_solutions:
            # Calculate solution score based on various factors
            score = self._calculate_solution_score(
                solution, classification, performance_analysis, context
            )
            
            if score > 0.3:  # Minimum threshold
                recommended.append((solution, score))
        
        # Sort by score
        recommended.sort(key=lambda x: x[1], reverse=True)
        
        # Return top solutions
        return [solution for solution, score in recommended[:3]]
    
    def _calculate_solution_score(self, solution: Solution,
                                classification: ProblemClassification,
                                performance_analysis: Optional[Dict[str, Any]],
                                context: Optional[Dict[str, Any]]) -> float:
        """Calculate solution relevance score."""
        score = solution.success_rate  # Base score from success rate
        
        # Adjust based on classification confidence
        score *= classification.confidence
        
        # Adjust based on severity match
        if classification.severity == ProblemSeverity.CRITICAL:
            if solution.solution_type in [SolutionType.QUICK_FIX, SolutionType.ENVIRONMENT_SETUP]:
                score *= 1.2
        elif classification.severity == ProblemSeverity.LOW:
            if solution.solution_type == SolutionType.DOCUMENTATION_REFERENCE:
                score *= 1.1
        
        # Adjust based on user context
        if context:
            user_level = context.get("experience_level", "intermediate")
            if user_level == "beginner" and solution.difficulty == "beginner":
                score *= 1.2
            elif user_level == "advanced" and solution.difficulty == "advanced":
                score *= 1.1
            elif user_level == "beginner" and solution.difficulty == "advanced":
                score *= 0.7
        
        # Adjust based on performance analysis
        if performance_analysis:
            if "memory_issue" in performance_analysis and "memory" in solution.title.lower():
                score *= 1.3
            if "timeout" in performance_analysis and "performance" in solution.title.lower():
                score *= 1.3
        
        return min(1.0, score)


class PerformanceAnalyzer:
    """Analyzes performance issues and provides optimization recommendations."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.performance_baselines = {
            "embedding_generation_ms": 100,
            "quantum_circuit_execution_ms": 500,
            "similarity_computation_ms": 50,
            "memory_usage_mb": 512,
            "batch_processing_efficiency": 0.8
        }
        
        self.logger = logger
        logger.info("Initialized PerformanceAnalyzer")
    
    def analyze_performance_issue(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance data to identify issues.
        
        Args:
            performance_data: Performance metrics and measurements
            
        Returns:
            Analysis results with recommendations
        """
        analysis = {
            "issues_detected": [],
            "bottlenecks": [],
            "recommendations": [],
            "optimization_opportunities": [],
            "resource_usage": {},
            "performance_score": 1.0
        }
        
        # Analyze timing metrics
        if "timing" in performance_data:
            timing_analysis = self._analyze_timing(performance_data["timing"])
            analysis["issues_detected"].extend(timing_analysis["issues"])
            analysis["bottlenecks"].extend(timing_analysis["bottlenecks"])
            analysis["recommendations"].extend(timing_analysis["recommendations"])
        
        # Analyze memory usage
        if "memory" in performance_data:
            memory_analysis = self._analyze_memory(performance_data["memory"])
            analysis["issues_detected"].extend(memory_analysis["issues"])
            analysis["recommendations"].extend(memory_analysis["recommendations"])
            analysis["resource_usage"]["memory"] = memory_analysis["usage"]
        
        # Analyze quantum performance
        if "quantum" in performance_data:
            quantum_analysis = self._analyze_quantum_performance(performance_data["quantum"])
            analysis["issues_detected"].extend(quantum_analysis["issues"])
            analysis["recommendations"].extend(quantum_analysis["recommendations"])
        
        # Calculate overall performance score
        analysis["performance_score"] = self._calculate_performance_score(performance_data)
        
        return analysis
    
    def _analyze_timing(self, timing_data: Dict[str, float]) -> Dict[str, List[str]]:
        """Analyze timing performance."""
        analysis = {"issues": [], "bottlenecks": [], "recommendations": []}
        
        for metric, value in timing_data.items():
            baseline = self.performance_baselines.get(metric)
            if baseline and value > baseline * 2:  # More than 2x baseline
                analysis["issues"].append(f"Slow {metric}: {value:.1f}ms (expected <{baseline}ms)")
                analysis["bottlenecks"].append(metric)
                
                # Specific recommendations
                if "embedding" in metric:
                    analysis["recommendations"].append("Consider using smaller embedding models or batch processing")
                elif "quantum" in metric:
                    analysis["recommendations"].append("Optimize quantum circuit depth or switch to faster backend")
                elif "similarity" in metric:
                    analysis["recommendations"].append("Use approximate similarity methods for large datasets")
        
        return analysis
    
    def _analyze_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory usage."""
        analysis = {"issues": [], "recommendations": [], "usage": {}}
        
        current_usage = memory_data.get("current_mb", 0)
        peak_usage = memory_data.get("peak_mb", 0)
        available_memory = memory_data.get("available_mb", 1024)
        
        analysis["usage"] = {
            "current_mb": current_usage,
            "peak_mb": peak_usage,
            "utilization": current_usage / available_memory if available_memory > 0 else 0
        }
        
        # Check for memory issues
        if current_usage > available_memory * 0.9:  # Using >90% of available memory
            analysis["issues"].append("High memory usage detected")
            analysis["recommendations"].extend([
                "Implement batch processing to reduce memory footprint",
                "Clear intermediate variables and call garbage collection",
                "Consider using memory-mapped files for large datasets"
            ])
        
        if peak_usage > available_memory:
            analysis["issues"].append("Memory usage exceeded available memory")
            analysis["recommendations"].append("Increase system memory or optimize data structures")
        
        return analysis
    
    def _analyze_quantum_performance(self, quantum_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze quantum-specific performance."""
        analysis = {"issues": [], "recommendations": []}
        
        circuit_depth = quantum_data.get("circuit_depth", 0)
        shot_count = quantum_data.get("shots", 1024)
        backend_type = quantum_data.get("backend", "unknown")
        
        # Analyze circuit complexity
        if circuit_depth > 20:
            analysis["issues"].append(f"Deep quantum circuit detected: {circuit_depth} layers")
            analysis["recommendations"].extend([
                "Optimize circuit depth using gate optimization techniques",
                "Consider approximate quantum algorithms",
                "Use hardware-aware circuit compilation"
            ])
        
        # Analyze shot count
        if shot_count > 10000:
            analysis["issues"].append(f"High shot count: {shot_count}")
            analysis["recommendations"].append("Reduce shot count for faster execution with acceptable accuracy")
        elif shot_count < 100:
            analysis["recommendations"].append("Consider increasing shot count for better accuracy")
        
        # Backend-specific recommendations
        if "simulator" not in backend_type.lower():
            analysis["recommendations"].append("Consider using simulator for development and testing")
        
        return analysis
    
    def _calculate_performance_score(self, performance_data: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-1)."""
        score = 1.0
        
        # Deduct based on timing issues
        if "timing" in performance_data:
            for metric, value in performance_data["timing"].items():
                baseline = self.performance_baselines.get(metric)
                if baseline and value > baseline:
                    ratio = value / baseline
                    deduction = min(0.3, (ratio - 1) * 0.1)  # Max 30% deduction
                    score -= deduction
        
        # Deduct based on memory issues
        if "memory" in performance_data:
            memory_usage = performance_data["memory"].get("current_mb", 0)
            available = performance_data["memory"].get("available_mb", 1024)
            if memory_usage > available * 0.8:  # >80% memory usage
                score -= 0.2
        
        return max(0.0, score)


class TroubleshootingKnowledgeBase:
    """Knowledge base for troubleshooting information."""
    
    def __init__(self):
        """Initialize troubleshooting knowledge base."""
        self.problem_descriptions = {}
        self.common_causes = {}
        self.prevention_tips = {}
        self.related_issues = {}
        
        self._initialize_knowledge_base()
        
        self.logger = logger
        logger.info("Initialized TroubleshootingKnowledgeBase")
    
    def _initialize_knowledge_base(self) -> None:
        """Initialize knowledge base with common problems and solutions."""
        # Installation problems
        self.problem_descriptions[ProblemCategory.INSTALLATION] = {
            "general": "Issues related to installing QuantumRerank and its dependencies",
            "specific": {
                "module_not_found": "Python cannot find the quantum_rerank module after installation",
                "dependency_conflict": "Conflicts between package versions during installation",
                "permission_error": "Insufficient permissions to install packages"
            }
        }
        
        self.common_causes[ProblemCategory.INSTALLATION] = [
            "Incorrect Python environment or version",
            "Missing or outdated package manager (pip/conda)",
            "Network connectivity issues during download",
            "Insufficient disk space",
            "Permission restrictions",
            "Conflicting package versions"
        ]
        
        # Quantum execution problems
        self.problem_descriptions[ProblemCategory.QUANTUM_EXECUTION] = {
            "general": "Issues with quantum circuit execution and quantum backend operations",
            "specific": {
                "backend_unavailable": "Quantum backend is not available or not responding",
                "circuit_error": "Errors in quantum circuit construction or execution",
                "state_preparation": "Issues with quantum state preparation from classical data"
            }
        }
        
        self.common_causes[ProblemCategory.QUANTUM_EXECUTION] = [
            "Invalid quantum backend configuration",
            "Quantum circuit too complex for backend",
            "Incorrect quantum state preparation",
            "Backend service downtime",
            "Invalid quantum parameters",
            "Circuit compilation errors"
        ]
        
        # Performance problems
        self.prevention_tips[ProblemCategory.PERFORMANCE] = [
            "Profile your code to identify bottlenecks",
            "Use appropriate batch sizes for your system",
            "Monitor memory usage during execution",
            "Choose optimal quantum shot counts",
            "Use caching for repeated computations",
            "Optimize embedding dimensions for your use case"
        ]
        
        # Configuration problems
        self.prevention_tips[ProblemCategory.CONFIGURATION] = [
            "Validate configuration files before use",
            "Keep backup copies of working configurations",
            "Use environment variables for sensitive settings",
            "Test configuration changes in development first",
            "Document your configuration choices"
        ]
    
    def get_problem_description(self, problem_type: str) -> str:
        """Get description for problem type."""
        descriptions = self.problem_descriptions.get(ProblemCategory(problem_type), {})
        return descriptions.get("general", "Unknown problem type")
    
    def get_common_causes(self, problem_type: str) -> List[str]:
        """Get common causes for problem type."""
        return self.common_causes.get(ProblemCategory(problem_type), [])
    
    def get_prevention_tips(self, problem_type: str) -> List[str]:
        """Get prevention tips for problem type."""
        return self.prevention_tips.get(ProblemCategory(problem_type), [])
    
    def get_related_issues(self, problem_type: str) -> List[str]:
        """Get related issues for problem type."""
        return self.related_issues.get(ProblemCategory(problem_type), [])


class IntelligentDiagnosticSystem:
    """
    AI-powered troubleshooting and diagnostic system.
    
    Integrates problem classification, solution recommendation,
    performance analysis, and knowledge base for comprehensive
    troubleshooting support.
    """
    
    def __init__(self):
        """Initialize diagnostic system."""
        self.problem_classifier = ProblemClassifier()
        self.solution_recommender = SolutionRecommender()
        self.performance_analyzer = PerformanceAnalyzer()
        self.knowledge_base = TroubleshootingKnowledgeBase()
        
        # Diagnostic history for learning
        self.diagnostic_history = []
        self.solution_feedback = defaultdict(list)
        
        self.logger = logger
        logger.info("Initialized IntelligentDiagnosticSystem")
    
    def diagnose_problem(self, symptoms: Dict[str, Any], 
                        context: Optional[Dict[str, Any]] = None) -> DiagnosisResult:
        """
        Comprehensive problem diagnosis with recommendations.
        
        Args:
            symptoms: Problem symptoms and error information
            context: Optional context about environment and user
            
        Returns:
            DiagnosisResult with complete analysis and recommendations
        """
        start_time = time.time()
        
        try:
            # 1. Classify the problem
            classification = self.problem_classifier.classify_problem(symptoms)
            
            # 2. Analyze performance if data available
            performance_analysis = None
            if "performance_data" in symptoms:
                performance_analysis = self.performance_analyzer.analyze_performance_issue(
                    symptoms["performance_data"]
                )
            
            # 3. Get solution recommendations
            recommended_solutions = self.solution_recommender.recommend_solutions(
                classification, performance_analysis, context
            )
            
            # 4. Get alternative solutions
            alternative_solutions = self._get_alternative_solutions(
                classification, recommended_solutions
            )
            
            # 5. Generate preventive measures
            preventive_measures = self._generate_preventive_measures(classification)
            
            # 6. Generate monitoring recommendations
            monitoring_recommendations = self._generate_monitoring_recommendations(
                classification, performance_analysis
            )
            
            # 7. Determine escalation path
            escalation_path = self._determine_escalation_path(classification, context)
            
            # 8. Calculate diagnosis confidence
            diagnosis_confidence = self._calculate_diagnosis_confidence(
                classification, len(recommended_solutions)
            )
            
            diagnosis_time = (time.time() - start_time) * 1000
            
            # Create diagnosis result
            result = DiagnosisResult(
                problem_classification=classification,
                recommended_solutions=recommended_solutions,
                alternative_solutions=alternative_solutions,
                performance_analysis=performance_analysis,
                preventive_measures=preventive_measures,
                monitoring_recommendations=monitoring_recommendations,
                escalation_path=escalation_path,
                diagnosis_confidence=diagnosis_confidence,
                diagnosis_time_ms=diagnosis_time
            )
            
            # Store for learning
            self.diagnostic_history.append({
                "symptoms": symptoms,
                "context": context,
                "result": result,
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Diagnosis failed: {e}")
            return DiagnosisResult(
                problem_classification=ProblemClassification(
                    problem_type=ProblemCategory.API_USAGE,
                    confidence=0.1,
                    severity=ProblemSeverity.MEDIUM
                ),
                recommended_solutions=[],
                diagnosis_confidence=0.1,
                diagnosis_time_ms=(time.time() - start_time) * 1000
            )
    
    def _get_alternative_solutions(self, classification: ProblemClassification,
                                 recommended_solutions: List[Solution]) -> List[Solution]:
        """Get alternative solutions not in main recommendations."""
        all_solutions = self.solution_recommender.solution_database.get(
            classification.problem_type, []
        )
        
        recommended_ids = {sol.solution_id for sol in recommended_solutions}
        alternatives = [
            sol for sol in all_solutions 
            if sol.solution_id not in recommended_ids
        ]
        
        return alternatives[:2]  # Return top 2 alternatives
    
    def _generate_preventive_measures(self, classification: ProblemClassification) -> List[str]:
        """Generate preventive measures based on problem type."""
        base_measures = self.knowledge_base.get_prevention_tips(
            classification.problem_type.value
        )
        
        # Add specific measures based on root causes
        specific_measures = []
        for cause in classification.root_causes:
            if "network" in cause.lower():
                specific_measures.append("Ensure stable network connectivity")
            elif "memory" in cause.lower():
                specific_measures.append("Monitor system memory usage")
            elif "permission" in cause.lower():
                specific_measures.append("Verify proper file and directory permissions")
        
        return base_measures + specific_measures
    
    def _generate_monitoring_recommendations(self, classification: ProblemClassification,
                                          performance_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []
        
        if classification.problem_type == ProblemCategory.PERFORMANCE:
            recommendations.extend([
                "Monitor system resource usage (CPU, memory, disk)",
                "Track execution times for key operations",
                "Set up alerts for performance degradation"
            ])
        
        if classification.problem_type == ProblemCategory.QUANTUM_EXECUTION:
            recommendations.extend([
                "Monitor quantum backend availability and status",
                "Track quantum circuit execution success rates",
                "Monitor quantum job queue times"
            ])
        
        if performance_analysis:
            if "memory" in performance_analysis.get("issues_detected", []):
                recommendations.append("Implement memory usage monitoring and alerts")
            if "timeout" in str(performance_analysis):
                recommendations.append("Set up timeout monitoring for long-running operations")
        
        return recommendations
    
    def _determine_escalation_path(self, classification: ProblemClassification,
                                 context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Determine when and how to escalate the problem."""
        if classification.severity == ProblemSeverity.CRITICAL:
            return "immediate_escalation"
        elif classification.urgency == "urgent":
            return "priority_escalation"
        elif classification.confidence < 0.5:
            return "expert_consultation"
        else:
            return None
    
    def _calculate_diagnosis_confidence(self, classification: ProblemClassification,
                                      solution_count: int) -> float:
        """Calculate overall diagnosis confidence."""
        base_confidence = classification.confidence
        
        # Adjust based on available solutions
        if solution_count == 0:
            base_confidence *= 0.5
        elif solution_count >= 3:
            base_confidence *= 1.1
        
        # Adjust based on severity (higher severity should have higher confidence threshold)
        if classification.severity == ProblemSeverity.CRITICAL and base_confidence < 0.8:
            base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    def create_troubleshooting_guide(self, problem_type: str) -> Dict[str, Any]:
        """Create comprehensive troubleshooting guide for specific problem type."""
        try:
            category = ProblemCategory(problem_type)
        except ValueError:
            return {"error": f"Unknown problem type: {problem_type}"}
        
        guide = {
            "problem_type": problem_type,
            "description": self.knowledge_base.get_problem_description(problem_type),
            "common_causes": self.knowledge_base.get_common_causes(problem_type),
            "diagnostic_steps": self._generate_diagnostic_steps(category),
            "solution_steps": self._generate_solution_steps(category),
            "prevention_tips": self.knowledge_base.get_prevention_tips(problem_type),
            "related_issues": self.knowledge_base.get_related_issues(problem_type),
            "escalation_criteria": self._get_escalation_criteria(category)
        }
        
        return guide
    
    def _generate_diagnostic_steps(self, category: ProblemCategory) -> List[str]:
        """Generate diagnostic steps for problem category."""
        base_steps = [
            "Gather error messages and stack traces",
            "Check system requirements and environment",
            "Verify installation and configuration"
        ]
        
        category_specific = {
            ProblemCategory.INSTALLATION: [
                "Check Python version compatibility",
                "Verify package manager functionality",
                "Test network connectivity"
            ],
            ProblemCategory.QUANTUM_EXECUTION: [
                "Verify quantum backend availability",
                "Test with simple quantum circuit",
                "Check quantum parameter validity"
            ],
            ProblemCategory.PERFORMANCE: [
                "Profile application performance",
                "Monitor system resource usage",
                "Identify performance bottlenecks"
            ]
        }
        
        return base_steps + category_specific.get(category, [])
    
    def _generate_solution_steps(self, category: ProblemCategory) -> List[str]:
        """Generate solution steps for problem category."""
        solutions = self.solution_recommender.solution_database.get(category, [])
        
        if not solutions:
            return ["Consult documentation or seek expert help"]
        
        # Extract key steps from top solutions
        solution_steps = []
        for solution in solutions[:2]:  # Top 2 solutions
            solution_steps.append(f"Try {solution.title}: {solution.description}")
            solution_steps.extend(solution.steps[:3])  # First 3 steps of each solution
        
        return solution_steps
    
    def _get_escalation_criteria(self, category: ProblemCategory) -> List[str]:
        """Get criteria for when to escalate the problem."""
        general_criteria = [
            "Problem persists after trying recommended solutions",
            "Problem causes system instability or data loss",
            "Problem blocks critical functionality"
        ]
        
        category_specific = {
            ProblemCategory.QUANTUM_EXECUTION: [
                "Quantum backend consistently unavailable",
                "Quantum circuits fail with unknown errors"
            ],
            ProblemCategory.PERFORMANCE: [
                "Performance degradation exceeds 50%",
                "System becomes unresponsive"
            ]
        }
        
        return general_criteria + category_specific.get(category, [])
    
    def record_solution_feedback(self, solution_id: str, success: bool,
                               user_rating: float, comments: str = "") -> None:
        """Record feedback on solution effectiveness."""
        feedback = {
            "solution_id": solution_id,
            "success": success,
            "rating": user_rating,
            "comments": comments,
            "timestamp": time.time()
        }
        
        self.solution_feedback[solution_id].append(feedback)
        
        # Update solution effectiveness
        solution_feedbacks = self.solution_feedback[solution_id]
        success_rate = sum(1 for f in solution_feedbacks if f["success"]) / len(solution_feedbacks)
        self.solution_recommender.solution_effectiveness[solution_id] = success_rate
    
    def get_diagnostic_statistics(self) -> Dict[str, Any]:
        """Get diagnostic system statistics."""
        if not self.diagnostic_history:
            return {"total_diagnoses": 0}
        
        total_diagnoses = len(self.diagnostic_history)
        
        # Calculate statistics
        problem_types = Counter()
        severities = Counter()
        avg_confidence = 0
        avg_diagnosis_time = 0
        
        for diagnosis in self.diagnostic_history:
            result = diagnosis["result"]
            problem_types[result.problem_classification.problem_type.value] += 1
            severities[result.problem_classification.severity.value] += 1
            avg_confidence += result.diagnosis_confidence
            avg_diagnosis_time += result.diagnosis_time_ms
        
        avg_confidence /= total_diagnoses
        avg_diagnosis_time /= total_diagnoses
        
        return {
            "total_diagnoses": total_diagnoses,
            "problem_type_distribution": dict(problem_types),
            "severity_distribution": dict(severities),
            "average_confidence": avg_confidence,
            "average_diagnosis_time_ms": avg_diagnosis_time,
            "solution_feedback_count": sum(len(feedback) for feedback in self.solution_feedback.values())
        }