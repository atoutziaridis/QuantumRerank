"""
Developer Portal for QuantumRerank Documentation.

This module provides comprehensive developer experience features including
personalized learning paths, interactive tutorials, code examples,
integration guides, and developer playground functionality.
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ExperienceLevel(Enum):
    """Developer experience levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TutorialType(Enum):
    """Types of tutorials."""
    QUICKSTART = "quickstart"
    DEEP_DIVE = "deep_dive"
    HANDS_ON = "hands_on"
    CONCEPTUAL = "conceptual"
    INTEGRATION = "integration"
    BEST_PRACTICES = "best_practices"


class ProgressStatus(Enum):
    """Learning progress status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class LearningPath:
    """Structured learning path for developers."""
    id: str
    title: str
    description: str
    experience_level: ExperienceLevel
    use_case: str
    estimated_time_minutes: int
    steps: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tutorial:
    """Interactive tutorial with code examples."""
    id: str
    title: str
    description: str
    tutorial_type: TutorialType
    experience_level: ExperienceLevel
    estimated_time_minutes: int
    content_sections: List[Dict[str, Any]] = field(default_factory=list)
    code_examples: List[Dict[str, Any]] = field(default_factory=list)
    exercises: List[Dict[str, Any]] = field(default_factory=list)
    quiz_questions: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CodeExample:
    """Executable code example with context."""
    id: str
    title: str
    description: str
    code: str
    language: str = "python"
    framework: str = ""
    complexity: str = "beginner"
    expected_output: str = ""
    setup_code: str = ""
    cleanup_code: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    explanation: str = ""
    related_examples: List[str] = field(default_factory=list)


@dataclass
class IntegrationGuide:
    """Integration guide for specific platforms/frameworks."""
    id: str
    title: str
    platform: str
    description: str
    difficulty: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    code_samples: List[CodeExample] = field(default_factory=list)
    troubleshooting: List[Dict[str, Any]] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)


@dataclass
class UserProgress:
    """Track user learning progress."""
    user_id: str
    learning_paths: Dict[str, ProgressStatus] = field(default_factory=dict)
    tutorials_completed: Set[str] = field(default_factory=set)
    examples_tried: Set[str] = field(default_factory=set)
    quiz_scores: Dict[str, float] = field(default_factory=dict)
    time_spent_minutes: int = 0
    achievements: List[str] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)


class TutorialManager:
    """Manages interactive tutorials and learning content."""
    
    def __init__(self):
        """Initialize tutorial manager."""
        self.tutorials = {}
        self.tutorial_templates = {}
        self.user_progress = {}
        
        # Initialize default tutorials
        self._initialize_default_tutorials()
        
        self.logger = logger
        logger.info("Initialized TutorialManager")
    
    def _initialize_default_tutorials(self) -> None:
        """Initialize default tutorials for QuantumRerank."""
        default_tutorials = [
            {
                "id": "quantum_basics",
                "title": "Quantum Computing Basics for Information Retrieval",
                "description": "Learn the fundamentals of quantum computing and how it applies to information retrieval",
                "tutorial_type": TutorialType.CONCEPTUAL,
                "experience_level": ExperienceLevel.BEGINNER,
                "estimated_time_minutes": 30,
                "content_sections": [
                    {
                        "title": "What is Quantum Computing?",
                        "content": "Quantum computing leverages quantum mechanical phenomena...",
                        "type": "text"
                    },
                    {
                        "title": "Quantum States and Superposition",
                        "content": "Quantum bits (qubits) can exist in superposition...",
                        "type": "text"
                    },
                    {
                        "title": "Quantum Fidelity in Information Retrieval",
                        "content": "Quantum fidelity measures similarity between quantum states...",
                        "type": "text"
                    }
                ],
                "tags": ["quantum", "basics", "theory"]
            },
            {
                "id": "quickstart_guide",
                "title": "QuantumRerank Quick Start",
                "description": "Get up and running with QuantumRerank in 15 minutes",
                "tutorial_type": TutorialType.QUICKSTART,
                "experience_level": ExperienceLevel.BEGINNER,
                "estimated_time_minutes": 15,
                "content_sections": [
                    {
                        "title": "Installation",
                        "content": "Install QuantumRerank using pip...",
                        "type": "code"
                    },
                    {
                        "title": "Basic Usage",
                        "content": "Create your first quantum-enhanced search...",
                        "type": "code"
                    }
                ],
                "tags": ["quickstart", "installation", "basic-usage"]
            },
            {
                "id": "swap_test_implementation",
                "title": "Implementing SWAP Test for Quantum Similarity",
                "description": "Deep dive into implementing the SWAP test algorithm",
                "tutorial_type": TutorialType.DEEP_DIVE,
                "experience_level": ExperienceLevel.INTERMEDIATE,
                "estimated_time_minutes": 45,
                "content_sections": [
                    {
                        "title": "SWAP Test Theory",
                        "content": "The SWAP test is a quantum algorithm...",
                        "type": "text"
                    },
                    {
                        "title": "Circuit Implementation",
                        "content": "Implement the SWAP test circuit using Qiskit...",
                        "type": "code"
                    },
                    {
                        "title": "Integration with Embeddings",
                        "content": "Connect SWAP test with classical embeddings...",
                        "type": "code"
                    }
                ],
                "tags": ["swap-test", "quantum-algorithms", "implementation"]
            }
        ]
        
        for tutorial_data in default_tutorials:
            tutorial = Tutorial(**tutorial_data)
            self.tutorials[tutorial.id] = tutorial
    
    def get_tutorial(self, tutorial_id: str) -> Optional[Tutorial]:
        """Get tutorial by ID."""
        return self.tutorials.get(tutorial_id)
    
    def get_tutorials_by_level(self, level: ExperienceLevel) -> List[Tutorial]:
        """Get tutorials filtered by experience level."""
        return [
            tutorial for tutorial in self.tutorials.values()
            if tutorial.experience_level == level
        ]
    
    def get_tutorials_by_type(self, tutorial_type: TutorialType) -> List[Tutorial]:
        """Get tutorials filtered by type."""
        return [
            tutorial for tutorial in self.tutorials.values()
            if tutorial.tutorial_type == tutorial_type
        ]
    
    def search_tutorials(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Tutorial]:
        """Search tutorials by content and metadata."""
        query_lower = query.lower()
        matching_tutorials = []
        
        for tutorial in self.tutorials.values():
            score = 0
            
            # Search in title and description
            if query_lower in tutorial.title.lower():
                score += 3
            if query_lower in tutorial.description.lower():
                score += 2
            
            # Search in tags
            for tag in tutorial.tags:
                if query_lower in tag.lower():
                    score += 1
            
            # Search in content sections
            for section in tutorial.content_sections:
                if query_lower in section.get("title", "").lower():
                    score += 1
                if query_lower in section.get("content", "").lower():
                    score += 0.5
            
            if score > 0:
                matching_tutorials.append((tutorial, score))
        
        # Sort by score and apply filters
        matching_tutorials.sort(key=lambda x: x[1], reverse=True)
        
        filtered_tutorials = []
        for tutorial, score in matching_tutorials:
            include = True
            
            if filters:
                if "experience_level" in filters:
                    if tutorial.experience_level.value not in filters["experience_level"]:
                        include = False
                
                if "tutorial_type" in filters:
                    if tutorial.tutorial_type.value not in filters["tutorial_type"]:
                        include = False
                
                if "max_time" in filters:
                    if tutorial.estimated_time_minutes > filters["max_time"]:
                        include = False
            
            if include:
                filtered_tutorials.append(tutorial)
        
        return filtered_tutorials
    
    def add_tutorial(self, tutorial: Tutorial) -> None:
        """Add new tutorial."""
        self.tutorials[tutorial.id] = tutorial
    
    def update_user_progress(self, user_id: str, tutorial_id: str, 
                           status: ProgressStatus, time_spent: int = 0) -> None:
        """Update user progress for tutorial."""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = UserProgress(user_id=user_id)
        
        progress = self.user_progress[user_id]
        progress.learning_paths[tutorial_id] = status
        progress.time_spent_minutes += time_spent
        progress.last_activity = time.time()
        
        if status == ProgressStatus.COMPLETED:
            progress.tutorials_completed.add(tutorial_id)
    
    def get_user_progress(self, user_id: str) -> Optional[UserProgress]:
        """Get user learning progress."""
        return self.user_progress.get(user_id)


class CodeExampleGenerator:
    """Generates and manages interactive code examples."""
    
    def __init__(self):
        """Initialize code example generator."""
        self.examples = {}
        self.example_templates = {}
        
        # Initialize default examples
        self._initialize_default_examples()
        
        self.logger = logger
        logger.info("Initialized CodeExampleGenerator")
    
    def _initialize_default_examples(self) -> None:
        """Initialize default code examples."""
        default_examples = [
            {
                "id": "basic_usage",
                "title": "Basic QuantumRerank Usage",
                "description": "Simple example showing how to use QuantumRerank for similarity search",
                "code": """
# Import QuantumRerank
from quantum_rerank import QuantumSimilarityEngine

# Initialize the engine
engine = QuantumSimilarityEngine()

# Create documents and query
documents = [
    "Quantum computing uses quantum mechanics",
    "Machine learning algorithms optimize parameters", 
    "Information retrieval finds relevant documents"
]
query = "quantum algorithms"

# Compute quantum-enhanced similarities
similarities = engine.compute_similarities(query, documents)

# Display results
for i, (doc, sim) in enumerate(zip(documents, similarities)):
    print(f"Document {i}: {sim:.3f} - {doc}")
""",
                "expected_output": """
Document 0: 0.892 - Quantum computing uses quantum mechanics
Document 2: 0.654 - Information retrieval finds relevant documents  
Document 1: 0.432 - Machine learning algorithms optimize parameters
""",
                "explanation": "This example demonstrates the basic usage of QuantumRerank to compute quantum-enhanced similarities between a query and documents.",
                "tags": ["basic", "similarity", "getting-started"],
                "complexity": "beginner"
            },
            {
                "id": "swap_test_circuit",
                "title": "SWAP Test Circuit Implementation",
                "description": "Implement a SWAP test circuit for quantum fidelity computation",
                "code": """
from quantum_rerank.quantum import SwapTestCircuit
import numpy as np

# Create two quantum states (as embeddings)
state1 = np.array([0.6, 0.8])  # Normalized embedding
state2 = np.array([0.8, 0.6])  # Another normalized embedding

# Initialize SWAP test circuit
swap_test = SwapTestCircuit(n_qubits=2)

# Encode states and perform SWAP test
circuit = swap_test.create_circuit(state1, state2)

# Execute circuit (simulation)
fidelity = swap_test.execute(circuit)

print(f"Quantum fidelity between states: {fidelity:.3f}")

# Visualize the circuit
print("Circuit diagram:")
print(circuit.draw())
""",
                "expected_output": """
Quantum fidelity between states: 0.894

Circuit diagram:
     ┌───┐ ░ ┌─────────────┐ ░ ┌───┐ ░ ┌─┐
q_0: ┤ H ├─░─┤ U(state1)   ├─░─┤ X ├─░─┤M├───
     └───┘ ░ └─────────────┘ ░ └─┬─┘ ░ └╥┘┌─┐
q_1: ──────░──┤ U(state2)   ├─░───■───░──╫─┤M├
           ░  └─────────────┘ ░       ░  ║ └╥┘
c: 2/══════════════════════════════════════╩══╩═
                                           0  1
""",
                "explanation": "This example shows how to implement and execute a SWAP test circuit to compute quantum fidelity between two quantum states.",
                "tags": ["swap-test", "quantum-circuit", "fidelity"],
                "complexity": "intermediate"
            },
            {
                "id": "hybrid_reranking",
                "title": "Hybrid Quantum-Classical Reranking",
                "description": "Combine classical and quantum methods for improved reranking",
                "code": """
from quantum_rerank import HybridReranker
from sentence_transformers import SentenceTransformer

# Initialize components
classical_model = SentenceTransformer('all-MiniLM-L6-v2')
quantum_reranker = HybridReranker(
    classical_model=classical_model,
    quantum_weight=0.3,
    classical_weight=0.7
)

# Documents to rerank
query = "quantum machine learning algorithms"
documents = [
    "Quantum algorithms for machine learning tasks",
    "Classical optimization in deep neural networks",
    "Quantum computing applications in AI",
    "Traditional information retrieval methods",
    "Hybrid quantum-classical approaches"
]

# Get initial classical rankings
classical_scores = quantum_reranker.get_classical_scores(query, documents)

# Apply quantum enhancement
reranked_results = quantum_reranker.rerank(
    query=query,
    documents=documents,
    top_k=3
)

# Display results
print("Hybrid Quantum-Classical Reranking Results:")
for i, (doc_idx, score) in enumerate(reranked_results):
    print(f"{i+1}. [{score:.3f}] {documents[doc_idx]}")
""",
                "expected_output": """
Hybrid Quantum-Classical Reranking Results:
1. [0.892] Quantum algorithms for machine learning tasks
2. [0.745] Hybrid quantum-classical approaches  
3. [0.678] Quantum computing applications in AI
""",
                "explanation": "This example demonstrates how to combine classical embeddings with quantum enhancement for improved document reranking.",
                "tags": ["hybrid", "reranking", "advanced"],
                "complexity": "advanced"
            }
        ]
        
        for example_data in default_examples:
            example = CodeExample(**example_data)
            self.examples[example.id] = example
    
    def get_example(self, example_id: str) -> Optional[CodeExample]:
        """Get code example by ID."""
        return self.examples.get(example_id)
    
    def get_examples_by_tags(self, tags: List[str]) -> List[CodeExample]:
        """Get examples that match any of the given tags."""
        matching_examples = []
        
        for example in self.examples.values():
            if any(tag in example.tags for tag in tags):
                matching_examples.append(example)
        
        return matching_examples
    
    def get_examples_by_complexity(self, complexity: str) -> List[CodeExample]:
        """Get examples filtered by complexity level."""
        return [
            example for example in self.examples.values()
            if example.complexity == complexity
        ]
    
    def generate_executable_examples(self, topic: str) -> List[CodeExample]:
        """Generate executable examples for a specific topic."""
        topic_lower = topic.lower()
        relevant_examples = []
        
        for example in self.examples.values():
            # Check if topic matches title, description, or tags
            if (topic_lower in example.title.lower() or
                topic_lower in example.description.lower() or
                any(topic_lower in tag.lower() for tag in example.tags)):
                relevant_examples.append(example)
        
        return relevant_examples
    
    def add_example(self, example: CodeExample) -> None:
        """Add new code example."""
        self.examples[example.id] = example
    
    def create_custom_example(self, title: str, description: str, code: str,
                            **kwargs) -> CodeExample:
        """Create a custom code example."""
        example_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        example = CodeExample(
            id=example_id,
            title=title,
            description=description,
            code=code,
            **kwargs
        )
        
        self.add_example(example)
        return example


class IntegrationGuideManager:
    """Manages integration guides for different platforms and frameworks."""
    
    def __init__(self):
        """Initialize integration guide manager."""
        self.guides = {}
        
        # Initialize default integration guides
        self._initialize_default_guides()
        
        self.logger = logger
        logger.info("Initialized IntegrationGuideManager")
    
    def _initialize_default_guides(self) -> None:
        """Initialize default integration guides."""
        default_guides = [
            {
                "id": "fastapi_integration",
                "title": "FastAPI Integration",
                "platform": "FastAPI",
                "description": "Integrate QuantumRerank with FastAPI web applications",
                "difficulty": "intermediate",
                "steps": [
                    {
                        "title": "Install Dependencies",
                        "description": "Install required packages",
                        "code": "pip install quantum-rerank fastapi uvicorn"
                    },
                    {
                        "title": "Create FastAPI App",
                        "description": "Set up basic FastAPI application with QuantumRerank",
                        "code": """
from fastapi import FastAPI, HTTPException
from quantum_rerank import QuantumSimilarityEngine
from pydantic import BaseModel

app = FastAPI(title="QuantumRerank API")
engine = QuantumSimilarityEngine()

class SearchRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 5

@app.post("/search")
async def quantum_search(request: SearchRequest):
    try:
        similarities = engine.compute_similarities(
            request.query, 
            request.documents
        )
        
        # Rank and return top results
        results = sorted(
            zip(request.documents, similarities),
            key=lambda x: x[1],
            reverse=True
        )[:request.top_k]
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
                    }
                ],
                "best_practices": [
                    "Use async/await for better performance",
                    "Implement proper error handling",
                    "Add request validation with Pydantic",
                    "Use dependency injection for engine initialization"
                ],
                "common_pitfalls": [
                    "Not handling quantum backend initialization errors",
                    "Forgetting to normalize embeddings before quantum processing",
                    "Not implementing proper timeout handling for quantum operations"
                ]
            },
            {
                "id": "jupyter_integration",
                "title": "Jupyter Notebook Integration",
                "platform": "Jupyter",
                "description": "Use QuantumRerank in Jupyter notebooks for research and experimentation",
                "difficulty": "beginner",
                "steps": [
                    {
                        "title": "Setup Environment",
                        "description": "Install and configure QuantumRerank in Jupyter",
                        "code": """
# Install in notebook cell
!pip install quantum-rerank

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from quantum_rerank import QuantumSimilarityEngine

# Enable interactive plots
%matplotlib inline
"""
                    },
                    {
                        "title": "Interactive Exploration",
                        "description": "Create interactive widgets for parameter exploration",
                        "code": """
from ipywidgets import interact, FloatSlider
import plotly.graph_objects as go

@interact(
    quantum_weight=FloatSlider(min=0, max=1, step=0.1, value=0.5),
    classical_weight=FloatSlider(min=0, max=1, step=0.1, value=0.5)
)
def explore_hybrid_weighting(quantum_weight, classical_weight):
    # Normalize weights
    total = quantum_weight + classical_weight
    if total > 0:
        quantum_weight /= total
        classical_weight /= total
    
    # Demo computation
    engine = QuantumSimilarityEngine(
        quantum_weight=quantum_weight,
        classical_weight=classical_weight
    )
    
    # Visualize results
    query = "quantum machine learning"
    docs = ["quantum algorithms", "classical ML", "hybrid approaches"]
    similarities = engine.compute_similarities(query, docs)
    
    fig = go.Figure(data=go.Bar(x=docs, y=similarities))
    fig.update_layout(title="Similarity Scores with Different Weightings")
    fig.show()
"""
                    }
                ]
            }
        ]
        
        for guide_data in default_guides:
            guide = IntegrationGuide(**guide_data)
            self.guides[guide.id] = guide
    
    def get_guide(self, guide_id: str) -> Optional[IntegrationGuide]:
        """Get integration guide by ID."""
        return self.guides.get(guide_id)
    
    def get_guides_by_platform(self, platform: str) -> List[IntegrationGuide]:
        """Get integration guides for specific platform."""
        return [
            guide for guide in self.guides.values()
            if guide.platform.lower() == platform.lower()
        ]
    
    def search_guides(self, query: str) -> List[IntegrationGuide]:
        """Search integration guides."""
        query_lower = query.lower()
        matching_guides = []
        
        for guide in self.guides.values():
            score = 0
            
            if query_lower in guide.title.lower():
                score += 3
            if query_lower in guide.platform.lower():
                score += 2
            if query_lower in guide.description.lower():
                score += 1
            
            if score > 0:
                matching_guides.append((guide, score))
        
        matching_guides.sort(key=lambda x: x[1], reverse=True)
        return [guide for guide, score in matching_guides]


class InteractiveDeveloperPlayground:
    """Interactive playground for experimenting with QuantumRerank."""
    
    def __init__(self):
        """Initialize developer playground."""
        self.playground_sessions = {}
        self.saved_experiments = {}
        
        self.logger = logger
        logger.info("Initialized InteractiveDeveloperPlayground")
    
    def create_playground_session(self, user_id: str) -> str:
        """Create new playground session."""
        session_id = f"playground_{uuid.uuid4().hex[:8]}"
        
        self.playground_sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "code_history": [],
            "results_history": [],
            "current_state": {}
        }
        
        return session_id
    
    def execute_code(self, session_id: str, code: str) -> Dict[str, Any]:
        """Execute code in playground session (simulation)."""
        if session_id not in self.playground_sessions:
            return {"error": "Invalid session ID"}
        
        session = self.playground_sessions[session_id]
        
        # Simulate code execution
        execution_result = {
            "success": True,
            "output": "Code executed successfully",
            "execution_time": 0.1,
            "warnings": [],
            "variables": {}
        }
        
        # Store in session history
        session["code_history"].append({
            "code": code,
            "timestamp": time.time(),
            "result": execution_result
        })
        session["last_activity"] = time.time()
        
        return execution_result
    
    def save_experiment(self, session_id: str, name: str, description: str) -> str:
        """Save playground experiment."""
        if session_id not in self.playground_sessions:
            return None
        
        session = self.playground_sessions[session_id]
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        self.saved_experiments[experiment_id] = {
            "name": name,
            "description": description,
            "user_id": session["user_id"],
            "code_history": session["code_history"].copy(),
            "created_at": time.time()
        }
        
        return experiment_id
    
    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load saved experiment."""
        return self.saved_experiments.get(experiment_id)
    
    def create_playground_config(self, topic: str) -> Dict[str, Any]:
        """Create playground configuration for specific topic."""
        config = {
            "topic": topic,
            "environment": {
                "python_version": "3.9+",
                "packages": ["quantum-rerank", "numpy", "matplotlib"],
                "memory_limit": "1GB",
                "timeout": 30
            },
            "starter_code": self._get_starter_code(topic),
            "available_datasets": self._get_available_datasets(topic),
            "suggested_exercises": self._get_suggested_exercises(topic)
        }
        
        return config
    
    def _get_starter_code(self, topic: str) -> str:
        """Get starter code for topic."""
        starter_codes = {
            "quantum_similarity": """
# QuantumRerank Similarity Playground
from quantum_rerank import QuantumSimilarityEngine

# Initialize engine
engine = QuantumSimilarityEngine()

# Try your own examples here
query = "your query here"
documents = ["doc1", "doc2", "doc3"]

# Compute similarities
similarities = engine.compute_similarities(query, documents)
print("Similarities:", similarities)
""",
            "swap_test": """
# SWAP Test Circuit Playground
from quantum_rerank.quantum import SwapTestCircuit
import numpy as np

# Create quantum states
state1 = np.array([0.6, 0.8])  # Normalized
state2 = np.array([0.8, 0.6])  # Normalized

# Initialize SWAP test
swap_test = SwapTestCircuit(n_qubits=2)

# Create and execute circuit
circuit = swap_test.create_circuit(state1, state2)
fidelity = swap_test.execute(circuit)

print(f"Quantum fidelity: {fidelity}")
""",
            "hybrid_reranking": """
# Hybrid Reranking Playground
from quantum_rerank import HybridReranker

# Initialize reranker
reranker = HybridReranker(quantum_weight=0.3, classical_weight=0.7)

# Sample data
query = "machine learning quantum"
docs = [
    "Quantum machine learning algorithms",
    "Classical neural networks",
    "Hybrid quantum-classical models"
]

# Rerank documents
results = reranker.rerank(query, docs)
print("Reranked results:", results)
"""
        }
        
        return starter_codes.get(topic, "# Start coding here\n")
    
    def _get_available_datasets(self, topic: str) -> List[Dict[str, str]]:
        """Get available datasets for topic."""
        datasets = {
            "quantum_similarity": [
                {"name": "Sample Documents", "description": "Collection of technical documents"},
                {"name": "Research Papers", "description": "Academic papers on quantum computing"}
            ],
            "information_retrieval": [
                {"name": "News Articles", "description": "News article dataset"},
                {"name": "Web Pages", "description": "Web page content samples"}
            ]
        }
        
        return datasets.get(topic, [])
    
    def _get_suggested_exercises(self, topic: str) -> List[str]:
        """Get suggested exercises for topic."""
        exercises = {
            "quantum_similarity": [
                "Compare quantum vs classical similarity for different document types",
                "Experiment with different quantum weights in hybrid approach",
                "Analyze performance on various embedding dimensions"
            ],
            "swap_test": [
                "Implement SWAP test for different qubit numbers",
                "Compare fidelity computation with classical methods",
                "Visualize quantum circuit execution"
            ]
        }
        
        return exercises.get(topic, [])


class DeveloperPortal:
    """
    Comprehensive developer experience portal for QuantumRerank.
    
    Integrates tutorials, code examples, integration guides,
    and interactive playground for complete developer experience.
    """
    
    def __init__(self):
        """Initialize developer portal."""
        self.tutorial_manager = TutorialManager()
        self.example_generator = CodeExampleGenerator()
        self.integration_guide = IntegrationGuideManager()
        self.playground = InteractiveDeveloperPlayground()
        
        self.user_profiles = {}
        self.learning_analytics = defaultdict(int)
        
        self.logger = logger
        logger.info("Initialized DeveloperPortal")
    
    def assess_experience_level(self, user_responses: Dict[str, Any]) -> ExperienceLevel:
        """Assess developer experience level based on responses."""
        score = 0
        
        # Experience with quantum computing
        quantum_exp = user_responses.get("quantum_experience", "none")
        if quantum_exp == "expert":
            score += 3
        elif quantum_exp == "intermediate":
            score += 2
        elif quantum_exp == "beginner":
            score += 1
        
        # Programming experience
        programming_exp = user_responses.get("programming_years", 0)
        if programming_exp >= 5:
            score += 2
        elif programming_exp >= 2:
            score += 1
        
        # ML/IR experience
        ml_exp = user_responses.get("ml_experience", "none")
        if ml_exp in ["expert", "intermediate"]:
            score += 1
        
        # Determine level
        if score >= 5:
            return ExperienceLevel.EXPERT
        elif score >= 3:
            return ExperienceLevel.ADVANCED
        elif score >= 1:
            return ExperienceLevel.INTERMEDIATE
        else:
            return ExperienceLevel.BEGINNER
    
    def generate_getting_started_path(self, user_id: str,
                                    experience_level: ExperienceLevel,
                                    use_case: str) -> LearningPath:
        """Generate personalized getting started path."""
        path_id = f"path_{user_id}_{int(time.time())}"
        
        # Base path structure
        path = LearningPath(
            id=path_id,
            title=f"Getting Started with QuantumRerank - {experience_level.value.title()}",
            description=f"Personalized learning path for {use_case} use case",
            experience_level=experience_level,
            use_case=use_case,
            estimated_time_minutes=0
        )
        
        # Add steps based on experience level
        if experience_level == ExperienceLevel.BEGINNER:
            path.steps = self._create_beginner_path(use_case)
        elif experience_level == ExperienceLevel.INTERMEDIATE:
            path.steps = self._create_intermediate_path(use_case)
        elif experience_level == ExperienceLevel.ADVANCED:
            path.steps = self._create_advanced_path(use_case)
        else:  # EXPERT
            path.steps = self._create_expert_path(use_case)
        
        # Calculate total estimated time
        path.estimated_time_minutes = sum(step.get("time_minutes", 15) for step in path.steps)
        
        return path
    
    def _create_beginner_path(self, use_case: str) -> List[Dict[str, Any]]:
        """Create learning path for beginners."""
        return [
            {
                "title": "Quantum Computing Basics",
                "type": "tutorial",
                "tutorial_id": "quantum_basics",
                "time_minutes": 30,
                "description": "Learn fundamental quantum computing concepts"
            },
            {
                "title": "Installation and Setup",
                "type": "tutorial",
                "tutorial_id": "quickstart_guide",
                "time_minutes": 15,
                "description": "Install and configure QuantumRerank"
            },
            {
                "title": "Basic Usage Examples",
                "type": "examples",
                "example_ids": ["basic_usage"],
                "time_minutes": 20,
                "description": "Try basic QuantumRerank functionality"
            },
            {
                "title": "Your First Application",
                "type": "hands_on",
                "playground_topic": "quantum_similarity",
                "time_minutes": 30,
                "description": "Build a simple quantum-enhanced search application"
            }
        ]
    
    def _create_intermediate_path(self, use_case: str) -> List[Dict[str, Any]]:
        """Create learning path for intermediate developers."""
        return [
            {
                "title": "Quick Overview",
                "type": "tutorial",
                "tutorial_id": "quickstart_guide",
                "time_minutes": 10,
                "description": "Quick introduction to QuantumRerank"
            },
            {
                "title": "SWAP Test Deep Dive",
                "type": "tutorial",
                "tutorial_id": "swap_test_implementation",
                "time_minutes": 45,
                "description": "Understand the core quantum algorithm"
            },
            {
                "title": "Hybrid Reranking",
                "type": "examples",
                "example_ids": ["hybrid_reranking"],
                "time_minutes": 25,
                "description": "Implement hybrid quantum-classical approaches"
            },
            {
                "title": "Integration Guide",
                "type": "integration",
                "guide_id": "fastapi_integration" if "api" in use_case else "jupyter_integration",
                "time_minutes": 40,
                "description": f"Integrate with your {use_case} workflow"
            }
        ]
    
    def _create_advanced_path(self, use_case: str) -> List[Dict[str, Any]]:
        """Create learning path for advanced developers."""
        return [
            {
                "title": "Architecture Overview",
                "type": "documentation",
                "doc_id": "quantum_overview",
                "time_minutes": 15,
                "description": "Understand system architecture"
            },
            {
                "title": "Advanced Examples",
                "type": "examples",
                "example_ids": ["swap_test_circuit", "hybrid_reranking"],
                "time_minutes": 40,
                "description": "Explore advanced implementation patterns"
            },
            {
                "title": "Custom Implementation",
                "type": "hands_on",
                "playground_topic": "custom_quantum_algorithm",
                "time_minutes": 60,
                "description": "Implement custom quantum algorithms"
            },
            {
                "title": "Performance Optimization",
                "type": "best_practices",
                "time_minutes": 30,
                "description": "Optimize for production deployment"
            }
        ]
    
    def _create_expert_path(self, use_case: str) -> List[Dict[str, Any]]:
        """Create learning path for expert developers."""
        return [
            {
                "title": "Research Papers and Theory",
                "type": "research",
                "time_minutes": 30,
                "description": "Review underlying research and theory"
            },
            {
                "title": "Advanced Customization",
                "type": "hands_on",
                "playground_topic": "quantum_algorithm_design",
                "time_minutes": 90,
                "description": "Design custom quantum algorithms"
            },
            {
                "title": "Contributing to QuantumRerank",
                "type": "documentation",
                "time_minutes": 20,
                "description": "Learn how to contribute to the project"
            }
        ]
    
    def create_interactive_tutorial(self, tutorial_id: str, user_id: str) -> Dict[str, Any]:
        """Create interactive tutorial session."""
        tutorial = self.tutorial_manager.get_tutorial(tutorial_id)
        if not tutorial:
            return {"error": "Tutorial not found"}
        
        # Create playground session for tutorial
        playground_session = self.playground.create_playground_session(user_id)
        
        # Get relevant code examples
        examples = self.example_generator.generate_executable_examples(tutorial.title)
        
        # Create interactive tutorial configuration
        interactive_config = {
            "tutorial": tutorial,
            "playground_session": playground_session,
            "code_examples": examples,
            "progress_tracking": True,
            "estimated_completion": tutorial.estimated_time_minutes
        }
        
        return interactive_config
    
    def get_user_recommendations(self, user_id: str) -> Dict[str, List[Any]]:
        """Get personalized recommendations for user."""
        user_progress = self.tutorial_manager.get_user_progress(user_id)
        
        recommendations = {
            "next_tutorials": [],
            "relevant_examples": [],
            "integration_guides": [],
            "advanced_topics": []
        }
        
        if user_progress:
            completed_tutorials = user_progress.tutorials_completed
            
            # Recommend next tutorials based on completed ones
            for tutorial in self.tutorial_manager.tutorials.values():
                if tutorial.id not in completed_tutorials:
                    # Check if prerequisites are met
                    prereqs_met = all(
                        prereq in completed_tutorials 
                        for prereq in tutorial.dependencies
                    )
                    if prereqs_met:
                        recommendations["next_tutorials"].append(tutorial)
            
            # Limit recommendations
            recommendations["next_tutorials"] = recommendations["next_tutorials"][:3]
        
        return recommendations
    
    def get_developer_portal_analytics(self) -> Dict[str, Any]:
        """Get developer portal usage analytics."""
        return {
            "total_users": len(self.user_profiles),
            "tutorials_created": len(self.tutorial_manager.tutorials),
            "examples_available": len(self.example_generator.examples),
            "integration_guides": len(self.integration_guide.guides),
            "playground_sessions": len(self.playground.playground_sessions),
            "learning_analytics": dict(self.learning_analytics)
        }