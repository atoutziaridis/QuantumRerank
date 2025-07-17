#!/usr/bin/env python3
"""
Semantic Quality Evaluation: Classical vs Quantum Reranking
=========================================================

Comprehensive evaluation to test whether quantum reranking provides better
semantic understanding compared to classical-only retrieval using real documents
and complex queries requiring deep semantic reasoning.

Following RAGBench best practices for unbiased evaluation.
"""

import os
import sys
import time
import json
import numpy as np
import random
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Statistical testing
from scipy.stats import wilcoxon, mannwhitneyu
import pandas as pd
from sklearn.metrics import ndcg_score

# Document fetching
import arxiv
import wikipedia
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

# Add quantum_rerank to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata


@dataclass
class SemanticEvaluationConfig:
    """Configuration for semantic quality evaluation."""
    # Document corpus
    target_documents: int = 200  # Real documents from multiple sources
    domains: List[str] = field(default_factory=lambda: ["computer_science", "physics", "medicine", "law", "general"])
    
    # Query complexity
    total_queries: int = 100  # Complex semantic queries
    multi_hop_queries: int = 40  # Queries requiring multi-document reasoning
    comparative_queries: int = 30  # Queries requiring comparison/contrast
    inference_queries: int = 30  # Queries requiring inference/deduction
    
    # Evaluation rigor
    alpha: float = 0.05
    statistical_power: float = 0.8
    bootstrap_iterations: int = 1000
    
    # Source diversity
    arxiv_papers: int = 50
    wikipedia_articles: int = 50
    pubmed_articles: int = 50
    legal_documents: int = 25
    general_web_content: int = 25


class DocumentFetcher:
    """Fetch real documents from multiple sources for evaluation."""
    
    def __init__(self, config: SemanticEvaluationConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QuantumRAG-Evaluation/1.0 (Research Evaluation)'
        })
        
    def fetch_arxiv_papers(self, count: int = 50) -> List[Document]:
        """Fetch academic papers from arXiv."""
        print(f"Fetching {count} papers from arXiv...")
        documents = []
        
        # Diverse CS and physics topics for semantic complexity
        search_queries = [
            "cat:cs.AI AND quantum",
            "cat:cs.CL AND semantic",
            "cat:cs.IR AND retrieval",
            "cat:cs.LG AND representation",
            "cat:physics.quant-ph AND information",
            "cat:cs.CV AND understanding",
            "cat:cs.AI AND reasoning",
            "cat:cs.NE AND networks"
        ]
        
        client = arxiv.Client()
        papers_per_query = max(1, count // len(search_queries))
        
        for query in search_queries:
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=papers_per_query,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                for paper in client.results(search):
                    if len(documents) >= count:
                        break
                        
                    # Extract meaningful content
                    content = f"Title: {paper.title}\n\n"
                    content += f"Abstract: {paper.summary}\n\n"
                    
                    # Add more content if available
                    if hasattr(paper, 'comment') and paper.comment:
                        content += f"Comments: {paper.comment}\n\n"
                    
                    categories = [cat.split('.')[-1] for cat in paper.categories]
                    
                    metadata = DocumentMetadata(
                        title=paper.title,
                        source="arxiv",
                        custom_fields={
                            "domain": "computer_science" if any("cs" in cat for cat in paper.categories) else "physics",
                            "authors": [author.name for author in paper.authors],
                            "categories": categories,
                            "published": str(paper.published),
                            "url": paper.entry_id,
                            "complexity": "high"
                        }
                    )
                    
                    documents.append(Document(
                        doc_id=f"arxiv_{paper.entry_id.split('/')[-1]}",
                        content=content,
                        metadata=metadata
                    ))
                    
                if len(documents) >= count:
                    break
                    
            except Exception as e:
                print(f"Error fetching arXiv papers for query '{query}': {e}")
                continue
        
        print(f"Successfully fetched {len(documents)} arXiv papers")
        return documents[:count]
    
    def fetch_wikipedia_articles(self, count: int = 50) -> List[Document]:
        """Fetch Wikipedia articles on complex topics."""
        print(f"Fetching {count} Wikipedia articles...")
        documents = []
        
        # Complex topics requiring semantic understanding
        topics = [
            # Computer Science & AI
            "Artificial intelligence", "Machine learning", "Natural language processing",
            "Computer vision", "Quantum computing", "Cryptography", "Algorithm",
            "Data structure", "Neural network", "Deep learning",
            
            # Physics & Science
            "Quantum mechanics", "Relativity", "Thermodynamics", "Electromagnetism",
            "Particle physics", "Cosmology", "String theory", "Quantum field theory",
            
            # Medicine & Biology
            "Molecular biology", "Genetics", "Immunology", "Neuroscience",
            "Pharmacology", "Epidemiology", "Cancer", "Metabolism",
            
            # Philosophy & Logic
            "Philosophy of mind", "Epistemology", "Logic", "Ethics",
            "Consciousness", "Free will", "Determinism", "Causality",
            
            # Complex Systems
            "Complex system", "Chaos theory", "Network theory", "Game theory",
            "Information theory", "Cybernetics", "Systems theory"
        ]
        
        random.shuffle(topics)
        
        for topic in topics[:count * 2]:  # Fetch more than needed in case some fail
            if len(documents) >= count:
                break
                
            try:
                # Search for the topic
                search_results = wikipedia.search(topic, results=1)
                if not search_results:
                    continue
                    
                page_title = search_results[0]
                page = wikipedia.page(page_title, auto_suggest=False)
                
                # Extract meaningful content
                content = f"Title: {page.title}\n\n"
                content += f"Summary: {page.summary}\n\n"
                
                # Add sections for more content
                if hasattr(page, 'content'):
                    content += page.content[:5000]  # Limit to avoid too long documents
                
                # Determine domain based on content
                content_lower = content.lower()
                if any(word in content_lower for word in ["computer", "algorithm", "software", "artificial intelligence"]):
                    domain = "computer_science"
                elif any(word in content_lower for word in ["physics", "quantum", "particle", "relativity"]):
                    domain = "physics"
                elif any(word in content_lower for word in ["medicine", "medical", "disease", "treatment", "biology"]):
                    domain = "medicine"
                elif any(word in content_lower for word in ["philosophy", "logic", "ethics", "consciousness"]):
                    domain = "philosophy"
                else:
                    domain = "general"
                
                metadata = DocumentMetadata(
                    title=page.title,
                    source="wikipedia",
                    custom_fields={
                        "domain": domain,
                        "url": page.url,
                        "categories": getattr(page, 'categories', []),
                        "complexity": "medium",
                        "length": len(content)
                    }
                )
                
                documents.append(Document(
                    doc_id=f"wiki_{hashlib.md5(page.title.encode()).hexdigest()[:8]}",
                    content=content,
                    metadata=metadata
                ))
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching Wikipedia article for '{topic}': {e}")
                continue
        
        print(f"Successfully fetched {len(documents)} Wikipedia articles")
        return documents[:count]
    
    def fetch_pubmed_abstracts(self, count: int = 50) -> List[Document]:
        """Fetch medical abstracts from PubMed."""
        print(f"Fetching {count} PubMed abstracts...")
        documents = []
        
        # Medical topics requiring semantic understanding
        search_terms = [
            "machine learning medicine",
            "artificial intelligence healthcare",
            "quantum biology",
            "computational neuroscience",
            "bioinformatics algorithms",
            "medical imaging AI",
            "drug discovery computational",
            "systems biology networks"
        ]
        
        abstracts_per_term = max(1, count // len(search_terms))
        
        for term in search_terms:
            try:
                # PubMed E-utilities API
                search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                search_params = {
                    'db': 'pubmed',
                    'term': term,
                    'retmax': abstracts_per_term,
                    'retmode': 'json'
                }
                
                response = self.session.get(search_url, params=search_params)
                if response.status_code != 200:
                    continue
                    
                search_data = response.json()
                pmids = search_data.get('esearchresult', {}).get('idlist', [])
                
                if not pmids:
                    continue
                
                # Fetch abstracts
                fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(pmids),
                    'retmode': 'xml'
                }
                
                response = self.session.get(fetch_url, params=fetch_params)
                if response.status_code != 200:
                    continue
                
                # Parse XML
                root = ET.fromstring(response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    if len(documents) >= count:
                        break
                        
                    try:
                        # Extract title
                        title_elem = article.find('.//ArticleTitle')
                        title = title_elem.text if title_elem is not None else "No title"
                        
                        # Extract abstract
                        abstract_elem = article.find('.//Abstract/AbstractText')
                        abstract = abstract_elem.text if abstract_elem is not None else "No abstract"
                        
                        # Extract PMID
                        pmid_elem = article.find('.//PMID')
                        pmid = pmid_elem.text if pmid_elem is not None else "unknown"
                        
                        content = f"Title: {title}\n\nAbstract: {abstract}"
                        
                        metadata = DocumentMetadata(
                            title=title,
                            source="pubmed",
                            custom_fields={
                                "domain": "medicine",
                                "pmid": pmid,
                                "search_term": term,
                                "complexity": "high",
                                "length": len(content)
                            }
                        )
                        
                        documents.append(Document(
                            doc_id=f"pubmed_{pmid}",
                            content=content,
                            metadata=metadata
                        ))
                        
                    except Exception as e:
                        print(f"Error parsing PubMed article: {e}")
                        continue
                
                time.sleep(0.5)  # Rate limiting for PubMed API
                
            except Exception as e:
                print(f"Error fetching PubMed articles for '{term}': {e}")
                continue
        
        print(f"Successfully fetched {len(documents)} PubMed abstracts")
        return documents[:count]
    
    def create_legal_documents(self, count: int = 25) -> List[Document]:
        """Create complex legal-style documents requiring semantic understanding."""
        print(f"Creating {count} legal documents...")
        documents = []
        
        legal_templates = [
            {
                "type": "contract_analysis",
                "content": """Contract Interpretation and Quantum Computing Licensing Agreement
                
This agreement concerns the licensing of quantum computing technologies with specific provisions 
for intellectual property protection. The licensee shall have exclusive rights to utilize patented 
quantum algorithms within designated geographical boundaries, subject to performance milestones.

Key provisions include: (1) Quantum supremacy demonstration requirements within 24 months, 
(2) Minimum 100-qubit system deployment, (3) Error correction protocols meeting industry standards,
(4) Joint research collaboration on quantum error mitigation techniques.

The agreement includes liability limitations for quantum decoherence events beyond scientific control,
force majeure clauses for quantum hardware failures, and dispute resolution through specialized
quantum technology arbitration panels.

Termination conditions involve breach of quantum fidelity requirements, failure to achieve
quantum advantage benchmarks, or violation of quantum information security protocols."""
            },
            {
                "type": "regulatory_analysis", 
                "content": """Artificial Intelligence Regulatory Compliance Framework

This framework establishes guidelines for AI system deployment in healthcare environments,
addressing algorithmic transparency, bias mitigation, and patient data protection requirements.

Healthcare AI systems must demonstrate: (1) Explainable decision-making processes for medical
diagnoses, (2) Validation across diverse patient populations, (3) Integration with existing
electronic health record systems, (4) Compliance with HIPAA privacy regulations.

The framework requires continuous monitoring of AI model performance, regular bias audits
across demographic groups, and maintenance of human oversight mechanisms for critical decisions.
Documentation must include model training data provenance, validation methodologies, and
performance metrics across different patient cohorts.

Enforcement mechanisms include periodic regulatory audits, mandatory reporting of AI-related
adverse events, and penalties for non-compliance with algorithmic transparency requirements."""
            }
        ]
        
        for i in range(count):
            template = random.choice(legal_templates)
            
            # Add variations to make each document unique
            content = template["content"]
            content += f"\n\nDocument ID: LEGAL-{i:03d}\n"
            content += f"Jurisdiction: {random.choice(['Federal', 'State', 'International'])}\n"
            content += f"Complexity Level: {random.choice(['Standard', 'Complex', 'Advanced'])}\n"
            
            metadata = DocumentMetadata(
                title=f"Legal Document {i+1}: {template['type'].replace('_', ' ').title()}",
                source="legal_corpus",
                custom_fields={
                    "domain": "law",
                    "document_type": template["type"],
                    "complexity": "high",
                    "length": len(content)
                }
            )
            
            documents.append(Document(
                doc_id=f"legal_{i:03d}",
                content=content,
                metadata=metadata
            ))
        
        print(f"Successfully created {len(documents)} legal documents")
        return documents
    
    def create_general_content(self, count: int = 25) -> List[Document]:
        """Create general knowledge content requiring semantic reasoning."""
        print(f"Creating {count} general knowledge documents...")
        documents = []
        
        topics = [
            "climate_change_technology",
            "space_exploration_ethics", 
            "renewable_energy_economics",
            "urban_planning_sustainability",
            "digital_privacy_society",
            "biotechnology_ethics",
            "automation_employment",
            "education_technology_impact"
        ]
        
        for i in range(count):
            topic = topics[i % len(topics)]
            
            content = f"""Analysis of {topic.replace('_', ' ').title()}

This document explores the complex interrelationships between technological advancement
and societal implications in the context of {topic.replace('_', ' ')}.

The analysis considers multiple perspectives including economic impacts, ethical considerations,
environmental consequences, and long-term sustainability factors. Key stakeholders include
policymakers, industry leaders, academic researchers, and affected communities.

Critical questions addressed: How do emerging technologies reshape traditional frameworks?
What are the unintended consequences of rapid technological adoption? How can society
balance innovation with equity and sustainability?

The document examines case studies, comparative analyses across different regions,
and projections for future development. Recommendations include policy frameworks,
research priorities, and implementation strategies for balanced technological progress.

Conclusion emphasizes the need for interdisciplinary collaboration, stakeholder engagement,
and adaptive governance mechanisms to navigate complex technological transitions.

Document {i+1} of {count} in the general knowledge corpus."""
            
            metadata = DocumentMetadata(
                title=f"Analysis: {topic.replace('_', ' ').title()}",
                source="general_corpus",
                custom_fields={
                    "domain": "general",
                    "topic": topic,
                    "complexity": "medium",
                    "length": len(content)
                }
            )
            
            documents.append(Document(
                doc_id=f"general_{i:03d}",
                content=content,
                metadata=metadata
            ))
        
        print(f"Successfully created {len(documents)} general knowledge documents")
        return documents
    
    def fetch_all_documents(self) -> List[Document]:
        """Fetch all documents from multiple sources."""
        print("🔍 FETCHING DIVERSE DOCUMENT CORPUS")
        print("=" * 50)
        
        all_documents = []
        
        # ArXiv papers
        if self.config.arxiv_papers > 0:
            arxiv_docs = self.fetch_arxiv_papers(self.config.arxiv_papers)
            all_documents.extend(arxiv_docs)
        
        # Wikipedia articles
        if self.config.wikipedia_articles > 0:
            wiki_docs = self.fetch_wikipedia_articles(self.config.wikipedia_articles)
            all_documents.extend(wiki_docs)
        
        # PubMed abstracts
        if self.config.pubmed_articles > 0:
            pubmed_docs = self.fetch_pubmed_abstracts(self.config.pubmed_articles)
            all_documents.extend(pubmed_docs)
        
        # Legal documents
        if self.config.legal_documents > 0:
            legal_docs = self.create_legal_documents(self.config.legal_documents)
            all_documents.extend(legal_docs)
        
        # General content
        if self.config.general_web_content > 0:
            general_docs = self.create_general_content(self.config.general_web_content)
            all_documents.extend(general_docs)
        
        print(f"\n📚 CORPUS SUMMARY")
        print(f"Total documents: {len(all_documents)}")
        
        # Domain distribution
        domain_counts = {}
        for doc in all_documents:
            domain = doc.metadata.custom_fields.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} documents")
        
        return all_documents


class ComplexQueryGenerator:
    """Generate complex queries requiring deep semantic understanding."""
    
    def __init__(self, config: SemanticEvaluationConfig):
        self.config = config
        
    def generate_multi_hop_queries(self) -> List[Dict[str, Any]]:
        """Generate queries requiring information from multiple documents."""
        queries = [
            {
                "query": "How do quantum computing principles apply to machine learning optimization, and what are the implications for neural network training algorithms?",
                "type": "multi_hop",
                "complexity": "high",
                "expected_domains": ["computer_science", "physics"],
                "reasoning_required": "synthesis_across_domains"
            },
            {
                "query": "What are the ethical considerations of using AI in healthcare diagnosis, particularly regarding algorithmic bias and patient privacy?",
                "type": "multi_hop", 
                "complexity": "high",
                "expected_domains": ["medicine", "computer_science", "law"],
                "reasoning_required": "ethical_analysis"
            },
            {
                "query": "How do quantum mechanical effects in biological systems relate to consciousness theories in neuroscience?",
                "type": "multi_hop",
                "complexity": "expert",
                "expected_domains": ["physics", "medicine", "philosophy"],
                "reasoning_required": "interdisciplinary_synthesis"
            },
            {
                "query": "What are the legal frameworks governing artificial intelligence development, and how do they address intellectual property issues in machine learning models?",
                "type": "multi_hop",
                "complexity": "high", 
                "expected_domains": ["law", "computer_science"],
                "reasoning_required": "regulatory_analysis"
            },
            {
                "query": "How do quantum error correction techniques compare to classical error correction methods in information theory?",
                "type": "multi_hop",
                "complexity": "expert",
                "expected_domains": ["physics", "computer_science"],
                "reasoning_required": "comparative_analysis"
            }
        ]
        
        # Extend to reach target count
        extended_queries = []
        for i in range(self.config.multi_hop_queries):
            base_query = queries[i % len(queries)]
            query_copy = base_query.copy()
            query_copy["id"] = f"multi_hop_{i:03d}"
            extended_queries.append(query_copy)
        
        return extended_queries
    
    def generate_comparative_queries(self) -> List[Dict[str, Any]]:
        """Generate queries requiring comparison and contrast."""
        queries = [
            {
                "query": "Compare and contrast quantum computing approaches versus classical computing methods for solving optimization problems in artificial intelligence.",
                "type": "comparative",
                "complexity": "high",
                "expected_domains": ["computer_science", "physics"],
                "reasoning_required": "comparative_analysis"
            },
            {
                "query": "What are the differences between supervised and unsupervised machine learning approaches in medical diagnosis applications?",
                "type": "comparative",
                "complexity": "medium",
                "expected_domains": ["computer_science", "medicine"],
                "reasoning_required": "method_comparison"
            },
            {
                "query": "How do legal frameworks for AI regulation differ across jurisdictions, and what are the implications for international technology companies?",
                "type": "comparative", 
                "complexity": "high",
                "expected_domains": ["law", "computer_science"],
                "reasoning_required": "regulatory_comparison"
            },
            {
                "query": "Compare the effectiveness of classical physics models versus quantum mechanical models in explaining biological phenomena.",
                "type": "comparative",
                "complexity": "expert",
                "expected_domains": ["physics", "medicine"],
                "reasoning_required": "theoretical_comparison"
            }
        ]
        
        extended_queries = []
        for i in range(self.config.comparative_queries):
            base_query = queries[i % len(queries)]
            query_copy = base_query.copy()
            query_copy["id"] = f"comparative_{i:03d}"
            extended_queries.append(query_copy)
        
        return extended_queries
    
    def generate_inference_queries(self) -> List[Dict[str, Any]]:
        """Generate queries requiring inference and deduction."""
        queries = [
            {
                "query": "Given the current limitations of quantum computing hardware, what are the most promising near-term applications in artificial intelligence?",
                "type": "inference",
                "complexity": "high",
                "expected_domains": ["computer_science", "physics"],
                "reasoning_required": "predictive_inference"
            },
            {
                "query": "What can we infer about the future of medical diagnosis from current trends in AI model interpretability and regulatory requirements?",
                "type": "inference",
                "complexity": "high",
                "expected_domains": ["medicine", "computer_science", "law"],
                "reasoning_required": "trend_analysis"
            },
            {
                "query": "Based on principles of quantum mechanics and information theory, what are the theoretical limits of computational efficiency?",
                "type": "inference",
                "complexity": "expert",
                "expected_domains": ["physics", "computer_science"],
                "reasoning_required": "theoretical_deduction"
            },
            {
                "query": "What implications can be drawn from the intersection of artificial intelligence and legal decision-making for the future of judicial systems?",
                "type": "inference",
                "complexity": "high",
                "expected_domains": ["law", "computer_science"],
                "reasoning_required": "systemic_inference"
            }
        ]
        
        extended_queries = []
        for i in range(self.config.inference_queries):
            base_query = queries[i % len(queries)]
            query_copy = base_query.copy()
            query_copy["id"] = f"inference_{i:03d}"
            extended_queries.append(query_copy)
        
        return extended_queries
    
    def generate_all_queries(self) -> List[Dict[str, Any]]:
        """Generate all complex queries."""
        print("🔍 GENERATING COMPLEX SEMANTIC QUERIES")
        print("=" * 50)
        
        all_queries = []
        
        # Multi-hop queries
        multi_hop = self.generate_multi_hop_queries()
        all_queries.extend(multi_hop)
        print(f"Generated {len(multi_hop)} multi-hop queries")
        
        # Comparative queries
        comparative = self.generate_comparative_queries()
        all_queries.extend(comparative)
        print(f"Generated {len(comparative)} comparative queries")
        
        # Inference queries
        inference = self.generate_inference_queries()
        all_queries.extend(inference)
        print(f"Generated {len(inference)} inference queries")
        
        print(f"\n📋 QUERY SUMMARY")
        print(f"Total queries: {len(all_queries)}")
        
        # Complexity distribution
        complexity_counts = {}
        for query in all_queries:
            complexity = query.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        for complexity, count in complexity_counts.items():
            print(f"  {complexity}: {count} queries")
        
        return all_queries


def main():
    """Run semantic quality evaluation."""
    print("🧬 SEMANTIC QUALITY EVALUATION: CLASSICAL vs QUANTUM")
    print("=" * 70)
    print("Testing whether quantum reranking provides better semantic understanding")
    print("using real documents and complex queries requiring deep reasoning.")
    print()
    
    # Initialize configuration
    config = SemanticEvaluationConfig()
    
    # Fetch documents
    print("Phase 1: Document Collection")
    print("-" * 30)
    fetcher = DocumentFetcher(config)
    documents = fetcher.fetch_all_documents()
    
    if len(documents) < 50:
        print("❌ Insufficient documents fetched. Need at least 50 for meaningful evaluation.")
        return
    
    # Generate queries
    print("\nPhase 2: Query Generation")
    print("-" * 30)
    query_generator = ComplexQueryGenerator(config)
    queries = query_generator.generate_all_queries()
    
    # Save data for reproducibility
    print(f"\nSaving corpus and queries for reproducibility...")
    with open("semantic_evaluation_corpus.json", "w") as f:
        corpus_data = {
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata.to_dict()
                }
                for doc in documents
            ],
            "queries": queries,
            "config": {
                "total_documents": len(documents),
                "total_queries": len(queries),
                "domains": list(set(doc.metadata.custom_fields.get("domain", "unknown") for doc in documents))
            }
        }
        json.dump(corpus_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Corpus and queries saved to semantic_evaluation_corpus.json")
    print(f"   Documents: {len(documents)}")
    print(f"   Queries: {len(queries)}")
    print(f"   Ready for classical vs quantum evaluation comparison")
    
    return documents, queries


if __name__ == "__main__":
    main()