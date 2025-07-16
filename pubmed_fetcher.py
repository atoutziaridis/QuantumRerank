"""
Real PubMed document fetcher for medical RAG testing.
Fetches actual medical documents from PubMed Central Open Access subset.
"""

import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import re
import random


class PubMedDocumentFetcher:
    """Fetches real medical documents from PubMed Central."""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = None  # Add your NCBI API key if you have one
        
    def search_pubmed(self, query: str, max_results: int = 20) -> List[str]:
        """Search PubMed and return PMIDs."""
        search_url = f"{self.base_url}esearch.fcgi"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get("esearchresult", {}).get("idlist", [])
            print(f"Found {len(pmids)} articles for query: {query}")
            return pmids
            
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def fetch_abstract(self, pmid: str) -> Optional[Dict[str, str]]:
        """Fetch abstract and metadata for a given PMID."""
        fetch_url = f"{self.base_url}efetch.fcgi"
        
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(fetch_url, params=params)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Extract article information
            article = root.find(".//Article")
            if article is None:
                return None
                
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"
            
            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Get journal info
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown journal"
            
            # Get keywords/mesh terms
            keywords = []
            for keyword in root.findall(".//Keyword"):
                if keyword.text:
                    keywords.append(keyword.text)
            
            # Get MeSH terms
            mesh_terms = []
            for mesh in root.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "keywords": keywords,
                "mesh_terms": mesh_terms,
                "full_text": f"{title}\n\n{abstract}"
            }
            
        except Exception as e:
            print(f"Error fetching PMID {pmid}: {e}")
            return None
    
    def fetch_medical_documents(self, num_docs: int = 20) -> List[Dict[str, str]]:
        """Fetch real medical documents from PubMed."""
        print(f"Fetching {num_docs} real medical documents from PubMed...")
        
        # Medical search queries to get diverse content
        search_queries = [
            "myocardial infarction treatment",
            "diabetes mellitus management", 
            "COPD exacerbation therapy",
            "hypertension cardiovascular risk",
            "pneumonia antibiotic treatment",
            "heart failure ACE inhibitors",
            "stroke thrombolytic therapy", 
            "sepsis intensive care",
            "asthma bronchodilator treatment",
            "atrial fibrillation anticoagulation"
        ]
        
        documents = []
        docs_per_query = max(1, num_docs // len(search_queries))
        
        for query in search_queries:
            if len(documents) >= num_docs:
                break
                
            print(f"Searching for: {query}")
            pmids = self.search_pubmed(query, docs_per_query + 5)  # Get extra in case some fail
            
            for pmid in pmids:
                if len(documents) >= num_docs:
                    break
                    
                doc_data = self.fetch_abstract(pmid)
                if doc_data and doc_data["abstract"]:  # Only keep documents with abstracts
                    documents.append(doc_data)
                    print(f"  Fetched: {doc_data['title'][:60]}...")
                
                # Rate limiting - NCBI allows 3 requests per second without API key
                time.sleep(0.4)
        
        print(f"Successfully fetched {len(documents)} real medical documents")
        return documents


def fetch_real_medical_documents(num_docs: int = 15) -> List[Dict[str, str]]:
    """Main function to fetch real medical documents."""
    fetcher = PubMedDocumentFetcher()
    return fetcher.fetch_medical_documents(num_docs)


if __name__ == "__main__":
    # Test the fetcher
    docs = fetch_real_medical_documents(5)
    
    print(f"\nFetched {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc['title']}")
        print(f"   Journal: {doc['journal']}")
        print(f"   Abstract length: {len(doc['abstract'])} chars")
        print(f"   MeSH terms: {len(doc['mesh_terms'])}")
        print()