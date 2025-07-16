"""
PMC XML Article Parser
Parses full-text XML articles from PMC Open Access into RAG-ready documents.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PMCArticle:
    """Structured representation of a PMC article."""
    pmc_id: str
    title: str
    abstract: str
    full_text: str
    sections: Dict[str, str]
    keywords: List[str]
    journal: str
    authors: List[str]
    publication_year: str
    medical_domain: str


class PMCXMLParser:
    """Parser for PMC XML articles."""
    
    def __init__(self):
        # Medical domain classification keywords
        self.domain_keywords = {
            'cardiology': [
                'cardiac', 'heart', 'myocardial', 'coronary', 'cardiovascular',
                'arrhythmia', 'infarction', 'hypertension', 'atherosclerosis'
            ],
            'diabetes': [
                'diabetes', 'insulin', 'glucose', 'glycemic', 'diabetic',
                'hyperglycemia', 'endocrine', 'metabolic', 'pancreatic'
            ],
            'respiratory': [
                'lung', 'pulmonary', 'respiratory', 'pneumonia', 'asthma',
                'COPD', 'bronchial', 'ventilation', 'breathing'
            ],
            'neurology': [
                'brain', 'neural', 'neurological', 'stroke', 'seizure',
                'alzheimer', 'parkinson', 'epilepsy', 'cognitive'
            ],
            'oncology': [
                'cancer', 'tumor', 'malignant', 'oncology', 'chemotherapy',
                'radiation', 'metastasis', 'carcinoma', 'lymphoma'
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical notation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\-\%\+\=\<\>\/]', ' ', text)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_text_from_element(self, element) -> str:
        """Extract text content from XML element, handling nested tags."""
        if element is None:
            return ""
        
        # Get text content including nested elements
        text_parts = []
        
        if element.text:
            text_parts.append(element.text)
        
        for child in element:
            text_parts.append(self.extract_text_from_element(child))
            if child.tail:
                text_parts.append(child.tail)
        
        return ' '.join(text_parts)
    
    def classify_medical_domain(self, title: str, abstract: str, keywords: List[str]) -> str:
        """Classify article into medical domain based on content."""
        # Combine all text for analysis
        combined_text = f"{title} {abstract} {' '.join(keywords)}".lower()
        
        domain_scores = {}
        
        for domain, domain_keywords in self.domain_keywords.items():
            score = sum(1 for keyword in domain_keywords if keyword in combined_text)
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no clear match
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general'
    
    def parse_xml_file(self, xml_path: Path) -> Optional[PMCArticle]:
        """Parse a single PMC XML file into structured article."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract basic metadata
            pmc_id = self.extract_pmc_id(root)
            title = self.extract_title(root)
            abstract = self.extract_abstract(root)
            
            # Extract full text body
            full_text, sections = self.extract_body_text(root)
            
            # Extract additional metadata
            keywords = self.extract_keywords(root)
            journal = self.extract_journal(root)
            authors = self.extract_authors(root)
            year = self.extract_publication_year(root)
            
            # Classify medical domain
            domain = self.classify_medical_domain(title, abstract, keywords)
            
            # Create article object
            article = PMCArticle(
                pmc_id=pmc_id,
                title=self.clean_text(title),
                abstract=self.clean_text(abstract),
                full_text=self.clean_text(full_text),
                sections=sections,
                keywords=keywords,
                journal=journal,
                authors=authors,
                publication_year=year,
                medical_domain=domain
            )
            
            return article
            
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None
    
    def extract_pmc_id(self, root) -> str:
        """Extract PMC ID from XML."""
        # Try different possible locations
        pmc_id_elem = root.find(".//article-id[@pub-id-type='pmc']")
        if pmc_id_elem is not None and pmc_id_elem.text:
            return pmc_id_elem.text.strip()
        
        # Fallback: try to extract from filename or other sources
        return "PMC_UNKNOWN"
    
    def extract_title(self, root) -> str:
        """Extract article title."""
        title_elem = root.find(".//article-title")
        if title_elem is not None:
            return self.extract_text_from_element(title_elem)
        return "No Title"
    
    def extract_abstract(self, root) -> str:
        """Extract abstract text."""
        abstract_parts = []
        
        # Look for abstract section
        abstract_elem = root.find(".//abstract")
        if abstract_elem is not None:
            # Handle structured abstracts
            for section in abstract_elem.findall(".//sec"):
                section_title = section.find("title")
                if section_title is not None:
                    abstract_parts.append(f"{self.extract_text_from_element(section_title)}:")
                
                for p in section.findall(".//p"):
                    abstract_parts.append(self.extract_text_from_element(p))
            
            # Handle simple abstracts
            if not abstract_parts:
                for p in abstract_elem.findall(".//p"):
                    abstract_parts.append(self.extract_text_from_element(p))
            
            # Fallback: get all text from abstract
            if not abstract_parts:
                abstract_parts.append(self.extract_text_from_element(abstract_elem))
        
        return ' '.join(abstract_parts) if abstract_parts else "No Abstract"
    
    def extract_body_text(self, root) -> tuple:
        """Extract main body text and sections."""
        sections = {}
        body_parts = []
        
        # Find main body
        body_elem = root.find(".//body")
        if body_elem is not None:
            # Extract sections
            for section in body_elem.findall(".//sec"):
                section_title_elem = section.find("title")
                section_title = "Section"
                
                if section_title_elem is not None:
                    section_title = self.extract_text_from_element(section_title_elem)
                
                # Extract paragraphs from this section
                section_text_parts = []
                for p in section.findall(".//p"):
                    section_text_parts.append(self.extract_text_from_element(p))
                
                section_text = ' '.join(section_text_parts)
                if section_text:
                    sections[section_title] = self.clean_text(section_text)
                    body_parts.append(section_text)
            
            # If no sections found, extract all paragraphs
            if not body_parts:
                for p in body_elem.findall(".//p"):
                    body_parts.append(self.extract_text_from_element(p))
        
        full_text = ' '.join(body_parts) if body_parts else "No Body Text"
        return full_text, sections
    
    def extract_keywords(self, root) -> List[str]:
        """Extract keywords/MeSH terms."""
        keywords = []
        
        # Extract keyword groups
        for kwd_group in root.findall(".//kwd-group"):
            for kwd in kwd_group.findall(".//kwd"):
                if kwd.text:
                    keywords.append(kwd.text.strip())
        
        return keywords
    
    def extract_journal(self, root) -> str:
        """Extract journal name."""
        journal_elem = root.find(".//journal-title")
        if journal_elem is not None and journal_elem.text:
            return journal_elem.text.strip()
        return "Unknown Journal"
    
    def extract_authors(self, root) -> List[str]:
        """Extract author names."""
        authors = []
        
        for contrib in root.findall(".//contrib[@contrib-type='author']"):
            # Extract given and surname
            given = contrib.find(".//given-names")
            surname = contrib.find(".//surname")
            
            if given is not None and surname is not None:
                author_name = f"{given.text} {surname.text}".strip()
                authors.append(author_name)
            elif surname is not None:
                authors.append(surname.text.strip())
        
        return authors
    
    def extract_publication_year(self, root) -> str:
        """Extract publication year."""
        year_elem = root.find(".//pub-date/year")
        if year_elem is not None and year_elem.text:
            return year_elem.text.strip()
        return "Unknown"
    
    def parse_directory(self, xml_dir: Path, max_articles: int = 100) -> List[PMCArticle]:
        """Parse all XML files in directory."""
        print(f"Parsing XML files from {xml_dir}")
        
        xml_files = list(xml_dir.glob("*.xml"))
        if not xml_files:
            print("No XML files found")
            return []
        
        print(f"Found {len(xml_files)} XML files")
        
        articles = []
        parsed_count = 0
        failed_count = 0
        
        for xml_file in xml_files:
            if len(articles) >= max_articles:
                break
            
            print(f"Parsing {xml_file.name}... ({parsed_count + 1}/{min(len(xml_files), max_articles)})")
            
            article = self.parse_xml_file(xml_file)
            if article:
                articles.append(article)
                parsed_count += 1
                print(f"  ✓ {article.title[:60]}... (Domain: {article.medical_domain})")
            else:
                failed_count += 1
                print(f"  ✗ Failed to parse")
        
        print(f"\nParsing complete:")
        print(f"  Successfully parsed: {parsed_count}")
        print(f"  Failed to parse: {failed_count}")
        
        # Print domain distribution
        domain_counts = {}
        for article in articles:
            domain_counts[article.medical_domain] = domain_counts.get(article.medical_domain, 0) + 1
        
        print(f"\nDomain distribution:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count} articles")
        
        return articles


def main():
    """Parse PMC XML files."""
    parser = PMCXMLParser()
    xml_dir = Path("./pmc_docs")
    
    if not xml_dir.exists():
        print(f"Directory {xml_dir} does not exist. Please download XML files first.")
        return
    
    # Parse articles
    articles = parser.parse_directory(xml_dir, max_articles=100)
    
    if articles:
        print(f"\n✅ Successfully parsed {len(articles)} PMC articles")
        print("Articles are ready for benchmark testing")
        
        # Save parsed articles for later use
        import pickle
        with open("parsed_pmc_articles.pkl", "wb") as f:
            pickle.dump(articles, f)
        print("Saved parsed articles to parsed_pmc_articles.pkl")
    else:
        print("❌ No articles parsed successfully")


if __name__ == "__main__":
    main()