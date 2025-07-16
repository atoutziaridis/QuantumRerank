"""
Realistic Noise Injection for Medical Documents
Simulates OCR errors, medical abbreviations, and other real-world corruptions.
"""

import random
import re
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    ocr_error_rate: float = 0.02  # 2% character corruption
    abbreviation_rate: float = 0.10  # 10% chance to abbreviate
    spacing_error_rate: float = 0.005  # 0.5% spacing issues
    line_break_rate: float = 0.003  # 0.3% random line breaks


class MedicalNoiseInjector:
    """Injects realistic noise into medical documents."""
    
    def __init__(self):
        # OCR character substitution patterns from real medical document scanning
        self.ocr_substitutions = {
            # Letter to number confusions
            'o': ['0', 'O'], 'O': ['0', 'o'], '0': ['O', 'o'],
            'l': ['1', 'I', '|'], 'I': ['l', '1', '|'], '1': ['l', 'I'],
            'S': ['5', '$'], '5': ['S'], 'G': ['6', '9'], '6': ['G', 'b'],
            'B': ['8'], '8': ['B', '3'], 'Z': ['2'], '2': ['Z'],
            
            # Letter to letter confusions
            'a': ['@', 'e'], 'e': ['a', 'c', 'o'], 'c': ['e', 'o'],
            'n': ['m', 'h', 'r'], 'm': ['rn', 'n'], 'h': ['n', 'b'],
            'u': ['v', 'n'], 'v': ['u', 'y'], 'y': ['v', 'g'],
            'r': ['n', 'f'], 'f': ['t', 'r'], 't': ['f', '1'],
            'i': ['j', 'l'], 'j': ['i', '1'], 'p': ['b'], 'q': ['g', '9'],
            'd': ['b', 'cl'], 'w': ['vv', 'vy'], 'x': ['><', 'k'],
            'k': ['lc', 'x']
        }
        
        # Medical abbreviation mappings (comprehensive medical terminology)
        self.medical_abbreviations = {
            # Cardiovascular
            'myocardial infarction': 'MI',
            'acute myocardial infarction': 'AMI',
            'congestive heart failure': 'CHF',
            'heart failure': 'HF',
            'blood pressure': 'BP',
            'heart rate': 'HR',
            'electrocardiogram': 'ECG',
            'coronary artery disease': 'CAD',
            'percutaneous coronary intervention': 'PCI',
            'atrial fibrillation': 'AFib',
            'ventricular tachycardia': 'VT',
            'cardiac catheterization': 'cath',
            
            # Diabetes/Endocrine
            'diabetes mellitus': 'DM',
            'type 1 diabetes': 'T1DM',
            'type 2 diabetes': 'T2DM',
            'diabetic ketoacidosis': 'DKA',
            'blood glucose': 'BG',
            'insulin': 'ins',
            'hemoglobin A1c': 'HbA1c',
            'fasting blood glucose': 'FBG',
            'oral glucose tolerance test': 'OGTT',
            
            # Respiratory
            'chronic obstructive pulmonary disease': 'COPD',
            'acute respiratory distress syndrome': 'ARDS',
            'shortness of breath': 'SOB',
            'dyspnea on exertion': 'DOE',
            'oxygen saturation': 'O2 sat',
            'arterial blood gas': 'ABG',
            'chest X-ray': 'CXR',
            'computed tomography': 'CT',
            'magnetic resonance imaging': 'MRI',
            'pulmonary embolism': 'PE',
            'pulmonary function test': 'PFT',
            'continuous positive airway pressure': 'CPAP',
            
            # Neurology
            'cerebrovascular accident': 'CVA',
            'transient ischemic attack': 'TIA',
            'electroencephalogram': 'EEG',
            'cerebrospinal fluid': 'CSF',
            'central nervous system': 'CNS',
            'peripheral nervous system': 'PNS',
            'Glasgow Coma Scale': 'GCS',
            
            # Oncology
            'chemotherapy': 'chemo',
            'radiation therapy': 'RT',
            'computed tomography': 'CT',
            'positron emission tomography': 'PET',
            'bone marrow transplant': 'BMT',
            
            # General Medical
            'intensive care unit': 'ICU',
            'emergency department': 'ED',
            'emergency room': 'ER',
            'operating room': 'OR',
            'intravenous': 'IV',
            'intramuscular': 'IM',
            'subcutaneous': 'SC',
            'by mouth': 'PO',
            'twice daily': 'BID',
            'three times daily': 'TID',
            'four times daily': 'QID',
            'as needed': 'PRN',
            'before meals': 'AC',
            'after meals': 'PC',
            'at bedtime': 'HS',
            'immediately': 'STAT',
            'white blood cell': 'WBC',
            'red blood cell': 'RBC',
            'complete blood count': 'CBC',
            'basic metabolic panel': 'BMP',
            'liver function test': 'LFT',
            'blood urea nitrogen': 'BUN',
            'creatinine': 'Cr',
            'urinalysis': 'UA',
            'follow-up': 'f/u',
            'history of': 'h/o',
            'physical examination': 'PE',
            'vital signs': 'VS',
            'temperature': 'temp',
            'respiratory rate': 'RR',
            'within normal limits': 'WNL',
            'no acute distress': 'NAD'
        }
        
        # Reverse mapping for expansion
        self.abbreviation_expansions = {v: k for k, v in self.medical_abbreviations.items()}
    
    def inject_ocr_errors(self, text: str, error_rate: float = 0.02) -> str:
        """Inject OCR-like character substitution errors."""
        chars = list(text)
        
        for i in range(len(chars)):
            if random.random() < error_rate:
                char = chars[i]
                
                # Try exact match first
                if char in self.ocr_substitutions:
                    chars[i] = random.choice(self.ocr_substitutions[char])
                # Try case-insensitive match
                elif char.lower() in self.ocr_substitutions:
                    replacement = random.choice(self.ocr_substitutions[char.lower()])
                    # Preserve original case
                    if char.isupper():
                        replacement = replacement.upper()
                    chars[i] = replacement
        
        return ''.join(chars)
    
    def inject_spacing_errors(self, text: str, error_rate: float = 0.005) -> str:
        """Inject spacing and line break errors."""
        # Remove some spaces
        words = text.split()
        result = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if random.random() < error_rate:
                # Randomly merge with next word (remove space)
                if i + 1 < len(words):
                    merged_word = word + words[i + 1]
                    result.append(merged_word)
                    i += 2  # Skip next word since we merged it
                else:
                    result.append(word)
                    i += 1
            else:
                result.append(word)
                i += 1
        
        text_with_spacing_errors = ' '.join(result)
        
        # Add random line breaks
        chars = list(text_with_spacing_errors)
        for i in range(len(chars)):
            if chars[i] == ' ' and random.random() < error_rate * 0.3:
                chars[i] = '\n'
        
        return ''.join(chars)
    
    def inject_medical_abbreviations(self, text: str, abbreviation_rate: float = 0.10) -> str:
        """Convert medical terms to abbreviations."""
        result = text
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_terms = sorted(self.medical_abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
        
        for full_term, abbreviation in sorted_terms:
            if random.random() < abbreviation_rate:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(full_term), re.IGNORECASE)
                result = pattern.sub(abbreviation, result, count=1)  # Replace only first occurrence
        
        return result
    
    def inject_abbreviation_expansions(self, text: str, expansion_rate: float = 0.05) -> str:
        """Expand some abbreviations back to full terms (mixed usage)."""
        result = text
        
        for abbreviation, full_term in self.abbreviation_expansions.items():
            if abbreviation in result and random.random() < expansion_rate:
                result = result.replace(abbreviation, full_term, 1)  # Replace only first occurrence
        
        return result
    
    def create_clean_version(self, text: str) -> str:
        """Return clean version (no noise)."""
        return text
    
    def create_ocr_noisy_version(self, text: str, config: NoiseConfig = None) -> str:
        """Create version with OCR noise."""
        if config is None:
            config = NoiseConfig()
        
        noisy_text = text
        
        # Apply OCR errors
        noisy_text = self.inject_ocr_errors(noisy_text, config.ocr_error_rate)
        
        # Apply spacing/line break errors
        noisy_text = self.inject_spacing_errors(noisy_text, config.spacing_error_rate)
        
        return noisy_text
    
    def create_abbreviation_noisy_version(self, text: str, config: NoiseConfig = None) -> str:
        """Create version with medical abbreviation noise."""
        if config is None:
            config = NoiseConfig()
        
        noisy_text = text
        
        # Apply medical abbreviations
        noisy_text = self.inject_medical_abbreviations(noisy_text, config.abbreviation_rate)
        
        # Occasionally expand some abbreviations (mixed usage)
        noisy_text = self.inject_abbreviation_expansions(noisy_text, 0.03)
        
        return noisy_text
    
    def create_all_noise_versions(self, text: str, config: NoiseConfig = None) -> Dict[str, str]:
        """Create all noise versions of the text."""
        if config is None:
            config = NoiseConfig()
        
        return {
            'clean': self.create_clean_version(text),
            'ocr_noise': self.create_ocr_noisy_version(text, config),
            'abbreviation_noise': self.create_abbreviation_noisy_version(text, config)
        }


def main():
    """Test the noise injector."""
    injector = MedicalNoiseInjector()
    
    # Test text
    test_text = """
    Patient presents with acute myocardial infarction and elevated blood pressure. 
    Electrocardiogram shows ST-elevation in leads II, III, and aVF. Emergency department 
    staff initiated percutaneous coronary intervention. Intensive care unit monitoring 
    recommended for chronic obstructive pulmonary disease exacerbation. Patient has 
    history of diabetes mellitus and congestive heart failure.
    """
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*60)
    
    # Generate noise versions
    noise_versions = injector.create_all_noise_versions(test_text)
    
    for noise_type, noisy_text in noise_versions.items():
        print(f"\n{noise_type.upper()} VERSION:")
        print(noisy_text)
        print("-" * 40)


if __name__ == "__main__":
    main()