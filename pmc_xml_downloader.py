"""
PMC Open Access XML Article Downloader
Downloads real full-text medical articles from PMC Open Access subset.
"""

import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
from pathlib import Path
import time
import random


class PMCDownloader:
    """Downloads XML articles from PMC Open Access subset."""
    
    def __init__(self, download_dir: str = "./pmc_docs"):
        self.base_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
    def get_available_archives(self) -> list:
        """Get list of available tar.gz archives from PMC FTP."""
        print(f"Fetching archive list from {self.base_url}")
        
        try:
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML directory listing
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links to tar.gz archives
            archives = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.tar.gz') and 'PMC00' in href:  # Start with smaller PMC000 archive
                    archives.append(href)
            
            print(f"Found {len(archives)} tar.gz archives available")
            
            # Return the smallest archive first (PMC000)
            archives.sort()
            return archives[:1]  # Just download the first (smallest) one
            
        except Exception as e:
            print(f"Error fetching archive list: {e}")
            return []
    
    def download_and_extract_archive(self, archive_name: str, max_files: int = 100) -> int:
        """Download and extract PMC archive."""
        import tarfile
        
        url = urllib.parse.urljoin(self.base_url, archive_name)
        archive_path = self.download_dir / archive_name
        
        try:
            print(f"Downloading {archive_name}...")
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r  Progress: {percent:.1f}% ({downloaded // (1024*1024)}MB)", end='')
            
            print(f"\n  ✓ Downloaded {archive_name}")
            
            # Extract XML files
            print(f"Extracting XML files...")
            extracted_count = 0
            
            with tarfile.open(archive_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.xml') and extracted_count < max_files:
                        # Extract to download directory with simple filename
                        xml_filename = f"pmc_{extracted_count:04d}.xml"
                        xml_path = self.download_dir / xml_filename
                        
                        with tar.extractfile(member) as source:
                            with open(xml_path, 'wb') as target:
                                target.write(source.read())
                        
                        extracted_count += 1
                        if extracted_count % 10 == 0:
                            print(f"  Extracted {extracted_count} XML files...")
            
            print(f"  ✓ Extracted {extracted_count} XML files")
            
            # Clean up archive
            archive_path.unlink()
            print(f"  Cleaned up {archive_name}")
            
            return extracted_count
            
        except Exception as e:
            print(f"  ✗ Failed to process {archive_name}: {e}")
            if archive_path.exists():
                archive_path.unlink()
            return 0
    
    def download_articles(self, num_articles: int = 100) -> int:
        """Download specified number of XML articles."""
        print(f"Starting download of {num_articles} PMC XML articles...")
        
        # Get available archives
        available_archives = self.get_available_archives()
        
        if not available_archives:
            print("No PMC archives found to download")
            return 0
        
        total_extracted = 0
        
        for archive_name in available_archives:
            if total_extracted >= num_articles:
                break
                
            print(f"Processing archive: {archive_name}")
            
            extracted_count = self.download_and_extract_archive(archive_name, num_articles - total_extracted)
            total_extracted += extracted_count
            
            if total_extracted >= num_articles:
                break
        
        print(f"\nDownload complete:")
        print(f"  Successfully extracted: {total_extracted} XML files")
        print(f"  Files saved to: {self.download_dir}")
        
        return total_extracted


def main():
    """Download PMC XML articles."""
    downloader = PMCDownloader()
    
    # Download 100 articles
    count = downloader.download_articles(100)
    
    if count > 0:
        print(f"\n✅ Successfully downloaded {count} PMC XML articles")
        print(f"Files are ready for parsing in: {downloader.download_dir}")
    else:
        print("❌ No articles downloaded")


if __name__ == "__main__":
    main()