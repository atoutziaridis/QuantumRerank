"""
Tests for QuantumRerank client functionality.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from quantum_rerank import Client
from quantum_rerank.exceptions import (
    QuantumRerankError,
    AuthenticationError, 
    RateLimitError,
    ValidationError
)


class TestClient:
    """Test cases for the QuantumRerank Client class."""
    
    def test_client_initialization(self):
        """Test client initialization with default and custom parameters."""
        # Test default initialization
        client = Client(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30
        assert client.max_retries == 3
        assert "Bearer test-key" in client.session.headers["Authorization"]
        
        # Test custom initialization
        client = Client(
            api_key="custom-key",
            base_url="https://api.example.com",
            timeout=60,
            max_retries=5
        )
        assert client.api_key == "custom-key"
        assert client.base_url == "https://api.example.com"
        assert client.timeout == 60
        assert client.max_retries == 5
    
    @patch('requests.Session.request')
    def test_rerank_success(self, mock_request):
        """Test successful rerank request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "text": "Quantum computing uses quantum mechanics",
                    "similarity_score": 0.95,
                    "rank": 1,
                    "metadata": {"source": "quantum"}
                },
                {
                    "text": "Machine learning is AI subset", 
                    "similarity_score": 0.75,
                    "rank": 2,
                    "metadata": {"source": "ml"}
                }
            ],
            "computation_time_ms": 145.5,
            "method_used": "hybrid",
            "query_metadata": {"model": "test"}
        }
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key")
        result = client.rerank(
            query="What is quantum computing?",
            documents=["Quantum computing uses quantum mechanics", "Machine learning is AI subset"],
            top_k=2,
            method="hybrid"
        )
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["json"]["query"] == "What is quantum computing?"
        assert call_args[1]["json"]["candidates"] == [
            "Quantum computing uses quantum mechanics", 
            "Machine learning is AI subset"
        ]
        assert call_args[1]["json"]["method"] == "hybrid"
        assert call_args[1]["json"]["top_k"] == 2
        
        # Verify response parsing
        assert result.query == "What is quantum computing?"
        assert result.method == "hybrid"
        assert result.processing_time_ms == 145.5
        assert len(result.documents) == 2
        
        # Check first document
        doc1 = result.documents[0]
        assert doc1.text == "Quantum computing uses quantum mechanics"
        assert doc1.score == 0.95
        assert doc1.rank == 1
        assert doc1.metadata["source"] == "quantum"
        
        # Check second document
        doc2 = result.documents[1]
        assert doc2.text == "Machine learning is AI subset"
        assert doc2.score == 0.75
        assert doc2.rank == 2
    
    def test_rerank_validation_errors(self):
        """Test rerank validation errors."""
        client = Client(api_key="test-key")
        
        # Test empty query
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            client.rerank(query="", documents=["doc1"])
        
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            client.rerank(query="   ", documents=["doc1"])
        
        # Test empty documents
        with pytest.raises(ValidationError, match="At least one document is required"):
            client.rerank(query="test query", documents=[])
        
        # Test invalid top_k
        with pytest.raises(ValidationError, match="top_k must be greater than 0"):
            client.rerank(query="test", documents=["doc1"], top_k=0)
        
        with pytest.raises(ValidationError, match="top_k must be greater than 0"):
            client.rerank(query="test", documents=["doc1"], top_k=-1)
    
    def test_rerank_top_k_adjustment(self):
        """Test that top_k is adjusted when greater than document count."""
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [{"text": "doc1", "similarity_score": 0.9, "rank": 1, "metadata": {}}],
                "computation_time_ms": 100.0,
                "method_used": "hybrid",
                "query_metadata": {}
            }
            mock_request.return_value = mock_response
            
            client = Client(api_key="test-key")
            client.rerank(
                query="test",
                documents=["doc1"],
                top_k=5  # Greater than document count
            )
            
            # Should adjust top_k to 1
            call_args = mock_request.call_args
            assert call_args[1]["json"]["top_k"] == 1
    
    @patch('requests.Session.request')
    def test_authentication_error(self, mock_request):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response
        
        client = Client(api_key="invalid-key")
        
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client.rerank(query="test", documents=["doc1"])
    
    @patch('requests.Session.request')
    def test_rate_limit_error(self, mock_request):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key")
        
        with pytest.raises(RateLimitError) as exc_info:
            client.rerank(query="test", documents=["doc1"])
        
        assert exc_info.value.retry_after == 60
        assert "Retry after 60 seconds" in str(exc_info.value)
    
    @patch('requests.Session.request')
    def test_validation_error_from_api(self, mock_request):
        """Test validation error from API response."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "detail": "Query exceeds maximum length"
        }
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key")
        
        with pytest.raises(ValidationError, match="Query exceeds maximum length"):
            client.rerank(query="test", documents=["doc1"])
    
    @patch('requests.Session.request')
    def test_health_check_success(self, mock_request):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0.0",
            "components": {"quantum_engine": "ok", "embeddings": "ok"}
        }
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key")
        health = client.health()
        
        assert health.status == "healthy"
        assert health.is_healthy == True
        assert health.version == "1.0.0"
        assert health.components["quantum_engine"] == "ok"
        
        # Verify correct endpoint was called
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert "/health" in call_args[1]["url"] or call_args[0][1].endswith("/health")
    
    @patch('requests.Session.request')
    def test_retry_logic_success(self, mock_request):
        """Test retry logic on temporary failures."""
        # First call fails with 503, second succeeds
        error_response = Mock()
        error_response.status_code = 503
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "results": [],
            "computation_time_ms": 100.0,
            "method_used": "hybrid",
            "query_metadata": {}
        }
        
        mock_request.side_effect = [error_response, success_response]
        
        client = Client(api_key="test-key")
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = client.rerank(query="test", documents=["doc1"])
        
        # Should have made 2 requests (first failed, second succeeded)
        assert mock_request.call_count == 2
        assert len(result.documents) == 0
    
    @patch('requests.Session.request')
    def test_retry_logic_max_retries(self, mock_request):
        """Test retry logic reaching max retries."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key", max_retries=2)
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(QuantumRerankError):
                client.rerank(query="test", documents=["doc1"])
        
        # Should have made max_retries attempts
        assert mock_request.call_count == 2
    
    @patch('requests.Session.request')
    def test_timeout_error(self, mock_request):
        """Test timeout error handling."""
        mock_request.side_effect = requests.exceptions.Timeout("Request timeout")
        
        client = Client(api_key="test-key", max_retries=1)
        
        with pytest.raises(QuantumRerankError, match="Request timeout"):
            client.rerank(query="test", documents=["doc1"])
    
    @patch('requests.Session.request')
    def test_connection_error(self, mock_request):
        """Test connection error handling."""
        mock_request.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        client = Client(api_key="test-key", max_retries=1)
        
        with pytest.raises(QuantumRerankError, match="Connection error"):
            client.rerank(query="test", documents=["doc1"])
    
    def test_context_manager(self):
        """Test client as context manager."""
        with Client(api_key="test-key") as client:
            assert client.api_key == "test-key"
            assert client.session is not None
        
        # Session should be closed after context exit
        # Note: We can't easily test session.close() was called without mocking
    
    @patch('requests.Session.request')
    def test_get_similarity_methods(self, mock_request):
        """Test get_similarity_methods."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "methods": {
                "classical": {"name": "Classical Cosine"},
                "quantum": {"name": "Quantum Fidelity"},
                "hybrid": {"name": "Hybrid Method"}
            }
        }
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key")
        methods = client.get_similarity_methods()
        
        assert "methods" in methods
        assert "classical" in methods["methods"]
        assert methods["methods"]["classical"]["name"] == "Classical Cosine"
    
    @patch('requests.Session.request')
    def test_get_limits(self, mock_request):
        """Test get_limits."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "limits": {
                "max_candidates": 100,
                "max_query_length": 5000
            }
        }
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key")
        limits = client.get_limits()
        
        assert "limits" in limits
        assert limits["limits"]["max_candidates"] == 100
    
    @patch('requests.Session.request')
    def test_validate_request(self, mock_request):
        """Test validate_request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        mock_request.return_value = mock_response
        
        client = Client(api_key="test-key")
        validation = client.validate_request(
            query="test query",
            documents=["doc1", "doc2"]
        )
        
        assert validation["valid"] == True
        assert isinstance(validation["warnings"], list)
        assert isinstance(validation["errors"], list)