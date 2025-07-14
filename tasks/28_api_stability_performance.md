# Task 28: API Stability & Performance

## Overview
Ensure the API is stable, performs well under load, and handles errors gracefully for production use.

## Objectives
- API consistently meets performance targets (<500ms, <100ms similarity)
- Handles errors gracefully without crashes
- Stable under production load patterns
- Proper monitoring and alerting

## Requirements

### Performance Targets (from PRD)
- **Similarity computation**: <100ms per request
- **Batch reranking**: <500ms for 50-100 documents  
- **Memory usage**: <2GB per instance
- **Throughput**: 100+ requests/minute per instance

### API Reliability

#### Request Validation
```python
# quantum_rerank/api/validation.py
from pydantic import BaseModel, validator
from typing import List, Optional

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    method: str = "hybrid"
    top_k: Optional[int] = None
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 10000:  # 10K character limit
            raise ValueError('Query too long (max 10000 characters)')
        return v.strip()
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v:
            raise ValueError('Documents list cannot be empty')
        if len(v) > 1000:  # Max 1000 documents
            raise ValueError('Too many documents (max 1000)')
        
        for i, doc in enumerate(v):
            if not doc or not doc.strip():
                raise ValueError(f'Document {i} cannot be empty')
            if len(doc) > 50000:  # 50K character limit per doc
                raise ValueError(f'Document {i} too long (max 50000 characters)')
        
        return [doc.strip() for doc in v]
    
    @validator('method')
    def validate_method(cls, v):
        valid_methods = ['classical', 'quantum', 'hybrid']
        if v not in valid_methods:
            raise ValueError(f'Invalid method. Must be one of: {valid_methods}')
        return v
    
    @validator('top_k')
    def validate_top_k(cls, v, values):
        if v is not None:
            if v < 1:
                raise ValueError('top_k must be positive')
            if 'documents' in values and v > len(values['documents']):
                raise ValueError('top_k cannot exceed number of documents')
        return v
```

#### Error Handling Middleware
```python
# quantum_rerank/api/middleware/error_handling.py
import traceback
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            processing_time = time.time() - start_time
            
            # Log error with context
            logger.error(
                f"Unhandled error in {request.method} {request.url.path}",
                extra={
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "processing_time": processing_time,
                    "request_id": request.headers.get("X-Request-ID"),
                    "user_agent": request.headers.get("User-Agent"),
                }
            )
            
            # Return structured error response
            error_response = {
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": request.headers.get("X-Request-ID"),
                "processing_time_ms": round(processing_time * 1000, 2)
            }
            
            response = JSONResponse(
                status_code=500,
                content=error_response
            )
            await response(scope, receive, send)
```

#### Circuit Breaker for Quantum Operations
```python
# quantum_rerank/core/circuit_breaker.py
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage in quantum similarity engine
quantum_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

def compute_quantum_similarity_with_breaker(embedding1, embedding2):
    try:
        return quantum_circuit_breaker.call(
            compute_quantum_similarity, embedding1, embedding2
        )
    except Exception:
        # Fallback to classical method
        return compute_classical_similarity(embedding1, embedding2)
```

### Performance Monitoring

#### Request Timing Middleware
```python
# quantum_rerank/api/middleware/timing.py
import time
import logging
from fastapi import Request

logger = logging.getLogger(__name__)

class TimingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Add timing to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                processing_time = time.time() - start_time
                
                # Add timing header
                headers = dict(message.get("headers", []))
                headers[b"x-processing-time"] = f"{processing_time:.3f}".encode()
                message["headers"] = list(headers.items())
                
                # Log request timing
                logger.info(
                    f"{request.method} {request.url.path}",
                    extra={
                        "processing_time": processing_time,
                        "status_code": message.get("status", 0),
                        "request_id": request.headers.get("X-Request-ID"),
                    }
                )
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
```

#### Memory Monitoring
```python
# quantum_rerank/core/memory_monitor.py
import psutil
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        usage = {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": memory_percent,
            "memory_limit_gb": self.max_memory_bytes / 1024 / 1024 / 1024
        }
        
        # Log warning if approaching limit
        if memory_info.rss > self.max_memory_bytes * 0.8:
            logger.warning(
                f"High memory usage: {usage['memory_mb']:.1f}MB "
                f"({usage['memory_percent']:.1f}%)"
            )
        
        return usage
    
    def is_memory_available(self) -> bool:
        """Check if we have memory available for new requests"""
        return self.process.memory_info().rss < self.max_memory_bytes * 0.9

# Global memory monitor
memory_monitor = MemoryMonitor()
```

### Load Testing

#### Performance Test Script
```python
# tests/load_test.py
import asyncio
import aiohttp
import time
import statistics
from typing import List

class LoadTester:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def single_request(self, session: aiohttp.ClientSession) -> float:
        """Make single rerank request and return response time"""
        start_time = time.time()
        
        payload = {
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a subset of AI",
                "Python is a programming language", 
                "Deep learning uses neural networks",
                "Data science involves analyzing data"
            ],
            "method": "hybrid"
        }
        
        try:
            async with session.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
                headers=self.headers
            ) as response:
                await response.json()
                return time.time() - start_time
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    async def run_load_test(self, concurrent_users: int, requests_per_user: int):
        """Run load test with specified parameters"""
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for concurrent users
            tasks = []
            for _ in range(concurrent_users):
                for _ in range(requests_per_user):
                    tasks.append(self.single_request(session))
            
            # Execute all requests
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            response_times = [r for r in results if isinstance(r, float)]
            errors = len([r for r in results if not isinstance(r, float)])
            
            if response_times:
                stats = {
                    "total_requests": len(tasks),
                    "successful_requests": len(response_times),
                    "errors": errors,
                    "total_time": total_time,
                    "requests_per_second": len(response_times) / total_time,
                    "avg_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
                    "max_response_time": max(response_times),
                    "min_response_time": min(response_times)
                }
                
                print(f"Results:")
                print(f"  Requests/sec: {stats['requests_per_second']:.1f}")
                print(f"  Avg response: {stats['avg_response_time']*1000:.1f}ms")
                print(f"  95th percentile: {stats['p95_response_time']*1000:.1f}ms")
                print(f"  Error rate: {errors/len(tasks)*100:.1f}%")
                
                return stats
            else:
                print("All requests failed!")
                return None

async def main():
    tester = LoadTester("http://localhost:8000", "your-api-key")
    
    # Test scenarios
    await tester.run_load_test(concurrent_users=10, requests_per_user=10)
    await tester.run_load_test(concurrent_users=50, requests_per_user=5)
    await tester.run_load_test(concurrent_users=100, requests_per_user=2)

if __name__ == "__main__":
    asyncio.run(main())
```

### Health Checks

#### Comprehensive Health Check
```python
# quantum_rerank/api/endpoints/health.py
from fastapi import APIRouter, HTTPException
from quantum_rerank.core.memory_monitor import memory_monitor
from quantum_rerank.core.quantum_similarity_engine import quantum_engine
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Memory check
    try:
        memory_info = memory_monitor.check_memory_usage()
        health_status["checks"]["memory"] = {
            "status": "ok" if memory_monitor.is_memory_available() else "warning",
            **memory_info
        }
    except Exception as e:
        health_status["checks"]["memory"] = {"status": "error", "error": str(e)}
    
    # Quantum engine check
    try:
        start_time = time.time()
        # Test with small embeddings
        test_embedding = [0.1] * 768
        similarity = quantum_engine.compute_classical_cosine(test_embedding, test_embedding)
        response_time = time.time() - start_time
        
        health_status["checks"]["quantum_engine"] = {
            "status": "ok",
            "response_time_ms": round(response_time * 1000, 2),
            "similarity": similarity
        }
    except Exception as e:
        health_status["checks"]["quantum_engine"] = {"status": "error", "error": str(e)}
    
    # Overall status
    failed_checks = [k for k, v in health_status["checks"].items() if v["status"] == "error"]
    if failed_checks:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    
    warning_checks = [k for k, v in health_status["checks"].items() if v["status"] == "warning"]
    if warning_checks:
        health_status["status"] = "degraded"
    
    return health_status

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    # Quick check if service can handle requests
    if not memory_monitor.is_memory_available():
        raise HTTPException(status_code=503, detail="Not ready - high memory usage")
    
    return {"status": "ready"}

@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    # Basic check if service is alive
    return {"status": "alive"}
```

### Graceful Shutdown
```python
# quantum_rerank/api/app.py
import signal
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.active_requests = 0
    
    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def wait_for_shutdown(self):
        await self.shutdown_event.wait()
        
        # Wait for active requests to complete (max 30 seconds)
        for i in range(30):
            if self.active_requests == 0:
                break
            logger.info(f"Waiting for {self.active_requests} active requests...")
            await asyncio.sleep(1)
        
        logger.info("Graceful shutdown complete")

# Initialize graceful shutdown
shutdown_handler = GracefulShutdown()
signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)
signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
```

## Testing Strategy

### Automated Performance Tests
```bash
#!/bin/bash
# test-performance.sh

echo "Running performance tests..."

# Start the service
docker-compose -f docker-compose.prod.yml up -d
sleep 30

# Run load tests
python tests/load_test.py

# Check if performance targets are met
RESPONSE_TIME=$(curl -s http://localhost/health/detailed | jq '.checks.quantum_engine.response_time_ms')

if (( $(echo "$RESPONSE_TIME < 100" | bc -l) )); then
    echo "✅ Performance target met: ${RESPONSE_TIME}ms < 100ms"
else
    echo "❌ Performance target missed: ${RESPONSE_TIME}ms >= 100ms"
    exit 1
fi

echo "✅ All performance tests passed"
```

## Success Criteria
- [ ] API meets all performance targets consistently
- [ ] Graceful error handling without crashes
- [ ] Memory usage stays under 2GB limit
- [ ] Circuit breaker prevents cascade failures
- [ ] Health checks provide accurate status
- [ ] Load testing shows stable performance
- [ ] Monitoring and alerting works correctly

## Timeline
- **Week 1**: Error handling and validation
- **Week 2**: Performance monitoring and circuit breaker
- **Week 3**: Load testing and optimization
- **Week 4**: Health checks and graceful shutdown

This ensures the API is production-ready and stable under real-world conditions.