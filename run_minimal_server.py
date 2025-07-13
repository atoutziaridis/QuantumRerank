#!/usr/bin/env python3
"""
Minimal server runner that bypasses quantum dependencies.

This script creates a basic FastAPI server for testing core functionality
without requiring quantum libraries.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment to disable quantum features
os.environ['QUANTUM_RERANK_DISABLE_QUANTUM'] = 'true'
os.environ['QUANTUM_RERANK_CONFIG'] = str(PROJECT_ROOT / 'config' / 'minimal.yaml')

def create_minimal_app():
    """Create a minimal FastAPI app for testing."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import json
    import time
    
    app = FastAPI(
        title="QuantumRerank Minimal API",
        description="Minimal API for testing basic functionality",
        version="1.0.0-minimal"
    )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "QuantumRerank Minimal API",
            "version": "1.0.0-minimal",
            "status": "operational",
            "features": {
                "quantum": False,
                "classical": True,
                "mode": "minimal"
            }
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {
                "api": "ok",
                "quantum": "disabled",
                "classical": "ok"
            }
        }
    
    @app.post("/v1/similarity")
    async def similarity(request: dict):
        """Simple similarity endpoint."""
        text1 = request.get("text1", "")
        text2 = request.get("text2", "")
        method = request.get("method", "classical")
        
        # Simple word overlap similarity
        if not text1 or not text2:
            return JSONResponse(
                status_code=400,
                content={"error": "Both text1 and text2 are required"}
            )
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            similarity_score = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            similarity_score = intersection / union if union > 0 else 0.0
        
        return {
            "similarity_score": similarity_score,
            "method": "classical_minimal",
            "computation_time_ms": 1.0,  # Mock timing
            "details": {
                "text1_words": len(words1),
                "text2_words": len(words2),
                "common_words": len(words1.intersection(words2))
            }
        }
    
    @app.post("/v1/rerank")
    async def rerank(request: dict):
        """Simple reranking endpoint."""
        query = request.get("query", "")
        candidates = request.get("candidates", [])
        top_k = request.get("top_k", len(candidates))
        
        if not query or not candidates:
            return JSONResponse(
                status_code=400,
                content={"error": "Query and candidates are required"}
            )
        
        # Simple scoring based on word overlap
        query_words = set(query.lower().split())
        
        scores = []
        for i, candidate in enumerate(candidates):
            candidate_words = set(candidate.lower().split())
            if query_words and candidate_words:
                intersection = len(query_words.intersection(candidate_words))
                union = len(query_words.union(candidate_words))
                score = intersection / union if union > 0 else 0.0
            else:
                score = 0.0
            
            scores.append({
                "index": i,
                "text": candidate,
                "score": score
            })
        
        # Sort by score and return top_k
        ranked = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
        
        return {
            "rankings": ranked,
            "query": query,
            "method": "classical_minimal",
            "total_candidates": len(candidates),
            "returned_candidates": len(ranked)
        }
    
    @app.get("/v1/metrics")
    async def metrics():
        """Basic metrics endpoint."""
        return {
            "requests_total": 100,  # Mock metrics
            "response_time_avg": 10.5,
            "error_rate": 0.01,
            "quantum_enabled": False,
            "classical_enabled": True
        }
    
    return app

def main():
    """Main entry point."""
    print("🚀 Starting QuantumRerank Minimal Server")
    print("📁 Project root:", PROJECT_ROOT)
    print("🔧 Quantum features: DISABLED")
    print("⚡ Classical features: ENABLED")
    
    try:
        import uvicorn
        
        app = create_minimal_app()
        
        print("\n🌟 Server starting...")
        print("📍 URL: http://localhost:8000")
        print("📚 Endpoints:")
        print("   GET  /           - Service info")
        print("   GET  /health     - Health check")
        print("   POST /v1/similarity - Text similarity")
        print("   POST /v1/rerank  - Document reranking")
        print("   GET  /v1/metrics - Basic metrics")
        print("\n🔄 Use Ctrl+C to stop")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)