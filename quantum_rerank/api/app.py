"""
FastAPI application factory for QuantumRerank API.

This module provides the main FastAPI application with dependency injection,
middleware setup, and endpoint registration using the application factory pattern.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .dependencies import initialize_dependencies, cleanup_dependencies
# Temporarily disable middleware for initial testing
# from .middleware import TimingMiddleware, ErrorHandlingMiddleware, LoggingMiddleware
# from .endpoints import rerank, similarity, batch, health, metrics
from ..utils.logging_config import get_logger
from ..config.manager import ConfigManager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting QuantumRerank API service")
    
    try:
        # Initialize dependencies
        config_path = getattr(app.state, 'config_path', None)
        await initialize_dependencies(config_path)
        
        logger.info("QuantumRerank API service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start QuantumRerank API service: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down QuantumRerank API service")
        
        try:
            await cleanup_dependencies()
            logger.info("QuantumRerank API service shutdown completed")
        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")


def create_app(
    config_path: Optional[str] = None,
    enable_cors: bool = True,
    debug: bool = False,
    include_middleware: bool = True,
    custom_openapi: bool = True
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config_path: Path to configuration file
        enable_cors: Whether to enable CORS middleware
        debug: Enable debug mode
        include_middleware: Whether to include custom middleware
        custom_openapi: Whether to use custom OpenAPI schema
        
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="QuantumRerank API",
        description="Quantum-enhanced semantic reranking for information retrieval",
        version="1.0.0",
        docs_url="/docs" if debug else "/docs",
        redoc_url="/redoc" if debug else "/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Store config path for lifespan manager
    app.state.config_path = config_path
    
    # Configure CORS if enabled
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on environment
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
    
    # Temporarily disable middleware and endpoints
    # # Add custom middleware if enabled
    # if include_middleware:
    #     # Add in reverse order of desired execution
    #     app.add_middleware(LoggingMiddleware, log_requests=True, log_responses=True)
    #     app.add_middleware(ErrorHandlingMiddleware, include_traceback=debug)
    #     app.add_middleware(TimingMiddleware, include_in_headers=True)
    # 
    # # Include endpoint routers
    # app.include_router(rerank.router, prefix="/v1", tags=["reranking"])
    # app.include_router(similarity.router, prefix="/v1", tags=["similarity"])
    # app.include_router(batch.router, prefix="/v1", tags=["batch"])
    # app.include_router(health.router, prefix="/v1", tags=["health"])
    # app.include_router(metrics.router, prefix="/v1", tags=["metrics"])
    
    # Add root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "QuantumRerank API",
            "version": "1.0.0",
            "status": "operational",
            "documentation": "/docs",
            "health": "/health",
            "quantum_enabled": True,
            "features": ["quantum_similarity", "classical_fallback", "hybrid_methods"]
        }
    
    # Add basic health endpoint
    @app.get("/health", include_in_schema=False)
    async def health():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "service": "QuantumRerank API",
            "version": "1.0.0",
            "quantum_dependencies": {
                "qiskit": "available",
                "pennylane": "available", 
                "torch": "available",
                "faiss": "available"
            }
        }
    
    # Add custom exception handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """Custom 404 handler."""
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "NOT_FOUND",
                    "message": "Endpoint not found",
                    "details": {
                        "path": str(request.url.path),
                        "method": request.method
                    }
                }
            }
        )
    
    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc):
        """Custom 405 handler."""
        return JSONResponse(
            status_code=405,
            content={
                "error": {
                    "code": "METHOD_NOT_ALLOWED",
                    "message": "Method not allowed for this endpoint",
                    "details": {
                        "path": str(request.url.path),
                        "method": request.method
                    }
                }
            }
        )
    
    # Custom OpenAPI schema
    if custom_openapi:
        def custom_openapi_schema():
            if app.openapi_schema:
                return app.openapi_schema
            
            openapi_schema = get_openapi(
                title="QuantumRerank API",
                version="1.0.0",
                description="""
# QuantumRerank API

A quantum-enhanced semantic reranking API for information retrieval systems.

## Features

- **Quantum Similarity**: Leverage quantum computing for enhanced similarity computation
- **Classical Fallback**: Robust classical methods for reliability
- **Hybrid Approach**: Best of both quantum and classical methods
- **Batch Processing**: Efficient handling of multiple documents
- **Real-time Monitoring**: Performance metrics and health checks

## Similarity Methods

1. **Classical**: Traditional cosine similarity using embeddings
2. **Quantum**: Quantum fidelity-based similarity computation
3. **Hybrid**: Weighted combination of classical and quantum methods

## Rate Limits

- Default: 100 requests per minute per API key
- Batch endpoints: 50 requests per minute per API key

## Authentication

Include your API key in the `X-API-Key` header for authenticated requests.
                """,
                routes=app.routes,
            )
            
            # Add custom schema information
            openapi_schema["info"]["contact"] = {
                "name": "QuantumRerank Support",
                "email": "support@quantumrerank.ai"
            }
            
            openapi_schema["info"]["license"] = {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
            
            # Add security schemes
            openapi_schema["components"]["securitySchemes"] = {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
            
            # Add global security requirement
            openapi_schema["security"] = [{"ApiKeyAuth": []}]
            
            # Add example servers
            openapi_schema["servers"] = [
                {"url": "https://api.quantumrerank.ai", "description": "Production server"},
                {"url": "https://staging-api.quantumrerank.ai", "description": "Staging server"},
                {"url": "http://localhost:8000", "description": "Development server"}
            ]
            
            # Add custom tags
            openapi_schema["tags"] = [
                {
                    "name": "reranking",
                    "description": "Document reranking operations"
                },
                {
                    "name": "similarity", 
                    "description": "Direct similarity computation"
                },
                {
                    "name": "batch",
                    "description": "Batch processing operations"
                },
                {
                    "name": "health",
                    "description": "Service health and status"
                },
                {
                    "name": "metrics",
                    "description": "Performance metrics and analytics"
                }
            ]
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi_schema
    
    # Add startup event for additional initialization
    @app.on_event("startup")
    async def startup_event():
        """Additional startup configuration."""
        logger.info("FastAPI application startup event triggered")
        
        # Log application configuration
        logger.info(
            "Application configuration",
            extra={
                "debug": debug,
                "cors_enabled": enable_cors,
                "middleware_enabled": include_middleware,
                "config_path": config_path
            }
        )
    
    return app


def create_production_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create production-ready FastAPI application.
    
    Args:
        config_path: Path to production configuration file
        
    Returns:
        Production-configured FastAPI application
    """
    return create_app(
        config_path=config_path,
        enable_cors=False,  # Configure CORS explicitly for production
        debug=False,
        include_middleware=True,
        custom_openapi=True
    )


def create_development_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create development FastAPI application.
    
    Args:
        config_path: Path to development configuration file
        
    Returns:
        Development-configured FastAPI application
    """
    return create_app(
        config_path=config_path,
        enable_cors=True,
        debug=True,
        include_middleware=True,
        custom_openapi=True
    )


def create_test_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create test FastAPI application.
    
    Args:
        config_path: Path to test configuration file
        
    Returns:
        Test-configured FastAPI application
    """
    return create_app(
        config_path=config_path,
        enable_cors=True,
        debug=True,
        include_middleware=False,  # Disable middleware for cleaner testing
        custom_openapi=False
    )


# Create default application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run development server
    uvicorn.run(
        "quantum_rerank.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )