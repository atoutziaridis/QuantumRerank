#!/usr/bin/env python3
"""
Validate API structure and code quality without running imports.

This script validates the API implementation by checking file structure,
syntax, and basic code patterns without requiring dependencies.
"""

import os
import ast
import sys
from pathlib import Path

def validate_file_structure():
    """Validate that all required API files exist."""
    print("Validating file structure...")
    
    api_dir = Path("quantum_rerank/api")
    required_files = [
        "models.py",
        "dependencies.py", 
        "app.py",
        "__init__.py"
    ]
    
    required_dirs = [
        "endpoints",
        "middleware",
        "services"
    ]
    
    endpoint_files = [
        "endpoints/__init__.py",
        "endpoints/rerank.py",
        "endpoints/similarity.py",
        "endpoints/batch.py",
        "endpoints/health.py",
        "endpoints/metrics.py"
    ]
    
    middleware_files = [
        "middleware/__init__.py",
        "middleware/timing.py",
        "middleware/error_handling.py",
        "middleware/logging.py"
    ]
    
    service_files = [
        "services/__init__.py",
        "services/similarity_service.py",
        "services/health_service.py"
    ]
    
    all_files = required_files + endpoint_files + middleware_files + service_files
    
    missing_files = []
    for file_path in all_files:
        full_path = api_dir / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    for dir_name in required_dirs:
        dir_path = api_dir / dir_name
        if not dir_path.exists():
            missing_files.append(f"{dir_path}/ (directory)")
    
    if missing_files:
        print(f"❌ Missing files/directories:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print(f"✅ All {len(all_files)} required files exist")
        return True

def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def validate_code_syntax():
    """Validate Python syntax of all API files."""
    print("Validating Python syntax...")
    
    api_dir = Path("quantum_rerank/api")
    python_files = []
    
    # Collect all Python files
    for root, dirs, files in os.walk(api_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    syntax_errors = []
    
    for file_path in python_files:
        is_valid, error = validate_python_syntax(file_path)
        if not is_valid:
            syntax_errors.append(f"{file_path}: {error}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"   - {error}")
        return False
    else:
        print(f"✅ All {len(python_files)} Python files have valid syntax")
        return True

def validate_fastapi_patterns():
    """Validate FastAPI patterns in endpoint files."""
    print("Validating FastAPI patterns...")
    
    endpoint_dir = Path("quantum_rerank/api/endpoints")
    endpoint_files = [
        "rerank.py",
        "similarity.py", 
        "batch.py",
        "health.py",
        "metrics.py"
    ]
    
    pattern_checks = []
    
    for file_name in endpoint_files:
        file_path = endpoint_dir / file_name
        if not file_path.exists():
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required FastAPI patterns
            checks = {
                "router_creation": "router = APIRouter()" in content,
                "router_decorator": "@router." in content,  # Any router decorator
                "async_function": "async def " in content,
                "pydantic_models": "response_model=" in content or "summary=" in content,
                "dependency_injection": "Depends(" in content
            }
            
            failed_checks = [name for name, passed in checks.items() if not passed]
            
            if failed_checks:
                pattern_checks.append(f"{file_name}: Missing patterns: {failed_checks}")
            
        except Exception as e:
            pattern_checks.append(f"{file_name}: Error reading file: {e}")
    
    if pattern_checks:
        print(f"❌ FastAPI pattern issues:")
        for issue in pattern_checks:
            print(f"   - {issue}")
        return False
    else:
        print(f"✅ All endpoint files follow FastAPI patterns")
        return True

def validate_pydantic_models():
    """Validate Pydantic model definitions."""
    print("Validating Pydantic models...")
    
    models_file = Path("quantum_rerank/api/models.py")
    
    if not models_file.exists():
        print("❌ models.py file not found")
        return False
    
    try:
        with open(models_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required model classes
        required_models = [
            "RerankRequest",
            "RerankResponse",
            "SimilarityRequest", 
            "SimilarityResponse",
            "BatchSimilarityRequest",
            "BatchSimilarityResponse",
            "HealthCheckResponse",
            "ErrorResponse"
        ]
        
        missing_models = []
        for model in required_models:
            if f"class {model}" not in content:
                missing_models.append(model)
        
        # Check for Pydantic patterns
        pydantic_patterns = [
            "from pydantic import BaseModel",
            "BaseModel",
            "Field("
        ]
        
        missing_patterns = []
        for pattern in pydantic_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        issues = []
        if missing_models:
            issues.append(f"Missing models: {missing_models}")
        if missing_patterns:
            issues.append(f"Missing Pydantic patterns: {missing_patterns}")
        
        if issues:
            print(f"❌ Pydantic model issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print(f"✅ All required Pydantic models found")
            return True
            
    except Exception as e:
        print(f"❌ Error validating models: {e}")
        return False

def validate_app_structure():
    """Validate app.py structure."""
    print("Validating app.py structure...")
    
    app_file = Path("quantum_rerank/api/app.py")
    
    if not app_file.exists():
        print("❌ app.py file not found")
        return False
    
    try:
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_patterns = [
            "from fastapi import FastAPI",
            "def create_app(",
            "lifespan",
            "app = FastAPI(",
            "app.include_router("
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"❌ App structure issues:")
            for pattern in missing_patterns:
                print(f"   - Missing: {pattern}")
            return False
        else:
            print(f"✅ app.py has correct structure")
            return True
            
    except Exception as e:
        print(f"❌ Error validating app.py: {e}")
        return False

def count_code_metrics():
    """Count basic code metrics."""
    print("Counting code metrics...")
    
    api_dir = Path("quantum_rerank/api")
    
    if not api_dir.exists():
        print("❌ API directory not found")
        return False
    
    total_files = 0
    total_lines = 0
    total_functions = 0
    total_classes = 0
    
    for root, dirs, files in os.walk(api_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                total_files += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    total_lines += len(lines)
                    
                    # Count functions and classes
                    total_functions += content.count('def ')
                    total_classes += content.count('class ')
                    
                except Exception:
                    pass
    
    print(f"✅ Code metrics:")
    print(f"   - Files: {total_files}")
    print(f"   - Lines of code: {total_lines}")
    print(f"   - Functions: {total_functions}")
    print(f"   - Classes: {total_classes}")
    
    return True

def main():
    """Run all validation tests."""
    print("🔍 Validating API Implementation Structure")
    print("=" * 50)
    
    tests = [
        validate_file_structure,
        validate_code_syntax,
        validate_pydantic_models,
        validate_fastapi_patterns,
        validate_app_structure,
        count_code_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        else:
            print()
    
    print()
    print("=" * 50)
    print(f"🔍 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 API implementation structure is valid!")
        return 0
    else:
        print("⚠️  Some validation checks failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)