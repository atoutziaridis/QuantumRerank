#!/bin/bash
# Build script for QuantumRerank Docker image

set -e

echo "🔨 Building QuantumRerank Docker image..."

# Get version from git or use default
VERSION=$(git describe --tags --always 2>/dev/null || echo "latest")

# Build the Docker image
echo "Building quantumrerank/server:latest..."
docker build -t quantumrerank/server:latest .

# Tag with version
echo "Tagging as quantumrerank/server:${VERSION}..."
docker tag quantumrerank/server:latest quantumrerank/server:${VERSION}

echo ""
echo "✅ Build completed successfully!"
echo "   Images created:"
echo "   - quantumrerank/server:latest"
echo "   - quantumrerank/server:${VERSION}"
echo ""
echo "🚀 To run: docker run -p 8000:8000 quantumrerank/server:latest"
echo "📋 To deploy: ./quick-start.sh"