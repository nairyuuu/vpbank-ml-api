#!/bin/bash

# VPBank ML API Setup Script
echo "🚀 VPBank ML API Setup"
echo "======================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Node.js is installed
check_nodejs() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js is installed: $NODE_VERSION"
        
        # Check if version is 18 or higher
        NODE_MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
        if [ "$NODE_MAJOR_VERSION" -ge 18 ]; then
            print_status "Node.js version is compatible (18+)"
        else
            print_warning "Node.js version should be 18 or higher. Current: $NODE_VERSION"
        fi
    else
        print_error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org"
        exit 1
    fi
}

# Check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_status "Docker is installed: $DOCKER_VERSION"
    else
        print_warning "Docker is not installed. Install Docker for containerization support."
    fi
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version)
        print_status "Docker Compose is installed: $COMPOSE_VERSION"
    else
        print_warning "Docker Compose is not installed. Install for full containerization support."
    fi
}

# Install npm dependencies
install_dependencies() {
    print_info "Installing npm dependencies..."
    
    if npm install; then
        print_status "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
}

# Setup environment file
setup_environment() {
    if [ ! -f ".env" ]; then
        print_info "Setting up environment configuration..."
        cp .env.example .env
        print_status "Environment file created (.env)"
        print_warning "Please edit .env file with your configuration before starting the application"
    else
        print_info ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    mkdir -p models
    mkdir -p logs
    
    print_status "Directories created"
}

# Check for ONNX models
check_models() {
    print_info "Checking for ONNX models..."
    
    MODELS_FOUND=0
    
    for model in "qr_model.onnx" "ibft_model.onnx" "topup_model.onnx"; do
        if [ -f "models/$model" ]; then
            print_status "Found: models/$model"
            MODELS_FOUND=$((MODELS_FOUND + 1))
        else
            print_warning "Missing: models/$model"
        fi
    done
    
    if [ $MODELS_FOUND -eq 0 ]; then
        print_warning "No ONNX models found in models/ directory"
        print_info "The system will use mock predictions until real models are added"
        print_info "See models/README.md for conversion instructions"
    else
        print_status "$MODELS_FOUND/3 models found"
    fi
}

# Test the application
test_application() {
    print_info "Testing application setup..."
    
    # Start the application in background
    npm start &
    APP_PID=$!
    
    # Wait a moment for startup
    sleep 5
    
    # Test health endpoint
    if curl -s http://localhost:3000/health/live > /dev/null; then
        print_status "Application started successfully"
        print_status "Health check passed"
    else
        print_error "Application health check failed"
    fi
    
    # Stop the application
    kill $APP_PID 2>/dev/null
    wait $APP_PID 2>/dev/null
}

# Show next steps
show_next_steps() {
    echo ""
    echo "🎉 Setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Add ONNX model files to models/ directory (optional)"
    echo "3. Start the application:"
    echo ""
    echo "   Development mode:"
    echo "   npm run dev"
    echo ""
    echo "   Production mode with Docker:"
    echo "   docker-compose up -d"
    echo ""
    echo "4. Test the API:"
    echo "   node test/api-test.js"
    echo ""
    echo "📚 Documentation: See README.md for detailed information"
}

# Main setup process
main() {
    echo ""
    print_info "Starting setup process..."
    echo ""
    
    check_nodejs
    check_docker
    create_directories
    setup_environment
    install_dependencies
    check_models
    
    print_status "Setup process completed"
    show_next_steps
}

# Check if help was requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "VPBank ML API Setup Script"
    echo ""
    echo "Usage: ./setup.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --test         Run application test after setup"
    echo ""
    echo "This script will:"
    echo "- Check system requirements (Node.js, Docker)"
    echo "- Install npm dependencies"
    echo "- Create necessary directories"
    echo "- Setup environment configuration"
    echo "- Check for ONNX models"
    exit 0
fi

# Run main setup
main

# Run test if requested
if [ "$1" = "--test" ]; then
    echo ""
    print_info "Running application test..."
    test_application
fi
