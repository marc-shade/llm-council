#!/bin/bash
# Start the LLM Council (CLI Edition)
# Uses Claude Code, Codex CLI, and Gemini CLI as council members

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "LLM Council - CLI Edition"
echo "Council Members: Claude Code, Codex CLI, Gemini CLI"
echo "=================================================="

# Check CLI tools are available
echo "Checking CLI tools..."
for cmd in claude codex gemini; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>/dev/null | head -1 || echo "installed")
        echo "  ✓ $cmd: $version"
    else
        echo "  ✗ $cmd: NOT FOUND"
        echo "    Please install $cmd CLI"
    fi
done

echo ""

# Set environment for CLI mode
export PROVIDER_MODE=cli

# Check if backend port is in use
if lsof -i :8001 > /dev/null 2>&1; then
    echo "Warning: Port 8001 already in use"
    echo "Kill existing process? (y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        kill $(lsof -t -i :8001) 2>/dev/null || true
        sleep 1
    fi
fi

# Start backend
echo "Starting backend on port 8001..."
cd "$SCRIPT_DIR"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!
sleep 2

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Failed to start backend!"
    exit 1
fi
echo "Backend running (PID: $BACKEND_PID)"

# Start frontend
echo ""
echo "Starting frontend on port 5173..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
sleep 3

echo ""
echo "=================================================="
echo "LLM Council is running!"
echo "=================================================="
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for either to exit
wait $BACKEND_PID $FRONTEND_PID
