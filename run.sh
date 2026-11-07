#!/bin/bash
# CivilityLens One-Click Startup Script

# Navigate to the project directory
cd "$(dirname "$0")"

echo "=========================================================="
echo "🛡️ Starting CivilityLens Cyberbullying Detection App..."
echo "=========================================================="

# Check if streamlit is installed and run
if command -v streamlit &> /dev/null; then
    streamlit run app.py
elif python3 -m streamlit --version &> /dev/null; then
    python3 -m streamlit run app.py
else
    echo "Streamlit not found in active PATH, launching standalone web interface..."
    python3 web_demo.py
fi
