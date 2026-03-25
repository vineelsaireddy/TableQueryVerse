#!/bin/bash

# Set up environment variables
OLLAMA_HOST="127.0.0.1:11434"  # Local host with specific port
OLLAMA_MODELS_PATH="$HOME/.ollama/models"  # Store models in user's home directory
OLLAMA_ORIGINS="*"  # Allow all origins for API requests

# Ensure we're in the right virtual environment
if [[ -z "${VIRTUAL_ENV}" ]] || [[ "${VIRTUAL_ENV}" != *"ollama-codellama"* ]]; then
    echo "Please activate the ollama-codellama environment first!"
    echo "Run: source ~/ollama-codellama/bin/activate"
    exit 1
fi

# Create necessary directories
mkdir -p "$HOME/.ollama/models"
mkdir -p "$HOME/bin"

# Extract the Ollama binary
tar xzf ollama-linux-amd64.tgz
mv ollama "$HOME/bin/"

# Make the binary executable
chmod +x "$HOME/bin/ollama"

# Add the binary location to PATH if not already there
if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
    echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
    # Also add to the current session
    export PATH="$HOME/bin:$PATH"
fi

# Export environment variables
{
    echo "export OLLAMA_HOST=$OLLAMA_HOST"
    echo "export OLLAMA_MODELS_PATH=$OLLAMA_MODELS_PATH"
    echo "export OLLAMA_ORIGINS=$OLLAMA_ORIGINS"
} >> "$HOME/.bashrc"

# Also export for current session
export OLLAMA_HOST=$OLLAMA_HOST
export OLLAMA_MODELS_PATH=$OLLAMA_MODELS_PATH
export OLLAMA_ORIGINS=$OLLAMA_ORIGINS

# Install required Python packages
pip install pandas requests datasets

# Try to install databench_eval
pip install databench_eval || echo "Warning: Could not install databench_eval. You may need to install it manually."

# Print installation complete message
echo "Ollama has been installed to $HOME/bin/ollama"
echo "Environment variables have been set in ~/.bashrc"
echo "Python dependencies have been installed in the ollama-codellama environment"
echo "To start Ollama, run: ollama serve"

# Additional instructions
echo "
Additional steps needed:
1. Start Ollama server: ollama serve
2. In a new terminal, activate the environment and pull the codellama model:
   source ~/ollama-codellama/bin/activate
   ollama pull codellama
3. Make sure databench_eval is installed correctly

Note: You may need to request appropriate SLURM resources to run the model.
Example SLURM script:

#!/bin/bash
#SBATCH --job-name=ollama_run
#SBATCH --output=ollama_%j.out
#SBATCH --error=ollama_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/ollama-codellama/bin/activate

ollama serve &
sleep 10  # wait for server to start
python your_script.py
"
