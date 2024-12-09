# Data Analysis AI Agent

An intelligent data analysis agent built with Pydantic that helps analyze datasets, create visualizations, and answer questions about your data using Ollama for AI capabilities.

## Features

- Load data from CSV and Excel files
- Automatic column analysis and data profiling
- Generate basic statistics and insights
- Create various types of visualizations (histograms, scatter plots, bar charts, box plots)
- Ask questions about your data using natural language (powered by Ollama)
- Generate comprehensive data analysis reports with AI insights

## Prerequisites

- Python 3.7+
- Ollama installed and running locally (default URL: http://localhost:11434)
- Required packages listed in requirements.txt

## Setup

1. Install Ollama by following the instructions at: https://github.com/ollama/ollama
2. Start the Ollama service
3. Pull the default model (llama2):
   ```bash
   ollama pull llama2
   ```
4. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the example script:
```bash
python example.py
```

The script will prompt you to:
1. Provide the path to your data file (CSV or Excel)
2. Choose from various analysis options:
   - Get basic statistics
   - Create visualizations
   - Ask questions about your data
   - Generate a comprehensive report

## Example Usage in Code

```python
from data_analysis_agent import DataAnalysisAgent

# Initialize the agent
agent = DataAnalysisAgent()

# Optionally configure Ollama settings
agent.model_name = "llama2"  # or any other model you have pulled
agent.ollama_url = "http://localhost:11434/api/generate"  # default URL

# Load data
agent.load_data("your_data.csv")

# Get basic statistics
stats = agent.get_basic_stats()

# Create a visualization
agent.create_visualization("histogram", "column_name")

# Ask a question about the data
answer = agent.analyze_question("What is the correlation between column A and column B?")

# Generate a report
report = agent.generate_report()
```

## Available Ollama Models

You can use any model that you have pulled with Ollama. Some recommended models for data analysis:
- llama2 (default)
- mistral
- codellama
- vicuna

To use a different model, set it when initializing the agent or change it later:
```python
agent = DataAnalysisAgent(model_name="mistral")
# or
agent.model_name = "mistral"
```

## Note

Make sure Ollama is running before using the agent. You can check if it's running by visiting http://localhost:11434 in your browser. 