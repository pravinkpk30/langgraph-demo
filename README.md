# LangGraph Course

**Video:** https://www.youtube.com/watch?v=jGg_1h0qzaM

Repository for all of the code written for the FreeCodeCamp LangGraph Course, including solutions for all exercises. This repo provides practical examples of using [LangGraph](https://github.com/langchain-ai/langgraph) for building agent-based applications through Python scripts and interactive Jupyter notebooks.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started (zsh/Mac)](#getting-started-zshmac)
  - [Using pyenv and uv](#using-pyenv-and-uv)
- [Usage](#usage)
- [Exercises](#exercises)
- [Requirements](#requirements)

---

## Overview

LangGraph is a Python framework for designing and managing the flow of tasks in your application using graph structures. This course demonstrates LangGraph concepts through step-by-step exercises, agent implementations, and Jupyter notebooks.

---

## Repository Structure

```
LangGraph-Course/
├── Agents/            # Python agents for various tasks (e.g., RAG_Agent, Drafter)
├── Exercises/         # Jupyter notebooks with exercise solutions
├── Graphs/            # Jupyter notebooks illustrating LangGraph concepts
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

**Notable Directories:**
- **Agents/**: Python scripts for agents such as Retrieval-Augmented Generation (RAG) and document drafting.
- **Exercises/**: Jupyter notebooks for each exercise (e.g. `Exercise_Graph1.ipynb`).
- **Graphs/**: Notebooks demonstrating LangGraph patterns (e.g., Hello World, Looping).

---

## Getting Started (zsh/Mac)

### Using pyenv and uv

#### 1. Clone the Repository

```zsh
git clone https://github.com/rdtiv/LangGraph-Course.git
cd LangGraph-Course
```

#### 2. Install pyenv (if not already installed)

```zsh
brew update
brew install pyenv
```

Add the following to your `~/.zshrc` if it's not already there:

```zsh
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```
Restart your terminal or source your `~/.zshrc`:

```zsh
source ~/.zshrc
```

#### 3. Install Python Version

```zsh
pyenv install 3.12.6
pyenv local 3.12.6
```

#### 4. Install uv

```zsh
pipx install uv           # Recommended, or:
pip install --user uv
```

If you don't have pipx, install it with:

```zsh
brew install pipx
pipx ensurepath
```

#### 5. Set Up Virtual Environment with uv

```zsh
uv venv .venv
source .venv/bin/activate
```

#### 6. Install Dependencies

```zsh
uv pip install -r requirements.txt
```

#### 7. (Optional) Set up Environment Variables

If you need API keys (such as for OpenAI), create a `.env` file in the root directory:

```zsh
echo "OPENAI_API_KEY=your_openai_key" > .env
# Add other variables as needed
```

#### 8. Start JupyterLab

```zsh
uv pip install jupyterlab  # Only if not already installed
jupyter lab
```

---

## Usage

- Open and run Jupyter notebooks in `Graphs/` and `Exercises/` for hands-on practice and exploration.
- Run agent scripts in `Agents/` for more advanced experiments.
- All code is designed to work in a local, isolated Python environment managed by pyenv and uv.

---

## Exercises

- Explore the `Exercises/` directory for self-contained solutions to LangGraph problems.
- Example notebooks:
  - `Exercise_Graph1.ipynb`: Agent state and basic graph usage.
  - `Exercise_Graph2.ipynb`: User input and graph visualization.
  - `Exercise_Graph3.ipynb`: Personalization and skills-based responses.
  - `Exercise_Graph4.ipynb`, `Exercise_Graph5.ipynb`: Advanced graph operations.

---

## Requirements

Core dependencies (see `requirements.txt` for full list):

- langgraph
- langchain
- ipython
- langchain_openai
- langchain_community
- dotenv
- typing
- chromadb
- langchain_chroma

Install all dependencies with:

```zsh
uv pip install -r requirements.txt
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=iamvaibhavmehra/LangGraph-Course-freeCodeCamp&type=Date)](https://www.star-history.com/#iamvaibhavmehra/LangGraph-Course-freeCodeCamp&Date)

# LangGraph Demo Project

A collection of LangGraph agents demonstrating various AI agent patterns and capabilities using Google's Gemini model.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [UV](https://github.com/astral-sh/uv) (Recommended for faster package management)
- Google API Key (for Gemini)

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd langgraph-demo
   ```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies using UV (recommended):
```bash
uv pip install -r requirements.txt
```
Or using pip:
```bash
pip install -r requirements.txt
```

4. Create a .env file in the root directory and add your Google API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

### 🤖 Available Agents

1. Simple Agent Bot
A basic agent that demonstrates a simple conversation flow.

Run with UV:

```bash
uv run agents/1.SimpleAgentBot.py
```
Run with Python:

```bash
python agents/1.SimpleAgentBot.py
```
2. Memory Agent Bot
An agent with conversation memory that persists the chat history.

Run with UV:

```bash
uv run agents/2.MemoryAgentBot.py
```
Run with Python:

```bash
python agents/2.MemoryAgentBot.py
```
3. ReAct Agent
An implementation of the ReAct (Reasoning and Acting) pattern with tool usage.

Run with UV:

```bash
uv run agents/3.ReActAgent.py
```
Run with Python:

```bash
python agents/3.ReActAgent.py
```
4. Drafter Agent
A document editing assistant that can create and modify text documents.

Run with UV:

```bash
uv run agents/4.DrafterAgent.py
```
Run with Python:

```bash
python agents/4.DrafterAgent.py
```
5. RAG Agent
A Retrieval-Augmented Generation agent that can answer questions about a PDF document.

Run with UV:

```bash
uv run agents/5.RAGAgent.py
```
Run with Python:

```bash
python agents/5.RAGAgent.py
```
Note for RAG Agent: Place your PDF file in the agents/ directory and update the pdf_path in the script if needed.

### 📁 Project Structure

```
langgraph-demo/
├── agents/                    # Agent implementations
│   ├── 1.SimpleAgentBot.py    # Basic agent
│   ├── 2.MemoryAgentBot.py    # Agent with memory
│   ├── 3.ReActAgent.py        # ReAct pattern agent
│   ├── 4.DrafterAgent.py      # Document editing agent
│   └── 5.RAGAgent.py          # RAG implementation
├── data/                      # Data files
├── graphs/                    # Jupyter notebook examples
├── exercise/                  # Practice exercises
├── .env.example               # Example environment variables
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

### 🛠️ Dependencies

1. langgraph
2. langchain
3. langchain-google-genai
4. google-generativeai
5. chromadb
6. pypdf
7. python-dotenv

### 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
