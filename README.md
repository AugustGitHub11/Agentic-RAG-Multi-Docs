# Agentic RAG Multi-Docs

An agentic Retrieval-Augmented Generation (RAG) system that answers questions over a collection of local documents using [LlamaIndex](https://www.llamaindex.ai/) and Claude.

## How It Works

For each document in `./datasets`, the system builds two tools:
- **Summary tool** — answers high-level questions using tree summarization
- **Vector search tool** — retrieves specific passages using semantic search

All tools are indexed with `ObjectIndex`, so when a query comes in, only the most relevant tools are retrieved and passed to the agent — keeping the context window efficient regardless of how many documents you have.

## Tech Stack

- **LLM**: Claude Haiku (`claude-haiku-4-5`) via Anthropic
- **Embeddings**: `nomic-embed-text-v2-moe` via Ollama
- **Framework**: LlamaIndex (`FunctionAgent`, `ObjectIndex`)

## Setup

```bash
pip install llama-index llama-index-llms-anthropic llama-index-embeddings-ollama python-dotenv
```

Create a `.env` file with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_key_here
```

Place your PDF documents in the `./datasets` folder.

## Usage

```bash
python agent.py What is LoRA and why is it used?
```

The agent will automatically pick the most relevant tools for your query and return a markdown-formatted answer.
