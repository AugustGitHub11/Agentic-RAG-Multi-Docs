# Agentic RAG Multi-Docs

An agentic Retrieval-Augmented Generation (RAG) system that answers questions over a collection of local documents using [LlamaIndex](https://www.llamaindex.ai/) and Claude, with automated evaluation using [Giskard](https://www.giskard.ai/).

## How It Works

For each document in `./datasets`, the system builds two tools:
- **Summary tool** — answers high-level questions using tree summarization
- **Vector search tool** — retrieves specific passages using semantic search

All tools are indexed with `ObjectIndex`, so when a query comes in, only the most relevant tools are retrieved and passed to the agent — keeping the context window efficient regardless of how many documents you have.

## Tech Stack

- **LLM**: Claude Haiku (`claude-haiku-4-5`) via Anthropic
- **Embeddings**: `nomic-embed-text-v2-moe` via Ollama
- **Framework**: LlamaIndex (`FunctionAgent`, `ObjectIndex`)
- **Evaluation**: Giskard RAG evaluation framework

## Setup

```bash
pip install llama-index llama-index-llms-anthropic llama-index-embeddings-ollama python-dotenv
```

For evaluation, install the additional dependencies:
```bash
pip install "giskard[llm]" litellm ollama
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

## Evaluation

The system is evaluated using Giskard's RAG evaluation framework (`evaluation.ipynb`). Giskard automatically generates a test set from the knowledge base and scores the agent across five components:

| Component | Description |
|---|---|
| **Generator** | How accurately the LLM produces answers from retrieved context |
| **Retriever** | How well relevant documents are fetched for a given query |
| **Rewriter** | How well the agent handles multi-turn and complex questions |
| **Router** | Whether the agent correctly identifies query intent |
| **Knowledge Base** | Consistency of coverage across topics in the document set |

The evaluation uses Claude (`anthropic/claude-haiku-4-5`) via LiteLLM as the judge LLM and `nomic-embed-text-v2-moe` via Ollama for embeddings. Generated test sets are saved to `./testsets/` for reuse.

### Latest Results

| Component | Score |
|---|---|
| Generator | 69% |
| Retriever | 65% |
| Rewriter | 64% |
| Router | 100% |
| Knowledge Base | 61% |
| **Overall Correctness** | **67%** |
