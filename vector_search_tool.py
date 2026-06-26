from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

async def create_vector_search_tool(document_fp: str, doc_name: str) -> QueryEngineTool:
    # Load the document
    document = SimpleDirectoryReader(input_files=[document_fp]).load_data()

    # Split the doc with a chunk_size of 1024 (a good default value)
    splitter = SentenceSplitter(chunk_size=1024)
    # Create nodes from the doc
    nodes = splitter.get_nodes_from_documents(document)

    # Setup the LLM model
    Settings.llm = Anthropic(model="claude-haiku-4-5")
    # Setup the embedding model
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text-v2-moe")

    # Create the vector search index
    vector_index = VectorStoreIndex(nodes)

    # Create the vector search query engine
    vector_query_engine = vector_index.as_query_engine()

    # Create the vector search tool
    vector_tool = QueryEngineTool.from_defaults(
        name=f"{doc_name}_vector_query_engine_tool",
        query_engine=vector_query_engine,
        description=(
            f"Useful for retrieve specific context from the {doc_name}."
        ),
    )

    return vector_tool