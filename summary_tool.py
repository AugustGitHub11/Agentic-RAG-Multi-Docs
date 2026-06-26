from llama_index.core.tools import QueryEngineTool
from llama_index.core import SummaryIndex
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

async def create_summary_tool(document_fp: str, doc_name: str) -> QueryEngineTool:
    # Load the document
    document = SimpleDirectoryReader(input_files=[document_fp]).load_data()

    # Split the doc with a chunk_size of 1024 (a good default value)
    splitter = SentenceSplitter(chunk_size=1024)
    # Create nodes from the doc
    nodes = splitter.get_nodes_from_documents(document)

    # Setup the LLM model
    Settings.llm = Anthropic(model="claude-haiku-4-5")

    # Create the summary index
    summary_index = SummaryIndex(nodes)

    # Create the summary query engine
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    # Create the summary tool
    summary_tool = QueryEngineTool.from_defaults(
        name=f"{doc_name}_summary_query_engine_tool",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarize on the {doc_name}."
        ),
    )

    return summary_tool