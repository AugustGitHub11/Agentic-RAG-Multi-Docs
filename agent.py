import asyncio
import sys
import dotenv
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionAgent
from llama_index.llms.anthropic import Anthropic

from summary_tool import create_summary_tool
from vector_search_tool import create_vector_search_tool

dotenv.load_dotenv()

SYSTEM_PROMPT = """\
You are an AI agent programmed to respond to questions purely based on a list of documents. 
Always utilize tools chosen by the tool retriever to generate answers, 
ensuring that responses are based directly on the provided materials rather than on any pre-existing knowledge.
All your responses should be formatted in markdown text.
"""


async def create_tools_for_papers(papers: list[str]) -> list:
    all_tools = []

    for paper in papers:
        print(f"Creating tools for {paper}")

        path = Path(paper)
        summary_tool = await create_summary_tool(document_fp=path, doc_name=path.stem)
        vector_tool = await create_vector_search_tool(document_fp=path, doc_name=path.stem)

        all_tools.extend([summary_tool, vector_tool])

    return all_tools


def create_tool_retriever(all_tools: list, similarity_top_k: int = 2):
    obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
    return obj_index.as_retriever(similarity_top_k=similarity_top_k)


def create_agent(obj_retriever, llm, verbose: bool = False) -> FunctionAgent:
    return FunctionAgent(
        system_prompt=SYSTEM_PROMPT,
        tool_retriever=obj_retriever,
        llm=llm,
        verbose=verbose,
    )


async def run_query(agent: FunctionAgent, user_msg: str) -> str:
    response = await agent.run(user_msg=user_msg)
    return str(response)


async def main():
    papers = [str(p) for p in Path("./datasets").iterdir() if p.is_file()]

    all_tools = await create_tools_for_papers(papers)
    obj_retriever = create_tool_retriever(all_tools, similarity_top_k=2)

    llm = Anthropic(model="claude-haiku-4-5")
    agent = create_agent(obj_retriever, llm, verbose=False)

    query = " ".join(sys.argv[1:])

    print(f"\nGenerating result for query: {query}")
    print("-" * 60)
    result = await run_query(agent, query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
