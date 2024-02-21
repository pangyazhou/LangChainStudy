"""
Tavily 的搜索 API是专为人工智能代理 (LLM) 构建的搜索引擎，可快速提供实时、准确和真实的结果。
"""

import os
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever


def simple():
    api_key = "tvly-xxx"
    retriever = TavilySearchAPIRetriever(api_key=api_key, k=3)
    document = retriever.invoke("what is the weather in SF?")
    print(document)


def test():
    api_key = os.getenv("TAVILY_API_KEY")
    print(api_key)


# 程序入口
if __name__ == '__main__':
    simple()
    # test()
