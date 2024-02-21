"""
大型语言模型（LLM）是LangChain的核心组件。
LangChain不为自己的LLM提供服务，而是提供一个标准接口来与许多不同的LLM进行交互。
具体来说，该接口是以字符串作为输入并返回字符串的接口。

1. How to write a custom LLM class
2. How to cache LLM responses
3. How to stream responses from an LLM
4. [How to track token usage in an LLM call)(./token_usage_tracking)
"""
import asyncio

from langchain_openai import OpenAI


llm = OpenAI()


def llm_invoke():
    # 正常调用
    result = llm.invoke(
        "What are some theories about the relationship between unemployment and inflation?"
    )
    print(result)
    # 流式响应
    for chunk in llm.stream(
            "What are some theories about the relationship between unemployment and inflation?"
    ):
        print(chunk, end="", flush=True)
    # 批量调用
    result = llm.batch(
        [
            "What are some theories about the relationship between unemployment and inflation?"
        ]
    )
    print(result)
    # 异步调用
    result = asyncio.run(llm.ainvoke(
        "What are some theories about the relationship between unemployment and inflation?"
    ))
    print(result)
    


# 程序入口
if __name__ == "__main__":
    llm_invoke()
    pass