"""
自定义代理
"""
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
        format_to_openai_tool_messages,
    )
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# python函数
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


def custom_agent_build() -> AgentExecutor:
    # 代理工具集
    tools = [get_word_length, TavilySearchResults(max_results=1)]
    # 提示词模型
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, but don't know current events",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # 绑定工具集
    llm_with_tools = llm.bind_tools(tools)
    # 创建代理
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    # 代理执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


def custom_agent_invoke():
    agent_executor = custom_agent_build()
    msg = "大不列颠有多少字符"
    msg = "大不列颠面积有多大"
    list(agent_executor.stream({"input": msg}))
    # for chunk in agent_executor.stream({"input": "How many letters in the word eudca"}):
    #     print(chunk, end="", flush=True)


if __name__ == "__main__":
    custom_agent_invoke()
    pass
