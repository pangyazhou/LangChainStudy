"""
通过model来判断使用那些工具提供检索结果
quickstart.py的简略版
functions agent已被OpenAI废弃
"""
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


# 构建代理执行器
def function_agent_build() -> AgentExecutor:
    tools = [TavilySearchResults(max_results=1)]
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

    # Construct the OpenAI Functions agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


def function_agent_invoke():
    agent_executor = function_agent_build()
    result = agent_executor.invoke({"input": "what is LangChain?"})
    print(result)


def function_agent_with_history_invoke():
    agent_executor = function_agent_build()
    result = agent_executor.invoke(
        {
            "input": "what's my name?",
            "chat_history": [
                HumanMessage(content="hi! my name is bob"),
                AIMessage(content="Hello Bob! How can I assist you today?"),
            ],
        }
    )
    print(result)
    pass


if __name__ == "__main__":
    function_agent_invoke()
    pass