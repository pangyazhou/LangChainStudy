"""
Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)
一次可以掉一个或多个函数
"""
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI


# 构建工具代理对象
def tool_agent_build() -> AgentExecutor:
    tools = [TavilySearchResults(max_results=1)]
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-tools-agent")
    # Choose the LLM that will drive the agent
    # Only certain models support this
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


# 执行工具代理
def tool_agent_invoke():
    agent_executor = tool_agent_build()
    result = agent_executor.invoke({"input": "what is LangChain?"})
    print(result)


if __name__ == "__main__":
    tool_agent_invoke()
    pass