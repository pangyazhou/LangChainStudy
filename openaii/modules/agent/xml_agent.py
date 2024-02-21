"""
如何使用在提示词时使用 XML 的代理
"""
from langchain import hub
from langchain.agents import AgentExecutor, create_xml_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI


# 构建xml代理对象
def xml_agent_build() -> AgentExecutor:
    tools = [TavilySearchResults(max_results=1)]
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/xml-agent-convo")
    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

    # Construct the XML agent
    agent = create_xml_agent(llm, tools, prompt)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


def xml_agent_invoke():
    agent_executor = xml_agent_build()
    result = agent_executor.invoke({"input": "what is LangChain?"})
    print(result)


if __name__ == "__main__":
    xml_agent_invoke()
    pass