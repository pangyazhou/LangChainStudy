from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_openai import ChatOpenAI


def self_ask_agent_build() -> AgentExecutor:
    tools = [TavilyAnswer(max_results=1, name="Intermediate Answer")]
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/self-ask-with-search")
    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

    # Construct the Self Ask With Search Agent
    agent = create_self_ask_with_search_agent(llm, tools, prompt)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor


# 调用结果报错
def self_ask_agent_invoke():
    agent_executor = self_ask_agent_build()
    result = agent_executor.invoke(
        {"input": "What is the hometown of the reigning men's U.S. Open champion?"}
    )
    print(result)


if __name__ == "__main__":
    self_ask_agent_invoke()