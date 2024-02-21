"""
构建代理时要做的第一件事就是决定它应该有权访问哪些工具
1.我们刚刚创建的检索器。这将让它轻松回答有关 LangSmith 的问题
2.一个搜索工具。这将使它能够轻松回答需要最新信息的问题
"""
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="sk-xxx")


def simple():
    # 加载检索数据
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    # 嵌入模型
    embeddings = OpenAIEmbeddings(openai_api_key="sk-xxx")
    # 创建索引
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    # 创建检索器
    retriever = vector.as_retriever()

    # 为检索器设置一个工具
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )
    # 设置 TAVILY_API_KEY
    # 创建tavily工具
    search = TavilySearchResults()
    # 创建我们想要使用的工具的列表
    tools = [retriever_tool, search]

    # Get the prompts to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke({"input": "how can langsmith help with testing?"})
    agent_executor.invoke({"input": "阜阳市后面两天天气情况如何"})


# 程序入口
if __name__ == '__main__':
    simple()
