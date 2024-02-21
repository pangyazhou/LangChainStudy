"""
构建一个具有两种工具的代理：一种用于在线查找，另一种用于查找我们加载到索引中的特定数据。
"""
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# 演示使用tavily工具在线查询
def tavily_search_demo_invoke():
    search = TavilySearchResults()
    result = search.invoke("what is the weather in SF")
    print(result)


# 使用检索器查询
def retriever_invoke():
    retriever = retriever_build()
    # 6.检索向量库
    result = retriever.get_relevant_documents("how to upload a dataset")
    for doc in result:
        print(doc.metadata)


# 根据自定义数据创建检索器
def retriever_build() -> VectorStoreRetriever:
    # 1.加载文档
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    # 2.分割文档
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(docs)
    # 3.引入嵌入模型
    embedding = OpenAIEmbeddings()
    # 4.插入向量数据库
    vector = FAISS.from_documents(documents, embedding)
    # 5.构建检索器
    retriever = vector.as_retriever()
    return retriever


# 将在线查询工具与检索器组合为检索工具
def agent_executor_build() -> AgentExecutor:
    retriever = retriever_build()
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )
    # 在线查询工具
    search = TavilySearchResults()
    # 构建工具集合
    tools = [search, retriever_tool]
    # 创建提示词
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # print(prompt)
    # 使用 LLM、提示符和工具来初始化代理。代理负责接收输入并决定采取什么操作
    agent = create_openai_functions_agent(llm, tools, prompt)
    # 关联代理agent与内部工具AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def normal_search_invoke():
    agent_executor = agent_executor_build()
    # 使用代理工具执行查询
    result = agent_executor.invoke({"input": "hi!"})
    print(result)


def retriever_search_invoke():
    agent_executor = agent_executor_build()
    # 查询自定义数据相关内容
    result = agent_executor.invoke({"input": "how can langsmith help with testing?"})
    print(result)


def tavily_search_invoke():
    agent_executor = agent_executor_build()
    # 查询在线内容
    result = agent_executor.invoke({"input": "whats the weather in sf?"})
    print(result)


# 有记忆查询
def history_in_memory_search_invoke():
    agent_executor = agent_executor_build()
    # Here we pass in an empty list of messages for chat_history because it is the first message in the chat
    result = agent_executor.invoke({"input": "hi! my name is bob", "chat_history": []})
    print(result)
    result = agent_executor.invoke(
        {
            "chat_history": [
                HumanMessage(content="hi! my name is bob"),
                AIMessage(content="Hello Bob! How can I assist you today?"),
            ],
            "input": "what's my name?",
        }
    )
    print(result)


# 自动跟踪历史记录
def auto_history_search_invoke():
    # 代理执行器
    agent_executor = agent_executor_build()
    # 消息历史对象
    message_history = ChatMessageHistory()
    # 构建自动跟踪历史记录的代理执行器
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    result = agent_with_chat_history.invoke(
        {"input": "hi! I'm bob"},
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        config={"configurable": {"session_id": "<foo>"}},
    )
    print(result)
    result = agent_with_chat_history.invoke(
        {"input": "what's my name?"},
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        config={"configurable": {"session_id": "<foo>"}},
    )
    print(result)


if __name__ == "__main__":
    # tavily_search_demo_invoke()
    # retriever_invoke()
    # retriever_search_invoke()
    # history_in_memory_search_invoke()
    auto_history_search_invoke()
    pass