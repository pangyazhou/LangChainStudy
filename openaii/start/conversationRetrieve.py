"""
对话检索功能模块示例
1.检索方法现在不应仅适用于最近的输入，而应考虑整个历史记录。
2.最终的 LLM 链同样应该考虑整个历史
"""
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

llm = ChatOpenAI(openai_api_key="sk-xxx")


# 简单示例
def simple():
    ## 检索功能
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

    # 为了更新检索，我们将创建一个新链。该链将接受最近的输入 ( input) 和对话历史记录 ( chat_history) 并使用 LLM 生成搜索查询。
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to "
         "the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    # 我们可以通过传入用户提出后续问题的实例来测试这一点。
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    text = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(text)
    # 我们可以创建一个新的链来继续与这些检索到的文档进行对话
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    # 端到端地测试
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    text = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(text)

# 程序入口
if __name__ == '__main__':
    simple()