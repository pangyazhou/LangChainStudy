"""
检索功能模块示例
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm = ChatOpenAI(openai_api_key="xxxx")


def simple():
    # 加载检索数据
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    # 嵌入模型
    embeddings = OpenAIEmbeddings(openai_api_key="xxxx")
    # 创建索引
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    # 创建查询链
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>

    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 创建检索器
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 调用链
    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])

    # LangSmith offers several features that can help with testing:...

def demo():
    output_parser = StrOutputParser()
    chain = llm | output_parser
    text = chain.invoke("langsmith如何帮助我们测试")

    print(text)

if __name__ == '__main__':
    simple()
    # demo()
    pass