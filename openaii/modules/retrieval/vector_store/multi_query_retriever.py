"""
TODO 搞不懂啥意思
"""
# Build a sample vectorDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
import logging


def multi_query_retriever_invoke():
    # Load blog post
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # VectorDB
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    question = "What are the approaches to Task Decomposition?"
    llm = ChatOpenAI(temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )

    # Set logging for the queries
    logging.basicConfig()
    logging.getLogger("openaii/modules/retrieval/vector_store").setLevel(logging.INFO)

    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    len(unique_docs)
    print(unique_docs)


if __name__ == '__main__':
    multi_query_retriever_invoke()
    pass