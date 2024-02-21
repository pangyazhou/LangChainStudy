"""
本程序演示如何使用本地存储的向量库执行嵌入文本的存储与搜索
开源、免费、本地部署的向量数据库
Chroma, Faiss, Lance
"""
import asyncio

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant


def vector_store_invoke():
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    raw_documents = TextLoader("../example_data/anthropic.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    # 相似性搜索
    query = "How to use a language model?"
    docs = db.similarity_search(query)
    for doc in docs:
        print(doc)

    # 使用向量作为查询参数
    embedding_vector = OpenAIEmbeddings().embed_query(query)
    print(embedding_vector)
    docs = db.similarity_search_by_vector(embedding_vector)
    print(docs[0].page_content)


# 异步向量库操作 Qdrant
# todo 调用失败，后续调研
async def async_vector_store_invoke():
    documents = TextLoader("../example_data/anthropic.txt").load()
    db = await Qdrant.afrom_documents(documents, OpenAIEmbeddings(), "http://localhost:6333")

    query = "How to use a language model?"
    docs = await db.asimilarity_search(query)
    print(docs[0].page_content)
    pass


if __name__ == '__main__':
    # vector_store_invoke()
    asyncio.run(async_vector_store_invoke())
    pass