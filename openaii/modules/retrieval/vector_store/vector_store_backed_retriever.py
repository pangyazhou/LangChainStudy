"""
矢量存储检索器是使用矢量存储来检索文档的检索器。
它是矢量存储类的轻量级包装器，使其符合检索器接口。
它使用向量存储实现的搜索方法（例如相似性搜索和 MMR）来查询向量存储中的文本。
本程序演示了基础的向量存储检索器的使用
"""
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def vector_store_invoke():
    loader = TextLoader("../example_data/state_of_union.txt")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    # 相关查询
    docs = retriever.get_relevant_documents("tell me what is usa")
    print(docs)
    # 最大边际相关性查询
    retriever = db.as_retriever(search_type="mmr")
    docs = retriever.get_relevant_documents("tell me what is usa")
    print(docs)
    # 相似度阈值
    retriever = db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )
    docs = retriever.get_relevant_documents("tell me what is usa")
    print(docs)
    # 指定前k个相似选项
    retriever = db.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents("tell me what is usa")
    print(docs)


if __name__ == '__main__':
    vector_store_invoke()
    pass