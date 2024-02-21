"""
缓存嵌入可以使用CacheBackedEmbeddings,
支持缓存的嵌入器是嵌入器的包装器，它将嵌入缓存在键值存储中。
对文本进行哈希处理，并将哈希值用作缓存中的密钥。
"""
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore, InMemoryByteStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# 使用本地文件存储向量数据
def cache_with_vector_store_invoke():
    underlying_embeddings = OpenAIEmbeddings()
    # 使用本地文件环境
    store = LocalFileStore("./cache/")
    # 使用内存缓存
    store = InMemoryByteStore()
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )
    # 嵌入之前为空
    print(list(store.yield_keys()))

    # 加载本地文件，嵌入每个块并存储到向量库
    raw_documents = TextLoader("example_data/deployment.md").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, cached_embedder)
    # print(list(store.yield_keys()))
    print(list(store.yield_keys())[:5])


if __name__ == '__main__':
    cache_with_vector_store_invoke()
    pass