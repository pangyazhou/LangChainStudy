"""
How to cache ChatModel responses

LangChain为聊天模型提供了可选的缓存层。这很有用，原因有两个：
如果您经常多次请求相同的完成，它可以通过减少您对 LLM 提供商的 API 调用次数来节省资金。
它可以通过减少您对 LLM 提供商的 API 调用次数来加快您的申请速度。
"""
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache

model = ChatOpenAI()

# 在内存中缓存
def cache_in_memory_invoke():
    set_llm_cache(InMemoryCache())
    print(model.invoke("给我讲一个关于中国国足队员傅欢的笑话"))
    print("=============================")
    print(model.invoke("给我讲一个关于中国国足队员傅欢的笑话"))
    pass


# 在数据库中缓存
def cache_in_sql_invoke():
    set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    print(model.invoke("给我讲一个关于中国国足队员傅欢的笑话"))
    print("=============================")
    print(model.invoke("给我讲一个关于中国国足队员傅欢的笑话"))


# 程序入口
if __name__ == "__main__":
    # cache_in_memory_invoke()
    cache_in_sql_invoke()
    pass