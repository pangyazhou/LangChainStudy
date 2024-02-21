"""
本程序使用OpenAI提供的OpenAIEmbeddings嵌入模型作为演示示例
"""
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()


# 文档嵌入模型
def embedding_documents_invoke():
    # 嵌入文档
    embeddings = embeddings_model.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ]
    )
    print(len(embeddings), len(embeddings[0]))
    for embed in embeddings:
        print(embed)


# 嵌入模型查询
def embedding_query_invoke():
    embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
    print(embedded_query[:5])


if __name__ == '__main__':
    # embedding_documents_invoke()
    embedding_query_invoke()
    pass