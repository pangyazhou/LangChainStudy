"""
根据语义相似性分割文本
在较高层次上，它会分成句子，然后分成 3 个句子为一组，然后合并嵌入空间中相似的句子。
"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


def semantic_chunking_splitter_invoke():
    # This is a long document we can split up.
    with open("example_data/state_of_union.txt") as f:
        state_of_the_union = f.read()
    # 使用嵌入模型计算语义相似性
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    docs = text_splitter.create_documents([state_of_the_union])
    for doc in docs:
        print(doc)


# 程序入口
if __name__ == "__main__":
    semantic_chunking_splitter_invoke()
    pass