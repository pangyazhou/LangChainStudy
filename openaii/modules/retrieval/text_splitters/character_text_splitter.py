"""
基于字符（默认为“”）进行分割，并通过字符数来测量块长度。
"""
from langchain.text_splitter import CharacterTextSplitter


def character_splitter_invoke():
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    # This is a long document we can split up.
    with open("example_data/deployment.md") as f:
        state_of_the_union = f.read()
    texts = text_splitter.create_documents([state_of_the_union])
    # for doc in texts:
    #     print(doc)

    # 传递元数据
    with open("example_data/deploy.md") as f:
        state_of_the_union2 = f.read()
    metadatas = [{"document": 1}, {"document": 2}]
    documents = text_splitter.create_documents(
        [state_of_the_union, state_of_the_union2], metadatas=metadatas
    )
    for doc in documents:
        print(doc)


# 程序入口
if __name__ == "__main__":
    character_splitter_invoke()
    pass