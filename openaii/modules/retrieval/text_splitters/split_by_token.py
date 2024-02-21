"""
语言模型有一个令牌限制。您不应超出令牌限制。
因此，当您将文本拆分为块时，最好计算标记的数量。
有很多标记器。当您计算文本中的标记时，您应该使用与语言模型中使用的相同的标记生成器。
"""
from langchain.text_splitter import CharacterTextSplitter


# tiktoken 快速分词器
def tiktoken_splitter_invoke():
    # This is a long document we can split up.
    with open("example_data/state_of_union.txt") as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=0
    )
    texts = text_splitter.split_text(state_of_the_union)
    for text in texts:
        print(text)

    from langchain.text_splitter import TokenTextSplitter

    text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

    texts = text_splitter.split_text(state_of_the_union)
    for text in texts:
        print(text)


# 程序入口
if __name__ == "__main__":
    tiktoken_splitter_invoke()
    pass