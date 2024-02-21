"""
字符递归分割器
它由字符列表参数化。它尝试按顺序分割它们，直到块足够小。
默认列表是 ["\n\n", "\n", " ", ""].
这样做的效果是尝试将所有段落（然后是句子，然后是单词）尽可能长时间地放在一起，因为这些通常看起来是语义相关性最强的文本片段。
1. 文本如何分割：按字符列表。
2. 如何测量块大小：按字符数。
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter


def recursive_character_text_splitter_invoke():
    # This is a long document we can split up.
    with open("example_data/state_of_union.txt") as f:
        state_of_the_union = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([state_of_the_union])
    for text in texts:
        print(text)


# 程序入口
if __name__ == "__main__":
    recursive_character_text_splitter_invoke()
    pass