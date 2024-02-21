"""
最简单的加载程序将文件作为文本读入，并将其全部放入一个文档中。
"""
from langchain_community.document_loaders import TextLoader


def text_loader_invoke():
    loader = TextLoader("example_data/deployment.md")
    doc = loader.load()
    print(doc)


# 程序入口
if __name__ == "__main__":
    text_loader_invoke()
    pass